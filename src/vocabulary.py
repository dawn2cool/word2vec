"""
vocabulary.py
=============
Builds and manages the vocabulary from a tokenised corpus.

Key responsibilities:
  - Count word frequencies
  - Filter rare words (min_count threshold)
  - Build word <-> index mappings
  - Compute the noise distribution used for negative sampling
      P_noise(w) ∝ freq(w)^(3/4)   (Mikolov et al., 2013)
  - Subsample frequent words during corpus iteration
      P_discard(w) = 1 - sqrt(t / freq_ratio(w))
"""

from __future__ import annotations

import re
import collections
import numpy as np
from typing import List, Dict, Tuple, Optional


class Vocabulary:
    """Stores the word-to-index mapping and all frequency-derived quantities."""

    def __init__(
        self,
        min_count: int = 5,
        subsample_threshold: float = 1e-3,
        noise_exponent: float = 0.75,
        unk_token: str = "<UNK>",
    ) -> None:
        self.min_count = min_count
        self.subsample_threshold = subsample_threshold
        self.noise_exponent = noise_exponent
        self.unk_token = unk_token

        # Populated by build()
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Dict[str, int] = {}
        self.vocab_size: int = 0
        self.total_words: int = 0          # sum of all counts (after min_count filter)
        self._noise_table: Optional[np.ndarray] = None   # cached table
        self._subsample_probs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, tokens: List[str]) -> "Vocabulary":
        """
        Build vocabulary from a flat list of tokens.

        Parameters
        ----------
        tokens : list of str
            The full training corpus as a flat token list.

        Returns
        -------
        self (for chaining)
        """
        raw_counts = collections.Counter(tokens)

        # Filter by minimum frequency
        filtered = {w: c for w, c in raw_counts.items() if c >= self.min_count}

        # Sort by descending frequency for reproducibility
        sorted_words = sorted(filtered.items(), key=lambda x: -x[1])

        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = {}

        for idx, (word, count) in enumerate(sorted_words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_counts[word] = count

        self.vocab_size = len(self.word2idx)
        self.total_words = sum(self.word_counts.values())

        # Pre-compute derived tables
        self._build_noise_table()
        self._build_subsample_probs()

        return self

    # ------------------------------------------------------------------
    # Noise distribution for negative sampling
    # ------------------------------------------------------------------

    def _build_noise_table(self, table_size: int = 1_000_000) -> None:
        """
        Construct a large alias / flat table for O(1) negative sampling.

        P_noise(w) ∝ count(w)^noise_exponent

        We fill an integer array of length `table_size` with word indices
        proportional to their noise probability.  Sampling a negative word
        is then just picking a random position in this table — extremely fast.
        """
        counts = np.array(
            [self.word_counts[self.idx2word[i]] for i in range(self.vocab_size)],
            dtype=np.float64,
        )
        powered = counts ** self.noise_exponent
        probs = powered / powered.sum()

        # Fill the table
        table = np.zeros(table_size, dtype=np.int32)
        cumulative = 0.0
        j = 0
        for i, p in enumerate(probs):
            cumulative += p * table_size
            while j < cumulative and j < table_size:
                table[j] = i
                j += 1
        self._noise_table = table

    def sample_negatives(self, n: int, exclude: Optional[set] = None) -> np.ndarray:
        """
        Draw `n` negative sample indices from the noise distribution.

        Parameters
        ----------
        n       : number of samples
        exclude : set of indices to never return (e.g. the positive word)
        """
        if self._noise_table is None:
            raise RuntimeError("Vocabulary not built yet.")
        indices = np.random.randint(0, len(self._noise_table), size=n * 2)
        samples = self._noise_table[indices]
        if exclude:
            mask = np.array([s not in exclude for s in samples])
            samples = samples[mask]
        return samples[:n]

    # ------------------------------------------------------------------
    # Subsampling
    # ------------------------------------------------------------------

    def _build_subsample_probs(self) -> None:
        """
        P_keep(w) = min(1, sqrt(t / f(w)) + t / f(w))
        where f(w) = count(w) / total_words,  t = subsample_threshold.

        Words with higher frequency are discarded more aggressively.
        """
        probs = np.zeros(self.vocab_size, dtype=np.float32)
        t = self.subsample_threshold
        for i in range(self.vocab_size):
            word = self.idx2word[i]
            f = self.word_counts[word] / self.total_words
            # Formula from the original paper (Mikolov 2013, Eq. 6)
            keep = min(1.0, (np.sqrt(f / t) + 1.0) * (t / f))
            probs[i] = keep
        self._subsample_probs = probs

    def subsample_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Return a filtered list of token IDs after applying subsampling.
        Each token is kept with probability P_keep(w).
        """
        if self._subsample_probs is None:
            return token_ids
        keep = np.random.random(len(token_ids)) < self._subsample_probs[token_ids]
        return [tid for tid, k in zip(token_ids, keep) if k]

    # ------------------------------------------------------------------
    # Corpus encoding helpers
    # ------------------------------------------------------------------

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices, skipping OOV words."""
        return [self.word2idx[t] for t in tokens if t in self.word2idx]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2word[i] for i in indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import json, os
        data = {
            "word2idx": self.word2idx,
            "word_counts": self.word_counts,
            "min_count": self.min_count,
            "subsample_threshold": self.subsample_threshold,
            "noise_exponent": self.noise_exponent,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        import json
        with open(path) as f:
            data = json.load(f)
        vocab = cls(
            min_count=data["min_count"],
            subsample_threshold=data["subsample_threshold"],
            noise_exponent=data["noise_exponent"],
        )
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        vocab.word_counts = data["word_counts"]
        vocab.vocab_size = len(vocab.word2idx)
        vocab.total_words = sum(vocab.word_counts.values())
        vocab._build_noise_table()
        vocab._build_subsample_probs()
        return vocab

    def __len__(self) -> int:
        return self.vocab_size

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __repr__(self) -> str:
        return (
            f"Vocabulary(size={self.vocab_size:,}, "
            f"total_tokens={self.total_words:,}, "
            f"min_count={self.min_count})"
        )
