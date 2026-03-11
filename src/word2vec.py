"""
word2vec.py
===========
Pure-NumPy implementation of Word2Vec.

Implements two variants:
  1. Skip-gram with Negative Sampling (SGNS)  ← default, recommended
  2. CBOW with Negative Sampling

Architecture
------------
Two embedding matrices are maintained:

  W_in  : (vocab_size, embed_dim)  — "input" / center-word embeddings
  W_out : (vocab_size, embed_dim)  — "output" / context-word embeddings

The final word vectors are typically taken from W_in (or averaged with W_out).

Loss Function — Negative Sampling
----------------------------------
For a positive pair (center c, context o) with k negative samples {n_1 .. n_k}:

  L = -log σ(v_o · u_c)  -  Σ_{i=1}^{k} log σ(-v_{n_i} · u_c)

where σ(x) = 1 / (1 + e^{-x}) is the sigmoid function,
      u_c  is the row of W_in  for center word c,
      v_o  is the row of W_out for context word o,
      v_n  is the row of W_out for each negative word.

Gradient Derivations
---------------------
Let s_o = v_o · u_c  and  s_i = v_{n_i} · u_c.

∂L/∂u_c  =  (σ(s_o) - 1) · v_o  +  Σ_i  σ(s_i) · v_{n_i}
                ↑ error on positive        ↑ error on negatives

∂L/∂v_o  =  (σ(s_o) - 1) · u_c

∂L/∂v_{n_i}  =  σ(s_i) · u_c

These are applied via SGD:  param -= lr * gradient

Numerical stability: sigmoid is clamped to avoid log(0).
"""

from __future__ import annotations

from typing import Tuple, Optional, Literal
import numpy as np

from .vocabulary import Vocabulary


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: σ(x) = 1 / (1 + e^{-x})."""
    # Clip to avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class Word2Vec:
    """
    Word2Vec model (Skip-gram or CBOW) trained with Negative Sampling.

    Parameters
    ----------
    vocab : Vocabulary
        Fully built vocabulary object.
    embed_dim : int
        Dimensionality of word vectors (typical: 50–300).
    n_negatives : int
        Number of negative samples per positive pair (typical: 5–20).
    model_type : 'skipgram' | 'cbow'
        Which variant to use.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embed_dim: int = 100,
        n_negatives: int = 5,
        model_type: Literal["skipgram", "cbow"] = "skipgram",
        seed: int = 42,
    ) -> None:
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.n_negatives = n_negatives
        self.model_type = model_type

        rng = np.random.default_rng(seed)

        # Initialise embeddings with small uniform values (standard practice)
        # Shape: (vocab_size, embed_dim)
        bound = 0.5 / embed_dim
        self.W_in  = rng.uniform(-bound, bound, (vocab.vocab_size, embed_dim)).astype(np.float32)
        self.W_out = np.zeros((vocab.vocab_size, embed_dim), dtype=np.float32)

        # Running loss for logging
        self.total_loss: float = 0.0
        self.n_updates: int = 0

    # ------------------------------------------------------------------
    # Forward + backward for a single (center, context) pair — Skip-gram
    # ------------------------------------------------------------------

    def _sgns_step(
        self,
        center_idx: int,
        context_idx: int,
        lr: float,
    ) -> float:
        """
        Perform one forward + backward + update step for SGNS.

        Returns
        -------
        loss : float
            The negative-sampling loss for this pair.
        """
        # Sample negative indices (exclude the positive context word)
        neg_indices = self.vocab.sample_negatives(
            self.n_negatives, exclude={context_idx}
        )

        # --- Forward pass ---
        u_c = self.W_in[center_idx]          # shape: (D,)
        v_o = self.W_out[context_idx]        # shape: (D,)
        v_negs = self.W_out[neg_indices]     # shape: (k, D)

        s_pos = np.dot(v_o, u_c)             # scalar: positive score
        s_neg = v_negs @ u_c                 # shape: (k,)  negative scores

        sig_pos = sigmoid(s_pos)             # σ(v_o · u_c)
        sig_neg = sigmoid(s_neg)             # σ(v_{n_i} · u_c)  for all i

        # --- Loss ---
        # L = -log σ(s_pos) - Σ log σ(-s_neg)
        #   = -log σ(s_pos) - Σ log (1 - σ(s_neg))
        eps = 1e-7
        loss = (
            -np.log(sig_pos + eps)
            - np.sum(np.log(1.0 - sig_neg + eps))
        )

        # --- Gradients ---
        # Error signals (scalars / vectors):
        err_pos = sig_pos - 1.0              # ∂L/∂s_pos  = σ(s_pos) - 1
        err_neg = sig_neg                    # ∂L/∂s_{n_i} = σ(s_{n_i})

        # Gradient w.r.t. u_c (center embedding):
        #   ∂L/∂u_c = err_pos * v_o + Σ_i err_neg_i * v_{n_i}
        grad_u_c = err_pos * v_o + err_neg @ v_negs   # shape: (D,)

        # Gradient w.r.t. v_o (positive context embedding):
        #   ∂L/∂v_o = err_pos * u_c
        grad_v_o = err_pos * u_c                        # shape: (D,)

        # Gradient w.r.t. each v_{n_i} (negative context embeddings):
        #   ∂L/∂v_{n_i} = err_neg_i * u_c
        grad_v_negs = np.outer(err_neg, u_c)            # shape: (k, D)

        # --- SGD updates ---
        self.W_in[center_idx]    -= lr * grad_u_c
        self.W_out[context_idx]  -= lr * grad_v_o

        # Accumulate gradients for negative samples
        # (multiple negatives may coincide — np.add.at handles this correctly)
        np.add.at(self.W_out, neg_indices, -lr * grad_v_negs)

        return float(loss)

    # ------------------------------------------------------------------
    # Batched Skip-gram step  (more cache-friendly for larger batches)
    # ------------------------------------------------------------------

    def sgns_batch_step(
        self,
        center_ids: np.ndarray,
        context_ids: np.ndarray,
        lr: float,
    ) -> float:
        """
        Process a batch of (center, context) pairs.

        Parameters
        ----------
        center_ids  : int array, shape (B,)
        context_ids : int array, shape (B,)
        lr          : current learning rate

        Returns
        -------
        mean_loss : float
        """
        B = len(center_ids)
        k = self.n_negatives

        # Sample all negatives at once: shape (B, k)
        neg_ids = np.array(
            [self.vocab.sample_negatives(k) for _ in range(B)],
            dtype=np.int32,
        )  # (B, k)

        # Gather embeddings
        U = self.W_in[center_ids]           # (B, D)
        V_pos = self.W_out[context_ids]     # (B, D)
        V_neg = self.W_out[neg_ids]         # (B, k, D)

        # Scores
        # s_pos[b] = dot(V_pos[b], U[b])
        s_pos = np.einsum("bd,bd->b", V_pos, U)            # (B,)
        # s_neg[b, j] = dot(V_neg[b,j], U[b])
        s_neg = np.einsum("bkd,bd->bk", V_neg, U)          # (B, k)

        sig_pos = sigmoid(s_pos)                            # (B,)
        sig_neg = sigmoid(s_neg)                            # (B, k)

        # Loss (mean over batch)
        eps = 1e-7
        loss = (
            -np.mean(np.log(sig_pos + eps))
            - np.mean(np.sum(np.log(1.0 - sig_neg + eps), axis=1))
        )

        # Gradients
        err_pos = (sig_pos - 1.0)[:, None]                 # (B, 1)
        err_neg = sig_neg                                   # (B, k)

        # ∂L/∂U[b] = err_pos[b]*V_pos[b] + Σ_j err_neg[b,j]*V_neg[b,j]
        grad_U = err_pos * V_pos + np.einsum("bk,bkd->bd", err_neg, V_neg)

        # ∂L/∂V_pos[b] = err_pos[b] * U[b]
        grad_V_pos = err_pos * U                            # (B, D)

        # ∂L/∂V_neg[b,j] = err_neg[b,j] * U[b]
        grad_V_neg = np.einsum("bk,bd->bkd", err_neg, U)   # (B, k, D)

        # SGD updates — one update per example (matching the per-pair original)
        # We do NOT divide by B: np.add.at accumulates per example correctly.
        np.add.at(self.W_in,  center_ids,  -lr * grad_U)
        np.add.at(self.W_out, context_ids, -lr * grad_V_pos)

        # Flatten neg_ids and grad_V_neg for scatter update
        flat_neg  = neg_ids.reshape(-1)                     # (B*k,)
        flat_grad = grad_V_neg.reshape(-1, self.embed_dim)  # (B*k, D)
        np.add.at(self.W_out, flat_neg, -lr * flat_grad)

        return float(loss)

    # ------------------------------------------------------------------
    # CBOW step
    # ------------------------------------------------------------------

    def cbow_step(
        self,
        context_ids: list,
        target_idx: int,
        lr: float,
    ) -> float:
        """
        One CBOW forward + backward + update.

        The "input" vector is the mean of the context word embeddings.
        The "output" is the target word embedding in W_out.

        Returns
        -------
        loss : float
        """
        neg_indices = self.vocab.sample_negatives(
            self.n_negatives, exclude={target_idx}
        )
        ctx = np.array(context_ids, dtype=np.int32)

        # Compute context mean (h)
        h = self.W_in[ctx].mean(axis=0)     # shape: (D,)

        v_t = self.W_out[target_idx]
        v_negs = self.W_out[neg_indices]

        s_pos = np.dot(v_t, h)
        s_neg = v_negs @ h

        sig_pos = sigmoid(s_pos)
        sig_neg = sigmoid(s_neg)

        eps = 1e-7
        loss = (
            -np.log(sig_pos + eps)
            - np.sum(np.log(1.0 - sig_neg + eps))
        )

        err_pos = sig_pos - 1.0
        err_neg = sig_neg

        # Gradient w.r.t. h
        grad_h = err_pos * v_t + err_neg @ v_negs       # (D,)

        # Distribute grad_h equally to all context words
        grad_ctx = grad_h / len(context_ids)             # (D,)

        grad_v_t = err_pos * h
        grad_v_negs = np.outer(err_neg, h)

        np.add.at(self.W_in, ctx, -lr * grad_ctx)
        self.W_out[target_idx] -= lr * grad_v_t
        np.add.at(self.W_out, neg_indices, -lr * grad_v_negs)

        return float(loss)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Return the W_in embedding for a word, or None if OOV."""
        if word not in self.vocab:
            return None
        return self.W_in[self.vocab.word2idx[word]].copy()

    def get_embeddings(self) -> np.ndarray:
        """Return a copy of the full W_in matrix."""
        return self.W_in.copy()

    def most_similar(
        self,
        word: str,
        top_n: int = 10,
    ) -> list:
        """
        Find the top_n most similar words by cosine similarity.

        Returns
        -------
        list of (word, similarity) tuples
        """
        if word not in self.vocab:
            return []

        query = self.W_in[self.vocab.word2idx[word]]
        # Normalise all vectors
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-9
        normed = self.W_in / norms
        query_norm = query / (np.linalg.norm(query) + 1e-9)

        similarities = normed @ query_norm
        similarities[self.vocab.word2idx[word]] = -1.0   # exclude self

        top_ids = np.argsort(-similarities)[:top_n]
        return [(self.vocab.idx2word[i], float(similarities[i])) for i in top_ids]

    def analogy(
        self,
        pos1: str,
        neg1: str,
        pos2: str,
        top_n: int = 5,
    ) -> list:
        """
        Compute   pos1 - neg1 + pos2   and return the nearest words.

        Example:  analogy('king', 'man', 'woman')  →  'queen'
        """
        words = [pos1, neg1, pos2]
        for w in words:
            if w not in self.vocab:
                return []

        vecs = [self.W_in[self.vocab.word2idx[w]] for w in words]
        target = vecs[0] - vecs[1] + vecs[2]

        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-9
        normed = self.W_in / norms
        target_norm = target / (np.linalg.norm(target) + 1e-9)

        similarities = normed @ target_norm
        # Exclude input words
        for w in words:
            similarities[self.vocab.word2idx[w]] = -1.0

        top_ids = np.argsort(-similarities)[:top_n]
        return [(self.vocab.idx2word[i], float(similarities[i])) for i in top_ids]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to a compressed .npz file."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            W_in=self.W_in,
            W_out=self.W_out,
            embed_dim=np.array(self.embed_dim),
            n_negatives=np.array(self.n_negatives),
        )
        print(f"Model saved to {path}.npz")

    @classmethod
    def load(cls, path: str, vocab: Vocabulary) -> "Word2Vec":
        """Load model weights from a .npz file."""
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        model = cls(
            vocab=vocab,
            embed_dim=int(data["embed_dim"]),
            n_negatives=int(data["n_negatives"]),
        )
        model.W_in  = data["W_in"]
        model.W_out = data["W_out"]
        return model

    def __repr__(self) -> str:
        return (
            f"Word2Vec(type={self.model_type}, "
            f"vocab={self.vocab.vocab_size:,}, "
            f"dim={self.embed_dim}, "
            f"neg={self.n_negatives})"
        )
