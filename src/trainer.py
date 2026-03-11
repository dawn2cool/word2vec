"""
trainer.py
==========
Training orchestration for Word2Vec.

Handles:
  - Epoch / step loop with progress reporting
  - Linear learning-rate decay (standard in word2vec)
  - Periodic checkpoint saving
  - Loss history tracking for plotting
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import numpy as np

from src.vocabulary import Vocabulary
from src.word2vec import Word2Vec
from src.preprocessing import (
    batch_skipgram_pairs,
    generate_cbow_pairs,
    load_text8,
)


class Trainer:
    """
    Training manager for Word2Vec (SGNS or CBOW).

    Parameters
    ----------
    model       : Word2Vec
    vocab       : Vocabulary
    corpus      : pre-encoded token list (after subsampling)
    n_epochs    : number of full passes over the corpus
    lr_start    : initial learning rate (0.025 is standard)
    lr_min      : minimum learning rate floor (lr_start / 100 is common)
    window_size : context window radius
    batch_size  : pairs per gradient step (only used for SGNS batch mode)
    log_every   : print loss every N steps
    save_every  : save checkpoint every N steps (0 = no checkpoints)
    checkpoint_dir : directory to save checkpoints
    """

    def __init__(
        self,
        model: Word2Vec,
        vocab: Vocabulary,
        corpus: List[int],
        n_epochs: int = 5,
        lr_start: float = 0.025,
        lr_min: float = 0.0001,
        window_size: int = 5,
        batch_size: int = 512,
        log_every: int = 10_000,
        save_every: int = 0,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.corpus = corpus
        self.n_epochs = n_epochs
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.window_size = window_size
        self.batch_size = batch_size
        self.log_every = log_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir

        self.loss_history: List[Tuple[int, float]] = []   # (global_step, loss)
        self.global_step: int = 0

        # Estimate total steps for LR decay
        avg_pairs_per_token = window_size   # rough estimate
        self.total_steps = n_epochs * len(corpus) * avg_pairs_per_token // batch_size

    # ------------------------------------------------------------------
    # Learning rate schedule
    # ------------------------------------------------------------------

    def _get_lr(self) -> float:
        """
        Linear decay:  lr(t) = lr_start * (1 - t / total_steps)
        Clamped to lr_min.
        """
        fraction = self.global_step / max(self.total_steps, 1)
        lr = self.lr_start * (1.0 - fraction)
        return max(lr, self.lr_min)

    # ------------------------------------------------------------------
    # Main training entry points
    # ------------------------------------------------------------------

    def train(self) -> "Trainer":
        """Run training for the configured number of epochs."""
        if self.model.model_type == "skipgram":
            return self._train_sgns()
        else:
            return self._train_cbow()

    def _train_sgns(self) -> "Trainer":
        """SGNS training loop — processes pairs in batches."""
        print(
            f"Training Skip-gram with Negative Sampling\n"
            f"  vocab={self.vocab.vocab_size:,}  dim={self.model.embed_dim}"
            f"  neg={self.model.n_negatives}\n"
            f"  epochs={self.n_epochs}  lr={self.lr_start}→{self.lr_min}"
            f"  window={self.window_size}  batch={self.batch_size}\n"
            f"  corpus tokens={len(self.corpus):,}"
        )

        for epoch in range(1, self.n_epochs + 1):
            # Re-subsample at each epoch (as in the original word2vec)
            corpus_epoch = self.vocab.subsample_tokens(self.corpus)
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for centers, contexts in batch_skipgram_pairs(
                corpus_epoch,
                window_size=self.window_size,
                batch_size=self.batch_size,
            ):
                lr = self._get_lr()
                loss = self.model.sgns_batch_step(centers, contexts, lr)
                epoch_loss += loss
                n_batches += 1
                self.global_step += 1

                if self.log_every > 0 and self.global_step % self.log_every == 0:
                    avg_loss = epoch_loss / n_batches
                    self.loss_history.append((self.global_step, avg_loss))
                    print(
                        f"  epoch {epoch:2d}  step {self.global_step:7,}"
                        f"  lr={lr:.5f}  loss={avg_loss:.4f}"
                        f"  ({time.time()-t0:.1f}s)"
                    )

                if self.save_every > 0 and self.global_step % self.save_every == 0:
                    self._save_checkpoint(epoch)

            # End-of-epoch summary
            avg = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{self.n_epochs}  avg_loss={avg:.4f}  time={elapsed:.1f}s")

        print("Training complete.")
        return self

    def _train_cbow(self) -> "Trainer":
        """CBOW training loop — processes one target at a time."""
        print(
            f"Training CBOW with Negative Sampling\n"
            f"  vocab={self.vocab.vocab_size:,}  dim={self.model.embed_dim}"
            f"  neg={self.model.n_negatives}\n"
            f"  epochs={self.n_epochs}  lr={self.lr_start}→{self.lr_min}"
            f"  window={self.window_size}"
        )

        for epoch in range(1, self.n_epochs + 1):
            corpus_epoch = self.vocab.subsample_tokens(self.corpus)
            epoch_loss = 0.0
            n_steps = 0
            t0 = time.time()

            for context_ids, target_idx in generate_cbow_pairs(
                corpus_epoch, window_size=self.window_size
            ):
                lr = self._get_lr()
                loss = self.model.cbow_step(context_ids, target_idx, lr)
                epoch_loss += loss
                n_steps += 1
                self.global_step += 1

                if self.log_every > 0 and n_steps % self.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    self.loss_history.append((self.global_step, avg_loss))
                    print(
                        f"  epoch {epoch:2d}  step {n_steps:7,}"
                        f"  lr={lr:.5f}  loss={avg_loss:.4f}"
                    )

            avg = epoch_loss / max(n_steps, 1)
            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{self.n_epochs}  avg_loss={avg:.4f}  time={elapsed:.1f}s")

        print("Training complete.")
        return self

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.checkpoint_dir,
            f"word2vec_epoch{epoch}_step{self.global_step}",
        )
        self.model.save(path)

    def save_loss_history(self, path: str = "results/loss_history.csv") -> None:
        """Write the loss log to a CSV file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write("global_step,loss\n")
            for step, loss in self.loss_history:
                f.write(f"{step},{loss:.6f}\n")
        print(f"Loss history saved to {path}")


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def train_on_sample(
    embed_dim: int = 100,
    n_negatives: int = 5,
    n_epochs: int = 30,
    window_size: int = 5,
    batch_size: int = 256,
    model_type: str = "skipgram",
    lr_start: float = 0.025,
    checkpoint_dir: str = "checkpoints",
    log_every: int = 500,
) -> Tuple[Word2Vec, Vocabulary, Trainer]:
    """
    Quick-start: train on the built-in sample corpus.
    Useful for testing / debugging.
    """
    from src.preprocessing import load_sample_corpus
    from src.vocabulary import Vocabulary

    tokens = load_sample_corpus()
    vocab = Vocabulary(min_count=2, subsample_threshold=1e-3).build(tokens)
    corpus = vocab.encode(tokens)
    corpus = vocab.subsample_tokens(corpus)

    model = Word2Vec(
        vocab=vocab,
        embed_dim=embed_dim,
        n_negatives=n_negatives,
        model_type=model_type,
    )

    trainer = Trainer(
        model=model,
        vocab=vocab,
        corpus=corpus,
        n_epochs=n_epochs,
        lr_start=lr_start,
        window_size=window_size,
        batch_size=batch_size,
        log_every=log_every,
        save_every=0,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.train()
    return model, vocab, trainer


def train_on_text8(
    data_dir: str = "data",
    embed_dim: int = 100,
    n_negatives: int = 5,
    n_epochs: int = 5,
    window_size: int = 5,
    batch_size: int = 512,
    max_tokens: Optional[int] = None,
    lr_start: float = 0.025,
    checkpoint_dir: str = "checkpoints",
    log_every: int = 10_000,
) -> Tuple[Word2Vec, Vocabulary, Trainer]:
    """
    Full training run on the text8 Wikipedia corpus.
    """
    tokens = load_text8(dest_dir=data_dir, max_tokens=max_tokens)
    print(f"Loaded {len(tokens):,} tokens.")

    vocab = Vocabulary(min_count=5, subsample_threshold=1e-3).build(tokens)
    print(vocab)

    corpus = vocab.encode(tokens)
    model = Word2Vec(vocab=vocab, embed_dim=embed_dim, n_negatives=n_negatives)

    trainer = Trainer(
        model=model,
        vocab=vocab,
        corpus=corpus,
        n_epochs=n_epochs,
        lr_start=lr_start,
        window_size=window_size,
        batch_size=batch_size,
        log_every=log_every,
        save_every=100_000,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.train()
    return model, vocab, trainer
