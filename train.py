#!/usr/bin/env python3
"""
train.py
========
Command-line interface for training Word2Vec.

Usage examples
--------------
# Quick smoke test on the built-in sample corpus:
    python train.py --corpus sample

# Train on text8 (downloads automatically, ~100 MB):
    python train.py --corpus text8 --dim 100 --epochs 5

# Train on a custom file:
    python train.py --corpus path/to/file.txt --dim 200 --epochs 3 --model cbow

# Full option list:
    python train.py --help
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.vocabulary import Vocabulary
from src.word2vec import Word2Vec
from src.trainer import Trainer
from src.preprocessing import load_text_file, load_text8, load_sample_corpus
from evaluate import full_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Word2Vec with pure NumPy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument(
        "--corpus", type=str, default="sample",
        help="'sample' | 'text8' | path to a .txt file",
    )
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--max_tokens", type=int, default=None)

    # Model
    p.add_argument("--model", type=str, default="skipgram", choices=["skipgram", "cbow"])
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--neg", type=int, default=5)
    p.add_argument("--window", type=int, default=5)

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.025)
    p.add_argument("--lr_min", type=float, default=0.0001)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--subsample", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--out", type=str, default="results/model")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--log_every", type=int, default=10_000)
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--plot", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # ── Load corpus ──────────────────────────────────────────────────────────
    print(f"\nLoading corpus: '{args.corpus}' ...")
    if args.corpus == "sample":
        tokens = load_sample_corpus()
        log_every = 500
    elif args.corpus == "text8":
        tokens = load_text8(dest_dir=args.data_dir, max_tokens=args.max_tokens)
        log_every = args.log_every
    else:
        if not os.path.isfile(args.corpus):
            sys.exit(f"File not found: {args.corpus}")
        tokens = load_text_file(args.corpus, max_tokens=args.max_tokens)
        log_every = args.log_every

    print(f"  Loaded {len(tokens):,} tokens.")

    # ── Build vocabulary ─────────────────────────────────────────────────────
    min_count = 2 if args.corpus == "sample" else args.min_count
    vocab = Vocabulary(
        min_count=min_count,
        subsample_threshold=args.subsample,
    ).build(tokens)
    print(vocab)

    # ── Encode corpus ────────────────────────────────────────────────────────
    corpus = vocab.encode(tokens)
    print(f"  Encoded corpus: {len(corpus):,} tokens (after OOV removal).")

    # ── Build model ──────────────────────────────────────────────────────────
    model = Word2Vec(
        vocab=vocab,
        embed_dim=args.dim,
        n_negatives=args.neg,
        model_type=args.model,
        seed=args.seed,
    )
    print(model)

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        vocab=vocab,
        corpus=corpus,
        n_epochs=args.epochs,
        lr_start=args.lr,
        lr_min=args.lr_min,
        window_size=args.window,
        batch_size=args.batch_size,
        log_every=log_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    vocab.save(args.out + "_vocab.json")
    trainer.save_loss_history("results/loss_history.csv")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    if args.evaluate:
        full_report(model, plot=args.plot, loss_history=trainer.loss_history)


if __name__ == "__main__":
    main()
