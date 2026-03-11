"""
evaluate.py
===========
Evaluation tools for trained word2vec embeddings.

Includes:
  1. Cosine-similarity nearest-neighbour lookup
  2. Word analogy solver  (king - man + woman ≈ queen)
  3. Intrinsic evaluation on built-in analogy test set
  4. t-SNE visualisation of word clusters
  5. Embedding quality heuristics (coverage, norm stats)
"""

from __future__ import annotations

import os
from typing import List, Tuple, Dict, Optional

import numpy as np

from src.word2vec import Word2Vec
from src.vocabulary import Vocabulary


# ---------------------------------------------------------------------------
# 1. Nearest neighbours
# ---------------------------------------------------------------------------

def nearest_neighbours(
    model: Word2Vec,
    query: str,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """Return top_n words closest to `query` by cosine similarity."""
    return model.most_similar(query, top_n=top_n)


def batch_nearest_neighbours(
    model: Word2Vec,
    queries: List[str],
    top_n: int = 5,
) -> Dict[str, List[Tuple[str, float]]]:
    """Run nearest-neighbour search for multiple words at once."""
    return {w: model.most_similar(w, top_n=top_n) for w in queries if w in model.vocab}


# ---------------------------------------------------------------------------
# 2. Analogy evaluation
# ---------------------------------------------------------------------------

# Built-in mini analogy test set covering semantic & syntactic relationships
ANALOGY_TESTS = [
    # (A, B, C, expected_D)  — A is to B as C is to D
    # Semantic
    ("king",   "man",   "woman",    "queen"),
    ("paris",  "france","germany",  "berlin"),
    ("london", "england","france",  "paris"),
    # Syntactic (will only work with larger corpus)
    ("big",    "bigger","small",    "smaller"),
    ("good",   "better","bad",      "worse"),
    ("king",   "kingdom","president","country"),
]


def evaluate_analogies(
    model: Word2Vec,
    test_set: Optional[List[Tuple[str, str, str, str]]] = None,
    top_n: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate embedding quality on word analogy tasks.

    A prediction is correct if the expected word appears in the top_n results.

    Parameters
    ----------
    model    : trained Word2Vec
    test_set : list of (A, B, C, D) tuples; uses ANALOGY_TESTS if None
    top_n    : consider top_n candidates for accuracy

    Returns
    -------
    dict with keys: total, correct, accuracy, details
    """
    if test_set is None:
        test_set = ANALOGY_TESTS

    correct = 0
    total = 0
    details = []

    for a, b, c, expected in test_set:
        words = [a, b, c, expected]
        if any(w not in model.vocab for w in words):
            if verbose:
                oov = [w for w in words if w not in model.vocab]
                print(f"  [SKIP] OOV words: {oov}  (analogy: {a}-{b}+{c}={expected})")
            continue

        results = model.analogy(a, b, c, top_n=top_n)
        predicted_words = [w for w, _ in results]
        hit = expected in predicted_words

        correct += int(hit)
        total += 1
        details.append({
            "query": f"{a} - {b} + {c}",
            "expected": expected,
            "got": predicted_words[:3],
            "correct": hit,
        })

        if verbose:
            mark = "✓" if hit else "✗"
            print(f"  {mark}  {a} - {b} + {c} = {expected}  |  predicted: {predicted_words[:3]}")

    accuracy = correct / total if total > 0 else 0.0
    if verbose:
        print(f"\nAnalogy accuracy: {correct}/{total} = {accuracy:.1%}")

    return {"total": total, "correct": correct, "accuracy": accuracy, "details": details}


# ---------------------------------------------------------------------------
# 3. Embedding statistics
# ---------------------------------------------------------------------------

def embedding_stats(model: Word2Vec) -> Dict:
    """Compute basic statistics about the learned embeddings."""
    W = model.W_in
    norms = np.linalg.norm(W, axis=1)

    stats = {
        "vocab_size": model.vocab.vocab_size,
        "embed_dim": model.embed_dim,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "weight_mean": float(W.mean()),
        "weight_std": float(W.std()),
    }
    return stats


def print_embedding_stats(model: Word2Vec) -> None:
    stats = embedding_stats(model)
    print("\n── Embedding statistics ──────────────────")
    for k, v in stats.items():
        print(f"  {k:20s}: {v}")
    print("──────────────────────────────────────────")


# ---------------------------------------------------------------------------
# 4. t-SNE visualisation
# ---------------------------------------------------------------------------

def plot_tsne(
    model: Word2Vec,
    words: Optional[List[str]] = None,
    n_words: int = 200,
    save_path: str = "results/tsne.png",
    perplexity: float = 30.0,
    seed: int = 42,
) -> None:
    """
    Reduce word vectors to 2-D with t-SNE and produce a scatter plot.

    Requires: matplotlib, scikit-learn (soft dependencies).

    Parameters
    ----------
    model     : trained Word2Vec
    words     : specific words to plot; if None, uses top-n most frequent
    n_words   : how many words to plot when `words` is None
    save_path : where to write the PNG
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("matplotlib and scikit-learn are required for t-SNE plotting.")
        print("Install with:  pip install matplotlib scikit-learn")
        return

    vocab = model.vocab

    if words is None:
        # Pick the n_words most frequent words
        sorted_words = sorted(
            vocab.word_counts.items(), key=lambda x: -x[1]
        )[:n_words]
        words = [w for w, _ in sorted_words if w in vocab]

    words = [w for w in words if w in vocab]
    if not words:
        print("No valid words for t-SNE.")
        return

    indices = [vocab.word2idx[w] for w in words]
    vectors = model.W_in[indices]

    # Normalise before t-SNE
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    vectors = vectors / norms

    n = len(vectors)
    perp = min(perplexity, n - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=1000,
        random_state=seed,
        init="pca",
    )
    coords = tsne.fit_transform(vectors)   # (n, 2)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.6)

    for i, word in enumerate(words):
        ax.annotate(
            word,
            xy=(coords[i, 0], coords[i, 1]),
            fontsize=7,
            alpha=0.85,
        )

    ax.set_title(f"t-SNE of Word Embeddings (dim={model.embed_dim}, n={n})")
    ax.axis("off")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE plot saved to {save_path}")


# ---------------------------------------------------------------------------
# 5. Loss curve plot
# ---------------------------------------------------------------------------

def plot_loss(
    loss_history: List[Tuple[int, float]],
    save_path: str = "results/loss_curve.png",
) -> None:
    """Plot the training loss curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting.")
        return

    if not loss_history:
        print("No loss history to plot.")
        return

    steps, losses = zip(*loss_history)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, linewidth=1.2, color="steelblue")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Avg. loss")
    ax.set_title("Word2Vec Training Loss")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved to {save_path}")


# ---------------------------------------------------------------------------
# 6. Full evaluation report
# ---------------------------------------------------------------------------

def full_report(
    model: Word2Vec,
    sample_words: Optional[List[str]] = None,
    analogy_tests: Optional[List[Tuple]] = None,
    plot: bool = True,
    loss_history: Optional[List] = None,
) -> None:
    """Print a comprehensive evaluation report."""
    print("\n" + "=" * 60)
    print("  WORD2VEC EVALUATION REPORT")
    print("=" * 60)

    print_embedding_stats(model)

    if sample_words is None:
        # Pick a few common words likely to be in any corpus
        candidates = [
            "king", "queen", "man", "woman", "paris", "london",
            "france", "germany", "dog", "cat", "good", "bad",
        ]
        sample_words = [w for w in candidates if w in model.vocab][:6]

    print("\n── Nearest neighbours ────────────────────")
    for word in sample_words:
        nn = model.most_similar(word, top_n=5)
        neighbours = ", ".join(f"{w}({s:.3f})" for w, s in nn)
        print(f"  {word:12s} → {neighbours}")

    print("\n── Analogy evaluation ────────────────────")
    evaluate_analogies(model, test_set=analogy_tests, verbose=True)

    if plot:
        print("\n── Visualisation ─────────────────────────")
        plot_tsne(model)
        if loss_history:
            plot_loss(loss_history)

    print("\n" + "=" * 60)
