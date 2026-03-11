# word2vec-numpy

A pure-NumPy implementation of **Word2Vec**, built from scratch. No PyTorch, TensorFlow, or other ML frameworks are used.

---

## Features

| Feature | Detail |
|---|---|
| **Algorithms** | Skip-gram + Negative Sampling (SGNS) and CBOW |
| **Batched training** | Vectorised mini-batch updates via NumPy einsum |
| **Subsampling** | Frequent-word subsampling (Mikolov 2013 §2.3) |
| **Noise distribution** | Unigram^0.75 table for O(1) negative sampling |
| **LR schedule** | Linear decay with configurable floor |
| **Evaluation** | Cosine NN lookup, word analogies, t-SNE, loss curves |
| **Checkpointing** | Save/load model weights as `.npz` |
| **CLI** | `train.py` with full argument parser |

---

## Repository Structure

```
word2vec-numpy/
├── src/
│   ├── __init__.py
│   ├── vocabulary.py      # vocab building, subsampling, noise table
│   ├── preprocessing.py   # text cleaning, tokenisation, pair generation
│   ├── word2vec.py        # model: forward pass, loss, gradients, SGD update
│   └── trainer.py         # training loop, LR schedule, checkpointing
├── evaluate.py            # analogy eval, nearest neighbours, t-SNE, plots
├── train.py               # CLI entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Install dependencies (only numpy is required; matplotlib + sklearn for plots)
pip install -r requirements.txt

# Train on the built-in sample corpus (runs in ~30 seconds):
python train.py --corpus sample --epochs 40 --dim 100 --evaluate

# Train on text8 (~100 MB Wikipedia corpus, downloads automatically):
python train.py --corpus text8 --epochs 5 --dim 100 --evaluate --plot

# Train on your own text file:
python train.py --corpus path/to/corpus.txt --dim 200 --epochs 3
```

### Use as a library

```python
from src import Vocabulary, Word2Vec, Trainer, load_sample_corpus

tokens = load_sample_corpus()
vocab  = Vocabulary(min_count=2).build(tokens)
corpus = vocab.encode(tokens)

model  = Word2Vec(vocab=vocab, embed_dim=100, n_negatives=5)
trainer = Trainer(model=model, vocab=vocab, corpus=corpus, n_epochs=10)
trainer.train()

# Query the embeddings
model.most_similar("king", top_n=5)
model.analogy("king", "man", "woman", top_n=5)   # → queen

# Save / load
model.save("results/model")
vocab.save("results/vocab.json")
```

---

## Algorithm: Skip-gram with Negative Sampling (SGNS)

### Objective

Given a centre word $w_c$, skip-gram predicts surrounding context words $w_o$ within a window of radius $w$.

**Without negative sampling**, the softmax over the full vocabulary is intractable ($O(V)$ per step). Negative sampling replaces it with a binary classification problem.

### Loss Function

For each positive pair $(w_c, w_o)$, sample $k$ noise words $\{w_{n_1}, \ldots, w_{n_k}\}$ from the noise distribution:

$$\mathcal{L} = -\log \sigma(\mathbf{v}_o \cdot \mathbf{u}_c) - \sum_{i=1}^{k} \log \sigma(-\mathbf{v}_{n_i} \cdot \mathbf{u}_c)$$

where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function
- $\mathbf{u}_c \in \mathbb{R}^d$ — the **input embedding** (row of `W_in`) for centre word $c$
- $\mathbf{v}_o \in \mathbb{R}^d$ — the **output embedding** (row of `W_out`) for context word $o$
- $\mathbf{v}_{n_i}$ — the output embedding for the $i$-th negative sample

The loss pushes $\mathbf{v}_o \cdot \mathbf{u}_c$ to be large (positive word should be predictable) and $\mathbf{v}_{n_i} \cdot \mathbf{u}_c$ to be small (negative words should be unpredictable).

### Gradient Derivations

Let $s_o = \mathbf{v}_o \cdot \mathbf{u}_c$ and $s_i = \mathbf{v}_{n_i} \cdot \mathbf{u}_c$.

**Gradient w.r.t. $\mathbf{u}_c$** (centre embedding):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_c}
= \underbrace{(\sigma(s_o) - 1)}_{\text{error on positive}} \cdot \mathbf{v}_o
+ \sum_{i=1}^{k} \underbrace{\sigma(s_i)}_{\text{error on negatives}} \cdot \mathbf{v}_{n_i}$$

**Gradient w.r.t. $\mathbf{v}_o$** (positive context embedding):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_o} = (\sigma(s_o) - 1) \cdot \mathbf{u}_c$$

**Gradient w.r.t. $\mathbf{v}_{n_i}$** (each negative context embedding):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{n_i}} = \sigma(s_i) \cdot \mathbf{u}_c$$

**Derivation sketch** (using the chain rule):

$$\frac{\partial}{\partial x}\left[-\log \sigma(x)\right] = \sigma(x) - 1 \qquad \frac{\partial}{\partial x}\left[-\log(1 - \sigma(x))\right] = \sigma(x)$$

Since $\frac{\partial s_o}{\partial \mathbf{u}_c} = \mathbf{v}_o$ and $\frac{\partial s_o}{\partial \mathbf{v}_o} = \mathbf{u}_c$, the chain rule directly gives the expressions above.

### SGD Update Rule

$$\mathbf{u}_c \leftarrow \mathbf{u}_c - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{u}_c}$$

where $\eta$ is the learning rate (linearly decayed from `lr_start` to `lr_min` over training).

---

## Noise Distribution

Negative samples are drawn from the **unigram distribution raised to the 3/4 power**:

$$P_{\text{noise}}(w) \propto \text{count}(w)^{3/4}$$

This was empirically found to outperform both the raw unigram and the uniform distributions. It is implemented as a large pre-computed lookup table (1 million entries) for O(1) sampling.

---

## Frequent-Word Subsampling

Each token is discarded during training with probability:

$$P_{\text{discard}}(w) = 1 - \min\!\left(1,\, \sqrt{\frac{t}{f(w)}} + \frac{t}{f(w)}\right)$$

where $f(w)$ is the word's relative frequency and $t$ is the subsampling threshold (default $10^{-3}$). This speeds up training and improves representations for rare words by reducing the dominance of frequent function words.

---

## CBOW Variant

CBOW predicts a target word from the **mean** of its context embeddings:

$$\mathbf{h} = \frac{1}{|C|} \sum_{c \in C} \mathbf{u}_c$$

The loss and negative sampling objective are identical to SGNS, but applied with $\mathbf{h}$ in place of $\mathbf{u}_c$. The gradient of $\mathcal{L}$ with respect to $\mathbf{h}$ is distributed equally back to all context word embeddings.

---

## CLI Reference

```
python train.py [options]

Data:
  --corpus        'sample' | 'text8' | /path/to/file.txt
  --max_tokens    Truncate corpus to N tokens
  --data_dir      Directory for downloaded datasets (default: data/)

Model:
  --model         skipgram | cbow  (default: skipgram)
  --dim           Embedding dimensionality (default: 100)
  --neg           Negative samples per pair (default: 5)
  --window        Context window radius (default: 5)
  --min_count     Minimum word frequency (default: 5)

Training:
  --epochs        Number of training epochs (default: 5)
  --lr            Initial learning rate (default: 0.025)
  --lr_min        Minimum learning rate (default: 0.0001)
  --batch_size    Pairs per gradient step (default: 512)
  --seed          Random seed (default: 42)

Output:
  --out           Model output path (default: results/model)
  --evaluate      Run evaluation after training
  --plot          Save t-SNE and loss curve plots
  --log_every     Log loss every N steps (default: 10000)
```

---

## Evaluation

After training:

```python
from evaluate import full_report
full_report(model, plot=True, loss_history=trainer.loss_history)
```

Outputs:
- Embedding norm statistics
- Nearest neighbours for sample words
- Word analogy accuracy
- t-SNE scatter plot → `results/tsne.png`
- Loss curve → `results/loss_curve.png`

---

## Design Decisions and Notes

**Why two embedding matrices?** `W_in` (centre) and `W_out` (context) are kept separate, as in the original paper. Using a single shared matrix (tying weights) is also possible and sometimes used, but the two-matrix variant generally performs better.

**Why not hierarchical softmax?** Negative sampling is faster in practice and achieves comparable or better results for large vocabularies. Hierarchical softmax has $O(\log V)$ per step vs $O(k)$ for NS, but $k \ll \log V$ only for very large $V$.

**Batching vs. per-pair updates:** The batched `sgns_batch_step` uses `np.einsum` for vectorised score computation and `np.add.at` for safe scattered gradient accumulation. Each example in the batch contributes its own gradient independently (not averaged), matching the original Mikolov implementation.

**Numerical stability:** Sigmoid inputs are clipped to $[-30, 30]$ before exponentiation, and a small $\epsilon = 10^{-7}$ is added inside log computations.

