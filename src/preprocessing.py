"""
preprocessing.py
================
Text loading, cleaning, tokenisation, and training-pair generation.

Supports:
  - Loading raw .txt files or the text8 corpus
  - Simple whitespace / punctuation tokenisation
  - Skip-gram pair generation with dynamic window size
  - CBOW context generation (used when model_type='cbow')
"""

from __future__ import annotations

import os
import re
import urllib.request
import zipfile
from typing import Iterator, List, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase and strip non-alphabetic characters."""
    text = text.lower()
    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str, clean: bool = True) -> List[str]:
    """Split text into a list of word tokens."""
    if clean:
        text = clean_text(text)
    return text.split()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_text_file(path: str, max_tokens: Optional[int] = None) -> List[str]:
    """Read a plain-text file and return a token list."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    tokens = tokenise(text)
    if max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


def download_text8(dest_dir: str = "data") -> str:
    """
    Download and extract the text8 corpus (~100 MB) if not already present.
    text8 is a cleaned slice of Wikipedia and is a standard word2vec benchmark.
    Returns the path to the extracted text file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "text8.zip")
    txt_path = os.path.join(dest_dir, "text8")

    if not os.path.exists(txt_path):
        url = "http://mattmahoney.net/dc/text8.zip"
        print(f"Downloading text8 from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)
        print("Done.")
    else:
        print(f"text8 already present at {txt_path}")

    return txt_path


def load_text8(dest_dir: str = "data", max_tokens: Optional[int] = None) -> List[str]:
    """Download (if needed) and load the text8 corpus."""
    path = download_text8(dest_dir)
    return load_text_file(path, max_tokens=max_tokens)


def load_sample_corpus() -> List[str]:
    """
    A small, self-contained corpus for quick testing.
    Covers several semantic / syntactic groups so analogies work.
    """
    sentences = [
        "the king rules the kingdom and the queen rules beside him",
        "the prince is son of the king and the princess is daughter of the queen",
        "man and woman are human beings living in society",
        "the dog barked loudly and the cat meowed softly",
        "paris is the capital of france and berlin is the capital of germany",
        "london is the capital of england and rome is the capital of italy",
        "the president leads the country and the prime minister leads the government",
        "machine learning is a field of artificial intelligence",
        "deep learning uses neural networks to learn representations",
        "word2vec learns word embeddings from large corpora of text",
        "language models predict the next word given previous words",
        "the sun rises in the east and sets in the west",
        "water flows downhill and fills rivers lakes and oceans",
        "scientists study nature and engineers build machines",
        "books contain knowledge and libraries store many books",
        "the teacher taught the students in the classroom",
        "the doctor healed the patient in the hospital",
        "the chef cooked delicious food in the restaurant kitchen",
        "the pilot flew the airplane through the cloudy sky",
        "the programmer wrote code to solve difficult problems",
        "mathematics is the language of science and logic",
        "physics describes the fundamental laws of nature and the universe",
        "chemistry studies atoms molecules and chemical reactions",
        "biology explores living organisms and their environments",
        "history records the events and stories of human civilization",
    ] * 200  # repeat to give enough training signal on this tiny corpus
    return tokenise(" ".join(sentences))


# ---------------------------------------------------------------------------
# Training pair generation
# ---------------------------------------------------------------------------

def generate_skipgram_pairs(
    token_ids: List[int],
    window_size: int = 5,
    dynamic_window: bool = True,
) -> Iterator[Tuple[int, int]]:
    """
    Yield (center, context) pairs for skip-gram training.

    Parameters
    ----------
    token_ids     : encoded token list (after subsampling)
    window_size   : maximum context window radius
    dynamic_window: if True, sample actual window size ~ Uniform(1, window_size)
                    (as in the original implementation)

    Yields
    ------
    (center_idx, context_idx) tuples
    """
    n = len(token_ids)
    for i, center in enumerate(token_ids):
        w = np.random.randint(1, window_size + 1) if dynamic_window else window_size
        start = max(0, i - w)
        end = min(n, i + w + 1)
        for j in range(start, end):
            if j != i:
                yield center, token_ids[j]


def generate_cbow_pairs(
    token_ids: List[int],
    window_size: int = 5,
    dynamic_window: bool = True,
) -> Iterator[Tuple[List[int], int]]:
    """
    Yield (context_list, target) pairs for CBOW training.

    Parameters
    ----------
    token_ids : encoded token list
    window_size : maximum context window radius
    dynamic_window : use random window size

    Yields
    ------
    (context_indices, target_idx) tuples
    """
    n = len(token_ids)
    for i, target in enumerate(token_ids):
        w = np.random.randint(1, window_size + 1) if dynamic_window else window_size
        start = max(0, i - w)
        end = min(n, i + w + 1)
        context = [token_ids[j] for j in range(start, end) if j != i]
        if context:
            yield context, target


def batch_skipgram_pairs(
    token_ids: List[int],
    window_size: int = 5,
    batch_size: int = 512,
    dynamic_window: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate skip-gram pairs in fixed-size numpy batches for efficiency.
    """
    centers, contexts = [], []
    for center, context in generate_skipgram_pairs(token_ids, window_size, dynamic_window):
        centers.append(center)
        contexts.append(context)
        if len(centers) == batch_size:
            yield np.array(centers, dtype=np.int32), np.array(contexts, dtype=np.int32)
            centers, contexts = [], []
    if centers:
        yield np.array(centers, dtype=np.int32), np.array(contexts, dtype=np.int32)
