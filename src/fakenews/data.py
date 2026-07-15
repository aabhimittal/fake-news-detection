"""Datasets.

The project is designed to run end-to-end with **zero external downloads** so
it works in offline CI. :func:`generate_synthetic_dataset` fabricates a small
but *learnable* corpus in which fake and real articles differ in both
vocabulary and style — exactly the signal the model is built to exploit.

To use a real corpus instead (e.g. the Kaggle "Fake and Real News" dataset)
just point :func:`load_dataset` at a CSV with ``text`` and ``label`` columns;
``label`` should be 1 for fake and 0 for real.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DEFAULT_DATASET_PATH

# --- building blocks for the synthetic generator ---------------------------

_REAL_TOPICS = [
    "the central bank", "the city council", "researchers at the university",
    "the health ministry", "the national weather service", "the supreme court",
    "the transport authority", "local farmers", "the census bureau",
    "the space agency", "the education board", "the trade delegation",
]
_REAL_VERBS = [
    "reported", "announced", "confirmed", "published a study showing",
    "released figures indicating", "stated in a briefing that",
    "approved a measure under which", "outlined a plan where",
]
_REAL_TAILS = [
    "inflation eased slightly over the quarter.",
    "the new policy would take effect next year.",
    "the results were consistent with earlier estimates.",
    "funding would be allocated across several districts.",
    "the changes followed a period of public consultation.",
    "officials expected a gradual improvement in the coming months.",
    "the data would be reviewed by independent experts.",
]

_FAKE_LEADS = [
    "SHOCKING", "BREAKING", "EXPOSED", "URGENT ALERT", "THEY DON'T WANT YOU TO KNOW",
    "UNBELIEVABLE", "BOMBSHELL", "WAKE UP",
]
_FAKE_CLAIMS = [
    "secret conspiracy to control your mind",
    "miracle cure doctors are hiding from you",
    "the truth about what really happened",
    "banned footage they tried to censor",
    "this one weird trick destroyed the establishment",
    "insiders admit the whole thing was a hoax",
    "government caught in unbelievable cover-up",
]
_FAKE_TAILS = [
    "Share before it gets DELETED!!!",
    "You won't BELIEVE what happens next!!",
    "Experts are STUNNED and refuse to comment!!!",
    "Click now — the mainstream media is LYING to you!!",
    "This will change EVERYTHING you thought you knew!!!",
]


def _real_article(rng: random.Random) -> str:
    return (
        f"{rng.choice(_REAL_TOPICS).capitalize()} {rng.choice(_REAL_VERBS)} "
        f"that {rng.choice(_REAL_TAILS)}"
    )


def _fake_article(rng: random.Random) -> str:
    return (
        f"{rng.choice(_FAKE_LEADS)}: {rng.choice(_FAKE_CLAIMS)}! "
        f"{rng.choice(_FAKE_TAILS)}"
    )


def generate_synthetic_dataset(
    n_per_class: int = 400,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a balanced DataFrame with ``text`` and ``label`` columns.

    ``label`` is 1 for fake, 0 for real.
    """
    rng = random.Random(random_state)
    rows = []
    for _ in range(n_per_class):
        rows.append({"text": _real_article(rng), "label": 0})
        rows.append({"text": _fake_article(rng), "label": 1})
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def generate_benchmark_dataset(
    n_per_class: int = 400,
    noise: float = 0.25,
    random_state: int = 42,
) -> pd.DataFrame:
    """A deliberately *harder* variant for benchmarking.

    The plain synthetic corpus is perfectly separable, so every classifier
    scores ~1.0 and a benchmark can't tell them apart. Here we blur the classes
    two ways, controlled by ``noise`` (0 = easy, 1 = very hard):

    * **Style crossover** — a fraction of *fake* stories are written in neutral
      wire-copy style, and a fraction of *real* ones adopt sensational phrasing.
    * **Label noise** — a fraction of labels are flipped outright.

    The result rewards models that pick up subtler cues, which is what makes a
    benchmark across classifiers meaningful.
    """
    rng = random.Random(random_state)
    rows = []
    for _ in range(n_per_class):
        for label in (0, 1):
            # Style crossover: with probability `noise/2`, write in the *other*
            # class's style while keeping the true label.
            flip_style = rng.random() < noise / 2
            if (label == 0) != flip_style:      # real style
                text = _real_article(rng)
            else:                               # fake style
                text = _fake_article(rng)
            # Label noise: occasionally flip the label outright.
            y = label
            if rng.random() < noise / 2:
                y = 1 - y
            rows.append({"text": text, "label": y})
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def load_dataset(
    path: Optional[Path] = None,
    *,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Load a CSV dataset, falling back to a freshly generated synthetic one.

    If ``path`` is given but does not exist, a :class:`FileNotFoundError` is
    raised. If ``path`` is ``None`` and the default sample file is missing, a
    synthetic dataset is generated on the fly.
    """
    if path is None:
        path = DEFAULT_DATASET_PATH
        if not path.exists():
            return generate_synthetic_dataset()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset {path} is missing required column(s): {sorted(missing)}"
        )
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def write_sample_dataset(path: Optional[Path] = None, n_per_class: int = 400) -> Path:
    """Materialise the synthetic dataset to disk (used by ``make data``)."""
    path = Path(path or DEFAULT_DATASET_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_dataset(n_per_class=n_per_class).to_csv(path, index=False)
    return path
