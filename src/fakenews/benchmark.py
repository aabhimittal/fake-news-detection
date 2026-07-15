"""Benchmark harness — compare classifiers with cross-validation.

A single train/test split gives one noisy number per model. To compare models
*fairly* we use **stratified k-fold cross-validation**: the data is split into
``k`` folds, each model is trained on ``k-1`` and tested on the held-out fold,
rotating through all folds. Reporting the mean ± standard deviation across folds
tells us both how good a model is and how *stable* that estimate is.

The harness runs on any ``text``/``label`` DataFrame, so it works equally on the
bundled synthetic data and on real corpora (see :func:`load_liar` and the
generic CSV loader in :mod:`fakenews.data`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import ModelConfig
from .models import build_pipeline

# The linear classifiers we compare out of the box.
DEFAULT_CLASSIFIERS = ("logistic", "passive_aggressive", "linear_svm", "naive_bayes")


@dataclass
class BenchmarkRow:
    classifier: str
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    fit_time: float

    def format(self) -> str:
        return (
            f"{self.classifier:>18} | "
            f"{self.accuracy_mean:.3f} ± {self.accuracy_std:.3f} | "
            f"{self.f1_mean:.3f} ± {self.f1_std:.3f} | "
            f"{self.fit_time:6.2f}s"
        )


def cross_validate_classifiers(
    df: pd.DataFrame,
    classifiers=DEFAULT_CLASSIFIERS,
    cv: int = 5,
    random_state: int = 42,
) -> List[BenchmarkRow]:
    """Cross-validate each named classifier on the same folds.

    Returns rows sorted by mean F1 (best first).
    """
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()

    # A single fold splitter shared across models => identical folds => fair.
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    rows: List[BenchmarkRow] = []
    for name in classifiers:
        pipeline = build_pipeline(ModelConfig(classifier=name, random_state=random_state))
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=splitter,
            scoring=["accuracy", "f1"],
            return_train_score=False,
        )
        rows.append(
            BenchmarkRow(
                classifier=name,
                accuracy_mean=float(np.mean(scores["test_accuracy"])),
                accuracy_std=float(np.std(scores["test_accuracy"])),
                f1_mean=float(np.mean(scores["test_f1"])),
                f1_std=float(np.std(scores["test_f1"])),
                fit_time=float(np.mean(scores["fit_time"])),
            )
        )

    rows.sort(key=lambda r: r.f1_mean, reverse=True)
    return rows


def format_table(rows: List[BenchmarkRow]) -> str:
    header = f"{'classifier':>18} | {'accuracy':>13} | {'F1 (fake)':>13} | {'fit':>7}"
    sep = "-" * len(header)
    lines = [header, sep] + [r.format() for r in rows]
    return "\n".join(lines)


# --- real-dataset loaders --------------------------------------------------

# The LIAR dataset labels statements on a 6-point truthfulness scale. We map it
# to a binary fake/real target the standard way.
_LIAR_FAKE = {"pants-fire", "false", "barely-true"}
_LIAR_REAL = {"half-true", "mostly-true", "true"}


def load_liar(path: Path) -> pd.DataFrame:
    """Load the LIAR dataset (``train.tsv`` etc.) as a binary text/label frame.

    LIAR is a tab-separated file whose 2nd column is the truthfulness label and
    3rd column is the statement. The 6-way label is collapsed to fake (1) / real
    (0). Download it from https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LIAR file not found: {path}")

    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    # Columns per the LIAR README: 0=id, 1=label, 2=statement, ...
    label_raw = df[1].str.strip().str.lower()
    statement = df[2].fillna("")

    label = label_raw.map(
        lambda v: 1 if v in _LIAR_FAKE else (0 if v in _LIAR_REAL else np.nan)
    )
    out = pd.DataFrame({"text": statement, "label": label}).dropna()
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)


def load_kaggle_fake_real(
    fake_csv: Path,
    real_csv: Path,
    text_col: str = "text",
) -> pd.DataFrame:
    """Load the Kaggle *Fake and Real News* dataset (two CSVs) into one frame.

    The dataset ships ``Fake.csv`` and ``True.csv``, each with a ``text`` column.
    """
    fake = pd.read_csv(fake_csv)
    real = pd.read_csv(real_csv)
    fake_df = pd.DataFrame({"text": fake[text_col].astype(str), "label": 1})
    real_df = pd.DataFrame({"text": real[text_col].astype(str), "label": 0})
    return pd.concat([fake_df, real_df], ignore_index=True)
