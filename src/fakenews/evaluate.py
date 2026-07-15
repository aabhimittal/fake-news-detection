"""Evaluation metrics and reporting.

For a fake-news classifier, *recall on the fake class* matters most — a missed
fake article keeps spreading — but precision matters too, because flagging real
news as fake erodes trust. We therefore report the full picture: accuracy plus
per-class precision/recall/F1 and the confusion matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABELS = {0: "real", 1: "fake"}


@dataclass
class EvaluationResult:
    accuracy: float
    precision_fake: float
    recall_fake: float
    f1_fake: float
    confusion: List[List[int]]
    report: str = field(repr=False)

    def as_dict(self) -> Dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "precision_fake": self.precision_fake,
            "recall_fake": self.recall_fake,
            "f1_fake": self.f1_fake,
            "confusion": self.confusion,
        }

    def summary(self) -> str:
        return (
            f"accuracy={self.accuracy:.3f}  "
            f"fake precision={self.precision_fake:.3f}  "
            f"fake recall={self.recall_fake:.3f}  "
            f"fake F1={self.f1_fake:.3f}"
        )


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> EvaluationResult:
    """Compute the standard metric bundle for a set of predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_fake=float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        recall_fake=float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        f1_fake=float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        report=classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["real", "fake"], zero_division=0
        ),
    )
