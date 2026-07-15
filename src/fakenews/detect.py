"""High-level detector API.

:class:`FakeNewsDetector` is the one class most callers need. It wraps the
scikit-learn pipeline with train / predict / explain / save / load methods and
returns friendly result objects instead of raw numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ModelConfig
from .data import load_dataset
from .evaluate import EvaluationResult, evaluate
from .models import build_pipeline, load_model, save_model


@dataclass
class Prediction:
    """A single classification result."""

    label: str            # "fake" or "real"
    is_fake: bool
    confidence: float     # probability of the predicted class, in [0, 1]

    def __str__(self) -> str:
        return f"{self.label.upper()} ({self.confidence:.1%} confidence)"


class FakeNewsDetector:
    """Train, persist and query a fake-news text classifier."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.pipeline = None  # set by fit() or load()

    # -- training ----------------------------------------------------------
    def fit(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> EvaluationResult:
        """Train on a DataFrame of ``text``/``label`` rows and return held-out metrics."""
        if df is None:
            df = load_dataset()

        X = df["text"].tolist()
        y = df["label"].astype(int).tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        self.pipeline = build_pipeline(self.config)
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        return evaluate(y_test, y_pred)

    # -- inference ---------------------------------------------------------
    def _check_ready(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Detector is not trained. Call fit() or load() first.")

    def _proba_of_fake(self, texts: Sequence[str]) -> np.ndarray:
        """Return P(fake) per text, working even for classifiers without predict_proba."""
        self._check_ready()
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(texts)[:, 1]
        # LinearSVC / PassiveAggressive expose decision_function; squash it.
        scores = np.asarray(self.pipeline.decision_function(texts), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    def predict(self, text: str) -> Prediction:
        """Classify a single document."""
        p_fake = float(self._proba_of_fake([text])[0])
        is_fake = p_fake >= 0.5
        return Prediction(
            label="fake" if is_fake else "real",
            is_fake=is_fake,
            confidence=p_fake if is_fake else 1.0 - p_fake,
        )

    def predict_batch(self, texts: Sequence[str]) -> List[Prediction]:
        probs = self._proba_of_fake(texts)
        out = []
        for p in probs:
            is_fake = p >= 0.5
            out.append(
                Prediction(
                    label="fake" if is_fake else "real",
                    is_fake=bool(is_fake),
                    confidence=float(p if is_fake else 1.0 - p),
                )
            )
        return out

    # -- explanation -------------------------------------------------------
    def explain(self, text: str, top_k: int = 8) -> List[Tuple[str, float]]:
        """Return the tokens that pushed *this* document toward its prediction.

        Only meaningful for the linear models (logistic / SVM / PA), whose
        per-feature weights are directly interpretable. Returns a list of
        ``(feature_name, signed_contribution)`` sorted by absolute impact.
        """
        self._check_ready()
        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return []

        # Transform through every step *up to* the classifier (clean -> features
        # -> scale), starting from a one-element batch.
        transformed = [text]
        for _name, step in self.pipeline.steps[:-1]:
            transformed = step.transform(transformed)
        # `transformed` is now the scaled feature row (1 x n_features).
        row = (
            np.asarray(transformed.todense()).ravel()
            if hasattr(transformed, "todense")
            else np.asarray(transformed).ravel()
        )

        names = self.pipeline.named_steps["features"].get_feature_names_out()
        contributions = row * clf.coef_.ravel()
        order = np.argsort(np.abs(contributions))[::-1][:top_k]
        return [(str(names[i]), float(contributions[i])) for i in order if row[i] != 0]

    # -- persistence -------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        self._check_ready()
        return save_model(self.pipeline, path)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FakeNewsDetector":
        detector = cls()
        detector.pipeline = load_model(path)
        return detector
