"""Model construction, training and persistence.

The estimator is a scikit-learn ``Pipeline`` so that preprocessing, feature
extraction and the classifier travel together — training and inference use the
exact same code path, which eliminates a whole class of train/serve skew bugs.

Pipeline shape::

    clean -> FeatureUnion( tfidf , stylometric ) -> linear classifier
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from .config import DEFAULT_MODEL_PATH, ModelConfig
from .features import StylometricFeatures, TextCleaner, build_tfidf


def _build_classifier(config: ModelConfig):
    """Instantiate the linear classifier named in the config."""
    name = config.classifier.lower()
    if name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=config.random_state)
    if name == "passive_aggressive":
        return PassiveAggressiveClassifier(
            max_iter=1000, random_state=config.random_state
        )
    if name == "naive_bayes":
        # MultinomialNB needs non-negative features; the stylometric block and
        # scaler below keep everything >= 0.
        return MultinomialNB()
    if name == "linear_svm":
        return LinearSVC(random_state=config.random_state)
    raise ValueError(
        f"Unknown classifier {config.classifier!r}. Choose from: logistic, "
        "passive_aggressive, naive_bayes, linear_svm."
    )


def build_pipeline(config: Optional[ModelConfig] = None) -> Pipeline:
    """Assemble the full clean -> features -> classifier pipeline."""
    config = config or ModelConfig()

    feature_blocks = [("tfidf", build_tfidf(config))]
    if config.use_stylometric:
        feature_blocks.append(("style", StylometricFeatures()))

    steps = [
        ("clean", TextCleaner()),
        ("features", FeatureUnion(feature_blocks)),
        # Scale to [0, 1] range without breaking sparsity — required so
        # MultinomialNB sees non-negative input and so mixed-scale features
        # (raw char counts vs. ratios) don't swamp the linear models.
        ("scale", MaxAbsScaler()),
        ("clf", _build_classifier(config)),
    ]
    return Pipeline(steps)


def save_model(pipeline: Pipeline, path: Optional[Path] = None) -> Path:
    """Persist a fitted pipeline with joblib."""
    path = Path(path or DEFAULT_MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: Optional[Path] = None) -> Pipeline:
    """Load a pipeline previously saved by :func:`save_model`."""
    path = Path(path or DEFAULT_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m fakenews.cli train` first."
        )
    return joblib.load(path)
