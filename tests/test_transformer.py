"""Tests for the optional transformer detector.

The lightweight tests (interface contracts) always run. The heavy end-to-end
test downloads and fine-tunes a real checkpoint, so it only runs when the
backend is installed *and* opted in via FAKENEWS_RUN_TRANSFORMER=1 — keeping the
default suite fast and offline.
"""

import os

import pytest

from fakenews.config import TransformerConfig
from fakenews.transformer import TransformerDetector

# Is the torch + transformers backend importable?
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

RUN_HEAVY = HAS_BACKEND and os.environ.get("FAKENEWS_RUN_TRANSFORMER") == "1"


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        TransformerDetector().predict("anything")


def test_config_defaults_are_cpu_friendly():
    cfg = TransformerConfig()
    assert cfg.model_name == "distilbert-base-uncased"
    assert cfg.epochs >= 1 and cfg.batch_size >= 1


@pytest.mark.skipif(HAS_BACKEND, reason="backend installed; error path not taken")
def test_helpful_error_without_backend():
    with pytest.raises(ImportError, match="fakenews\\[transformer\\]"):
        TransformerDetector().fit()


@pytest.mark.skipif(not RUN_HEAVY, reason="set FAKENEWS_RUN_TRANSFORMER=1 to run")
def test_end_to_end_fine_tune(tmp_path):
    from fakenews.data import generate_synthetic_dataset

    cfg = TransformerConfig(epochs=2, batch_size=16, max_length=64, random_state=0)
    detector = TransformerDetector(cfg)
    result = detector.fit(generate_synthetic_dataset(n_per_class=60, random_state=0))
    assert result.accuracy >= 0.85

    assert detector.predict("SHOCKING secret cure they tried to censor!!!").is_fake
    assert not detector.predict(
        "The central bank reported inflation eased over the quarter."
    ).is_fake

    path = detector.save(tmp_path / "tf")
    restored = TransformerDetector.load(path)
    assert restored.predict("BREAKING bombshell truth exposed!!!").is_fake
