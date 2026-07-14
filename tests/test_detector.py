import pytest

from fakenews.config import ModelConfig
from fakenews.data import generate_synthetic_dataset
from fakenews.detect import FakeNewsDetector


@pytest.fixture(scope="module")
def trained_detector():
    detector = FakeNewsDetector(ModelConfig(random_state=0))
    result = detector.fit(generate_synthetic_dataset(n_per_class=200))
    return detector, result


def test_model_learns_the_signal(trained_detector):
    _, result = trained_detector
    # The synthetic classes are strongly separable; anything below 0.85 means
    # the pipeline is broken.
    assert result.accuracy >= 0.85
    assert result.recall_fake >= 0.85


def test_predicts_obvious_fake_and_real(trained_detector):
    detector, _ = trained_detector
    fake = detector.predict("SHOCKING: secret miracle cure they tried to censor!!!")
    real = detector.predict(
        "The central bank reported that inflation eased slightly over the quarter."
    )
    assert fake.is_fake is True
    assert real.is_fake is False
    assert 0.0 <= fake.confidence <= 1.0


def test_explain_returns_signed_contributions(trained_detector):
    detector, _ = trained_detector
    contributions = detector.explain("SHOCKING conspiracy hoax exposed!!!")
    assert contributions  # non-empty for the linear model
    assert all(isinstance(name, str) for name, _ in contributions)


def test_save_and_load_roundtrip(trained_detector, tmp_path):
    detector, _ = trained_detector
    path = detector.save(tmp_path / "model.joblib")
    reloaded = FakeNewsDetector.load(path)
    original = detector.predict("BREAKING bombshell truth exposed!!!")
    restored = reloaded.predict("BREAKING bombshell truth exposed!!!")
    assert original.label == restored.label


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        FakeNewsDetector().predict("anything")
