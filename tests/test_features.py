import numpy as np

from fakenews.features import (
    STYLOMETRIC_FEATURE_NAMES,
    StylometricFeatures,
    _stylometric_vector,
)


def test_stylometric_vector_length_matches_names():
    vec = _stylometric_vector("Hello world!")
    assert vec.shape == (len(STYLOMETRIC_FEATURE_NAMES),)


def test_empty_document_is_all_zero():
    assert np.all(_stylometric_vector("") == 0)


def test_shouting_increases_uppercase_ratio():
    calm = _stylometric_vector("this is fine and normal text")
    shout = _stylometric_vector("THIS IS ABSOLUTELY SHOCKING NEWS")
    idx = STYLOMETRIC_FEATURE_NAMES.index("uppercase_ratio")
    assert shout[idx] > calm[idx]


def test_clickbait_ratio_detects_lexicon():
    idx = STYLOMETRIC_FEATURE_NAMES.index("clickbait_ratio")
    assert _stylometric_vector("shocking secret conspiracy")[idx] > 0
    assert _stylometric_vector("the council met today")[idx] == 0


def test_transformer_produces_matrix():
    X = ["one document", "another one here"]
    out = StylometricFeatures().fit_transform(X)
    assert out.shape == (2, len(STYLOMETRIC_FEATURE_NAMES))
