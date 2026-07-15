"""Feature engineering.

Two complementary views of a document are combined:

1. **Lexical** — a TF-IDF matrix over word n-grams. This captures *what* is
   being said (vocabulary, topical phrasing, clickbait n-grams).
2. **Stylometric** — a handful of interpretable, scale-free statistics that
   capture *how* it is said (shouting in capitals, exclamation spam, clickbait
   trigger words). Fake stories are stylistically distinct even when they
   discuss the same topics, so these features add signal that bag-of-words
   alone misses.

Both are exposed as scikit-learn transformers so they slot into a ``Pipeline``
and are serialised together with the model.
"""

from __future__ import annotations

import re
from typing import List, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import ModelConfig
from .preprocess import clean_text

# Words that disproportionately show up in sensational / low-credibility news.
CLICKBAIT_LEXICON = frozenset(
    """
    shocking miracle breaking exposed secret conspiracy hoax unbelievable
    outrage banned censored urgent alert exclusive bombshell insane destroyed
    slammed epic viral trick cure instantly guaranteed truth wake sheeple
    """.split()
)

# Names of the stylometric columns, in the order produced below. Exposed so the
# model can print human-readable explanations.
STYLOMETRIC_FEATURE_NAMES: List[str] = [
    "char_count",
    "word_count",
    "avg_word_len",
    "uppercase_ratio",
    "exclamation_ratio",
    "question_ratio",
    "digit_ratio",
    "clickbait_ratio",
    "unique_word_ratio",
]

_WORD_RE = re.compile(r"[A-Za-z']+")


def _stylometric_vector(raw: str) -> np.ndarray:
    """Compute the interpretable style features for a single raw document."""
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)

    n_chars = len(raw)
    words = _WORD_RE.findall(raw)
    n_words = len(words)
    lower_words = [w.lower() for w in words]

    if n_words == 0:
        return np.zeros(len(STYLOMETRIC_FEATURE_NAMES), dtype=np.float64)

    n_upper = sum(1 for c in raw if c.isupper())
    n_alpha = sum(1 for c in raw if c.isalpha())
    clickbait_hits = sum(1 for w in lower_words if w in CLICKBAIT_LEXICON)

    return np.array(
        [
            n_chars,
            n_words,
            sum(len(w) for w in words) / n_words,
            (n_upper / n_alpha) if n_alpha else 0.0,
            raw.count("!") / n_words,
            raw.count("?") / n_words,
            sum(c.isdigit() for c in raw) / max(n_chars, 1),
            clickbait_hits / n_words,
            len(set(lower_words)) / n_words,
        ],
        dtype=np.float64,
    )


class StylometricFeatures(BaseEstimator, TransformerMixin):
    """Turn raw documents into the interpretable style-statistics matrix."""

    def fit(self, X: Sequence[str], y=None):  # noqa: N803 (sklearn convention)
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:  # noqa: N803
        return np.vstack([_stylometric_vector(doc) for doc in X])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(STYLOMETRIC_FEATURE_NAMES, dtype=object)


class TextCleaner(BaseEstimator, TransformerMixin):
    """Pipeline-friendly wrapper around :func:`fakenews.preprocess.clean_text`."""

    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        return [clean_text(doc, remove_stopwords=self.remove_stopwords) for doc in X]


def build_tfidf(config: ModelConfig) -> TfidfVectorizer:
    """Construct the TF-IDF vectoriser from a :class:`ModelConfig`."""
    return TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
    )
