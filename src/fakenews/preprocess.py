"""Text normalisation.

The classifier is only as good as the text it sees. This module performs the
cheap, deterministic cleaning steps that help a bag-of-words model generalise:
lower-casing, stripping URLs / handles, collapsing whitespace and removing
noise characters. We deliberately *keep* punctuation counts available to the
stylometric feature extractor, so cleaning here is conservative.
"""

from __future__ import annotations

import re
from typing import Iterable, List

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HANDLE_RE = re.compile(r"[@#]\w+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTISPACE_RE = re.compile(r"\s+")

# A very small, dependency-free stop-word list. Using our own avoids pulling in
# NLTK downloads at runtime (which fail in offline / CI environments).
STOP_WORDS = frozenset(
    """
    a an and are as at be by for from has he in is it its of on that the to was
    were will with this these those they them their her his she you your i we our
    but or if then than so such not no nor can could would should do does did done
    """.split()
)


def clean_text(text: str, *, remove_stopwords: bool = False) -> str:
    """Return a normalised version of ``text`` suitable for TF-IDF.

    Args:
        text: raw document.
        remove_stopwords: drop common English stop words. Left off by default
            because TF-IDF already down-weights them and n-grams benefit from
            keeping the connective tissue.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HANDLE_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()

    if remove_stopwords and text:
        text = " ".join(tok for tok in text.split() if tok not in STOP_WORDS)
    return text


def clean_corpus(texts: Iterable[str], **kwargs) -> List[str]:
    """Vectorised convenience wrapper around :func:`clean_text`."""
    return [clean_text(t, **kwargs) for t in texts]
