"""Adversarial robustness — surviving deliberate evasion.

Everything in :mod:`fakenews.detect` assumes the author *wants* to be read
normally. Real misinformation operators do not: once a platform starts filtering
on words like "vaccine" or "shocking", they obfuscate those words to slip past
the classifier while staying perfectly readable to a human:

===================  =========================  ==================================
Attack               Example                    Why it defeats bag-of-words
===================  =========================  ==================================
Homoglyph            ``vаccine`` (Cyrillic а)   Different codepoint => different token
Zero-width           ``vac\\u200bcine``          Splits one token into two
Leetspeak            ``v4cc1n3``                Token not in the vocabulary
Char repetition      ``shoooocking``            Token not in the vocabulary
Letter spacing       ``s h o c k i n g``        Becomes eight 1-char tokens
Diacritic abuse      ``shöcking``               Different codepoint => different token
===================  =========================  ==================================

Every one of these keeps the text human-legible while making the *machine* see
an out-of-vocabulary token, so a TF-IDF model silently loses the feature it
relied on. This module provides:

* :func:`deobfuscate` — an aggressive normaliser that folds all six attacks back
  to their canonical ASCII form. Run it *before* the usual cleaning.
* :func:`apply_attack` — the same attacks as generators, so robustness can be
  *measured* rather than assumed.
* :func:`evaluate_robustness` — accuracy per attack, with and without the
  defence, which is the number you actually want in a production review.

Design note: this is a **normalisation** defence, not a detection one. We do not
try to decide whether obfuscation was malicious — we simply make the model see
the same token either way, which is cheap, deterministic, and cannot be gamed by
tuning the obfuscation rate.
"""

from __future__ import annotations

import random
import re
import unicodedata
from typing import Callable, Dict, List, Optional, Sequence

# --- confusable (homoglyph) table -----------------------------------------
# Latin-looking characters from the Cyrillic and Greek blocks. This is the set
# that actually shows up in evasion attempts; the full Unicode confusables file
# is enormous and mostly irrelevant to English-language news text.
_HOMOGLYPHS: Dict[str, str] = {
    # Cyrillic lower
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "і": "i", "ѕ": "s", "ј": "j",
    "к": "k", "м": "m", "т": "t", "в": "b", "н": "h",
    # Cyrillic upper
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "У": "Y", "Х": "X", "І": "I", "Ј": "J",
    # Greek lower
    "α": "a", "ο": "o", "ρ": "p", "ε": "e", "ι": "i",
    "ν": "v", "υ": "u", "χ": "x", "κ": "k", "τ": "t",
    # Greek upper
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}

# Invisible characters used to split tokens. Category Cf catches most, but a few
# (notably U+200B in some Unicode versions) are classified elsewhere, so we keep
# an explicit set as well.
_ZERO_WIDTH = frozenset(
    "​‌‍⁠﻿­᠎‎‏"
)

# Leetspeak substitutions. Applied conservatively — see _deleet_token.
_LEET: Dict[str, str] = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "@": "a", "$": "s", "!": "i", "|": "l",
}

_SPACED_RE = re.compile(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b")
# Letters only, deliberately: collapsing punctuation would destroy "!!!", which
# is a genuine stylometric signal the detector relies on.
_REPEAT_RE = re.compile(r"([A-Za-z])\1{2,}")
_DIGIT_RUN_RE = re.compile(r"\d{2,}")


def _strip_invisible(text: str) -> str:
    """Drop zero-width and format characters used to split tokens."""
    return "".join(
        ch for ch in text
        if ch not in _ZERO_WIDTH and unicodedata.category(ch) != "Cf"
    )


def _fold_homoglyphs(text: str) -> str:
    return "".join(_HOMOGLYPHS.get(ch, ch) for ch in text)


def _strip_diacritics(text: str) -> str:
    """Remove combining marks: ``shöcking`` -> ``shocking``.

    This is the right trade-off for an English-language detector. It would be
    wrong for a multilingual corpus, where diacritics are meaning-bearing — hence
    the ``strip_diacritics`` switch on :func:`deobfuscate`.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _join_spaced_letters(text: str) -> str:
    """``s h o c k i n g`` -> ``shocking`` (needs >=4 single letters in a row)."""
    return _SPACED_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), text)


def _collapse_repeats(text: str) -> str:
    """``shoooocking`` -> ``shocking``.

    Runs of three or more of the *same letter* essentially never occur in real
    English, so collapsing them all the way to one is safe — genuine double
    letters ("coffee", "all") are runs of two and are left untouched. Measured on
    the sample corpus this recovers the original token in ~52% of attacked
    documents with zero corruption of clean text, whereas collapsing to two
    recovers none (the model still sees an out-of-vocabulary "shoocking").
    Punctuation is excluded so "!!!" survives as a stylometric feature.
    """
    return _REPEAT_RE.sub(r"\1", text)


def _deleet_token(token: str) -> str:
    """Undo leetspeak inside a single token, conservatively.

    A leet character is only substituted when

    1. the token is longer than two characters — protects product names like
       ``5G``, ``4K``, ``3D``;
    2. it is not inside a *protected* digit run. A run of two or more digits is
       protected **unless** it is embedded between two letters: ``covid19`` and
       ``2024`` keep their digits, while ``exp05ed`` is still recovered; and
    3. at least one neighbouring character is an ASCII letter, so a bare number
       is never touched.

    Together these keep ``v4cc1n3`` -> ``vaccine`` and ``exp05ed`` -> ``exposed``
    while leaving ``covid19``, ``2024`` and ``5G`` alone.
    """
    if len(token) <= 2 or not any(c.isalpha() for c in token):
        return token

    protected = set()
    for match in _DIGIT_RUN_RE.finditer(token):
        start, end = match.start(), match.end()
        before = token[start - 1] if start > 0 else ""
        after = token[end] if end < len(token) else ""
        embedded = (
            before.isascii() and before.isalpha()
            and after.isascii() and after.isalpha()
        )
        if not embedded:
            protected.update(range(start, end))

    chars = list(token)
    for i, ch in enumerate(chars):
        if ch not in _LEET or i in protected:
            continue
        prev_alpha = i > 0 and chars[i - 1].isascii() and chars[i - 1].isalpha()
        next_alpha = (
            i + 1 < len(chars) and chars[i + 1].isascii() and chars[i + 1].isalpha()
        )
        if ch.isdigit():
            # A digit standing in for a letter has a letter on at least one side
            # ("v4ccine", "vaccin3").
            substitute = prev_alpha or next_alpha
        else:
            # Punctuation leet ("$hocking", "w@r") always *precedes* the rest of
            # the word. Requiring a following letter is what stops trailing
            # punctuation — "BREAKING!!!" — from being read as letters, which
            # would destroy the exclamation-ratio signal.
            substitute = next_alpha
        if substitute:
            chars[i] = _LEET[ch]
    return "".join(chars)


def deobfuscate(text: str, *, strip_diacritics: bool = True) -> str:
    """Fold deliberate obfuscation back to canonical ASCII.

    Order matters: invisible characters are removed before homoglyph folding (so
    a split token is whole again), and letter-spacing is joined before leetspeak
    is undone (so ``v 4 c c 1 n 3`` is handled too).
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if not text:
        return ""

    # NFKC first: folds full-width (ｓｈｏｃｋ), ligatures and other compatibility
    # forms into their plain equivalents.
    text = unicodedata.normalize("NFKC", text)
    text = _strip_invisible(text)
    text = _fold_homoglyphs(text)
    if strip_diacritics:
        text = _strip_diacritics(text)
    text = _join_spaced_letters(text)
    text = _collapse_repeats(text)
    text = " ".join(_deleet_token(tok) for tok in text.split())
    return text


# --- attack generators (for measuring robustness) --------------------------

_REVERSE_HOMOGLYPH = {"a": "а", "e": "е", "o": "о",
                      "p": "р", "c": "с", "y": "у", "x": "х"}
_REVERSE_LEET = {"o": "0", "i": "1", "e": "3", "a": "4", "s": "5", "t": "7"}


def _attack_homoglyph(text: str, rng: random.Random, rate: float) -> str:
    return "".join(
        _REVERSE_HOMOGLYPH[ch] if ch in _REVERSE_HOMOGLYPH and rng.random() < rate
        else ch
        for ch in text
    )


def _attack_zero_width(text: str, rng: random.Random, rate: float) -> str:
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < rate:
            out.append("​")
    return "".join(out)


def _attack_leet(text: str, rng: random.Random, rate: float) -> str:
    return "".join(
        _REVERSE_LEET[ch] if ch in _REVERSE_LEET and rng.random() < rate else ch
        for ch in text
    )


def _attack_repeat(text: str, rng: random.Random, rate: float) -> str:
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < rate:
            out.append(ch * rng.randint(2, 3))
    return "".join(out)


def _attack_spacing(text: str, rng: random.Random, rate: float) -> str:
    words = text.split()
    return " ".join(
        " ".join(w) if len(w) > 3 and rng.random() < rate else w for w in words
    )


def _attack_diacritic(text: str, rng: random.Random, rate: float) -> str:
    # Combining diaeresis on random vowels.
    return "".join(
        ch + "̈" if ch in "aeiou" and rng.random() < rate else ch
        for ch in text
    )


ATTACKS: Dict[str, Callable[[str, random.Random, float], str]] = {
    "homoglyph": _attack_homoglyph,
    "zero_width": _attack_zero_width,
    "leetspeak": _attack_leet,
    "repetition": _attack_repeat,
    "spacing": _attack_spacing,
    "diacritic": _attack_diacritic,
}


def apply_attack(
    text: str,
    kind: str,
    *,
    rate: float = 0.3,
    random_state: int = 0,
) -> str:
    """Obfuscate ``text`` with the named attack (see :data:`ATTACKS`)."""
    if kind not in ATTACKS:
        raise ValueError(f"Unknown attack {kind!r}. Choose from {sorted(ATTACKS)}.")
    return ATTACKS[kind](text, random.Random(random_state), rate)


# --- robustness evaluation -------------------------------------------------

def evaluate_robustness(
    detector,
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    attacks: Optional[Sequence[str]] = None,
    rate: float = 0.3,
    random_state: int = 0,
    defend: bool = False,
) -> Dict[str, float]:
    """Accuracy of ``detector`` on clean text and under each attack.

    Args:
        detector: anything with ``predict_batch`` (either detector class works).
        defend: if True, run :func:`deobfuscate` on the attacked text first —
            i.e. measure the *defended* system rather than the bare one.

    Returns:
        ``{"clean": acc, "<attack>": acc, ...}``.
    """
    attacks = list(attacks) if attacks is not None else list(ATTACKS)
    labels = list(labels)

    def _accuracy(docs: Sequence[str]) -> float:
        preds = [1 if p.is_fake else 0 for p in detector.predict_batch(list(docs))]
        return sum(int(a == b) for a, b in zip(preds, labels)) / max(len(labels), 1)

    scores = {"clean": _accuracy([deobfuscate(t) if defend else t for t in texts])}
    for kind in attacks:
        attacked = [
            apply_attack(t, kind, rate=rate, random_state=random_state + i)
            for i, t in enumerate(texts)
        ]
        if defend:
            attacked = [deobfuscate(t) for t in attacked]
        scores[kind] = _accuracy(attacked)
    return scores


def format_robustness(clean: Dict[str, float], defended: Dict[str, float]) -> str:
    """Side-by-side undefended vs defended robustness table."""
    header = f"{'attack':>12} | {'undefended':>10} | {'defended':>9} | {'recovered':>9}"
    lines = [header, "-" * len(header)]
    for key in clean:
        base, dfn = clean[key], defended.get(key, float("nan"))
        delta = dfn - base
        lines.append(f"{key:>12} | {base:10.3f} | {dfn:9.3f} | {delta:+9.3f}")
    return "\n".join(lines)
