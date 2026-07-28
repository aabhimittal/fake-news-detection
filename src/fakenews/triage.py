"""Production triage — calibrated confidence, abstention, and cost-aware cutoffs.

A research classifier answers "fake or real?". A deployed moderation system has
to answer a harder question: **"is this call safe to make automatically, or does
it need a human?"** Three things are missing from a bare classifier:

1. **Calibration.** A model's 0.9 should mean "right about 90% of the time".
   Margin-based classifiers are usually over-confident, so the raw score is not
   a probability and thresholds set on it do not mean what you think.
   :class:`ProbabilityCalibrator` fixes the scores; :func:`expected_calibration_error`
   and :func:`brier_score` measure whether it worked.

2. **Abstention.** Acting on a 51% score is indefensible. A
   :class:`TriagePolicy` auto-decides only outside a middle band and routes the
   uncertain remainder to a human queue. :func:`fit_policy` chooses the *widest*
   band that keeps the automated error rate under a stated budget — turning
   "how accurate is the model" into "how much can we safely automate".

3. **Asymmetric costs.** Missing a fake story and wrongly flagging a real one
   are not equally bad, and the ratio is a policy decision, not a modelling one.
   :func:`cost_optimal_threshold` picks the cutoff that minimises expected cost.

Everything here operates on ``(y_true, p_fake)`` arrays, so it works with either
detector — or any external scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Routing decisions produced by a TriagePolicy.
AUTO_FAKE = "auto_fake"
AUTO_REAL = "auto_real"
REVIEW = "review"


# --- calibration -----------------------------------------------------------

def brier_score(y_true: Sequence[int], p_fake: Sequence[float]) -> float:
    """Mean squared error of the probabilities (lower is better)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_fake, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: Sequence[int],
    p_fake: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error: average |confidence - accuracy| over bins.

    Scores are bucketed by predicted confidence; within each bucket we compare
    the mean predicted probability to the observed frequency. ECE is the
    sample-weighted mean of those gaps — 0 is perfectly calibrated.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_fake, dtype=float)
    if y.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin by *confidence in the predicted class*, the standard ECE formulation.
    pred = (p >= 0.5).astype(float)
    confidence = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y).astype(float)

    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


class ProbabilityCalibrator:
    """Map raw scores to calibrated probabilities.

    Two methods:

    * ``"isotonic"`` — a monotone step function fitted to the validation data.
      Flexible and non-parametric, but needs a few hundred points or it overfits.
    * ``"platt"`` — a one-dimensional logistic regression (Platt scaling). Two
      parameters only, so it is the right choice on small validation sets.

    Fit on a *held-out* split; calibrating on the training scores just relearns
    the model's own over-confidence.
    """

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "platt"):
            raise ValueError(f"Unknown method {method!r}: use 'isotonic' or 'platt'.")
        self.method = method
        self._model = None

    def fit(self, p_fake: Sequence[float], y_true: Sequence[int]) -> "ProbabilityCalibrator":
        p = np.asarray(p_fake, dtype=float).reshape(-1, 1)
        y = np.asarray(y_true, dtype=int)
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._model.fit(p.ravel(), y)
        else:
            from sklearn.linear_model import LogisticRegression

            self._model = LogisticRegression(max_iter=1000)
            self._model.fit(p, y)
        return self

    def transform(self, p_fake: Sequence[float]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator is not fitted. Call fit() first.")
        p = np.asarray(p_fake, dtype=float)
        if self.method == "isotonic":
            out = self._model.predict(p)
        else:
            out = self._model.predict_proba(p.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    def fit_transform(self, p_fake, y_true) -> np.ndarray:
        return self.fit(p_fake, y_true).transform(p_fake)


# --- abstention / triage policy -------------------------------------------

@dataclass
class TriagePolicy:
    """Route each document to an automated decision or a human queue.

    ``p_fake >= high`` -> auto-remove as fake; ``p_fake <= low`` -> auto-approve
    as real; anything between is sent for review.
    """

    low: float = 0.2
    high: float = 0.8

    def __post_init__(self):
        if not 0.0 <= self.low <= self.high <= 1.0:
            raise ValueError(
                f"Need 0 <= low <= high <= 1, got low={self.low}, high={self.high}."
            )

    def route(self, p_fake: float) -> str:
        if p_fake >= self.high:
            return AUTO_FAKE
        if p_fake <= self.low:
            return AUTO_REAL
        return REVIEW

    def route_all(self, p_fake: Sequence[float]) -> List[str]:
        return [self.route(float(p)) for p in p_fake]


@dataclass
class CoverageReport:
    """How much work the policy automates, and how well."""

    coverage: float          # fraction decided automatically
    automated_accuracy: float
    review_rate: float
    n_auto: int
    n_review: int
    n_auto_errors: int

    def summary(self) -> str:
        return (
            f"coverage={self.coverage:.1%}  "
            f"auto-accuracy={self.automated_accuracy:.3f}  "
            f"to-review={self.review_rate:.1%} "
            f"({self.n_review} of {self.n_auto + self.n_review})"
        )


def evaluate_policy(
    policy: TriagePolicy,
    y_true: Sequence[int],
    p_fake: Sequence[float],
) -> CoverageReport:
    """Measure coverage and automated accuracy for a policy."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_fake, dtype=float)
    routes = np.array(policy.route_all(p))

    auto_mask = routes != REVIEW
    n_auto = int(auto_mask.sum())
    n_review = int((~auto_mask).sum())
    total = max(n_auto + n_review, 1)

    if n_auto:
        auto_pred = (routes[auto_mask] == AUTO_FAKE).astype(int)
        errors = int((auto_pred != y[auto_mask]).sum())
        accuracy = 1.0 - errors / n_auto
    else:
        errors, accuracy = 0, float("nan")

    return CoverageReport(
        coverage=n_auto / total,
        automated_accuracy=accuracy,
        review_rate=n_review / total,
        n_auto=n_auto,
        n_review=n_review,
        n_auto_errors=errors,
    )


def fit_policy(
    y_true: Sequence[int],
    p_fake: Sequence[float],
    *,
    max_error_rate: float = 0.02,
    n_steps: int = 50,
) -> TriagePolicy:
    """Widest auto-decision band whose automated error rate stays within budget.

    This inverts the usual question. Instead of "how accurate is the model?" it
    answers **"given that we tolerate at most `max_error_rate` mistakes on the
    calls we make automatically, how much of the queue can we automate?"** —
    which is the number an operations team actually plans against.

    Symmetric bands (``low = 1 - high``) are searched from widest to narrowest,
    and the first one meeting the budget wins. If none does, the most
    conservative band is returned, so the caller degrades to "review everything"
    rather than silently exceeding the budget.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_fake, dtype=float)

    # Widest coverage first: high=0.5 means automate everything.
    for high in np.linspace(0.5, 0.999, n_steps):
        candidate = TriagePolicy(low=float(1.0 - high), high=float(high))
        report = evaluate_policy(candidate, y, p)
        if report.n_auto == 0:
            continue
        if (1.0 - report.automated_accuracy) <= max_error_rate:
            return candidate
    return TriagePolicy(low=0.001, high=0.999)


def risk_coverage_curve(
    y_true: Sequence[int],
    p_fake: Sequence[float],
    n_points: int = 20,
) -> List[Tuple[float, float]]:
    """``[(coverage, error_rate), ...]`` as the abstention band widens.

    The classic selective-prediction diagnostic: a good model's error rate falls
    steeply as you abstain more, meaning its confidence is informative.
    """
    out: List[Tuple[float, float]] = []
    for high in np.linspace(0.5, 0.999, n_points):
        report = evaluate_policy(
            TriagePolicy(low=float(1.0 - high), high=float(high)), y_true, p_fake
        )
        if report.n_auto:
            out.append((report.coverage, 1.0 - report.automated_accuracy))
    return out


# --- cost-sensitive thresholds --------------------------------------------

def cost_optimal_threshold(
    y_true: Sequence[int],
    p_fake: Sequence[float],
    *,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 1.0,
    n_steps: int = 101,
) -> Tuple[float, float]:
    """Threshold minimising expected cost, plus that cost per item.

    A false positive censors legitimate news; a false negative lets
    misinformation spread. Their relative cost is a *policy* choice, and it moves
    the optimal cutoff away from the default 0.5.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_fake, dtype=float)
    if y.size == 0:
        return 0.5, float("nan")

    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.0, 1.0, n_steps):
        pred = (p >= t).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        cost = (fp * cost_false_positive + fn * cost_false_negative) / y.size
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t, best_cost


def triage_report(
    y_true: Sequence[int],
    p_fake: Sequence[float],
    policy: Optional[TriagePolicy] = None,
) -> Dict[str, object]:
    """One-call summary: calibration quality plus routing behaviour."""
    policy = policy or fit_policy(y_true, p_fake)
    report = evaluate_policy(policy, y_true, p_fake)
    return {
        "policy": policy,
        "coverage": report.coverage,
        "automated_accuracy": report.automated_accuracy,
        "review_rate": report.review_rate,
        "brier": brier_score(y_true, p_fake),
        "ece": expected_calibration_error(y_true, p_fake),
    }
