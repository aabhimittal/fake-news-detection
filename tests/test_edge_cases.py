"""Industrial edge cases.

The other test modules check that the happy path is *correct*. This one checks
that the system behaves sanely on the inputs production actually delivers:
empty fields, 200 kB documents, emoji, mixed scripts, malformed CSVs,
single-class training data, degenerate graphs and concurrent requests.

The standard each case is held to is one of:

* **works** — produces a sensible result, or
* **fails loudly** — raises a clear, typed error the caller can handle.

What is never acceptable is a silent wrong answer, a crash deep inside a
third-party library, or a hang.
"""

import threading

import numpy as np
import pandas as pd
import pytest

from fakenews.adversarial import apply_attack, deobfuscate, evaluate_robustness
from fakenews.config import ModelConfig, PropagationConfig
from fakenews.data import generate_synthetic_dataset, load_dataset
from fakenews.detect import FakeNewsDetector
from fakenews.features import _stylometric_vector
from fakenews.preprocess import clean_text
from fakenews.propagation import (
    build_social_graph,
    choose_seeds,
    select_monitors,
    simulate,
)
from fakenews.triage import (
    TriagePolicy,
    brier_score,
    cost_optimal_threshold,
    evaluate_policy,
    expected_calibration_error,
    fit_policy,
)


# --------------------------------------------------------------------------
# Degenerate text input
# --------------------------------------------------------------------------

DEGENERATE_TEXTS = [
    "",                       # empty
    "   ",                    # whitespace only
    "\n\t\r\n",               # whitespace control chars
    "a",                      # single char
    "123456",                 # digits only
    "!!!???...",              # punctuation only
    "\x00\x01\x02",           # control bytes
    "<p>markup &amp; entities</p>",
    "https://example.com",    # URL only, stripped to nothing by the cleaner
    "@user #tag",             # handles only
]

UNICODE_TEXTS = [
    "🔥🔥 BREAKING 🔥🔥",       # emoji
    "假新闻正在传播",            # CJK
    "أخبار كاذبة عاجلة",        # RTL Arabic
    "Ψευδείς ειδήσεις",        # Greek
    "Новости фейк",            # Cyrillic
    "🇺🇸 flag sequence 👨‍👩‍👧‍👦 ZWJ family",
    "é́́combining marks",
]


@pytest.mark.parametrize("text", DEGENERATE_TEXTS + UNICODE_TEXTS)
def test_clean_text_never_raises(text):
    out = clean_text(text)
    assert isinstance(out, str)


@pytest.mark.parametrize("text", DEGENERATE_TEXTS + UNICODE_TEXTS)
def test_stylometric_features_are_finite(text):
    """No NaN/inf from division by zero on empty or exotic documents."""
    vec = _stylometric_vector(text)
    assert np.all(np.isfinite(vec)), f"non-finite stylometric features for {text!r}"


@pytest.mark.parametrize("text", DEGENERATE_TEXTS + UNICODE_TEXTS)
def test_deobfuscate_never_raises(text):
    assert isinstance(deobfuscate(text), str)


def test_clean_text_handles_non_string_types():
    for value in (None, 123, 4.5, True):
        assert isinstance(clean_text(value), str)


def test_stylometric_handles_none():
    assert np.all(np.isfinite(_stylometric_vector(None)))


def test_very_long_document():
    """A 200 kB document must not blow up or take pathological time."""
    huge = ("BREAKING shocking secret conspiracy exposed!!! " * 4000)[:200_000]
    assert len(clean_text(huge)) > 0
    assert np.all(np.isfinite(_stylometric_vector(huge)))
    assert isinstance(deobfuscate(huge), str)


def test_detector_predicts_on_degenerate_text():
    """An empty or exotic document must still get a well-formed prediction."""
    detector = FakeNewsDetector(ModelConfig(random_state=0))
    detector.fit(generate_synthetic_dataset(n_per_class=80, random_state=0))
    for text in DEGENERATE_TEXTS + UNICODE_TEXTS:
        pred = detector.predict(text)
        assert pred.label in ("fake", "real")
        assert 0.0 <= pred.confidence <= 1.0
        assert np.isfinite(pred.confidence)


def test_predict_batch_matches_predict():
    """Batch and single-document paths must agree exactly."""
    detector = FakeNewsDetector(ModelConfig(random_state=0))
    detector.fit(generate_synthetic_dataset(n_per_class=80, random_state=0))
    texts = ["SHOCKING secret!!!", "The council met today.", "", "🔥"]
    batch = detector.predict_batch(texts)
    for text, batched in zip(texts, batch):
        single = detector.predict(text)
        assert single.label == batched.label
        assert single.confidence == pytest.approx(batched.confidence, abs=1e-9)


def test_empty_batch_returns_empty():
    detector = FakeNewsDetector(ModelConfig(random_state=0))
    detector.fit(generate_synthetic_dataset(n_per_class=60, random_state=0))
    assert detector.predict_batch([]) == []
    assert len(detector.predict_proba([])) == 0


# --------------------------------------------------------------------------
# Degenerate datasets
# --------------------------------------------------------------------------

def test_single_class_training_fails_loudly():
    """Training on one class can't work — it must raise, not return garbage."""
    df = pd.DataFrame({"text": ["a fake story"] * 20, "label": [1] * 20})
    with pytest.raises(ValueError):
        FakeNewsDetector(ModelConfig(random_state=0)).fit(df)


def test_extreme_class_imbalance_still_trains():
    """99:1 imbalance is normal in moderation; it must train and predict."""
    rows = [{"text": f"the council reported figure {i}", "label": 0} for i in range(198)]
    rows += [{"text": "SHOCKING secret conspiracy!!!", "label": 1} for _ in range(2)]
    result = FakeNewsDetector(ModelConfig(random_state=0)).fit(pd.DataFrame(rows))
    assert 0.0 <= result.accuracy <= 1.0
    # Metrics must be defined (zero_division guard) even if the rare class is missed.
    assert np.isfinite(result.precision_fake)
    assert np.isfinite(result.recall_fake)


def test_duplicate_rows_do_not_break_training():
    df = generate_synthetic_dataset(n_per_class=40, random_state=0)
    doubled = pd.concat([df, df], ignore_index=True)
    assert FakeNewsDetector(ModelConfig(random_state=0)).fit(doubled).accuracy >= 0.0


def test_missing_column_raises_clear_error(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("headline,verdict\nsome text,1\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_dataset(csv)


def test_nan_rows_are_dropped(tmp_path):
    csv = tmp_path / "nan.csv"
    csv.write_text("text,label\nreal story,0\n,1\nfake story,\nanother,1\n")
    df = load_dataset(csv)
    assert not df["text"].isna().any()
    assert not df["label"].isna().any()
    assert len(df) == 2  # only the fully-populated rows survive


def test_float_and_string_labels_are_coerced(tmp_path):
    csv = tmp_path / "labels.csv"
    csv.write_text("text,label\nreal story,0.0\nfake story,1.0\n")
    df = load_dataset(csv)
    assert df["label"].tolist() == [0, 1]
    assert df["label"].dtype.kind in "iu"


def test_embedded_delimiters_and_newlines_survive_csv(tmp_path):
    """Quoted commas and newlines inside a document must round-trip."""
    csv = tmp_path / "tricky.csv"
    csv.write_text(
        'text,label\n"a story, with a comma\nand a newline",1\n"plain",0\n'
    )
    df = load_dataset(csv)
    assert len(df) == 2
    assert "," in df.iloc[0]["text"] and "\n" in df.iloc[0]["text"]


# --------------------------------------------------------------------------
# Serving: determinism, persistence, concurrency
# --------------------------------------------------------------------------

def test_training_is_deterministic():
    """Same seed => identical probabilities. Required for reproducible incidents."""
    df = generate_synthetic_dataset(n_per_class=100, random_state=3)
    texts = ["SHOCKING secret exposed!!!", "The ministry published figures."]
    a = FakeNewsDetector(ModelConfig(random_state=7)); a.fit(df)
    b = FakeNewsDetector(ModelConfig(random_state=7)); b.fit(df)
    np.testing.assert_allclose(a.predict_proba(texts), b.predict_proba(texts))


def test_save_load_preserves_predictions_exactly(tmp_path):
    detector = FakeNewsDetector(ModelConfig(random_state=0))
    detector.fit(generate_synthetic_dataset(n_per_class=80, random_state=0))
    texts = ["BREAKING bombshell!!!", "Officials confirmed the report.", ""]
    before = detector.predict_proba(texts)

    restored = FakeNewsDetector.load(detector.save(tmp_path / "m.joblib"))
    np.testing.assert_allclose(before, restored.predict_proba(texts))


def test_concurrent_predictions_are_consistent():
    """A loaded model is read-only at serve time; parallel calls must not corrupt it."""
    detector = FakeNewsDetector(ModelConfig(random_state=0))
    detector.fit(generate_synthetic_dataset(n_per_class=80, random_state=0))
    texts = ["SHOCKING secret conspiracy!!!"] * 20
    expected = detector.predict_proba(texts)

    results, errors = [], []

    def worker():
        try:
            results.append(detector.predict_proba(texts))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent predictions raised: {errors}"
    for got in results:
        np.testing.assert_allclose(got, expected)


def test_large_batch_prediction():
    detector = FakeNewsDetector(ModelConfig(random_state=0))
    detector.fit(generate_synthetic_dataset(n_per_class=60, random_state=0))
    probs = detector.predict_proba(["SHOCKING secret!!!"] * 5000)
    assert len(probs) == 5000
    assert np.all(np.isfinite(probs))


def test_explain_on_empty_text_returns_empty_not_error():
    detector = FakeNewsDetector(ModelConfig(random_state=0))
    detector.fit(generate_synthetic_dataset(n_per_class=60, random_state=0))
    assert detector.explain("") == []


# --------------------------------------------------------------------------
# Graph / propagation degeneracies
# --------------------------------------------------------------------------

def _tiny_config(**kw):
    base = dict(n_nodes=60, n_simulations=3, max_steps=6, random_state=1)
    base.update(kw)
    return PropagationConfig(**base)


def test_monitor_budget_larger_than_graph():
    """Asking for more monitors than there are nodes must clamp, not crash."""
    cfg = _tiny_config(strategy="degree", n_monitors=10_000)
    graph = build_social_graph(cfg)
    monitors = select_monitors(graph, cfg, exclude=set())
    assert len(monitors) <= graph.number_of_nodes()


@pytest.mark.parametrize("strategy", ["degree", "betweenness", "random", "acquaintance", "greedy"])
def test_zero_budget_selects_nobody(strategy):
    cfg = _tiny_config(strategy=strategy, n_monitors=0, greedy_sims=2, greedy_pool=10)
    graph = build_social_graph(cfg)
    assert select_monitors(graph, cfg, exclude=set()) == set()


def test_unknown_strategy_raises():
    cfg = _tiny_config(strategy="does-not-exist")
    graph = build_social_graph(cfg)
    with pytest.raises(ValueError, match="Unknown strategy"):
        select_monitors(graph, cfg, exclude=set())


def test_disconnected_graph_cascade_is_bounded():
    """A cascade can never exceed the seed's connected component."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])        # component A (3 nodes)
    graph.add_edges_from([(10, 11), (11, 12)])    # component B, unreachable
    cfg = _tiny_config(activation_prob=1.0, recovery_prob=0.0, n_simulations=3)
    result = simulate(graph, cfg, seeds=[0])
    assert result.total_reached <= 3


def test_single_node_graph():
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0)
    cfg = _tiny_config(n_monitors=1, n_simulations=2)
    result = simulate(graph, cfg, seeds=[0])
    assert result.total_reached == 1  # the seed itself, nobody else


def test_seeds_outnumbering_nodes_are_handled():
    cfg = _tiny_config(n_seeds=10_000)
    graph = build_social_graph(cfg)
    seeds = choose_seeds(graph, cfg)
    assert len(seeds) <= graph.number_of_nodes()


def test_activation_probability_extremes():
    """p=0 must never spread; p=1 with no recovery must saturate the component."""
    graph = build_social_graph(_tiny_config())
    never = simulate(graph, _tiny_config(activation_prob=0.0, strategy="none"), seeds=[0])
    always = simulate(
        graph,
        _tiny_config(activation_prob=1.0, recovery_prob=0.0, strategy="none", max_steps=30),
        seeds=[0],
    )
    assert never.total_reached == 1
    assert always.total_reached == graph.number_of_nodes()


# --------------------------------------------------------------------------
# Triage numerical edge cases
# --------------------------------------------------------------------------

def test_metrics_on_empty_input_are_nan_not_crash():
    assert np.isnan(brier_score([], []))
    assert np.isnan(expected_calibration_error([], []))


def test_policy_rejects_inverted_band():
    with pytest.raises(ValueError):
        TriagePolicy(low=0.9, high=0.1)


def test_policy_with_all_identical_probabilities():
    """A model that outputs 0.5 for everything must abstain, not guess."""
    y = [0, 1] * 25
    p = [0.5] * 50
    report = evaluate_policy(TriagePolicy(0.2, 0.8), y, p)
    assert report.coverage == 0.0
    assert report.n_review == 50
    assert np.isnan(report.automated_accuracy)  # nothing automated => undefined


def test_fit_policy_on_single_class_labels():
    """All-real validation data must not crash the policy search."""
    y = [0] * 40
    p = list(np.linspace(0.0, 0.4, 40))
    policy = fit_policy(y, p, max_error_rate=0.05)
    assert 0.0 <= policy.low <= policy.high <= 1.0


def test_cost_threshold_moves_with_asymmetric_costs():
    """Expensive false negatives must push the threshold down (catch more fakes)."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.25, 300), 0, 1)
    t_balanced, _ = cost_optimal_threshold(y, p, cost_false_positive=1, cost_false_negative=1)
    t_fn_costly, _ = cost_optimal_threshold(y, p, cost_false_positive=1, cost_false_negative=20)
    assert t_fn_costly <= t_balanced


def test_probabilities_at_exact_boundaries():
    policy = TriagePolicy(low=0.2, high=0.8)
    assert policy.route(0.8) == "auto_fake"   # inclusive upper bound
    assert policy.route(0.2) == "auto_real"   # inclusive lower bound
    assert policy.route(0.5) == "review"
    assert policy.route(0.0) == "auto_real"
    assert policy.route(1.0) == "auto_fake"


# --------------------------------------------------------------------------
# Adversarial edge cases
# --------------------------------------------------------------------------

def test_deobfuscate_is_idempotent():
    """Normalising twice must equal normalising once — required for stable caching."""
    for text in ["v4cc1n3", "SHOCKING!!!", "s h o c k i n g", "vаccine", "🔥 fine"]:
        once = deobfuscate(text)
        assert deobfuscate(once) == once


def test_deobfuscate_preserves_legitimate_tokens():
    """Regression guard: the normaliser must not corrupt ordinary text."""
    for text in ["covid19", "2024", "5G network", "COVID-19", "he paid $5", "100%"]:
        assert deobfuscate(text) == text


def test_attacks_do_not_change_length_class_or_crash():
    text = "SHOCKING vaccine conspiracy exposed"
    for kind in ("homoglyph", "zero_width", "leetspeak", "repetition", "spacing", "diacritic"):
        attacked = apply_attack(text, kind, rate=0.5, random_state=0)
        assert isinstance(attacked, str) and attacked
        # A zero rate must be an exact no-op.
        assert apply_attack(text, kind, rate=0.0, random_state=0) == text


def test_unknown_attack_raises():
    with pytest.raises(ValueError, match="Unknown attack"):
        apply_attack("text", "not-an-attack")


def test_robustness_evaluation_on_empty_corpus():
    detector = FakeNewsDetector(ModelConfig(random_state=0))
    detector.fit(generate_synthetic_dataset(n_per_class=60, random_state=0))
    scores = evaluate_robustness(detector, [], [], attacks=["leetspeak"])
    assert set(scores) == {"clean", "leetspeak"}
