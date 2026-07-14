# Architecture & design notes

This document explains *how* the code is organised and the reasoning behind the
main engineering decisions. For the theory, see [`CONCEPTS.md`](CONCEPTS.md).

## Module map

```
fakenews/
├── config.py        ModelConfig, PropagationConfig  — all hyper-parameters
├── data.py          synthetic generator + CSV loader
├── preprocess.py    clean_text() — deterministic normalisation
├── features.py      TextCleaner, StylometricFeatures, build_tfidf()
├── models.py        build_pipeline(), save_model(), load_model()
├── evaluate.py      evaluate() -> EvaluationResult
├── detect.py        FakeNewsDetector — the public façade
├── propagation.py   graph + Independent Cascade + containment strategies
└── cli.py           argparse entry point (train / predict / simulate / make-data)
```

Dependencies flow one way: `cli` → `detect`/`propagation` → `models`/`features`
→ `config`. No cycles; every module is importable and testable in isolation.

## Key design decisions

### 1. Everything is one scikit-learn `Pipeline`

`build_pipeline` returns a single estimator:

```
clean → FeatureUnion(tfidf, stylometric) → MaxAbsScaler → linear classifier
```

Bundling preprocessing *inside* the model means:

- **No train/serve skew** — inference runs the identical transforms as training.
- **One artefact** — `joblib.dump` on the pipeline persists the fitted
  vectoriser, scaler and classifier together. There is nothing to keep in sync.
- **Trivial swapping** — the classifier is chosen by name in `ModelConfig`.

`MaxAbsScaler` is used (not `StandardScaler`) because it preserves sparsity and
keeps features non-negative, which is required for `MultinomialNB` and stops the
raw-count stylometric features from swamping the `[0,1]` ratios.

### 2. Configuration as dataclasses

Every knob lives in `ModelConfig` / `PropagationConfig`. Experiments are
reproducible (nothing hidden in function defaults) and the CLI exposes the same
options without duplicating values. `random_state` is threaded everywhere so
runs are deterministic.

### 3. Offline-first, reproducible data

`data.py` generates a *learnable* synthetic corpus, so `git clone && make train`
works with no downloads and no network — essential for CI and for anyone
evaluating the repo quickly. Swapping in a real CSV is a one-flag change; the
loader normalises column names and dtypes.

### 4. A thin façade over the pipeline

`FakeNewsDetector` wraps the raw pipeline with friendly return types
(`Prediction`), probability handling that works even for classifiers without
`predict_proba` (SVM/PA get a sigmoid over `decision_function`), and an
`explain()` method that walks the fitted transformers to attribute a decision to
named features. Callers never touch numpy arrays directly.

### 5. Simulation separated from strategy

`propagation.py` cleanly splits three concerns:

- **`build_social_graph` / `choose_seeds`** — construct the world.
- **`select_monitors`** — the *policy* (which nodes to immunise); adding a new
  strategy means adding one branch here.
- **`_single_cascade` / `simulate`** — the *mechanics* of diffusion, independent
  of policy.

This makes it easy to benchmark a new containment idea against the existing five
without touching the simulator, and `compare_strategies` guarantees every policy
is evaluated on the *same* graph and seed set for a fair comparison.

## Testing strategy

`tests/` (25 tests, pytest) covers each module at the right altitude:

- **Unit** — preprocessing rules, stylometric feature values, dataset shape and
  determinism, graph size, monitor-budget selection.
- **Behavioural** — the detector actually learns the signal (`accuracy ≥ 0.85`),
  round-trips through save/load, and explains linear decisions; containment
  actually reduces spread (`degree < none`).

`conftest.py` puts `src/` on the path so tests run without an editable install.
Simulation tests use small graphs and few Monte-Carlo runs to stay fast (<2 s
for the whole suite) while still asserting the qualitative result.

## Extending the project

- **Better detection** — swap the linear head for a fine-tuned transformer;
  keep the `FakeNewsDetector` interface so the CLI/API are unchanged.
- **Richer features** — add source-credibility, readability, or
  network-metadata blocks to the `FeatureUnion`.
- **Smarter containment** — implement the greedy influence-maximisation
  heuristic (CELF) as a new `select_monitors` branch and benchmark it.
- **Real-time** — feed a stream into the `passive_aggressive` model with
  `partial_fit` for online updates.
