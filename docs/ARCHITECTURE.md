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
├── detect.py        FakeNewsDetector — the linear public façade
├── transformer.py   TransformerDetector — optional fine-tuned DistilBERT
├── benchmark.py     cross-validation harness + LIAR/Kaggle loaders
├── adversarial.py   evasion attacks + deobfuscation defence
├── triage.py        calibration, abstention policy, cost-aware cutoffs
├── earlydetect.py   diffusion trees, cascade-shape features, early detection
├── propagation.py   graph + Independent Cascade + containment strategies
└── cli.py           argparse entry point (train / predict / simulate / benchmark / make-data)
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

This makes it easy to benchmark a new containment idea against the existing six
without touching the simulator, and `compare_strategies` guarantees every policy
is evaluated on the *same* graph and seed set for a fair comparison.

### 6. Optional heavy dependencies stay optional

`transformer.py` needs `torch` + `transformers`, which the core package must not
require. The rule enforced here: **never import the heavy backend at module
top-level.** A tiny `_require_backend()` helper imports them lazily inside the
methods that use them and raises an actionable `ImportError` (pointing at
`pip install "fakenews[transformer]"`) otherwise. So `import fakenews` stays
light, the linear pipeline works with no deep-learning stack installed, and the
transformer is a genuine drop-in that shares the `FakeNewsDetector` interface
(`fit`/`predict`/`save`/`load`) — the CLI selects between them with `--arch`.

### 7. Benchmarking is split-agnostic

`benchmark.py` takes any `text`/`label` DataFrame and cross-validates every
classifier on a *single shared* `StratifiedKFold` splitter, so the models are
compared on identical folds. Real-corpus specifics (LIAR's 6-way scale, Kaggle's
two-file layout) live in dedicated loaders that normalise down to the same
`text`/`label` frame, keeping the harness itself corpus-agnostic.

### 8. Defences are measured, not assumed

`adversarial.py` ships the **attacks alongside the defence**, and `triage.py`
ships calibration *metrics* alongside the calibrators. This is deliberate: a
normaliser or a confidence score that is never adversarially measured is a
guess. Every claim in the docs is reproducible from a CLI subcommand
(`robustness`, `triage`, `early-detect`).

### 9. Numeric surfaces take arrays, not detectors

`triage.py` operates on `(y_true, p_fake)` rather than on a detector object, so
it works with the linear model, the transformer, or an external scorer, and is
trivially testable without training anything. `predict_proba` on both detectors
is the single adapter into it.

## Testing strategy

`tests/` (126 tests, pytest) covers each module at the right altitude:

- **Unit** — preprocessing rules, stylometric feature values, dataset shape and
  determinism, graph size, monitor-budget selection, LIAR label mapping.
- **Behavioural** — the detector actually learns the signal (`accuracy ≥ 0.85`),
  round-trips through save/load, and explains linear decisions; containment
  actually reduces spread and greedy matches the best heuristic; cross-validated
  classifiers beat chance.
- **Opt-in / guarded** — the transformer's end-to-end fine-tune test downloads a
  real checkpoint, so it runs only when `torch`+`transformers` are present *and*
  `FAKENEWS_RUN_TRANSFORMER=1` is set. Its interface contracts (predict-before-
  fit, config defaults, the no-backend error path) always run.

- **Industrial edge cases** (`test_edge_cases.py`) — the inputs production
  actually delivers: empty/whitespace/control-byte documents, 200 kB inputs,
  emoji, CJK, RTL and mixed scripts, malformed and NaN-bearing CSVs, embedded
  delimiters, single-class and 99:1-imbalanced training sets, degenerate graphs,
  probability boundaries, determinism, save/load fidelity and concurrent
  prediction. The standard is *works* or *fails loudly* — never a silent wrong
  answer. This suite found a real bug on its first run (empty batch crashing
  inside scikit-learn), which is now fixed and guarded.

`conftest.py` puts `src/` on the path so tests run without an editable install.
Simulation and greedy tests use small graphs and few Monte-Carlo runs to stay
fast while still asserting the qualitative result.

## Extending the project

- **Richer features** — add source-credibility, readability, or
  network-metadata blocks to the `FeatureUnion`.
- **Larger transformers** — point `TransformerConfig.model_name` at a bigger
  checkpoint (RoBERTa, DeBERTa) or add more epochs for real corpora.
- **Temporal containment** — let monitors be *placed* mid-cascade based on early
  detection signal, rather than pre-positioned.
- **Real-time** — feed a stream into the `passive_aggressive` model with
  `partial_fit` for online updates.
