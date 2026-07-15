"""Command-line interface.

Examples
--------
Train and save a model on the bundled synthetic data::

    python -m fakenews.cli train

Score a headline::

    python -m fakenews.cli predict "SHOCKING: doctors HATE this one weird trick!!!"

Run the propagation experiment and print the containment comparison::

    python -m fakenews.cli simulate --strategy degree --nodes 500
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import ModelConfig, PropagationConfig
from .data import load_dataset, write_sample_dataset
from .detect import FakeNewsDetector
from .propagation import simulate_campaign


def _cmd_train(args: argparse.Namespace) -> int:
    df = load_dataset(args.dataset) if args.dataset else load_dataset()
    print(f"Loaded {len(df)} documents "
          f"({int(df['label'].sum())} fake / {int((df['label'] == 0).sum())} real)")

    if args.arch == "transformer":
        from .transformer import TransformerDetector
        from .config import TransformerConfig

        detector = TransformerDetector(TransformerConfig(model_name=args.model_name))
        print(f"Fine-tuning {args.model_name} (this downloads weights on first run)...")
    else:
        detector = FakeNewsDetector(ModelConfig(classifier=args.classifier))

    result = detector.fit(df)
    print("\nHeld-out evaluation")
    print("-------------------")
    print(result.report)
    print(result.summary())

    path = detector.save(args.output)
    print(f"\nModel saved to {path}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    if args.arch == "transformer":
        from .transformer import TransformerDetector

        loader = TransformerDetector.load
    else:
        loader = FakeNewsDetector.load

    try:
        detector = loader(args.model)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    text = args.text if args.text else sys.stdin.read()
    prediction = detector.predict(text)
    print(f"\nVerdict: {prediction}")

    # Feature attributions are only available for the linear model.
    if args.explain and hasattr(detector, "explain"):
        contributions = detector.explain(text)
        if contributions:
            print("\nTop contributing features (+ pushes toward FAKE):")
            for name, weight in contributions:
                arrow = "fake" if weight > 0 else "real"
                print(f"  {weight:+.3f}  {name:<24} -> {arrow}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    config = PropagationConfig(
        n_nodes=args.nodes,
        n_seeds=args.seeds,
        n_monitors=args.monitors,
        activation_prob=args.activation,
        n_simulations=args.runs,
        greedy_sims=args.greedy_sims,
        greedy_pool=args.greedy_pool,
    )
    print(f"Network: {config.n_nodes} users (scale-free), "
          f"{config.n_seeds} initial spreaders, "
          f"{config.n_monitors} fact-checker budget, "
          f"{config.n_simulations} Monte-Carlo runs\n")

    results = simulate_campaign(config)
    baseline = results["none"].total_reached

    print(f"{'strategy':>13} | {'reached':>8} | {'peak':>6} | {'reduction vs none':>18}")
    print("-" * 56)
    for strat, res in results.items():
        reduction = (
            0.0 if strat == "none" or baseline == 0
            else 100.0 * (baseline - res.total_reached) / baseline
        )
        print(f"{strat:>13} | {res.total_reached:8.1f} | "
              f"{res.peak_active:6.1f} | {reduction:16.1f}%")
    return 0


def _cmd_makedata(args: argparse.Namespace) -> int:
    path = write_sample_dataset(args.output, n_per_class=args.n)
    print(f"Wrote synthetic dataset ({2 * args.n} rows) to {path}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import cross_validate_classifiers, format_table
    from .data import generate_benchmark_dataset

    if args.dataset:
        df = load_dataset(args.dataset)
        source = args.dataset
    else:
        # A deliberately noisy synthetic corpus so the classifiers actually
        # differ — a perfectly separable dataset would score them all at 1.0.
        df = generate_benchmark_dataset(n_per_class=args.n, noise=args.noise)
        source = f"synthetic benchmark (noise={args.noise})"

    print(f"Dataset: {source} — {len(df)} documents, {args.cv}-fold cross-validation\n")
    rows = cross_validate_classifiers(df, cv=args.cv)
    print(format_table(rows))
    print(f"\nBest by F1: {rows[0].classifier}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fakenews",
        description="Spot fake news and simulate stopping its propagation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train and save the classifier")
    p_train.add_argument("--dataset", help="path to a CSV with text,label columns")
    p_train.add_argument("--output", help="where to save the model")
    p_train.add_argument(
        "--arch",
        default="linear",
        choices=["linear", "transformer"],
        help="linear TF-IDF pipeline (default) or a fine-tuned transformer",
    )
    p_train.add_argument(
        "--classifier",
        default="logistic",
        choices=["logistic", "passive_aggressive", "naive_bayes", "linear_svm"],
        help="linear classifier head (only used with --arch linear)",
    )
    p_train.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="HuggingFace checkpoint to fine-tune (only used with --arch transformer)",
    )
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="classify a document")
    p_pred.add_argument("text", nargs="?", help="text to classify (or pass via stdin)")
    p_pred.add_argument("--model", help="path to a saved model")
    p_pred.add_argument(
        "--arch",
        default="linear",
        choices=["linear", "transformer"],
        help="which kind of saved model to load",
    )
    p_pred.add_argument("--explain", action="store_true", help="show feature attributions")
    p_pred.set_defaults(func=_cmd_predict)

    p_sim = sub.add_parser("simulate", help="run the propagation/containment experiment")
    p_sim.add_argument("--nodes", type=int, default=500)
    p_sim.add_argument("--seeds", type=int, default=5, help="initial spreaders")
    p_sim.add_argument("--monitors", type=int, default=25, help="fact-checker budget")
    p_sim.add_argument("--activation", type=float, default=0.08)
    p_sim.add_argument("--runs", type=int, default=40, help="Monte-Carlo repetitions")
    p_sim.add_argument("--strategy", default="degree")  # kept for discoverability
    p_sim.add_argument("--greedy-sims", type=int, default=12,
                       help="Monte-Carlo runs per greedy marginal-gain evaluation")
    p_sim.add_argument("--greedy-pool", type=int, default=60,
                       help="restrict greedy search to the top-N degree nodes")
    p_sim.set_defaults(func=_cmd_simulate)

    p_data = sub.add_parser("make-data", help="write the synthetic sample dataset")
    p_data.add_argument("--output", help="destination CSV path")
    p_data.add_argument("--n", type=int, default=400, help="rows per class")
    p_data.set_defaults(func=_cmd_makedata)

    p_bench = sub.add_parser("benchmark", help="cross-validate and compare classifiers")
    p_bench.add_argument("--dataset", help="CSV with text,label columns (else synthetic)")
    p_bench.add_argument("--cv", type=int, default=5, help="number of CV folds")
    p_bench.add_argument("--n", type=int, default=400, help="rows per class (synthetic)")
    p_bench.add_argument("--noise", type=float, default=0.25,
                         help="difficulty of the synthetic benchmark, 0..1")
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
