"""Plot the misinformation-cascade curves for every containment strategy.

Requires the optional ``viz`` extra::

    pip install matplotlib
    python scripts/plot_propagation.py --out docs/propagation.png

Produces one line per strategy showing the number of users actively sharing
the fake story at each time step. A good defence flattens and lowers the curve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fakenews.config import PropagationConfig  # noqa: E402
from fakenews.propagation import simulate_campaign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--monitors", type=int, default=25)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--out", default="docs/propagation.png")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 1

    config = PropagationConfig(
        n_nodes=args.nodes, n_monitors=args.monitors, n_simulations=args.runs
    )
    results = simulate_campaign(config)

    plt.figure(figsize=(9, 5.5))
    for strategy, res in results.items():
        plt.plot(res.timeline, label=f"{strategy} (reached {res.total_reached:.0f})", linewidth=2)

    plt.title("Active sharers of a fake story over time, by containment strategy")
    plt.xlabel("Simulation step")
    plt.ylabel("Users actively sharing")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130)
    print(f"Saved figure to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
