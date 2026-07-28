"""Early detection from cascade shape — classifying a story by *how* it spreads.

Parts 1 and 2 of this project are usually treated separately: read the text to
decide if it is fake, then contain it. This module closes the loop with a third
option that uses **no text at all**.

The empirical basis (Vosoughi, Roy & Aral, *Science* 2018) is that false and true
stories spread differently. False stories travel **deeper** and reach people
through longer person-to-person chains, whereas true stories are more often
**broadcast** — one large account posts, many people see it directly, and the
chain stops. Goel et al. captured this with **structural virality**: the average
pairwise distance in the diffusion tree, which separates "one-to-many broadcast"
(low) from "many-hop viral spread" (high) even when both reach the same number
of people.

That gives a detector with two properties the text classifier cannot match:

* **Language-independent** — it never reads the story, so it works across
  languages, and on images or video where there is no text to classify.
* **Evasion-resistant** — an author can rewrite wording to dodge a text model
  (see :mod:`fakenews.adversarial`), but cannot easily fake the shape of a real
  crowd's resharing behaviour.

The cost is that it needs the story to have started spreading. The central
question is therefore **how early** a reliable call can be made, which
:func:`evaluate_early_detection` answers by sweeping the observation window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .config import PropagationConfig
from .propagation import build_social_graph

# Feature order used by the classifier; exposed for interpretability.
CASCADE_FEATURE_NAMES: List[str] = [
    "size",
    "max_depth",
    "max_breadth",
    "structural_virality",
    "depth_over_size",
    "breadth_over_size",
    "mean_depth",
    "growth_rate",
    "leaf_fraction",
]


# --- cascade simulation with a diffusion tree ------------------------------

def simulate_cascade_tree(
    graph: nx.Graph,
    seed: int,
    *,
    activation_prob: float,
    max_steps: int,
    rng: np.random.Generator,
    seed_boost: float = 1.0,
    decay: float = 1.0,
) -> nx.DiGraph:
    """Run one Independent Cascade and return the **diffusion tree**.

    Unlike :func:`fakenews.propagation.simulate`, which only counts how many
    users were reached, this records *who infected whom* — the parent pointers
    that make the shape of the spread measurable.

    Args:
        seed_boost: multiplier on the activation probability for the seed's own
            edges. A large value produces **broadcast** cascades (one hub reaches
            many people directly), the pattern typical of true news.
        decay: per-hop multiplier on the activation probability. ``decay < 1``
            makes chains die out quickly (shallow); ``decay == 1`` sustains
            person-to-person spread (deep), the pattern typical of false news.

    Returns:
        A directed tree rooted at ``seed``; each node carries a ``depth``
        attribute (hops from the seed) and a ``time`` attribute (step activated).
    """
    tree = nx.DiGraph()
    tree.add_node(seed, depth=0, time=0)

    infected = {seed}
    frontier = [seed]

    for step in range(1, max_steps + 1):
        if not frontier:
            break
        next_frontier: List[int] = []
        for u in frontier:
            depth_u = tree.nodes[u]["depth"]
            # Seed edges may be boosted; every further hop decays.
            prob = activation_prob * (seed_boost if u == seed else 1.0)
            prob *= decay ** depth_u
            prob = min(max(prob, 0.0), 1.0)
            for v in graph.neighbors(u):
                if v in infected:
                    continue
                if rng.random() < prob:
                    infected.add(v)
                    tree.add_node(v, depth=depth_u + 1, time=step)
                    tree.add_edge(u, v)
                    next_frontier.append(v)
        frontier = next_frontier

    return tree


def structural_virality(tree: nx.DiGraph) -> float:
    """Average pairwise distance in the diffusion tree (Wiener index / n(n-1)).

    This is the quantity that separates a *broadcast* from a *viral* cascade of
    the same size. A star (one poster, N direct resharers) has virality ~2; a
    long chain has virality growing with its length.

    Returns 0.0 for a single-node cascade, where the measure is undefined.
    """
    n = tree.number_of_nodes()
    if n < 2:
        return 0.0
    undirected = tree.to_undirected()
    total = 0
    for _src, lengths in nx.all_pairs_shortest_path_length(undirected):
        total += sum(lengths.values())
    return float(total / (n * (n - 1)))


def cascade_features(
    tree: nx.DiGraph,
    *,
    observe_until: Optional[int] = None,
) -> Dict[str, float]:
    """Shape statistics for a cascade, optionally truncated to an early window.

    Args:
        observe_until: keep only nodes activated at ``time <= observe_until``.
            This is what makes *early* detection measurable — it simulates seeing
            only the first few steps of a spread that is still in progress.
    """
    if observe_until is not None:
        keep = [n for n, d in tree.nodes(data=True) if d.get("time", 0) <= observe_until]
        tree = tree.subgraph(keep).copy()

    n = tree.number_of_nodes()
    if n == 0:
        return {name: 0.0 for name in CASCADE_FEATURE_NAMES}

    depths = [d.get("depth", 0) for _, d in tree.nodes(data=True)]
    max_depth = max(depths)
    # Breadth = the largest number of nodes sharing a depth.
    counts = np.bincount(np.asarray(depths, dtype=int))
    max_breadth = int(counts.max())
    times = [d.get("time", 0) for _, d in tree.nodes(data=True)]
    span = max(max(times), 1)
    # A leaf is someone who received the story but never passed it on.
    leaves = sum(1 for node in tree.nodes if tree.out_degree(node) == 0)

    return {
        "size": float(n),
        "max_depth": float(max_depth),
        "max_breadth": float(max_breadth),
        "structural_virality": structural_virality(tree),
        "depth_over_size": max_depth / n,
        "breadth_over_size": max_breadth / n,
        "mean_depth": float(np.mean(depths)),
        "growth_rate": n / span,
        "leaf_fraction": leaves / n,
    }


# --- labelled cascade datasets --------------------------------------------

@dataclass
class CascadeRegime:
    """Parameters describing how one class of story spreads."""

    name: str
    label: int
    activation_prob: float
    seed_boost: float
    decay: float
    seed_from_hubs: bool


# Controlling for the popularity confound
# --------------------------------------
# The obvious way to model this — seed "real" stories from hubs and "fake" ones
# from ordinary users — produces a detector that looks excellent and is
# worthless: it hits 98% accuracy after a *single* step purely because a hub's
# cascade is 4x larger immediately. That is reading the poster's follower count,
# not the shape of the spread, and it would collapse the moment a fake story is
# posted by a popular account.
#
# So both regimes below are seeded from the **same pool** with **identical hop-0
# dynamics**; they differ only in ``decay``, i.e. whether resharing chains
# persist beyond the first hop. Any signal therefore has to come from the
# cascade's evolving structure, which is exactly the claim being tested.
BROADCAST_REGIME = CascadeRegime(
    name="broadcast (real-like)", label=0,
    activation_prob=0.13, seed_boost=1.0, decay=0.30, seed_from_hubs=False,
)
VIRAL_REGIME = CascadeRegime(
    name="viral (fake-like)", label=1,
    activation_prob=0.13, seed_boost=1.0, decay=1.0, seed_from_hubs=False,
)


def generate_cascade_dataset(
    n_per_class: int = 150,
    *,
    config: Optional[PropagationConfig] = None,
    observe_until: Optional[int] = None,
    min_size: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate labelled cascades and return their shape features.

    Cascades smaller than ``min_size`` are discarded: a story seen by two people
    has no measurable shape, and in production you would not attempt a structural
    call on it either. Keep this threshold low — filtering hard on size
    reintroduces the popularity bias the regimes are designed to avoid.
    """
    config = config or PropagationConfig(n_nodes=400, random_state=random_state)
    graph = build_social_graph(config)
    degrees = dict(graph.degree)
    hubs = sorted(degrees, key=degrees.get, reverse=True)[: max(10, len(graph) // 20)]
    ordinary = [n for n in graph.nodes if n not in set(hubs)]

    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, float]] = []

    for regime in (BROADCAST_REGIME, VIRAL_REGIME):
        pool = hubs if regime.seed_from_hubs else ordinary
        collected, attempts = 0, 0
        while collected < n_per_class and attempts < n_per_class * 40:
            attempts += 1
            seed = int(rng.choice(pool))
            tree = simulate_cascade_tree(
                graph, seed,
                activation_prob=regime.activation_prob,
                max_steps=config.max_steps,
                rng=rng,
                seed_boost=regime.seed_boost,
                decay=regime.decay,
            )
            if tree.number_of_nodes() < min_size:
                continue
            features = cascade_features(tree, observe_until=observe_until)
            features["label"] = regime.label
            rows.append(features)
            collected += 1

    return pd.DataFrame(rows)


# --- early-detection evaluation -------------------------------------------

def evaluate_early_detection(
    windows: Sequence[Optional[int]] = (1, 2, 3, 5, None),
    *,
    n_per_class: int = 150,
    config: Optional[PropagationConfig] = None,
    random_state: int = 42,
    cv: int = 4,
) -> pd.DataFrame:
    """How accurately can we classify a cascade after only *k* steps?

    Returns one row per observation window with cross-validated accuracy and F1.
    ``None`` means "observe the whole cascade" — the upper bound the early
    windows are trying to approach.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate

    rows = []
    for window in windows:
        df = generate_cascade_dataset(
            n_per_class=n_per_class,
            config=config,
            observe_until=window,
            random_state=random_state,
        )
        X = df[CASCADE_FEATURE_NAMES].to_numpy()
        y = df["label"].to_numpy()
        # A small forest: the features interact non-linearly (size *and* depth
        # together are what distinguish the regimes), and it needs no scaling.
        clf = RandomForestClassifier(
            n_estimators=100, random_state=random_state, min_samples_leaf=2
        )
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_validate(clf, X, y, cv=splitter, scoring=["accuracy", "f1"])
        rows.append({
            "window": "full" if window is None else f"{window} steps",
            "accuracy": float(np.mean(scores["test_accuracy"])),
            "f1": float(np.mean(scores["test_f1"])),
            "n_cascades": len(df),
        })
    return pd.DataFrame(rows)


def feature_importances(
    n_per_class: int = 200,
    *,
    observe_until: Optional[int] = None,
    random_state: int = 42,
) -> pd.Series:
    """Which shape features carry the signal (sorted, most important first)."""
    from sklearn.ensemble import RandomForestClassifier

    df = generate_cascade_dataset(
        n_per_class=n_per_class, observe_until=observe_until, random_state=random_state
    )
    clf = RandomForestClassifier(
        n_estimators=200, random_state=random_state, min_samples_leaf=2
    )
    clf.fit(df[CASCADE_FEATURE_NAMES].to_numpy(), df["label"].to_numpy())
    return pd.Series(
        clf.feature_importances_, index=CASCADE_FEATURE_NAMES
    ).sort_values(ascending=False)
