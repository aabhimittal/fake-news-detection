"""Modelling and containing the spread of misinformation.

Detecting a fake story is only half the battle — once it is loose on a social
network it propagates from user to user. We model this with the classic
**Independent Cascade (IC)** diffusion process on top of a scale-free graph
(Barabasi-Albert), which reproduces the hub-and-spoke structure of real social
networks.

Containment = choosing a small set of *monitor / fact-checker* nodes that, once
they see the story, immediately debunk it and refuse to propagate further
(they become "immunised" and block the cascade through them). The interesting
question is **which** nodes to spend a limited budget on. We implement and
compare several graph-theoretic strategies:

* ``degree``        — immunise the highest-degree hubs (cheap, strong baseline).
* ``betweenness``   — immunise the best "bridges" between communities.
* ``acquaintance``  — pick a random node, immunise a random neighbour of it.
  This famously targets hubs *without* needing global knowledge of the graph,
  which matters when the platform can only sample local structure.
* ``random``        — the null baseline.
* ``none``          — no intervention, to measure the untamed cascade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import networkx as nx
import numpy as np

from .config import PropagationConfig

STRATEGIES = ("none", "degree", "betweenness", "random", "acquaintance")


# --- network construction --------------------------------------------------

def build_social_graph(config: PropagationConfig) -> nx.Graph:
    """Build a synthetic scale-free social network."""
    return nx.barabasi_albert_graph(
        n=config.n_nodes,
        m=config.attachment,
        seed=config.random_state,
    )


def choose_seeds(graph: nx.Graph, config: PropagationConfig) -> List[int]:
    """Pick the initial spreaders — the highest-degree nodes are the worst case."""
    ranked = sorted(graph.degree, key=lambda kv: kv[1], reverse=True)
    return [node for node, _ in ranked[: config.n_seeds]]


# --- containment strategies ------------------------------------------------

def select_monitors(
    graph: nx.Graph,
    config: PropagationConfig,
    exclude: Optional[Set[int]] = None,
) -> Set[int]:
    """Return the set of nodes to immunise, per the configured strategy."""
    exclude = exclude or set()
    budget = config.n_monitors
    strategy = config.strategy.lower()
    rng = np.random.default_rng(config.random_state)

    candidates = [n for n in graph.nodes if n not in exclude]

    if strategy == "none" or budget <= 0:
        return set()

    if strategy == "degree":
        ranked = sorted(candidates, key=lambda n: graph.degree(n), reverse=True)
        return set(ranked[:budget])

    if strategy == "betweenness":
        # k-sampling keeps betweenness tractable on larger graphs.
        k = min(len(graph), max(50, budget * 4))
        bc = nx.betweenness_centrality(graph, k=k, seed=config.random_state)
        ranked = sorted(candidates, key=lambda n: bc.get(n, 0.0), reverse=True)
        return set(ranked[:budget])

    if strategy == "random":
        pick = rng.choice(candidates, size=min(budget, len(candidates)), replace=False)
        return set(int(x) for x in pick)

    if strategy == "acquaintance":
        # Repeatedly: sample a random node, immunise one of its random neighbours.
        chosen: Set[int] = set()
        attempts = 0
        while len(chosen) < budget and attempts < budget * 50:
            attempts += 1
            node = int(rng.choice(candidates))
            neighbours = [x for x in graph.neighbors(node) if x not in exclude]
            if neighbours:
                chosen.add(int(rng.choice(neighbours)))
        return chosen

    raise ValueError(f"Unknown strategy {config.strategy!r}. Choose from {STRATEGIES}.")


# --- diffusion simulation --------------------------------------------------

@dataclass
class CascadeResult:
    """Outcome of one or more Independent-Cascade runs."""

    strategy: str
    total_reached: float          # mean number of users who ever shared the story
    peak_active: float            # mean simultaneous sharers at the busiest step
    steps_to_peak: float
    timeline: List[float] = field(default_factory=list)  # mean active per step
    n_simulations: int = 1

    def summary(self) -> str:
        return (
            f"[{self.strategy:>12}] reached={self.total_reached:6.1f}  "
            f"peak={self.peak_active:6.1f}  steps_to_peak={self.steps_to_peak:4.1f}"
        )


def _single_cascade(
    graph: nx.Graph,
    seeds: Sequence[int],
    immunised: Set[int],
    config: PropagationConfig,
    rng: np.random.Generator,
) -> tuple[List[int], int]:
    """Run one Independent-Cascade + SIR-recovery simulation.

    Returns ``(timeline, total_reached)`` where ``timeline`` is the number of
    currently-active (sharing) nodes at each step and ``total_reached`` is the
    number of distinct users who ever shared the story.
    """
    # State: susceptible (default), infected (sharing), recovered/immunised.
    infected = set(s for s in seeds if s not in immunised)
    recovered: Set[int] = set(immunised)
    ever = set(infected)          # every node that was ever infected
    newly_infected = set(infected)

    active_timeline: List[int] = [len(infected)]

    for _ in range(config.max_steps):
        # 1. Active nodes try to infect susceptible neighbours (IC step).
        activated: Set[int] = set()
        for u in newly_infected:
            for v in graph.neighbors(u):
                if v in ever or v in recovered:
                    continue
                if rng.random() < config.activation_prob:
                    activated.add(v)

        # 2. Some currently-active nodes recover / lose interest (SIR step).
        recovering = {u for u in infected if rng.random() < config.recovery_prob}
        infected -= recovering
        recovered |= recovering

        infected |= activated
        ever |= activated
        newly_infected = activated
        active_timeline.append(len(infected))

        if not infected:
            break

    # Pad the timeline so every run has the same length for averaging.
    while len(active_timeline) < config.max_steps + 1:
        active_timeline.append(0)
    return active_timeline, len(ever)


def simulate(
    graph: nx.Graph,
    config: PropagationConfig,
    seeds: Optional[Sequence[int]] = None,
) -> CascadeResult:
    """Monte-Carlo average of the cascade under the configured strategy."""
    seeds = list(seeds) if seeds is not None else choose_seeds(graph, config)
    immunised = select_monitors(graph, config, exclude=set(seeds))

    timelines = np.zeros((config.n_simulations, config.max_steps + 1))
    reached = np.zeros(config.n_simulations)

    base_rng = np.random.default_rng(config.random_state)
    for i in range(config.n_simulations):
        # Independent stream per simulation for reproducible variance.
        rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
        timeline, total_reached = _single_cascade(graph, seeds, immunised, config, rng)
        timelines[i] = timeline
        reached[i] = total_reached

    mean_timeline = timelines.mean(axis=0)
    peak_idx = int(np.argmax(mean_timeline))
    return CascadeResult(
        strategy=config.strategy,
        total_reached=float(reached.mean()),
        peak_active=float(mean_timeline[peak_idx]),
        steps_to_peak=float(peak_idx),
        timeline=mean_timeline.tolist(),
        n_simulations=config.n_simulations,
    )


# --- experiment driver -----------------------------------------------------

def compare_strategies(
    config: PropagationConfig,
    strategies: Sequence[str] = STRATEGIES,
) -> Dict[str, CascadeResult]:
    """Run every containment strategy on the *same* graph and seed set."""
    graph = build_social_graph(config)
    seeds = choose_seeds(graph, config)
    results: Dict[str, CascadeResult] = {}
    for strat in strategies:
        cfg = PropagationConfig(**{**config.__dict__, "strategy": strat})
        results[strat] = simulate(graph, cfg, seeds=seeds)
    return results


def simulate_campaign(config: Optional[PropagationConfig] = None) -> Dict[str, CascadeResult]:
    """Convenience entry point: build a network and compare all strategies."""
    return compare_strategies(config or PropagationConfig())
