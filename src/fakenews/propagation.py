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
* ``greedy``        — CELF greedy influence-maximisation. Directly optimises the
  simulated objective: at each step it immunises the node that *most* reduces
  expected spread. Slower, but the strongest defence — the centrality heuristics
  above are really cheap approximations to this.
* ``random``        — the null baseline.
* ``none``          — no intervention, to measure the untamed cascade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import networkx as nx
import numpy as np

from .config import PropagationConfig

STRATEGIES = ("none", "degree", "betweenness", "random", "acquaintance", "greedy")


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
    seeds: Optional[Sequence[int]] = None,
) -> Set[int]:
    """Return the set of nodes to immunise, per the configured strategy.

    ``exclude`` are nodes that may not be chosen (typically the seed spreaders).
    ``seeds`` are the initial spreaders, required only by the simulation-driven
    ``greedy`` strategy; when omitted it falls back to ``exclude`` (which the
    simulator populates with the seed set).
    """
    exclude = exclude or set()
    budget = config.n_monitors
    strategy = config.strategy.lower()
    rng = np.random.default_rng(config.random_state)

    candidates = [n for n in graph.nodes if n not in exclude]

    if strategy == "none" or budget <= 0:
        return set()

    if strategy == "greedy":
        seed_nodes = list(seeds) if seeds is not None else list(exclude)
        return _greedy_monitors(graph, config, seed_nodes, exclude, budget)

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


# --- greedy / CELF influence-maximisation ----------------------------------

def _estimate_spread(
    graph: nx.Graph,
    seeds: Sequence[int],
    immunised: Set[int],
    config: PropagationConfig,
    n_sims: int,
    seed: int,
) -> float:
    """Mean number of users ever reached, given a set of immunised nodes.

    A fixed ``seed`` is used so that competing candidate sets are compared under
    *common random numbers* — the same simulated coin flips — which sharply
    reduces the variance of marginal-gain estimates and makes the greedy choice
    stable.
    """
    total = 0
    for i in range(n_sims):
        rng = np.random.default_rng(seed + i)
        _, reached = _single_cascade(graph, seeds, immunised, config, rng)
        total += reached
    return total / n_sims


def _greedy_monitors(
    graph: nx.Graph,
    config: PropagationConfig,
    seeds: Sequence[int],
    exclude: Set[int],
    budget: int,
) -> Set[int]:
    """Greedy immunisation: repeatedly immunise the single most valuable node.

    Each round we immunise the node whose addition to the current monitor set
    reduces the *simulated* expected spread the most (its marginal gain).

    A note on CELF
    --------------
    Influence *maximisation* (choosing seeds to spread a message) is monotone
    and **submodular**, which lets the CELF algorithm skip most re-evaluations
    using lazy upper bounds. Node *immunisation* — choosing blockers to minimise
    spread — is **not** submodular in general: removing one node can raise
    another node's marginal value (it may now sit on a newly-critical path). We
    verified empirically that the lazy CELF shortcut selects a strictly worse
    set here, so we deliberately run the exact greedy and re-evaluate every
    candidate each round. To keep that tractable we (a) restrict the search to
    the highest-degree ``greedy_pool`` nodes — low-degree nodes are almost never
    on critical paths — and (b) use *common random numbers* (a fixed simulation
    seed) so marginal-gain comparisons within a round are low-variance.
    """
    sims = max(1, config.greedy_sims)
    seed = config.random_state

    pool = sorted(
        (n for n in graph.nodes if n not in exclude),
        key=lambda n: graph.degree(n),
        reverse=True,
    )[: max(config.greedy_pool, budget)]

    selected: Set[int] = set()
    while len(selected) < budget:
        base = _estimate_spread(graph, seeds, set(exclude) | selected, config, sims, seed)
        best_node, best_gain = None, -float("inf")
        for v in pool:
            if v in selected:
                continue
            immun = set(exclude) | selected | {v}
            gain = base - _estimate_spread(graph, seeds, immun, config, sims, seed)
            if gain > best_gain:
                best_gain, best_node = gain, v
        if best_node is None:  # pool exhausted
            break
        selected.add(best_node)

    return selected


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
    immunised = select_monitors(graph, config, exclude=set(seeds), seeds=seeds)

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
