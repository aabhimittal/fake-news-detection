from fakenews.config import PropagationConfig
from fakenews.propagation import (
    STRATEGIES,
    build_social_graph,
    choose_seeds,
    compare_strategies,
    select_monitors,
    simulate,
)


def _fast_config(**kw):
    base = dict(n_nodes=200, n_simulations=8, max_steps=20, random_state=1)
    base.update(kw)
    return PropagationConfig(**base)


def test_graph_has_requested_size():
    g = build_social_graph(_fast_config())
    assert g.number_of_nodes() == 200


def test_seed_selection_returns_high_degree_nodes():
    cfg = _fast_config(n_seeds=3)
    g = build_social_graph(cfg)
    seeds = choose_seeds(g, cfg)
    assert len(seeds) == 3
    # Seeds should be among the higher-degree half of the network.
    median_deg = sorted(d for _, d in g.degree)[g.number_of_nodes() // 2]
    assert all(g.degree(s) >= median_deg for s in seeds)


def test_monitor_budget_respected():
    cfg = _fast_config(strategy="degree", n_monitors=15)
    g = build_social_graph(cfg)
    monitors = select_monitors(g, cfg)
    assert len(monitors) == 15


def test_none_strategy_selects_nobody():
    cfg = _fast_config(strategy="none")
    g = build_social_graph(cfg)
    assert select_monitors(g, cfg) == set()


def test_containment_reduces_spread():
    """A degree-targeted defence must beat doing nothing."""
    results = compare_strategies(_fast_config(n_monitors=30, activation_prob=0.1))
    assert results["degree"].total_reached < results["none"].total_reached


def test_all_strategies_run():
    cfg = _fast_config(n_monitors=10)
    g = build_social_graph(cfg)
    seeds = choose_seeds(g, cfg)
    for strat in STRATEGIES:
        c = PropagationConfig(**{**cfg.__dict__, "strategy": strat})
        res = simulate(g, config=c, seeds=seeds)
        assert res.total_reached >= 0
        assert len(res.timeline) == cfg.max_steps + 1
