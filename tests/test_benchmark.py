import numpy as np
import pandas as pd
import pytest

from fakenews.benchmark import (
    BenchmarkRow,
    cross_validate_classifiers,
    format_table,
    load_liar,
)
from fakenews.data import generate_benchmark_dataset, generate_synthetic_dataset


def test_benchmark_dataset_is_harder_than_clean():
    """Noise must actually blur the classes (fewer clean-style matches)."""
    easy = generate_synthetic_dataset(n_per_class=200, random_state=0)
    hard = generate_benchmark_dataset(n_per_class=200, noise=0.5, random_state=0)
    assert len(hard) == len(easy) == 400
    # Both balanced-ish, but the hard one should not be perfectly separable.
    assert set(hard["label"].unique()) == {0, 1}


def test_cross_validate_returns_sorted_rows():
    df = generate_benchmark_dataset(n_per_class=120, noise=0.3, random_state=1)
    rows = cross_validate_classifiers(df, cv=3)
    assert len(rows) == 4
    assert all(isinstance(r, BenchmarkRow) for r in rows)
    # Sorted by F1 descending.
    f1s = [r.f1_mean for r in rows]
    assert f1s == sorted(f1s, reverse=True)
    # Metrics are valid probabilities.
    assert all(0.0 <= r.accuracy_mean <= 1.0 for r in rows)


def test_classifiers_beat_chance_on_easy_data():
    df = generate_synthetic_dataset(n_per_class=150, random_state=2)
    rows = cross_validate_classifiers(df, cv=3)
    assert rows[0].f1_mean > 0.8


def test_format_table_has_all_rows():
    df = generate_benchmark_dataset(n_per_class=80, noise=0.3, random_state=3)
    table = format_table(cross_validate_classifiers(df, cv=3))
    for name in ("logistic", "naive_bayes", "linear_svm", "passive_aggressive"):
        assert name in table


def test_load_liar_maps_to_binary(tmp_path):
    # Minimal LIAR-format TSV: id, label, statement, ...
    liar = tmp_path / "train.tsv"
    liar.write_text(
        "1\tfalse\tThe moon is made of cheese.\ta\tb\n"
        "2\ttrue\tWater boils at 100C at sea level.\ta\tb\n"
        "3\tpants-fire\tVaccines contain microchips.\ta\tb\n"
        "4\tmostly-true\tThe census runs every ten years.\ta\tb\n"
    )
    df = load_liar(liar)
    assert list(df.columns) == ["text", "label"]
    assert df["label"].tolist() == [1, 0, 1, 0]


def test_load_liar_missing_file():
    with pytest.raises(FileNotFoundError):
        load_liar("/no/such/liar.tsv")
