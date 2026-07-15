import pytest

from fakenews.data import (
    generate_synthetic_dataset,
    load_dataset,
    write_sample_dataset,
)


def test_synthetic_dataset_is_balanced():
    df = generate_synthetic_dataset(n_per_class=50)
    assert len(df) == 100
    assert set(df["label"].unique()) == {0, 1}
    assert df["label"].sum() == 50


def test_synthetic_dataset_is_deterministic():
    a = generate_synthetic_dataset(n_per_class=20, random_state=7)
    b = generate_synthetic_dataset(n_per_class=20, random_state=7)
    assert a.equals(b)


def test_load_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("/no/such/file.csv")


def test_write_and_reload_roundtrip(tmp_path):
    path = write_sample_dataset(tmp_path / "sample.csv", n_per_class=30)
    df = load_dataset(path)
    assert len(df) == 60
    assert list(df.columns) == ["text", "label"]
