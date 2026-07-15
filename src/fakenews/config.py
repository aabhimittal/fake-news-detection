"""Central configuration objects.

Keeping every tunable knob in a couple of dataclasses means experiments are
reproducible (nothing is hidden inside a function) and the CLI can expose the
same options without duplicating defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# Resolve important directories relative to the installed package so the code
# works whether it is run from the repo root, a script, or an installed wheel.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_MODEL_PATH = MODELS_DIR / "fakenews_pipeline.joblib"
DEFAULT_DATASET_PATH = DATA_DIR / "sample_news.csv"


@dataclass
class ModelConfig:
    """Hyper-parameters for the text-classification pipeline."""

    # TF-IDF vectoriser
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 0.95
    sublinear_tf: bool = True

    # Which linear classifier to place on top of the features.
    # One of: "logistic", "passive_aggressive", "naive_bayes", "linear_svm".
    classifier: str = "logistic"

    # Train / test split
    test_size: float = 0.2
    random_state: int = 42

    # Whether to append hand-crafted stylometric features to the TF-IDF matrix.
    use_stylometric: bool = True


@dataclass
class PropagationConfig:
    """Parameters for the social-network propagation simulation."""

    n_nodes: int = 500          # number of users in the synthetic network
    attachment: int = 3         # Barabasi-Albert edges per new node (scale-free)
    random_state: int = 42

    # Independent Cascade activation probability on each edge.
    activation_prob: float = 0.08

    # SIR-style recovery probability (a user stops sharing / loses interest).
    recovery_prob: float = 0.10

    max_steps: int = 30         # simulation horizon
    n_seeds: int = 5            # number of initial spreaders of the fake story
    n_monitors: int = 25        # budget for the containment strategy

    # Which containment strategy to evaluate.
    # One of: "none", "degree", "betweenness", "random", "acquaintance".
    strategy: str = "degree"

    n_simulations: int = 40     # Monte-Carlo repetitions to average over
