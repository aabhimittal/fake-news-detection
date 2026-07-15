"""fakenews — algorithms to spot fake news and stop its propagation.

The package is organised as a small, readable pipeline:

    data  ->  preprocess  ->  features  ->  models  ->  detect
                                              |
                                          propagation

Every module is independently importable and unit-tested. The high level
entry points most users want are :class:`fakenews.detect.FakeNewsDetector`
and :func:`fakenews.propagation.simulate_campaign`.
"""

from .config import ModelConfig, PropagationConfig

__all__ = ["ModelConfig", "PropagationConfig", "__version__"]

__version__ = "0.1.0"
