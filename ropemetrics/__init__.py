"""
ropemetrics — Framework-agnostic jump rope counter powered by pose estimation.

Quick start:
    from ropemetrics import JumpCounter, JumpCounterConfig
    from ropemetrics.strategies import AnkleStrategy
    from ropemetrics.adapters import MediaPipeLandmarkProvider

    cfg      = JumpCounterConfig()
    counter  = JumpCounter(strategy=AnkleStrategy(cfg), config=cfg)
    provider = MediaPipeLandmarkProvider()

    # in your video loop:
    counter.update(provider, results)
    print(counter.count)
"""

from ropemetrics.base import (
    JumpCounterConfig,
    JumpEvent,
    JumpStrategy,
    Landmark,
    LandmarkName,
    LandmarkProvider,
)
from ropemetrics.counter import JumpCounter

__version__ = "0.1.0"
__all__ = [
    "JumpCounter",
    "JumpCounterConfig",
    "JumpEvent",
    "JumpStrategy",
    "Landmark",
    "LandmarkName",
    "LandmarkProvider",
]
