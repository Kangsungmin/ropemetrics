# Contributing to ropemetrics

Thank you for your interest in contributing! ropemetrics welcomes improvements
to counting algorithms, new pose backend adapters, and bug fixes.

## Ways to Contribute

- **New strategy** — implement a new `JumpStrategy` subclass in `ropemetrics/strategies/`
- **New adapter** — connect a new pose backend (MoveNet, YOLOv8, etc.) in `ropemetrics/adapters/`
- **Bug fix** — open an issue first, then submit a PR
- **Test** — increase coverage in `tests/`

## Adding a New Strategy

1. Create `ropemetrics/strategies/your_strategy.py`
2. Inherit from `JumpStrategy` and implement all abstract methods:

```python
from ropemetrics.base import JumpStrategy, JumpCounterConfig, LandmarkName
from typing import Optional, Sequence
import numpy as np

class MyStrategy(JumpStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"

    def required_landmarks(self) -> Sequence[LandmarkName]:
        return ("LEFT_HIP", "RIGHT_HIP")

    def extract_signal(self, landmarks) -> Optional[float]:
        lh = landmarks.get("LEFT_HIP")
        rh = landmarks.get("RIGHT_HIP")
        if lh is None or rh is None:
            return None
        return float((lh[1] + rh[1]) / 2)

    def reset(self) -> None:
        pass
```

3. Add tests in `tests/`
4. Export from `ropemetrics/strategies/__init__.py`

## Adding a New Adapter

1. Create `ropemetrics/adapters/your_backend.py`
2. Inherit from `LandmarkProvider`:

```python
from ropemetrics.base import LandmarkProvider, Landmark, LandmarkName
from typing import Optional
import numpy as np

class MyBackendProvider(LandmarkProvider):
    def get_landmark(self, results, name: LandmarkName) -> Optional[Landmark]:
        # extract [x, y, z, visibility] from your backend's results
        ...
```

3. Add the optional dependency to `pyproject.toml` under `[project.optional-dependencies]`

## Development Setup

```bash
git clone https://github.com/Kangsungmin/ropemetrics.git
cd ropemetrics
pip install -e ".[dev]"
pytest tests/ -v
```

## Pull Request Guidelines

- One feature or fix per PR
- All existing tests must pass (`pytest tests/ -v`)
- Add tests for new behaviour
- Keep `ropemetrics/base.py` and `ropemetrics/counter.py` free of backend-specific imports

## Attribution

By contributing, you agree that your contributions will be licensed under
the Apache License 2.0 and that attribution to the original project must
be preserved as described in the NOTICE file.
