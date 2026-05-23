# ropemetrics

**Framework-agnostic jump rope counter powered by pose estimation.**

ropemetrics is an open-source Python library for accurate jump rope repetition
counting. Built around a strategy pattern, it decouples the counting algorithm
from the pose backend — connect MediaPipe, MoveNet, YOLOv8-Pose, or any custom
provider through a single `LandmarkProvider` interface.

---

## Features

- **Backend-agnostic** — plug in any pose estimation framework via adapters
- **Swappable algorithms** — choose or implement a `JumpStrategy` without touching core logic
- **Config profiles** — `slow` / `fast` / `child` presets for different use cases
- **Event callbacks** — `on_jump(JumpEvent)` for real-time integration
- **Lightweight core** — only `numpy` required; pose backends are optional extras

---

## Installation

```bash
# Core only (numpy)
pip install ropemetrics

# With MediaPipe adapter
pip install "ropemetrics[mediapipe]"

# With MoveNet adapter (TensorFlow)
pip install "ropemetrics[movenet]"
```

---

## Quick Start

```python
import mediapipe as mp
from mediapipe.tasks.python import vision

from ropemetrics import JumpCounter, JumpCounterConfig
from ropemetrics.strategies import AnkleStrategy
from ropemetrics.adapters import MediaPipeLandmarkProvider

# 1. Set up MediaPipe PoseLandmarker
options  = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.python.BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.PoseLandmarker.create_from_options(options)

# 2. Assemble counter
cfg      = JumpCounterConfig()           # or .slow() / .fast() / .child()
counter  = JumpCounter(
    strategy = AnkleStrategy(cfg),
    config   = cfg,
    on_jump  = lambda e: print(f"Jump #{e.count}!"),
)
provider = MediaPipeLandmarkProvider()

# 3. Run on video frames
import cv2
cap, ts = cv2.VideoCapture(0), 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
    counter.update(provider, results)
    ts += 33
```

---

## Switching Strategies

```python
from ropemetrics.strategies import AnkleStrategy, HipShoulderStrategy

# Ankle Y-axis (default, most sensitive)
counter = JumpCounter(strategy=AnkleStrategy(cfg), config=cfg)

# Hip + Shoulder weighted center (stable under partial occlusion)
counter = JumpCounter(strategy=HipShoulderStrategy(cfg), config=cfg)
```

---

## Config Profiles

| Profile | Use case | jump_threshold | cooldown_frames | history_size | min_jump_frames | visibility_min |
|---------|----------|---------------|-----------------|--------------|-----------------|----------------|
| `JumpCounterConfig()` | Default | 0.025 | 8 | 10 | 3 | 0.5 |
| `.slow()` | Beginners / adults | 0.040 | 12 | 12 | 4 | 0.5 |
| `.fast()` | Double-unders | 0.015 | 5 | 7 | 2 | 0.4 |
| `.child()` | Children / students | 0.018 | 8 | 8 | 2 | 0.35 |

---

## Adding a Custom Backend

```python
from ropemetrics.base import LandmarkProvider, Landmark, LandmarkName
from typing import Optional
import numpy as np

class MoveNetProvider(LandmarkProvider):
    MOVENET_INDEX = {"LEFT_ANKLE": 15, "RIGHT_ANKLE": 16, ...}

    def get_landmark(self, results, name: LandmarkName) -> Optional[Landmark]:
        idx = self.MOVENET_INDEX.get(name)
        if idx is None:
            return None
        kp = results[idx]          # [y, x, confidence]
        return np.array([kp[1], kp[0], 0.0, kp[2]], dtype=np.float32)
```

---

## Attribution

If you use ropemetrics in your product or service, include the following notice:

```
This product uses ropemetrics (https://github.com/Kangsungmin/ropemetrics)
Copyright 2026 Kangsungmin — Licensed under the Apache License 2.0
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.
Commercial use is permitted with attribution.
