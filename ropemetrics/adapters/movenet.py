"""
MoveNet (TensorFlow Hub) adapter for ropemetrics.

Install:
    pip install ropemetrics[movenet]

Usage:
    import tensorflow_hub as hub
    import numpy as np
    from ropemetrics.adapters import MoveNetLandmarkProvider

    model   = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    movenet = model.signatures["serving_default"]

    provider = MoveNetLandmarkProvider()

    # in video loop:
    input_tensor = tf.cast(tf.image.resize_with_pad(frame, 192, 192), dtype=tf.int32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    results      = movenet(input=input_tensor)["output_0"].numpy()  # shape [1,1,17,3]
    counter.update(provider, results)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ropemetrics.base import Landmark, LandmarkName, LandmarkProvider


# MoveNet 17-point COCO landmark index map (ropemetrics LandmarkName → MoveNet index)
LANDMARK_INDEX: dict[str, int] = {
    "NOSE":           0,
    "LEFT_EYE":       1,
    "RIGHT_EYE":      2,
    "LEFT_EAR":       3,
    "RIGHT_EAR":      4,
    "LEFT_SHOULDER":  5,
    "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW":     7,
    "RIGHT_ELBOW":    8,
    "LEFT_WRIST":     9,
    "RIGHT_WRIST":   10,
    "LEFT_HIP":      11,
    "RIGHT_HIP":     12,
    "LEFT_KNEE":     13,
    "RIGHT_KNEE":    14,
    "LEFT_ANKLE":    15,
    "RIGHT_ANKLE":   16,
}


class MoveNetLandmarkProvider(LandmarkProvider):
    """
    LandmarkProvider adapter for MoveNet (TensorFlow Hub).

    MoveNet 추론 결과(numpy 배열)를 ropemetrics Landmark 형식으로 변환한다.

    결과 배열 형식:
        shape: [batch, person, 17, 3]
        각 관절: [y, x, confidence] (정규화 0~1)

    Args:
        person_index: 다중 인원 결과에서 추적할 인원 인덱스. 기본값 0.

    Example:
        provider = MoveNetLandmarkProvider()
        # results: numpy array, shape [1, 1, 17, 3]
        counter.update(provider, results)
    """

    def __init__(self, person_index: int = 0) -> None:
        self._person_index = person_index

    def get_landmark(
        self,
        results: object,
        name: LandmarkName,
    ) -> Optional[Landmark]:
        # results 유효성 검사
        if results is None:
            return None

        if not isinstance(results, np.ndarray):
            return None

        # shape 검사: 최소 [batch, person, 17, 3] 필요
        if results.ndim < 4 or results.shape[-1] < 3 or results.shape[-2] < 17:
            return None

        if self._person_index >= results.shape[0]:
            return None

        idx = LANDMARK_INDEX.get(name)
        if idx is None:
            return None

        # results[person_index][0][idx] → [y, x, confidence]
        keypoint = results[self._person_index][0][idx]
        y, x, confidence = float(keypoint[0]), float(keypoint[1]), float(keypoint[2])

        return np.array([x, y, 0.0, confidence], dtype=np.float32)

    def get_landmarks(
        self,
        results: object,
        names: Sequence[LandmarkName],
    ) -> dict[LandmarkName, Optional[Landmark]]:
        return {name: self.get_landmark(results, name) for name in names}
