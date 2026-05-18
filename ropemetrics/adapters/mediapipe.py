"""
MediaPipe Tasks API adapter for ropemetrics.

Install:
    pip install ropemetrics[mediapipe]

Usage:
    from ropemetrics.adapters import MediaPipeLandmarkProvider
    provider = MediaPipeLandmarkProvider()
    counter.update(provider, results)   # results = landmarker.detect_for_video(...)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ropemetrics.base import Landmark, LandmarkName, LandmarkProvider


# MediaPipe Pose 33-point COCO landmark index map
LANDMARK_INDEX: dict[str, int] = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1,  "LEFT_EYE": 2,  "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7,  "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11,  "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,     "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,     "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,     "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,     "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,     "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,       "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,      "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,     "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,      "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}


class MediaPipeLandmarkProvider(LandmarkProvider):
    """
    LandmarkProvider adapter for MediaPipe Tasks API (PoseLandmarker).

    Compatible with mediapipe>=0.10.30 Tasks API.
    Pass the raw PoseLandmarkerResult from landmarker.detect_for_video()
    directly to counter.update().

    Args:
        person_index: Index of the person to track in multi-person results.
                      Defaults to 0 (first detected person).

    Example:
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        options  = vision.PoseLandmarkerOptions(...)
        detector = vision.PoseLandmarker.create_from_options(options)

        provider = MediaPipeLandmarkProvider()

        # in video loop:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results  = detector.detect_for_video(mp_image, timestamp_ms)
        counter.update(provider, results)
    """

    def __init__(self, person_index: int = 0) -> None:
        self._person_index = person_index

    def get_landmark(
        self,
        results: object,
        name: LandmarkName,
    ) -> Optional[Landmark]:
        landmarks = getattr(results, "pose_landmarks", None)
        if not landmarks or self._person_index >= len(landmarks):
            return None

        person = landmarks[self._person_index]
        idx    = LANDMARK_INDEX.get(name)
        if idx is None or idx >= len(person):
            return None

        lm = person[idx]
        return np.array(
            [lm.x, lm.y, lm.z,
             lm.visibility if hasattr(lm, "visibility") else 1.0],
            dtype=np.float32,
        )

    def get_landmarks(
        self,
        results: object,
        names: Sequence[LandmarkName],
    ) -> dict[LandmarkName, Optional[Landmark]]:
        return {name: self.get_landmark(results, name) for name in names}
