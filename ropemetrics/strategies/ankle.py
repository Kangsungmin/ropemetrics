from __future__ import annotations

from typing import Optional, Sequence

from ropemetrics.base import JumpCounterConfig, JumpStrategy, Landmark, LandmarkName


class AnkleStrategy(JumpStrategy):
    """
    발목 Y좌표 평균으로 점프를 감지하는 전략.

    signal = (LEFT_ANKLE.y + RIGHT_ANKLE.y) / 2

    특징:
      - 발목 이동 거리가 가장 크므로 작은 점프도 민감하게 포착
      - 한쪽 발목만 보이면 단독값으로 fallback (외발 점프 대응)
      - 양쪽 모두 visibility_min 미달이면 None → 해당 프레임 스킵
      - 내부 상태 없음 → reset()이 no-op
    """

    @property
    def name(self) -> str:
        return "ankle"

    def required_landmarks(self) -> Sequence[LandmarkName]:
        return ("LEFT_ANKLE", "RIGHT_ANKLE")

    def extract_signal(
        self,
        landmarks: dict[LandmarkName, Optional[Landmark]],
    ) -> Optional[float]:
        left  = landmarks.get("LEFT_ANKLE")
        right = landmarks.get("RIGHT_ANKLE")

        left_ok  = left  is not None and float(left[3])  >= self.config.visibility_min
        right_ok = right is not None and float(right[3]) >= self.config.visibility_min

        if not left_ok and not right_ok:
            return None

        if left_ok and right_ok:
            return float((left[1] + right[1]) / 2)

        # 한쪽만 유효한 경우 — 가림 또는 외발 점프
        return float(left[1] if left_ok else right[1])

    def reset(self) -> None:
        pass
