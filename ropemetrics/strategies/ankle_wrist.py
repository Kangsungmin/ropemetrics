from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

from ropemetrics.base import JumpCounterConfig, JumpStrategy, Landmark, LandmarkName


class AnkleWristStrategy(JumpStrategy):
    """
    발목 Y좌표 + 손목 움직임을 결합한 Level 2 점프 감지 전략.

    Level 1(AnkleStrategy)과 동일한 발목 Y 신호를 사용하되,
    손목이 충분히 움직이고 있을 때만 신호를 반환한다.

    줄넘기를 돌릴 때 손목은 원운동으로 Y좌표가 주기적으로 변한다.
    손목 정지 상태(일반 점프, 오탐)에서는 None을 반환하여 카운트를 차단한다.

    손목 움직임 판정:
        wrist_history_size 프레임 동안의 양 손목 Y 진폭 평균이
        wrist_motion_threshold 이상이면 움직임으로 간주한다.

    Args:
        config: JumpCounterConfig (wrist_motion_threshold, wrist_history_size 사용)
    """

    def __init__(self, config: JumpCounterConfig) -> None:
        super().__init__(config)
        self._left_wrist_buf:  deque[float] = deque(maxlen=config.wrist_history_size)
        self._right_wrist_buf: deque[float] = deque(maxlen=config.wrist_history_size)

    @property
    def name(self) -> str:
        return "ankle_wrist"

    def required_landmarks(self) -> Sequence[LandmarkName]:
        return ("LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_WRIST", "RIGHT_WRIST")

    def extract_signal(
        self,
        landmarks: dict[LandmarkName, Optional[Landmark]],
    ) -> Optional[float]:
        left_ankle  = landmarks.get("LEFT_ANKLE")
        right_ankle = landmarks.get("RIGHT_ANKLE")
        left_wrist  = landmarks.get("LEFT_WRIST")
        right_wrist = landmarks.get("RIGHT_WRIST")

        vis = self.config.visibility_min

        # 발목 신호 추출 (AnkleStrategy와 동일한 로직)
        left_ok  = left_ankle  is not None and float(left_ankle[3])  >= vis
        right_ok = right_ankle is not None and float(right_ankle[3]) >= vis

        if not left_ok and not right_ok:
            return None

        if left_ok and right_ok:
            ankle_signal = float((left_ankle[1] + right_ankle[1]) / 2)
        else:
            ankle_signal = float(left_ankle[1] if left_ok else right_ankle[1])

        # 손목 이력 업데이트
        if left_wrist is not None and float(left_wrist[3]) >= vis:
            self._left_wrist_buf.append(float(left_wrist[1]))
        if right_wrist is not None and float(right_wrist[3]) >= vis:
            self._right_wrist_buf.append(float(right_wrist[1]))

        # 손목 움직임 게이트: 버퍼가 충분히 쌓이지 않았으면 통과 불가
        if (len(self._left_wrist_buf) < self.config.wrist_history_size and
                len(self._right_wrist_buf) < self.config.wrist_history_size):
            return None

        # 각 손목의 Y 진폭(max - min) 계산
        left_amplitude  = (max(self._left_wrist_buf)  - min(self._left_wrist_buf)
                           if len(self._left_wrist_buf) >= 2 else 0.0)
        right_amplitude = (max(self._right_wrist_buf) - min(self._right_wrist_buf)
                           if len(self._right_wrist_buf) >= 2 else 0.0)

        # 유효한 손목 중 하나라도 임계값 이상이면 움직임으로 판정
        amplitudes = [a for a in (left_amplitude, right_amplitude) if a > 0]
        if not amplitudes:
            return None

        wrist_motion = max(amplitudes)
        if wrist_motion < self.config.wrist_motion_threshold:
            return None

        return ankle_signal

    def reset(self) -> None:
        self._left_wrist_buf.clear()
        self._right_wrist_buf.clear()
