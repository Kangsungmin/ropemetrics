from __future__ import annotations

from typing import Optional, Sequence

from ropemetrics.base import JumpCounterConfig, JumpStrategy, Landmark, LandmarkName


class GroundContactStrategy(JumpStrategy):
    """
    지면 기준선 캘리브레이션 기반 점프 감지 전략.

    signal = ankle_y / ground_y
      - 지면 접촉 시 ≈ 1.0 (높은 값)
      - 공중 시 < 1.0 (낮은 값)

    초기 calibration_frames 프레임 동안 발목 Y 최댓값으로 ground_y를 결정한다.
    카메라 거리·신장에 무관한 정규화 신호를 제공하므로
    jump_threshold를 0.03 수준으로 고정할 수 있다.

    캘리브레이션 전에는 raw ankle_y를 반환한다 (JumpCounter 워밍업이 커버).

    Args:
        config             : JumpCounterConfig (ground_contact() 프리셋 권장)
        calibration_frames : 지면 기준선 수집 프레임 수 (기본 20)
    """

    def __init__(
        self,
        config: JumpCounterConfig,
        calibration_frames: int = 20,
    ) -> None:
        super().__init__(config)
        self._calibration_frames = calibration_frames
        self._ground_y: Optional[float] = None
        self._calib_buf: list[float] = []

    @property
    def name(self) -> str:
        return "ground_contact"

    @property
    def is_calibrated(self) -> bool:
        return self._ground_y is not None

    def required_landmarks(self) -> Sequence[LandmarkName]:
        return ("LEFT_ANKLE", "RIGHT_ANKLE")

    def extract_signal(
        self,
        landmarks: dict[LandmarkName, Optional[Landmark]],
    ) -> Optional[float]:
        left  = landmarks.get("LEFT_ANKLE")
        right = landmarks.get("RIGHT_ANKLE")

        vis = self.config.visibility_min
        left_ok  = left  is not None and float(left[3])  >= vis
        right_ok = right is not None and float(right[3]) >= vis

        if not left_ok and not right_ok:
            return None

        if left_ok and right_ok:
            ankle_y = float((left[1] + right[1]) / 2)
        else:
            ankle_y = float(left[1] if left_ok else right[1])

        # 캘리브레이션: 초기 N프레임의 최댓값 = 지면 기준선
        if self._ground_y is None:
            self._calib_buf.append(ankle_y)
            if len(self._calib_buf) >= self._calibration_frames:
                self._ground_y = max(self._calib_buf)
            return ankle_y  # 수집 중에는 raw 반환

        return ankle_y / self._ground_y

    def reset(self) -> None:
        self._ground_y = None
        self._calib_buf.clear()
