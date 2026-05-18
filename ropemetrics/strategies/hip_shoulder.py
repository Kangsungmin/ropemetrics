from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np

from ropemetrics.base import JumpCounterConfig, JumpStrategy, Landmark, LandmarkName


class HipShoulderStrategy(JumpStrategy):
    """
    엉덩이·어깨 가중 평균으로 점프를 감지하는 전략.

    signal = hip_mid * HIP_WEIGHT + shoulder_mid * (1 - HIP_WEIGHT)
      hip_mid      = (LEFT_HIP.y + RIGHT_HIP.y) / 2
      shoulder_mid = (LEFT_SHOULDER.y + RIGHT_SHOULDER.y) / 2

    AnkleStrategy 대비 장점:
      - 엉덩이·어깨는 화면 중앙부라 가려질 확률이 낮음
      - 무게중심에 가까운 신호라 노이즈 내성이 높음
      - 다중 인원 환경에서 부분 가림이 많을 때 유리

    캘리브레이션:
      초기 calibration_frames 프레임 동안 기준선(baseline_y)을 수집한다.
      기준선이 확보되면 신호를 기준선 대비 상대값으로 사용 가능.
      → 키가 다른 학생에도 동일한 threshold 적용 가능.

    Args:
        config             : JumpCounterConfig
        calibration_frames : 기준선 수집 프레임 수 (기본 15)
    """

    HIP_WEIGHT: float = 0.7

    def __init__(
        self,
        config: JumpCounterConfig,
        calibration_frames: int = 15,
    ) -> None:
        super().__init__(config)
        self._calibration_frames = calibration_frames
        self._baseline_y: Optional[float] = None
        self._calib_buf: list[float] = []

    @property
    def name(self) -> str:
        return "hip_shoulder"

    @property
    def is_calibrated(self) -> bool:
        return self._baseline_y is not None

    def required_landmarks(self) -> Sequence[LandmarkName]:
        return ("LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER")

    def extract_signal(
        self,
        landmarks: dict[LandmarkName, Optional[Landmark]],
    ) -> Optional[float]:
        lh = landmarks.get("LEFT_HIP")
        rh = landmarks.get("RIGHT_HIP")
        ls = landmarks.get("LEFT_SHOULDER")
        rs = landmarks.get("RIGHT_SHOULDER")

        vis = self.config.visibility_min
        hip_ok = (
            lh is not None and float(lh[3]) >= vis and
            rh is not None and float(rh[3]) >= vis
        )
        shoulder_ok = (
            ls is not None and float(ls[3]) >= vis and
            rs is not None and float(rs[3]) >= vis
        )

        if not hip_ok:
            return None  # 엉덩이 없으면 신호 불가

        hip_mid = float((lh[1] + rh[1]) / 2)

        if shoulder_ok:
            shoulder_mid = float((ls[1] + rs[1]) / 2)
            signal = hip_mid * self.HIP_WEIGHT + shoulder_mid * (1 - self.HIP_WEIGHT)
        else:
            signal = hip_mid  # 어깨 미감지 시 엉덩이만으로 fallback

        # 캘리브레이션: 초기 N프레임으로 기준선 수집
        if self._baseline_y is None:
            self._calib_buf.append(signal)
            if len(self._calib_buf) >= self._calibration_frames:
                self._baseline_y = float(np.mean(self._calib_buf))
            return signal  # 수집 중에는 raw 신호 그대로 반환

        return signal

    def reset(self) -> None:
        self._baseline_y = None
        self._calib_buf.clear()
