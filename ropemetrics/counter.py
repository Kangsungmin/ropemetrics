from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import numpy as np

from ropemetrics.base import (
    JumpCounterConfig,
    JumpEvent,
    JumpStrategy,
    LandmarkProvider,
)


class JumpCounter:
    """
    전략 패턴 기반 줄넘기 횟수 카운터.

    - LandmarkProvider : 포즈 백엔드 추상화 (MediaPipe 등)
    - JumpStrategy     : 알고리즘 추상화 (발목 / 엉덩이-어깨 등)
    - on_jump          : 점프 감지 시 JumpEvent를 받는 콜백

    Args:
        strategy : 점프 감지 알고리즘 인스턴스
        config   : 카운팅 파라미터. None이면 JumpCounterConfig() 기본값 사용.
        on_jump  : 점프 감지 시 호출되는 콜백. None이면 무시.

    Example:
        cfg      = JumpCounterConfig.child()
        strategy = AnkleStrategy(cfg)
        provider = MediaPipeLandmarkProvider()
        counter  = JumpCounter(strategy, cfg, on_jump=lambda e: print(e.count))

        while True:
            results = pose_analyzer.process(frame)
            counter.update(provider, results)
    """

    def __init__(
        self,
        strategy: JumpStrategy,
        config:   Optional[JumpCounterConfig] = None,
        on_jump:  Optional[Callable[[JumpEvent], None]] = None,
    ) -> None:
        self._strategy = strategy
        self._config   = config or JumpCounterConfig()
        self._on_jump  = on_jump

        self._history: deque[float] = deque(maxlen=self._config.history_size)
        self._count       = 0
        self._cooldown    = 0
        self._in_air      = False
        self._peak_y: Optional[float] = None
        self._frame_index = 0

    # ── 공개 속성 ──────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return self._count

    @property
    def strategy(self) -> JumpStrategy:
        return self._strategy

    @property
    def config(self) -> JumpCounterConfig:
        return self._config

    # ── 메인 업데이트 ──────────────────────────────────────────────────────

    def update(
        self,
        provider: LandmarkProvider,
        results:  object,
    ) -> bool:
        """
        한 프레임을 처리하고 새 점프 감지 여부를 반환한다.

        Args:
            provider : LandmarkProvider 구현체 (MediaPipeLandmarkProvider 등)
            results  : provider.process()가 반환한 원시 결과

        Returns:
            True  — 이 프레임에서 새 점프가 카운트됨 (on_jump도 호출됨)
            False — 아님
        """
        self._frame_index += 1

        # 1. 전략이 필요한 랜드마크 일괄 조회
        landmarks = provider.get_landmarks(results, self._strategy.required_landmarks())

        # 2. 알고리즘에서 스칼라 신호 추출 (None이면 프레임 스킵)
        signal = self._strategy.extract_signal(landmarks)
        if signal is None:
            return False

        self._history.append(signal)

        # 3. 쿨다운 소진 중
        if self._cooldown > 0:
            self._cooldown -= 1
            return False

        # 4. 워밍업 — 히스토리가 다 찰 때까지 대기
        if len(self._history) < self._config.history_size:
            return False

        history   = np.array(self._history)
        current   = history[-1]
        prev_avg  = float(np.mean(history[:-self._config.min_jump_frames]))
        threshold = self._config.jump_threshold

        # 5. 공중 진입 판정 (Y가 줄어듦 = 몸이 올라감)
        if not self._in_air and (prev_avg - current) > threshold:
            self._in_air = True
            self._peak_y = current

        # 6. 착지 판정 (Y가 다시 커짐 = 몸이 내려옴)
        if self._in_air and self._peak_y is not None:
            if (current - self._peak_y) > threshold:
                self._count   += 1
                self._in_air   = False
                self._cooldown = self._config.cooldown_frames

                event = JumpEvent(
                    count         = self._count,
                    strategy_name = self._strategy.name,
                    peak_y        = self._peak_y,
                    frame_index   = self._frame_index,
                )
                if self._on_jump is not None:
                    self._on_jump(event)

                return True

            # 7. 최고점 갱신 (더 낮은 Y = 더 높이 올라간 것)
            self._peak_y = min(self._peak_y, current)

        return False

    # ── 초기화 ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """카운트, 상태, 히스토리를 모두 초기화한다."""
        self._count       = 0
        self._in_air      = False
        self._peak_y      = None
        self._cooldown    = 0
        self._frame_index = 0
        self._history.clear()
        self._strategy.reset()
