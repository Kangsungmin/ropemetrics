from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import numpy as np


# ── 타입 별칭 ─────────────────────────────────────────────────────────────────
Landmark     = np.ndarray   # shape (4,) — [x, y, z, visibility], 정규화 0~1
LandmarkName = str          # "LEFT_ANKLE" 등 COCO 33-point 키


# ── LandmarkProvider ─────────────────────────────────────────────────────────

class LandmarkProvider(ABC):
    """
    포즈 추정 백엔드를 추상화하는 인터페이스.

    JumpCounter/Strategy는 이 인터페이스만 알면 되므로
    MediaPipe, MoveNet, OpenPose 등 백엔드를 교체해도 카운팅 로직은 수정 불필요.
    """

    @abstractmethod
    def get_landmark(
        self,
        results: object,
        name: LandmarkName,
    ) -> Optional[Landmark]:
        """
        이름으로 단일 랜드마크를 반환한다.

        Returns:
            np.ndarray([x, y, z, visibility], dtype=float32),
            랜드마크 부재 또는 신뢰도 미달이면 None.
        """
        ...

    def get_landmarks(
        self,
        results: object,
        names: Sequence[LandmarkName],
    ) -> dict[LandmarkName, Optional[Landmark]]:
        """복수 랜드마크를 한 번에 반환한다. 기본 구현은 get_landmark 루프."""
        return {name: self.get_landmark(results, name) for name in names}


# ── JumpEvent ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class JumpEvent:
    """
    점프 감지 시 on_jump 콜백에 전달되는 불변 데이터.

    frozen=True: 이벤트는 사실(fact)이므로 콜백 체인 어디에서도 변조 불가.
    """
    count:         int
    strategy_name: str
    peak_y:        Optional[float]
    frame_index:   int
    metadata:      dict = field(default_factory=dict, compare=False)


# ── JumpCounterConfig ─────────────────────────────────────────────────────────

@dataclass
class JumpCounterConfig:
    """
    JumpCounter 및 전략에 주입되는 파라미터 묶음.

    config.py 전역 상수 직접 의존을 끊어 단위 테스트를 단순화하고
    프리셋 팩토리(slow/fast/child)로 상황별 설정을 쉽게 교체한다.

    Fields:
        jump_threshold   : Y좌표 변화 임계값 (정규화, 0~1). 클수록 둔감.
        min_jump_frames  : 이동평균에서 제외할 최근 프레임 수. 오탐 필터.
        cooldown_frames  : 카운트 후 무시 프레임. 연속 오탐 방지.
        history_size     : 이동평균 윈도우 프레임 수.
        visibility_min   : 랜드마크 최소 visibility. 미달 시 해당 프레임 스킵.
    """
    jump_threshold:  float = 0.025
    min_jump_frames: int   = 3
    cooldown_frames: int   = 8
    history_size:    int   = 10
    visibility_min:  float = 0.5

    @classmethod
    def slow(cls) -> JumpCounterConfig:
        """
        느린 점프 / 성인 초보자용.
        임계값을 높여 기립 흔들림(~0.01~0.02)을 무시하고
        쿨다운을 길게 잡아 연속 오탐을 방지한다.
        """
        return cls(
            jump_threshold  = 0.04,
            min_jump_frames = 4,
            cooldown_frames = 12,
            history_size    = 12,
            visibility_min  = 0.5,
        )

    @classmethod
    def fast(cls) -> JumpCounterConfig:
        """
        빠른 점프 / 이단뛰기용.
        이단뛰기는 초당 4~5회(약 200ms/회).
        30fps 기준 cooldown=5 → 167ms 확보.
        history=7(233ms)로 반응 속도를 높인다.
        """
        return cls(
            jump_threshold  = 0.015,
            min_jump_frames = 2,
            cooldown_frames = 5,
            history_size    = 7,
            visibility_min  = 0.4,
        )

    @classmethod
    def child(cls) -> JumpCounterConfig:
        """
        어린이 / 중학생 초급용.
        점프 높이가 성인 대비 60~70% 수준이므로 임계값을 낮추고,
        자세가 불규칙하므로 visibility_min을 낮춰 가려짐에 관대하게 처리.
        """
        return cls(
            jump_threshold  = 0.018,
            min_jump_frames = 2,
            cooldown_frames = 8,
            history_size    = 8,
            visibility_min  = 0.35,
        )


# ── JumpStrategy ──────────────────────────────────────────────────────────────

class JumpStrategy(ABC):
    """
    점프 감지 알고리즘 인터페이스.

    구현체:
      - AnkleStrategy       (strategies/ankle.py)       — 발목 Y축 방식
      - HipShoulderStrategy (strategies/hip_shoulder.py) — 엉덩이+어깨 중심 방식

    설계 원칙:
      - Strategy는 JumpCounter가 넘겨준 랜드마크 딕셔너리에서
        스칼라 신호 하나(extract_signal)만 반환한다.
      - 히스토리·쿨다운 등 상태 관리는 JumpCounter가 담당.
      - reset()은 전략 내부 상태(캘리브레이션 버퍼 등)만 초기화.
    """

    def __init__(self, config: JumpCounterConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """전략 식별자. JumpEvent.strategy_name에 사용."""
        ...

    @abstractmethod
    def required_landmarks(self) -> Sequence[LandmarkName]:
        """
        이 전략이 필요로 하는 랜드마크 이름 목록.
        JumpCounter가 미리 fetch하여 extract_signal에 전달한다.
        """
        ...

    @abstractmethod
    def extract_signal(
        self,
        landmarks: dict[LandmarkName, Optional[Landmark]],
    ) -> Optional[float]:
        """
        랜드마크 딕셔너리에서 단일 스칼라 신호를 추출한다.

        반환값 규칙:
          - 몸이 낮을수록(지면에 가까울수록) 값이 커야 한다.
            (정규화 Y좌표계: 화면 아래 = Y 큰 값)
          - 필요 랜드마크 부재 / 신뢰도 미달이면 None 반환.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """전략 내부 상태(있는 경우)를 초기화한다."""
        ...
