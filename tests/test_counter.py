import numpy as np
import pytest
from unittest.mock import MagicMock

from ropemetrics.base import JumpCounterConfig, JumpEvent, LandmarkProvider
from ropemetrics.counter import JumpCounter
from ropemetrics.strategies.ankle import AnkleStrategy
from ropemetrics.strategies.hip_shoulder import HipShoulderStrategy


# ── 테스트용 헬퍼 ─────────────────────────────────────────────────────────────

class FixedProvider(LandmarkProvider):
    """지정된 Y값을 모든 랜드마크에 동일하게 반환하는 테스트 전용 Provider."""

    def __init__(self, y: float, visibility: float = 1.0) -> None:
        self._lm = np.array([0.5, y, 0.0, visibility], dtype=np.float32)

    def get_landmark(self, results, name):
        return self._lm

    def get_landmarks(self, results, names):
        return {n: self._lm for n in names}


def _make_counter(cfg=None, on_jump=None) -> JumpCounter:
    cfg = cfg or JumpCounterConfig()
    return JumpCounter(strategy=AnkleStrategy(cfg), config=cfg, on_jump=on_jump)


def _run(counter: JumpCounter, y_sequence: list, visibility: float = 1.0) -> None:
    results = MagicMock()
    for y in y_sequence:
        counter.update(FixedProvider(y, visibility), results)


# ── 기본 동작 ─────────────────────────────────────────────────────────────────

def test_no_jump_when_still():
    """정지 상태에서 카운트가 올라가면 안 된다."""
    counter = _make_counter()
    _run(counter, [0.8] * 20)
    assert counter.count == 0


def test_single_jump_detected():
    """지면→공중→착지 시퀀스에서 최소 1회 카운트된다."""
    counter = _make_counter()
    _run(counter, [0.8] * 8 + [0.72] * 5 + [0.8] * 8)
    assert counter.count >= 1


def test_reset_clears_count():
    """reset() 후 count가 0이어야 한다."""
    counter = _make_counter()
    _run(counter, [0.8] * 8 + [0.72] * 5 + [0.8] * 8)
    counter.reset()
    assert counter.count == 0


def test_multiple_jumps_count_correctly():
    """쿨다운이 끝나면 두 번째 점프도 카운트된다."""
    cfg     = JumpCounterConfig(cooldown_frames=3)
    counter = _make_counter(cfg)
    ground, air = [0.80] * 10, [0.72] * 5
    _run(counter, ground + air + ground + air + ground)
    assert counter.count >= 2


# ── 콜백 및 이벤트 ────────────────────────────────────────────────────────────

def test_on_jump_callback_receives_event():
    """점프 감지 시 on_jump 콜백이 올바른 JumpEvent와 함께 호출된다."""
    events: list[JumpEvent] = []
    counter = _make_counter(on_jump=lambda e: events.append(e))
    _run(counter, [0.8] * 8 + [0.72] * 5 + [0.8] * 8)
    assert len(events) >= 1
    ev = events[0]
    assert ev.count         == 1
    assert ev.strategy_name == "ankle"
    assert ev.peak_y        is not None
    assert ev.frame_index   > 0


def test_jump_event_is_immutable():
    """JumpEvent는 frozen dataclass이므로 필드 변경 시 예외가 발생한다."""
    ev = JumpEvent(count=1, strategy_name="ankle", peak_y=0.72, frame_index=10)
    with pytest.raises(Exception):
        ev.count = 2  # type: ignore


# ── 설정 프로파일 ─────────────────────────────────────────────────────────────

def test_config_slow_is_more_conservative():
    default, slow = JumpCounterConfig(), JumpCounterConfig.slow()
    assert slow.jump_threshold  > default.jump_threshold
    assert slow.cooldown_frames > default.cooldown_frames


def test_config_fast_is_more_sensitive():
    default, fast = JumpCounterConfig(), JumpCounterConfig.fast()
    assert fast.jump_threshold  < default.jump_threshold
    assert fast.cooldown_frames < default.cooldown_frames


def test_config_child_has_lower_visibility_min():
    default, child = JumpCounterConfig(), JumpCounterConfig.child()
    assert child.visibility_min < default.visibility_min


# ── 가시성 및 폴백 ────────────────────────────────────────────────────────────

def test_low_visibility_skips_frame():
    """visibility 미달 랜드마크는 프레임을 스킵하여 카운트가 0이어야 한다."""
    counter = _make_counter()
    _run(counter, [0.8] * 8 + [0.72] * 5 + [0.8] * 8, visibility=0.1)
    assert counter.count == 0


def test_single_ankle_fallback():
    """한쪽 발목만 유효해도 신호를 추출한다."""
    strategy  = AnkleStrategy(JumpCounterConfig())
    landmarks = {
        "LEFT_ANKLE":  np.array([0.5, 0.72, 0.0, 1.0], dtype=np.float32),
        "RIGHT_ANKLE": np.array([0.5, 0.80, 0.0, 0.1], dtype=np.float32),
    }
    assert strategy.extract_signal(landmarks) == pytest.approx(0.72)


# ── 전략 교체 ─────────────────────────────────────────────────────────────────

def test_strategy_names():
    cfg = JumpCounterConfig()
    assert AnkleStrategy(cfg).name        == "ankle"
    assert HipShoulderStrategy(cfg).name  == "hip_shoulder"


def test_hip_shoulder_strategy_detects_jump():
    """HipShoulderStrategy도 동일한 JumpCounter 인터페이스로 점프를 감지한다."""
    cfg      = JumpCounterConfig()
    counter  = JumpCounter(strategy=HipShoulderStrategy(cfg), config=cfg)
    results  = MagicMock()

    class HipShoulderProvider(LandmarkProvider):
        def __init__(self, y):
            self._lm = np.array([0.5, y, 0.0, 1.0], dtype=np.float32)
        def get_landmark(self, results, name):
            return self._lm
        def get_landmarks(self, results, names):
            return {n: self._lm for n in names}

    for y in [0.55] * 20 + [0.47] * 5 + [0.55] * 10:
        counter.update(HipShoulderProvider(y), results)

    assert counter.count >= 1


def test_strategy_reset_called_on_counter_reset():
    """counter.reset()이 strategy.reset()도 호출하는지 검증한다."""
    cfg      = JumpCounterConfig()
    strategy = MagicMock(spec=AnkleStrategy)
    strategy.name = "ankle"
    strategy.required_landmarks.return_value = ("LEFT_ANKLE", "RIGHT_ANKLE")
    strategy.extract_signal.return_value = 0.8
    strategy.config = cfg
    counter = JumpCounter(strategy=strategy, config=cfg)
    counter.reset()
    strategy.reset.assert_called_once()
