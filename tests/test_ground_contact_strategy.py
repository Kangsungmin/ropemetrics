import numpy as np
import pytest
from unittest.mock import MagicMock

from ropemetrics.base import JumpCounterConfig, LandmarkProvider
from ropemetrics.counter import JumpCounter
from ropemetrics.strategies.ground_contact import GroundContactStrategy


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _lm(y: float, vis: float = 1.0) -> np.ndarray:
    return np.array([0.5, y, 0.0, vis], dtype=np.float32)


def _make_landmarks(left_y: float, right_y: float, vis: float = 1.0) -> dict:
    return {
        "LEFT_ANKLE":  _lm(left_y, vis),
        "RIGHT_ANKLE": _lm(right_y, vis),
    }


class FixedProvider(LandmarkProvider):
    """지정된 Y값을 모든 랜드마크에 동일하게 반환하는 테스트 전용 Provider."""

    def __init__(self, y: float, visibility: float = 1.0) -> None:
        self._lm = np.array([0.5, y, 0.0, visibility], dtype=np.float32)

    def get_landmark(self, results, name):
        return self._lm

    def get_landmarks(self, results, names):
        return {n: self._lm for n in names}


def _run(counter: JumpCounter, y_sequence: list, visibility: float = 1.0) -> None:
    results = MagicMock()
    for y in y_sequence:
        counter.update(FixedProvider(y, visibility), results)


# ── 캘리브레이션 ───────────────────────────────────────────────────────────────

def test_calibration_completes_after_n_frames():
    """N프레임 공급 후 is_calibrated == True."""
    strategy = GroundContactStrategy(JumpCounterConfig(), calibration_frames=5)
    lm = _make_landmarks(0.85, 0.85)
    for _ in range(5):
        strategy.extract_signal(lm)
    assert strategy.is_calibrated is True


def test_not_calibrated_before_n_frames():
    """N-1프레임에서는 is_calibrated == False."""
    strategy = GroundContactStrategy(JumpCounterConfig(), calibration_frames=5)
    lm = _make_landmarks(0.85, 0.85)
    for _ in range(4):
        strategy.extract_signal(lm)
    assert strategy.is_calibrated is False


def test_ground_y_is_max_of_calib_buf():
    """ground_y == 캘리브레이션 버퍼의 최댓값."""
    strategy = GroundContactStrategy(JumpCounterConfig(), calibration_frames=5)
    y_values = [0.80, 0.85, 0.83, 0.84, 0.82]
    for y in y_values:
        strategy.extract_signal(_make_landmarks(y, y))
    assert strategy._ground_y == pytest.approx(max(y_values))


def test_reset_clears_calibration():
    """reset() 후 is_calibrated == False, _calib_buf가 빈 리스트."""
    strategy = GroundContactStrategy(JumpCounterConfig(), calibration_frames=3)
    lm = _make_landmarks(0.85, 0.85)
    for _ in range(3):
        strategy.extract_signal(lm)
    assert strategy.is_calibrated is True
    strategy.reset()
    assert strategy.is_calibrated is False
    assert strategy._calib_buf == []


# ── 신호 변환 ─────────────────────────────────────────────────────────────────

def test_signal_raw_before_calibration():
    """캘리브레이션 전에는 raw ankle_y를 반환한다."""
    strategy = GroundContactStrategy(JumpCounterConfig(), calibration_frames=10)
    lm = _make_landmarks(0.85, 0.85)
    result = strategy.extract_signal(lm)
    assert result == pytest.approx(0.85)


def test_signal_normalized_after_calibration():
    """캘리브레이션 완료 후 signal == ankle_y / ground_y."""
    strategy = GroundContactStrategy(JumpCounterConfig(), calibration_frames=3)
    ground_y = 0.85
    for _ in range(3):
        strategy.extract_signal(_make_landmarks(ground_y, ground_y))

    air_y = 0.70
    result = strategy.extract_signal(_make_landmarks(air_y, air_y))
    assert result == pytest.approx(air_y / ground_y)


def test_signal_on_ground_is_approximately_one():
    """지면 접촉 시 signal ≈ 1.0."""
    strategy = GroundContactStrategy(JumpCounterConfig(), calibration_frames=5)
    ground_y = 0.85
    for _ in range(5):
        strategy.extract_signal(_make_landmarks(ground_y, ground_y))

    result = strategy.extract_signal(_make_landmarks(ground_y, ground_y))
    assert result == pytest.approx(1.0)


# ── Fallback / visibility ─────────────────────────────────────────────────────

def test_single_ankle_fallback_left():
    """왼쪽 발목만 유효 시 LEFT_ANKLE.y를 단독값으로 사용."""
    cfg = JumpCounterConfig(visibility_min=0.5)
    strategy = GroundContactStrategy(cfg, calibration_frames=1)
    lm = {
        "LEFT_ANKLE":  _lm(0.80, vis=1.0),
        "RIGHT_ANKLE": _lm(0.80, vis=0.1),  # visibility 미달
    }
    result = strategy.extract_signal(lm)
    assert result == pytest.approx(0.80)


def test_single_ankle_fallback_right():
    """오른쪽 발목만 유효 시 RIGHT_ANKLE.y를 단독값으로 사용."""
    cfg = JumpCounterConfig(visibility_min=0.5)
    strategy = GroundContactStrategy(cfg, calibration_frames=1)
    lm = {
        "LEFT_ANKLE":  _lm(0.80, vis=0.1),  # visibility 미달
        "RIGHT_ANKLE": _lm(0.82, vis=1.0),
    }
    result = strategy.extract_signal(lm)
    assert result == pytest.approx(0.82)


def test_both_ankles_invisible_returns_none():
    """양쪽 발목 모두 visibility 미달 시 None 반환."""
    cfg = JumpCounterConfig(visibility_min=0.5)
    strategy = GroundContactStrategy(cfg, calibration_frames=1)
    lm = {
        "LEFT_ANKLE":  _lm(0.80, vis=0.1),
        "RIGHT_ANKLE": _lm(0.80, vis=0.1),
    }
    result = strategy.extract_signal(lm)
    assert result is None


# ── required_landmarks / name ─────────────────────────────────────────────────

def test_required_landmarks():
    """required_landmarks()가 두 발목 이름을 포함한다."""
    strategy = GroundContactStrategy(JumpCounterConfig())
    required = strategy.required_landmarks()
    assert "LEFT_ANKLE"  in required
    assert "RIGHT_ANKLE" in required


def test_name():
    """name 프로퍼티가 'ground_contact'를 반환한다."""
    assert GroundContactStrategy(JumpCounterConfig()).name == "ground_contact"


# ── JumpCounter 통합 ──────────────────────────────────────────────────────────

def test_jump_sequence_counts_one():
    """지면→공중→착지 시퀀스에서 1회 카운트된다."""
    cfg      = JumpCounterConfig.ground_contact()
    strategy = GroundContactStrategy(cfg, calibration_frames=5)
    counter  = JumpCounter(strategy, cfg)

    # calibration: 5프레임 (ground_y = 0.85)
    # warmup: history_size=6 프레임 필요 → 캘리브레이션과 겹침
    ground = [0.85] * 10
    air    = [0.78] * 6   # 0.78/0.85 ≈ 0.918 → 약 0.08 감소 > threshold 0.03
    land   = [0.85] * 8

    _run(counter, ground + air + land)
    assert counter.count >= 1


def test_no_jump_when_still():
    """정지 상태에서는 카운트가 올라가지 않는다."""
    cfg      = JumpCounterConfig.ground_contact()
    strategy = GroundContactStrategy(cfg, calibration_frames=5)
    counter  = JumpCounter(strategy, cfg)
    _run(counter, [0.85] * 30)
    assert counter.count == 0


def test_ground_contact_preset_fields():
    """ground_contact() 프리셋의 핵심 파라미터 값을 검증한다."""
    cfg = JumpCounterConfig.ground_contact()
    assert cfg.jump_threshold  == pytest.approx(0.03)
    assert cfg.history_size    == 6
    assert cfg.cooldown_frames == 8
