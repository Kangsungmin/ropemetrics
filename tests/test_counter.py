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


# ── JumpCounterConfig ─────────────────────────────────────────────────────────

def test_config_default_values():
    """기본 생성자의 모든 필드가 base.py에 정의된 기본값과 일치해야 한다."""
    cfg = JumpCounterConfig()
    assert cfg.jump_threshold  == 0.025
    assert cfg.min_jump_frames == 3
    assert cfg.cooldown_frames == 8
    assert cfg.history_size    == 10
    assert cfg.visibility_min  == 0.5


def test_config_fast_has_smaller_history_size():
    """fast 프리셋의 history_size는 기본값보다 작아야 한다."""
    default, fast = JumpCounterConfig(), JumpCounterConfig.fast()
    assert fast.history_size < default.history_size


def test_config_slow_has_larger_history_size():
    """slow 프리셋의 history_size는 기본값보다 커야 한다."""
    default, slow = JumpCounterConfig(), JumpCounterConfig.slow()
    assert slow.history_size > default.history_size


# ── JumpEvent ─────────────────────────────────────────────────────────────────

def test_jump_event_metadata_defaults_to_empty_dict():
    """metadata 인자 없이 생성했을 때 ev.metadata가 빈 딕셔너리여야 한다."""
    ev = JumpEvent(count=1, strategy_name="ankle", peak_y=0.72, frame_index=10)
    assert ev.metadata == {}


def test_jump_event_fields_accessible():
    """count, strategy_name, peak_y, frame_index 필드가 생성 후 정상 접근되어야 한다."""
    ev = JumpEvent(count=3, strategy_name="hip_shoulder", peak_y=0.45, frame_index=42)
    assert ev.count         == 3
    assert ev.strategy_name == "hip_shoulder"
    assert ev.peak_y        == pytest.approx(0.45)
    assert ev.frame_index   == 42


# ── AnkleStrategy ─────────────────────────────────────────────────────────────

def test_ankle_both_valid_returns_average():
    """양쪽 발목 모두 visibility=1.0일 때 (left.y + right.y) / 2를 반환한다."""
    strategy  = AnkleStrategy(JumpCounterConfig())
    landmarks = {
        "LEFT_ANKLE":  np.array([0.5, 0.72, 0.0, 1.0], dtype=np.float32),
        "RIGHT_ANKLE": np.array([0.5, 0.80, 0.0, 1.0], dtype=np.float32),
    }
    assert strategy.extract_signal(landmarks) == pytest.approx(0.76)


def test_ankle_both_invalid_returns_none():
    """양쪽 발목 모두 visibility 미달(0.1)일 때 None을 반환한다."""
    strategy  = AnkleStrategy(JumpCounterConfig())
    landmarks = {
        "LEFT_ANKLE":  np.array([0.5, 0.72, 0.0, 0.1], dtype=np.float32),
        "RIGHT_ANKLE": np.array([0.5, 0.80, 0.0, 0.1], dtype=np.float32),
    }
    assert strategy.extract_signal(landmarks) is None


def test_ankle_right_only_fallback():
    """왼쪽 발목이 visibility 미달일 때 오른쪽 발목의 y값 단독으로 반환한다."""
    strategy  = AnkleStrategy(JumpCounterConfig())
    landmarks = {
        "LEFT_ANKLE":  np.array([0.5, 0.72, 0.0, 0.1], dtype=np.float32),
        "RIGHT_ANKLE": np.array([0.5, 0.80, 0.0, 1.0], dtype=np.float32),
    }
    assert strategy.extract_signal(landmarks) == pytest.approx(0.80)


def test_ankle_required_landmarks_returns_both_ankles():
    """required_landmarks()가 LEFT_ANKLE과 RIGHT_ANKLE 두 항목을 모두 포함한다."""
    strategy = AnkleStrategy(JumpCounterConfig())
    required = strategy.required_landmarks()
    assert "LEFT_ANKLE"  in required
    assert "RIGHT_ANKLE" in required


def test_ankle_reset_is_noop():
    """reset() 호출 후 예외 없이 종료되며, 연속 호출에도 내부 상태에 영향이 없다."""
    strategy = AnkleStrategy(JumpCounterConfig())
    strategy.reset()
    strategy.reset()
    landmarks = {
        "LEFT_ANKLE":  np.array([0.5, 0.72, 0.0, 1.0], dtype=np.float32),
        "RIGHT_ANKLE": np.array([0.5, 0.80, 0.0, 1.0], dtype=np.float32),
    }
    assert strategy.extract_signal(landmarks) == pytest.approx(0.76)


# ── HipShoulderStrategy ───────────────────────────────────────────────────────

def _make_hip_shoulder_landmarks(hip_y=0.55, shoulder_y=0.35,
                                  hip_vis=1.0, shoulder_vis=1.0):
    """hip/shoulder Y값과 visibility를 지정한 랜드마크 dict 생성."""
    def lm(y, vis):
        return np.array([0.5, y, 0.0, vis], dtype=np.float32)
    return {
        "LEFT_HIP":       lm(hip_y, hip_vis),
        "RIGHT_HIP":      lm(hip_y, hip_vis),
        "LEFT_SHOULDER":  lm(shoulder_y, shoulder_vis),
        "RIGHT_SHOULDER": lm(shoulder_y, shoulder_vis),
    }


def test_hip_shoulder_calibration_completes_after_n_frames():  # N-14
    """calibration_frames=5로 설정하고 5번 extract_signal 호출 후 is_calibrated == True 검증."""
    strategy = HipShoulderStrategy(JumpCounterConfig(), calibration_frames=5)
    landmarks = _make_hip_shoulder_landmarks()
    for _ in range(5):
        strategy.extract_signal(landmarks)
    assert strategy.is_calibrated is True


def test_hip_shoulder_not_calibrated_before_n_frames():  # N-15
    """calibration_frames=5로 설정하고 4번 호출 후 is_calibrated == False 검증."""
    strategy = HipShoulderStrategy(JumpCounterConfig(), calibration_frames=5)
    landmarks = _make_hip_shoulder_landmarks()
    for _ in range(4):
        strategy.extract_signal(landmarks)
    assert strategy.is_calibrated is False


def test_hip_shoulder_reset_clears_calibration():  # N-16
    """캘리브레이션 완료 후 reset() 호출 시 is_calibrated == False이고 _calib_buf가 빈 리스트임을 검증."""
    strategy = HipShoulderStrategy(JumpCounterConfig(), calibration_frames=5)
    landmarks = _make_hip_shoulder_landmarks()
    for _ in range(5):
        strategy.extract_signal(landmarks)
    assert strategy.is_calibrated is True
    strategy.reset()
    assert strategy.is_calibrated is False
    assert strategy._calib_buf == []


def test_hip_shoulder_hip_not_ok_returns_none():  # N-17
    """hip visibility 미달 시 None 반환. hip_vis=0.1(미달)로 설정."""
    strategy = HipShoulderStrategy(JumpCounterConfig())
    landmarks = _make_hip_shoulder_landmarks(hip_vis=0.1)
    result = strategy.extract_signal(landmarks)
    assert result is None


def test_hip_shoulder_shoulder_fallback_uses_hip_only():  # N-18
    """shoulder visibility 미달 시 hip_mid 단독값 반환. shoulder_vis=0.1(미달), hip_y=0.55이면 0.55 반환."""
    strategy = HipShoulderStrategy(JumpCounterConfig())
    landmarks = _make_hip_shoulder_landmarks(hip_y=0.55, shoulder_vis=0.1)
    result = strategy.extract_signal(landmarks)
    assert result == pytest.approx(0.55)


def test_hip_shoulder_weighted_signal_formula():  # N-19
    """hip_y=0.60, shoulder_y=0.30일 때 0.60 * 0.7 + 0.30 * 0.3 = 0.51 반환 검증."""
    strategy = HipShoulderStrategy(JumpCounterConfig())
    landmarks = _make_hip_shoulder_landmarks(hip_y=0.60, shoulder_y=0.30)
    result = strategy.extract_signal(landmarks)
    assert result == pytest.approx(0.51)


def test_hip_shoulder_required_landmarks():  # N-20
    """required_landmarks()가 4개 랜드마크(LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER) 모두 포함하는지 검증."""
    strategy = HipShoulderStrategy(JumpCounterConfig())
    required = strategy.required_landmarks()
    assert "LEFT_HIP"       in required
    assert "RIGHT_HIP"      in required
    assert "LEFT_SHOULDER"  in required
    assert "RIGHT_SHOULDER" in required


# ── JumpCounter 상태 관리 ─────────────────────────────────────────────────────

def test_counter_frame_index_increments_every_update():  # N-01
    """update() 5회 호출 후 _frame_index가 정확히 5인지 검증한다."""
    counter = _make_counter()
    results = MagicMock()
    for _ in range(5):
        counter.update(FixedProvider(0.8), results)
    assert counter._frame_index == 5


def test_counter_returns_true_only_on_jump_frame():  # N-02
    """착지 시점 프레임에서만 True가 반환되며, True 개수가 감지된 점프 수와 일치하는지 검증한다."""
    cfg     = JumpCounterConfig(cooldown_frames=3)
    counter = _make_counter(cfg)
    results = MagicMock()
    y_sequence = [0.8] * 10 + [0.72] * 5 + [0.8] * 10
    return_values = [
        counter.update(FixedProvider(y), results)
        for y in y_sequence
    ]
    true_count = sum(1 for v in return_values if v is True)
    assert true_count == counter.count
    assert true_count >= 1


def test_counter_cooldown_suppresses_count_during_cooldown():  # N-03
    """착지 후 쿨다운 기간 내에 추가 점프 시퀀스를 주입해도 count가 올라가지 않음을 검증한다."""
    cfg     = JumpCounterConfig(cooldown_frames=20)
    counter = _make_counter(cfg)
    results = MagicMock()
    # 첫 번째 점프 — 착지까지 완료
    for y in [0.8] * 10 + [0.72] * 5 + [0.8] * 5:
        counter.update(FixedProvider(y), results)
    assert counter.count == 1
    # 쿨다운 5프레임만 소진 (cooldown_frames=20 이므로 아직 쿨다운 중)
    for _ in range(5):
        counter.update(FixedProvider(0.8), results)
    # 쿨다운 중에 또 다른 점프 시퀀스 주입
    for y in [0.72] * 5 + [0.8] * 5:
        counter.update(FixedProvider(y), results)
    assert counter.count == 1


def test_counter_warmup_prevents_early_count():  # N-04
    """history_size=10인 카운터에 10프레임 미만의 데이터만 주입하면 count가 0이어야 한다."""
    cfg     = JumpCounterConfig(history_size=10)
    counter = _make_counter(cfg)
    results = MagicMock()
    # 워밍업 미완료 상태에서 점프처럼 보이는 시퀀스 주입 (총 9프레임)
    for y in [0.8] * 5 + [0.72] * 4:
        counter.update(FixedProvider(y), results)
    assert counter.count == 0


def test_counter_signal_none_skips_frame():  # N-05
    """extract_signal이 항상 None을 반환하는 전략에서 카운트가 0임을 검증한다."""
    cfg      = JumpCounterConfig()
    strategy = MagicMock(spec=AnkleStrategy)
    strategy.name = "ankle"
    strategy.required_landmarks.return_value = ("LEFT_ANKLE", "RIGHT_ANKLE")
    strategy.extract_signal.return_value = None
    strategy.config = cfg
    counter = JumpCounter(strategy=strategy, config=cfg)
    results = MagicMock()
    for y in [0.8] * 10 + [0.72] * 5 + [0.8] * 10:
        counter.update(FixedProvider(y), results)
    assert counter.count == 0


def test_counter_reset_resets_all_internal_state():  # N-06
    """점프 감지 후 reset() 호출 시 모든 내부 상태가 초기값으로 돌아오는지 검증한다."""
    counter = _make_counter()
    _run(counter, [0.8] * 10 + [0.72] * 5 + [0.8] * 10)
    assert counter.count >= 1
    counter.reset()
    assert counter.count       == 0
    assert counter._in_air     is False
    assert counter._cooldown   == 0
    assert counter._peak_y     is None
    assert counter._frame_index == 0
    assert len(counter._history) == 0


def test_counter_no_on_jump_callback_does_not_raise():  # N-07
    """on_jump=None(기본값)인 카운터에서 점프가 감지될 때 예외가 발생하지 않음을 검증한다."""
    counter = _make_counter(on_jump=None)
    try:
        _run(counter, [0.8] * 10 + [0.72] * 5 + [0.8] * 10)
    except Exception as exc:
        pytest.fail(f"on_jump=None 상태에서 예외 발생: {exc}")
    assert counter.count >= 1


def test_counter_multiple_callbacks_accumulate_count():  # N-08
    """두 번 점프 시 콜백이 받는 JumpEvent.count가 1, 2 순서로 단조 증가하는지 검증한다."""
    events: list[JumpEvent] = []
    cfg     = JumpCounterConfig(cooldown_frames=3)
    counter = _make_counter(cfg, on_jump=lambda e: events.append(e))
    ground, air = [0.80] * 10, [0.72] * 5
    _run(counter, ground + air + ground + air + ground)
    assert len(events) >= 2
    assert events[0].count == 1
    assert events[1].count == 2


# ── AnkleWristStrategy ────────────────────────────────────────────────────────

class WristProvider(LandmarkProvider):
    """발목과 손목 Y값을 독립적으로 지정할 수 있는 테스트 전용 Provider."""

    def __init__(self, ankle_y: float, wrist_y: float, visibility: float = 1.0) -> None:
        self._ankle_y = ankle_y
        self._wrist_y = wrist_y
        self._vis = visibility

    def get_landmark(self, results, name):
        y = self._wrist_y if "WRIST" in name else self._ankle_y
        return np.array([0.5, y, 0.0, self._vis], dtype=np.float32)

    def get_landmarks(self, results, names):
        return {n: self.get_landmark(results, n) for n in names}


def _run_wrist(counter, ankle_y_seq, wrist_y_seq, visibility=1.0):
    """발목/손목 Y 시퀀스를 함께 공급한다."""
    results = MagicMock()
    for ankle_y, wrist_y in zip(ankle_y_seq, wrist_y_seq):
        counter.update(WristProvider(ankle_y, wrist_y, visibility), results)


def test_ankle_wrist_strategy_name():
    """전략 이름이 'ankle_wrist'여야 한다."""
    from ropemetrics.strategies import AnkleWristStrategy
    cfg = JumpCounterConfig()
    assert AnkleWristStrategy(cfg).name == "ankle_wrist"


def test_ankle_wrist_required_landmarks():
    """발목 2개 + 손목 2개 총 4개 랜드마크를 요구해야 한다."""
    from ropemetrics.strategies import AnkleWristStrategy
    cfg = JumpCounterConfig()
    lms = AnkleWristStrategy(cfg).required_landmarks()
    assert "LEFT_ANKLE"  in lms
    assert "RIGHT_ANKLE" in lms
    assert "LEFT_WRIST"  in lms
    assert "RIGHT_WRIST" in lms


def test_ankle_wrist_counts_with_wrist_motion():
    """손목이 충분히 움직이는 상황에서 점프가 카운트되어야 한다."""
    from ropemetrics.strategies import AnkleWristStrategy
    cfg     = JumpCounterConfig(wrist_motion_threshold=0.015, wrist_history_size=5)
    counter = JumpCounter(strategy=AnkleWristStrategy(cfg), config=cfg)

    ground, air = 0.80, 0.70
    # 손목은 0.40~0.60 사이를 오가며 진폭 0.20 (임계값 초과)
    wrist_oscillation = [0.40, 0.50, 0.60, 0.50, 0.40] * 10

    ankle_seq = [ground] * 10 + [air] * 5 + [ground] * 10
    wrist_seq = wrist_oscillation[:len(ankle_seq)]

    _run_wrist(counter, ankle_seq, wrist_seq)
    assert counter.count >= 1


def test_ankle_wrist_no_count_without_wrist_motion():
    """손목이 정지 상태이면 점프 시퀀스에서도 카운트되지 않아야 한다."""
    from ropemetrics.strategies import AnkleWristStrategy
    cfg     = JumpCounterConfig(wrist_motion_threshold=0.015, wrist_history_size=5)
    counter = JumpCounter(strategy=AnkleWristStrategy(cfg), config=cfg)

    ground, air = 0.80, 0.70
    wrist_still = [0.50] * 30  # 손목 정지 — 진폭 0.0

    ankle_seq = [ground] * 10 + [air] * 5 + [ground] * 15
    wrist_seq = wrist_still[:len(ankle_seq)]

    _run_wrist(counter, ankle_seq, wrist_seq)
    assert counter.count == 0


def test_ankle_wrist_reset_clears_wrist_buffer():
    """reset() 후 손목 버퍼가 초기화되어야 한다."""
    from ropemetrics.strategies import AnkleWristStrategy
    cfg      = JumpCounterConfig(wrist_history_size=5)
    strategy = AnkleWristStrategy(cfg)

    # 버퍼에 데이터 채우기
    lm = lambda y: np.array([0.5, y, 0.0, 1.0], dtype=np.float32)
    for y in [0.40, 0.50, 0.60, 0.50, 0.40]:
        strategy.extract_signal({
            "LEFT_ANKLE": lm(0.80), "RIGHT_ANKLE": lm(0.80),
            "LEFT_WRIST": lm(y),    "RIGHT_WRIST": lm(y),
        })

    strategy.reset()
    assert len(strategy._left_wrist_buf)  == 0
    assert len(strategy._right_wrist_buf) == 0


def test_ankle_wrist_no_count_before_buffer_fills():
    """손목 버퍼가 채워지기 전에는 카운트되지 않아야 한다."""
    from ropemetrics.strategies import AnkleWristStrategy
    cfg     = JumpCounterConfig(wrist_motion_threshold=0.015, wrist_history_size=20)
    counter = JumpCounter(strategy=AnkleWristStrategy(cfg), config=cfg)

    # 손목 버퍼(20프레임) 미만으로 공급
    ankle_seq = [0.80] * 5 + [0.70] * 5 + [0.80] * 5
    wrist_seq = [0.40, 0.60] * 8  # 진폭 충분하나 버퍼 미충족
    wrist_seq = wrist_seq[:len(ankle_seq)]

    _run_wrist(counter, ankle_seq, wrist_seq)
    assert counter.count == 0
