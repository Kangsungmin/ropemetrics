import numpy as np
import pytest

from ropemetrics.adapters.movenet import LANDMARK_INDEX, MoveNetLandmarkProvider


# ── 테스트용 헬퍼 ─────────────────────────────────────────────────────────────

def make_movenet_results(person_index: int = 0) -> np.ndarray:
    """shape [1, 1, 17, 3] 형태의 더미 MoveNet 결과를 반환한다."""
    data = np.zeros((1, 1, 17, 3), dtype=np.float32)
    data[0][0][15] = [0.80, 0.44, 0.95]  # LEFT_ANKLE: [y, x, confidence]
    return data


# ── MoveNetLandmarkProvider 테스트 ────────────────────────────────────────────

def test_movenet_returns_landmark_array():
    """정상 results에서 shape (4,) float32 배열을 반환해야 한다."""
    provider = MoveNetLandmarkProvider()
    results  = make_movenet_results()

    lm = provider.get_landmark(results, "LEFT_ANKLE")

    assert lm is not None
    assert isinstance(lm, np.ndarray)
    assert lm.shape == (4,)
    assert lm.dtype == np.float32


def test_movenet_xy_order_is_correct():
    """MoveNet은 [y, x, conf] 순이므로 반환 배열의 [0]이 x, [1]이 y여야 한다."""
    provider = MoveNetLandmarkProvider()
    results  = make_movenet_results()
    # LEFT_ANKLE: data[0][0][15] = [y=0.80, x=0.44, confidence=0.95]

    lm = provider.get_landmark(results, "LEFT_ANKLE")

    assert lm is not None
    assert pytest.approx(lm[0], abs=1e-5) == 0.44   # x
    assert pytest.approx(lm[1], abs=1e-5) == 0.80   # y
    assert pytest.approx(lm[3], abs=1e-5) == 0.95   # confidence


def test_movenet_unknown_landmark_returns_none():
    """매핑에 없는 랜드마크 이름은 None을 반환해야 한다."""
    provider = MoveNetLandmarkProvider()
    results  = make_movenet_results()

    lm = provider.get_landmark(results, "UNKNOWN_JOINT")

    assert lm is None


def test_movenet_none_results_returns_none():
    """results=None이면 None을 반환해야 한다."""
    provider = MoveNetLandmarkProvider()

    lm = provider.get_landmark(None, "LEFT_ANKLE")

    assert lm is None


def test_movenet_all_required_landmarks_mapped():
    """AnkleStrategy/HipShoulderStrategy 필수 랜드마크 6개가 LANDMARK_INDEX에 존재해야 한다."""
    required = [
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
    ]

    for name in required:
        assert name in LANDMARK_INDEX, f"{name}이 LANDMARK_INDEX에 없습니다."
