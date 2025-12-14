import numpy as np
from translator.capture.diff import detect_change


def test_detect_change_without_history() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert detect_change(None, img)


def test_detect_change_with_same_image() -> None:
    before = np.zeros((10, 10, 3), dtype=np.uint8)
    after = np.zeros((10, 10, 3), dtype=np.uint8)
    assert not detect_change(before, after)


def test_detect_change_with_different_image() -> None:
    before = np.zeros((10, 10, 3), dtype=np.uint8)
    after = np.ones((10, 10, 3), dtype=np.uint8) * 255 # Full white
    
    # Difference should be max (1.0), so > threshold
    assert detect_change(before, after, threshold=0.1)


def test_detect_change_shape_mismatch() -> None:
    before = np.zeros((10, 10, 3), dtype=np.uint8)
    after = np.zeros((20, 20, 3), dtype=np.uint8)
    
    # Should return True if shapes differ
    assert detect_change(before, after)
