from translator.capture.diff import detect_change


def test_detect_change_without_history() -> None:
    assert detect_change(None, b"abc")


def test_detect_change_with_same_bytes() -> None:
    before = b"abc"
    after = b"abc"
    assert not detect_change(before, after, threshold=0.5)
