import pytest

from ave_emotion.windows import window_bounds


def test_window_bounds():
    assert window_bounds(6, 3, 1.5) == [(0, 3), (1.5, 4.5), (3, 6), (4.5, 6)]
    assert window_bounds(3.2, 3, 1.5) == [(0, 3), (1.5, 3.2)]
    assert window_bounds(0.3, 3, 1.5) == [(0, 0.3)]
    assert window_bounds(0, 3, 1.5) == []
    assert window_bounds(6, 2, 2) == [(0, 2), (2, 4), (4, 6)]
