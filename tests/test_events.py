"""Unit tests for the event-window construction and normalisation rules (CPU only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hive_diff.data.events import build_event_windows, normalize_per_channel_quantile, normalize_per_frame_maxabs  # noqa: E402


def test_windows_4bin_centered():
    T, B = 3, 4
    stack = np.arange(B * (T + 1), dtype=np.float32)[:, None, None] * np.ones((1, 2, 2), np.float32)
    out = build_event_windows(stack, T, B, (1, 3, 5, 7))
    for k in range(T):
        c = 4 * k + 3
        assert out[k, 0, 0, 0] == c
        assert out[k, 1, 0, 0] == sum(range(c - 1, c + 2))
        assert out[k, 3, 0, 0] == sum(range(c - 3, c + 4))


def test_windows_12bin_training():
    T, B = 2, 12
    stack = np.arange(B * (T + 1), dtype=np.float32)[:, None, None] * np.ones((1, 1, 1), np.float32)
    out = build_event_windows(stack, T, B, (0, 4, 8, 12))
    k, end = 1, 12 * 1 + 11
    assert out[k, 0, 0, 0] == 0
    assert out[k, 1, 0, 0] == sum(range(end - 2, end + 2))
    assert out[k, 3, 0, 0] == sum(range(end - 6, end + 6))


def test_normalisers():
    ev = np.zeros((2, 4, 3, 3), np.float32); ev[0, 1] = 4.0; ev[1, 2] = -2.0
    m = normalize_per_frame_maxabs(ev)
    assert abs(m[0, 1, 0, 0] - 4.0 / (4.0 + 1e-8)) < 1e-6 and m[1, 2, 0, 0] < 0
    q = normalize_per_channel_quantile(ev, 0.995)
    assert q.max() <= 1.0 and q.min() >= -1.0 and q[0, 1, 0, 0] == 1.0 and q[1, 2, 0, 0] == -1.0


if __name__ == "__main__":
    test_windows_4bin_centered(); test_windows_12bin_training(); test_normalisers(); print("events tests OK")
