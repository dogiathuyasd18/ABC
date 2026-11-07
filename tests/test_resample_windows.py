import numpy as np

from src.prep.config import WINDOW_SAMPLES, CleanConfig, WindowConfig
from src.prep.gaps import split_on_gaps
from src.prep.hdf5 import init_file, iter_segments, save_session
from src.prep.resampling import ResampledSegment, resample_session
from src.prep.windows import count_windows, index_segment, window_starts
from src.data.dataset import HeadPoseWindowDataset


def _uniform(n, dt=0.01, yaw_step=0.002):
    t = np.arange(n) * dt
    angles = yaw_step * np.arange(n)
    q = np.stack([np.zeros(n), np.sin(angles / 2), np.zeros(n), np.cos(angles / 2)], axis=1)
    return t, q


def test_gap_split_does_not_interpolate_long_dropout():
    t1, q1 = _uniform(50)
    t2, q2 = _uniform(50)
    t2 = t2 + t1[-1] + 0.5
    t = np.concatenate([t1, t2])
    q = np.concatenate([q1, q2])
    cfg = CleanConfig()
    assert cfg.max_interpolate_gap_seconds < 0.5
    pieces = split_on_gaps(t, q, cfg)
    assert len(pieces) == 2
    result = resample_session(t, q, cfg)
    assert result.n_segments == 2
    assert result.segments[0].timestamps[-1] < t2[0]


def test_resample_does_not_extrapolate():
    t, q = _uniform(20, dt=0.011)
    result = resample_session(t, q)
    grid = result.segments[0].timestamps
    assert grid[0] >= t[0] - 1e-12
    assert grid[-1] <= t[-1] + 1e-9


def test_windows_never_cross_gap():
    lengths = [300, 300]
    cfg = WindowConfig()
    assert cfg.window_samples == 420
    assert count_windows(lengths, cfg) == 0
    left = index_segment("u", "s", "0", 300, cfg)
    right = index_segment("u", "s", "1", 300, cfg)
    assert left == []
    assert right == []
    assert count_windows([500, 500], cfg) == len(window_starts(500, 420, 20)) * 2


def test_window_length_is_exactly_420():
    starts = window_starts(1000, 420, 20)
    assert starts[0] == 0
    assert starts[1] == 20
    assert 1000 - starts[-1] >= 420


def test_hdf5_roundtrip_and_dataset(tmp_path):
    path = tmp_path / "t.h5"
    init_file(path, resampled=True)
    t, q = _uniform(500)
    segment = ResampledSegment(t, q.astype(np.float32), 500, 500, 0, float(t[-1] - t[0]))
    save_session(path, "user-a", "sess-1", "m.xror", "a.tar", [segment], {"resampled": True, "sampling_rate_hz": 100.0})
    rows = list(iter_segments(path))
    assert len(rows) == 1
    assert rows[0]["quaternions"].shape == (500, 4)
    data = HeadPoseWindowDataset(path, ["user-a"], context=400, horizon=20, stride=20)
    assert len(data) == len(window_starts(500, 420, 20))
    window = data[0]
    assert tuple(window.shape) == (420, 4)
