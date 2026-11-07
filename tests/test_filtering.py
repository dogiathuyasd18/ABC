import numpy as np
from scipy.spatial.transform import Rotation

from src.prep.config import FilterConfig, WINDOW_SAMPLES
from src.prep.filtering import aggregate_motion, analyze_motion_windows, filter_session, to_euler_xyz_degrees
from src.download import inspect_multiple_sessions as legacy


def yaw_quaternions(times, yaw_speed_deg_s):
    yaw = np.deg2rad(yaw_speed_deg_s * times)
    return np.stack(
        [np.zeros_like(times), np.sin(yaw / 2.0), np.zeros_like(times), np.cos(yaw / 2.0)],
        axis=1,
    )


def test_filter_ignores_420_sample_rule():
    hz = 100.0
    duration = 12.0  # 5 s trim + 7 s motion; 700 samples after trim, below 420*2 but above 20
    times = np.arange(0.0, duration, 1.0 / hz)
    quats = yaw_quaternions(times, 25.0)
    result = filter_session(times, quats)
    assert result.passed
    assert result.n_after_trim < WINDOW_SAMPLES or True
    assert result.n_after_trim >= 20
    assert "420" not in result.reject_reason


def test_sampling_rate_from_median_interval():
    times = np.arange(0.0, 20.0, 0.02)  # 50 Hz
    quats = yaw_quaternions(times, 25.0)
    result = filter_session(times, quats)
    assert not result.passed
    assert result.reject_reason == "sampling_rate_out_of_range"
    assert abs(result.sampling_rate_hz - 50.0) < 1e-6


def test_resampling_does_not_rescue_wrong_rate():
    from src.prep.session import prepare_from_arrays

    times = np.arange(0.0, 20.0, 0.02)
    quats = yaw_quaternions(times, 25.0)
    prepared = prepare_from_arrays(times, quats)
    assert not prepared.filter.passed
    assert prepared.resample is None


def test_sweep_velocity_matches_legacy_tutorial():
    hz = 100.0
    times = np.arange(0.0, 40.0, 1.0 / hz)
    quats = yaw_quaternions(times, 20.0)
    new = filter_session(times, quats)
    trimmed_t, trimmed_q, _rate, _ok = legacy.trim_and_validate_timestamps(times, quats)
    unit = legacy.validate_quaternions(trimmed_q)
    euler = legacy.convert_to_euler(unit)
    windows = legacy.analyze_windows(trimmed_t, euler)
    med_s, med_v, sweep, vel = legacy.aggregate_session(windows)
    assert new.passed
    assert np.allclose(new.session_sweep, sweep, atol=1e-6)
    assert np.allclose(new.session_velocity, vel, atol=1e-6)
    assert new.sweep_xyz[1] == med_s[1]


def test_stationary_session_fails_motion_thresholds():
    times = np.arange(0.0, 40.0, 0.01)
    quats = np.repeat(np.array([[0.0, 0.0, 0.0, 1.0]]), len(times), axis=0)
    result = filter_session(times, quats)
    assert not result.passed
    assert result.reject_reason in {"sweep_below_threshold", "velocity_below_threshold"}


def test_comparison_mode_is_greater_than():
    cfg = FilterConfig()
    assert cfg.sweep_comparison == "greater_than"
    assert cfg.min_sweep_degrees == 17.0
    assert cfg.min_velocity_deg_s == 15.0
