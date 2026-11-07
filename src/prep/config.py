"""Pinned challenge and preprocessing rules.

Do not change thresholds between corpus builds. Bump SPEC_VERSION or
PREPROCESSING_VERSION if a rule must change, and write a new HDF5 file.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = "2.0"
PREPROCESSING_VERSION = "2.0"
SPEC_VERSION = "9th-abc-challenge-data-page"
QUATERNION_ORDER_STORED = "x,y,z,w"
QUATERNION_AXES_XROR = ("i", "j", "k", "1")
EULER_ORDER = "xyz"

TARGET_SAMPLE_RATE_HZ = 100.0
TARGET_SAMPLE_INTERVAL = 1.0 / TARGET_SAMPLE_RATE_HZ

# Session filtering — official 9th ABC Challenge data page. Not training rules.
TRIM_SECONDS = 5.0
MOTION_WINDOW_SECONDS = 5.0
MIN_SAMPLES_AFTER_TRIM = 20
MIN_VALID_SAMPLES_PER_WINDOW = 10
MIN_SAMPLING_RATE_HZ = 85.0
MAX_SAMPLING_RATE_HZ = 110.0
MIN_SWEEP_DEGREES = 17.0
MIN_VELOCITY_DEGREES_PER_SEC = 15.0
SWEEP_COMPARISON = "greater_than"
VELOCITY_COMPARISON = "greater_than"

# Training usability — evaluated only after clean + segment + 100 Hz resample.
CONTEXT_SAMPLES = 400
HORIZON_SAMPLES = 20
WINDOW_SAMPLES = CONTEXT_SAMPLES + HORIZON_SAMPLES
DEFAULT_WINDOW_STRIDE = 20

NEAR_ZERO_NORM = 1e-8
UNIT_NORM_ATOL = 1e-5
DUPLICATE_QUAT_ATOL = 1e-6
TIMESTAMP_EPS = 1e-9

# Intervals longer than this are counted as dropouts in reports, not as ticks.
DROPOUT_INTERVAL_SECONDS = 2.0 / MIN_SAMPLING_RATE_HZ  # ~23.5 ms at 85 Hz

# Large geodesic step between native samples is flagged, never auto-deleted.
DISCONTINUITY_DEGREES = 30.0

# SLERP fills only short missing ticks. 0.5 s is the challenge's example of
# a gap that must NOT be interpolated. 40 ms covers three missed ticks at
# 85 Hz (slowest accepted native rate) and four ticks at 100 Hz.
MAX_INTERPOLATE_GAP_SECONDS = 0.04
GAP_THRESHOLD_RATIONALE = (
    "Challenge: SLERP short dropouts; split longer ones; "
    "do not invent half a second of motion. "
    "Native band 85–110 Hz ⇒ tick 9.1–11.8 ms. "
    "0.04 s ≤ three missed ticks at 85 Hz. Gaps above this start a new segment."
)


@dataclass(frozen=True)
class FilterConfig:
    spec_version: str = SPEC_VERSION
    trim_seconds: float = TRIM_SECONDS
    motion_window_seconds: float = MOTION_WINDOW_SECONDS
    min_samples_after_trim: int = MIN_SAMPLES_AFTER_TRIM
    min_valid_samples_per_window: int = MIN_VALID_SAMPLES_PER_WINDOW
    min_sampling_rate_hz: float = MIN_SAMPLING_RATE_HZ
    max_sampling_rate_hz: float = MAX_SAMPLING_RATE_HZ
    min_sweep_degrees: float = MIN_SWEEP_DEGREES
    min_velocity_deg_s: float = MIN_VELOCITY_DEGREES_PER_SEC
    sweep_comparison: str = SWEEP_COMPARISON
    velocity_comparison: str = VELOCITY_COMPARISON
    euler_order: str = EULER_ORDER
    quaternion_axes_xror: tuple = QUATERNION_AXES_XROR


@dataclass(frozen=True)
class CleanConfig:
    preprocessing_version: str = PREPROCESSING_VERSION
    near_zero_norm: float = NEAR_ZERO_NORM
    unit_norm_atol: float = UNIT_NORM_ATOL
    duplicate_quat_atol: float = DUPLICATE_QUAT_ATOL
    timestamp_eps: float = TIMESTAMP_EPS
    dropout_interval_seconds: float = DROPOUT_INTERVAL_SECONDS
    discontinuity_degrees: float = DISCONTINUITY_DEGREES
    max_interpolate_gap_seconds: float = MAX_INTERPOLATE_GAP_SECONDS
    gap_threshold_rationale: str = GAP_THRESHOLD_RATIONALE
    target_sample_rate_hz: float = TARGET_SAMPLE_RATE_HZ
    do_not_sort_timestamps: bool = True
    do_not_extrapolate: bool = True
    do_not_delete_fast_motion: bool = True


@dataclass(frozen=True)
class WindowConfig:
    context_samples: int = CONTEXT_SAMPLES
    horizon_samples: int = HORIZON_SAMPLES
    stride: int = DEFAULT_WINDOW_STRIDE

    @property
    def window_samples(self) -> int:
        return self.context_samples + self.horizon_samples


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    report_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    download_state: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "download_state.json"
    )
    session_log: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "session_filter.csv"
    )
    reject_log: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "session_rejects.csv"
    )
    intermediate_h5: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "head_pose_clean.h5"
    )
    resampled_h5: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "head_pose_v2.h5"
    )
    legacy_h5: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "head_pose.h5"
    )


@dataclass(frozen=True)
class DownloadConfig:
    repo_id: str = "cschell/boxrr-23"
    repo_type: str = "dataset"
    batch_size: int = 50
    max_retries: int = 3
    retry_sleep_seconds: float = 2.0
    min_free_gb: float = 10.0


def config_manifest(filter_cfg=None, clean_cfg=None, window_cfg=None) -> dict:
    filter_cfg = filter_cfg or FilterConfig()
    clean_cfg = clean_cfg or CleanConfig()
    window_cfg = window_cfg or WindowConfig()
    return {
        "schema_version": SCHEMA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "spec_version": SPEC_VERSION,
        "quaternion_order": QUATERNION_ORDER_STORED,
        "filter": asdict(filter_cfg),
        "clean": asdict(clean_cfg),
        "window": asdict(window_cfg),
    }
