
"""Training window index.

Windows are never stored in HDF5.
A usable window contains exactly context + horizon samples
from one continuous resampled segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from src.prep.config import (
    CONTEXT_SAMPLES,
    HORIZON_SAMPLES,
    WindowConfig,
)


@dataclass(frozen=True)
class WindowIndex:
    user_id: str
    session_id: str
    segment_id: str
    start: int
    context: int
    horizon: int

    @property
    def end(self) -> int:
        return self.start + self.context + self.horizon

    @property
    def window_samples(self) -> int:
        return self.context + self.horizon

    @property
    def target_start(self) -> int:
        return self.start + self.context

    @property
    def context_range(self) -> tuple[int, int]:
        return self.start, self.target_start

    @property
    def target_range(self) -> tuple[int, int]:
        return self.target_start, self.end


def _validate_integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")

    value = int(value)

    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")

    return value


def _validate_config(cfg: WindowConfig) -> None:
    context = _validate_integer(
        "context_samples",
        cfg.context_samples,
        minimum=1,
    )
    horizon = _validate_integer(
        "horizon_samples",
        cfg.horizon_samples,
        minimum=1,
    )
    stride = _validate_integer(
        "stride",
        cfg.stride,
        minimum=1,
    )
    window = _validate_integer(
        "window_samples",
        cfg.window_samples,
        minimum=1,
    )

    if window != context + horizon:
        raise ValueError(
            "window_samples must equal context_samples + horizon_samples"
        )


def window_starts(
    n_samples: int,
    window: int,
    stride: int,
) -> list[int]:
    n_samples = _validate_integer("n_samples", n_samples)
    window = _validate_integer("window", window, minimum=1)
    stride = _validate_integer("stride", stride, minimum=1)

    if n_samples < window:
        return []

    return list(range(0, n_samples - window + 1, stride))


def index_segment(
    user_id: str,
    session_id: str,
    segment_id: str,
    n_samples: int,
    cfg: WindowConfig | None = None,
) -> list[WindowIndex]:
    cfg = cfg or WindowConfig()
    _validate_config(cfg)

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")

    if segment_id is None or not str(segment_id).strip():
        raise ValueError("segment_id must be non-empty")

    starts = window_starts(
        n_samples=n_samples,
        window=cfg.window_samples,
        stride=cfg.stride,
    )

    return [
        WindowIndex(
            user_id=user_id,
            session_id=session_id,
            segment_id=str(segment_id),
            start=start,
            context=cfg.context_samples,
            horizon=cfg.horizon_samples,
        )
        for start in starts
    ]


def session_is_trainable(
    segment_lengths: list[int],
    cfg: WindowConfig | None = None,
) -> bool:
    cfg = cfg or WindowConfig()
    _validate_config(cfg)

    return any(
        window_starts(
            length,
            cfg.window_samples,
            cfg.stride,
        )
        for length in segment_lengths
    )


def count_windows(
    segment_lengths: list[int],
    cfg: WindowConfig | None = None,
) -> int:
    cfg = cfg or WindowConfig()
    _validate_config(cfg)

    return sum(
        len(
            window_starts(
                length,
                cfg.window_samples,
                cfg.stride,
            )
        )
        for length in segment_lengths
    )
