"""Training window index. Windows are never stored in HDF5.

A usable window is exactly context + horizon samples from one continuous
resampled segment. Windows never cross sessions or gap cuts.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.prep.config import CONTEXT_SAMPLES, HORIZON_SAMPLES, WindowConfig


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


def window_starts(n_samples: int, window: int, stride: int) -> list[int]:
    if n_samples < window or window <= 0 or stride <= 0:
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
    starts = window_starts(n_samples, cfg.window_samples, cfg.stride)
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


def session_is_trainable(segment_lengths: list[int], cfg: WindowConfig | None = None) -> bool:
    cfg = cfg or WindowConfig()
    return any(length >= cfg.window_samples for length in segment_lengths)


def count_windows(segment_lengths: list[int], cfg: WindowConfig | None = None) -> int:
    cfg = cfg or WindowConfig()
    return sum(len(window_starts(length, cfg.window_samples, cfg.stride)) for length in segment_lengths)
