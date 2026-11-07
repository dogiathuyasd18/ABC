"""Validate an HDF5 corpus and print filtering vs usability counts."""

import argparse
from pathlib import Path

from src.prep.config import PROJECT_ROOT, WINDOW_SAMPLES
from src.prep.hdf5 import list_users, session_summaries
from src.prep.windows import count_windows, session_is_trainable
from src.data.splits import split_users


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=PROJECT_ROOT / "data" / "processed" / "head_pose_v2.h5")
    parser.add_argument("--legacy-h5", type=Path, default=PROJECT_ROOT / "data" / "processed" / "head_pose.h5")
    args = parser.parse_args()
    path = args.h5 if args.h5.exists() else args.legacy_h5
    users = list_users(path)
    train_users, val_users, test_users = split_users(users) if users else ([], [], [])
    rows = session_summaries(path)
    trainable = sum(1 for row in rows if session_is_trainable(row["segment_lengths"]))
    windows = sum(count_windows(row["segment_lengths"]) for row in rows)
    print(f"file: {path}")
    print(f"users: {len(users)}")
    print(f"train/val/test users: {len(train_users)}/{len(val_users)}/{len(test_users)}")
    print(f"sessions: {len(rows)}")
    print(f"sessions with a {WINDOW_SAMPLES}-sample window: {trainable}")
    print(f"usable windows (stride 20): {windows}")
    print(f"resampled samples: {sum(row['n_resampled'] for row in rows)}")
    print(f"interpolated samples: {sum(row['n_interpolated'] for row in rows)}")


if __name__ == "__main__":
    main()
