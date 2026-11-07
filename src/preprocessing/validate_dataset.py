import argparse
from pathlib import Path

import h5py
import numpy as np

from src.data.dataset import MIN_WINDOW, list_users, resample_session, split_users


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=PROJECT_ROOT / "data" / "processed" / "head_pose.h5")
    args = parser.parse_args()

    users = list_users(args.h5)
    train_users, val_users = split_users(users)
    sessions = 0
    samples = 0
    short = 0
    rates = []
    resampled = 0

    with h5py.File(args.h5, "r") as h5_file:
        print("quaternion_order:", h5_file.attrs.get("quaternion_order"))
        for user_id in users:
            for session_id in h5_file["users"][user_id]["sessions"]:
                group = h5_file["users"][user_id]["sessions"][session_id]
                count = int(group["quaternions"].shape[0])
                sessions += 1
                samples += count
                if count < MIN_WINDOW:
                    short += 1
                rates.append(float(group.attrs["sampling_rate"]))
                pieces = resample_session(group["timestamps"][:], group["quaternions"][:])
                resampled += sum(len(piece) for piece in pieces)

    print(f"Users: {len(users)}")
    print(f"Train users: {len(train_users)} | Val users: {len(val_users)}")
    print(f"Val user ids: {val_users}")
    print(f"Sessions: {sessions}")
    print(f"Raw samples: {samples}")
    print(f"Sessions shorter than {MIN_WINDOW} samples: {short}")
    print(f"Sampling rate Hz min/median/max: {min(rates):.2f} / {np.median(rates):.2f} / {max(rates):.2f}")
    print(f"Samples after 100 Hz SLERP: {resampled}")


if __name__ == "__main__":
    main()
