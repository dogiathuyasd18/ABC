import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr

from src.data.dataset import load_segments


H5_PATH = Path("data/processed/head_pose_v2.h5")
PAIRED_PATH = Path("outputs/gru_vs_kinematics_paired.csv")
OUT_DIR = Path("outputs")

DT = 0.01
THRESHOLD = 5.0
HORIZON = 20
CONTEXT_COMMON = 100
CONTEXT_LONG = 400


def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)

    return np.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], axis=-1)


def quat_inv(q):
    result = q.copy()
    result[..., :3] *= -1
    return result


def relative_rotvec(q0, q1):
    rel = quat_mul(quat_inv(q0), q1)
    rel /= np.maximum(
        np.linalg.norm(rel, axis=-1, keepdims=True), 1e-12
    )
    rel = np.where(rel[..., 3:4] < 0, -rel, rel)
    return Rotation.from_quat(rel).as_rotvec()


def acceleration_metrics(q):
    # q: [context, 4], quaternion order xyzw
    rotvec = relative_rotvec(q[:-1], q[1:])
    omega_body = rotvec / DT

    # Transform angular velocity into a common world frame.
    omega_world = Rotation.from_quat(q[:-1]).apply(omega_body)

    # Finite difference in world coordinates.
    alpha = np.diff(omega_world, axis=0) / DT

    alpha_mag = np.linalg.norm(alpha, axis=-1)
    speed = np.linalg.norm(omega_world, axis=-1)

    return {
        "mean_angular_speed": float(np.mean(speed)),
        "rms_angular_accel": float(
            np.sqrt(np.mean(alpha_mag ** 2))
        ),
        "p95_angular_accel": float(
            np.percentile(alpha_mag, 95)
        ),
        "max_angular_accel": float(np.max(alpha_mag)),
    }


def add_correlations(df, metrics):
    print("\n=== CORRELATION WITH MODEL ERRORS ===")

    targets = [
        ("GRU error", "gru_error"),
        ("Kinematics error", "kin_error"),
        ("GRU - Kinematics error (positive = Kin advantage)",
         "error_difference"),
    ]

    for metric in metrics:
        x = df[metric].to_numpy()

        print(f"\n{metric}")

        for label, column in targets:
            y = df[column].to_numpy()

            r, p = pearsonr(x, y)
            rho, sp = spearmanr(x, y)

            print(f"  {label}")
            print(f"    Pearson r:  {r:.4f} (p={p:.3g})")
            print(f"    Spearman:   {rho:.4f} (p={sp:.3g})")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(PAIRED_PATH)

    users = sorted(df["user_id"].astype(str).unique())
    segments = load_segments(H5_PATH, users)

    segment_map = {
        (str(u), str(s), str(name).zfill(4)): q
        for u, s, name, q in segments
    }

    rows = []
    missing = 0

    for row in df.itertuples(index=False):
        key = (
            str(row.user_id),
            str(row.session_id),
            str(row.sequence_name).zfill(4),
        )

        quats = segment_map.get(key)

        if quats is None:
            missing += 1
            continue

        target = int(row.target_index)

        # Target index is the final sample of the 20-step horizon.
        # Common 100-sample history immediately preceding the horizon.
        start100 = target - HORIZON - CONTEXT_COMMON + 1
        end = target - HORIZON + 1

        q100 = quats[start100:end]

        if len(q100) != CONTEXT_COMMON:
            missing += 1
            continue

        metrics100 = acceleration_metrics(q100)

        # Also calculate metrics over the longer 400-sample history.
        start400 = target - HORIZON - CONTEXT_LONG + 1
        q400 = quats[start400:end]

        metrics400 = {}
        if len(q400) == CONTEXT_LONG:
            metrics400 = {
                f"{name}_400": value
                for name, value in acceleration_metrics(q400).items()
            }

        rows.append({
            "user_id": key[0],
            "session_id": key[1],
            "sequence_name": key[2],
            "target_index": target,
            "gru_error": float(row.gru_error),
            "kin_error": float(row.kin_error),
            "error_difference": (
                float(row.gru_error) - float(row.kin_error)
            ),
            **metrics100,
            **metrics400,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError("No paired windows matched HDF5 segments.")

    # Four groups using the 5-degree threshold.
    gru_good = result["gru_error"] <= THRESHOLD
    kin_good = result["kin_error"] <= THRESHOLD

    result["group"] = np.select(
        [
            gru_good & kin_good,
            gru_good & ~kin_good,
            ~gru_good & kin_good,
        ],
        [
            "both_good",
            "gru_only_good",
            "kin_only_good",
        ],
        default="both_bad",
    )

    print("\n=== MATCHING ===")
    print("Paired CSV rows:", len(df))
    print("Matched with HDF5:", len(result))
    print("Missing:", missing)

    print("\n=== FOUR GROUPS ===")
    group_order = [
        "both_good",
        "gru_only_good",
        "kin_only_good",
        "both_bad",
    ]

    for group in group_order:
        subset = result[result["group"] == group]
        print(f"\n{group}: {len(subset)} ({len(subset)/len(result)*100:.2f}%)")

        if len(subset) == 0:
            continue

        print(
            subset[
                [
                    "gru_error",
                    "kin_error",
                    "rms_angular_accel",
                    "p95_angular_accel",
                    "max_angular_accel",
                    "mean_angular_speed",
                ]
            ].mean().round(4).to_string()
        )

    metrics = [
        "rms_angular_accel",
        "p95_angular_accel",
        "max_angular_accel",
        "mean_angular_speed",
    ]

    add_correlations(result, metrics)

    print("\n=== ACCELERATION QUANTILE GROUPS ===")

    result["accel_level"] = pd.qcut(
        result["rms_angular_accel"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )

    accel_summary = (
        result.groupby("accel_level", observed=True)
        .agg(
            windows=("gru_error", "size"),
            mean_accel=("rms_angular_accel", "mean"),
            gru_mean_error=("gru_error", "mean"),
            kin_mean_error=("kin_error", "mean"),
            mean_advantage=("error_difference", "mean"),
            gru_only_good=("group", lambda x: (x == "gru_only_good").sum()),
            kin_only_good=("group", lambda x: (x == "kin_only_good").sum()),
            both_bad=("group", lambda x: (x == "both_bad").sum()),
        )
    )

    print(accel_summary.round(4).to_string())

    # Save detailed per-target data.
    out_csv = OUT_DIR / "paired_acceleration_analysis.csv"
    result.to_csv(out_csv, index=False)

    # Save group summary.
    group_summary = (
        result.groupby("group", observed=True)
        .agg(
            windows=("gru_error", "size"),
            gru_mean_error=("gru_error", "mean"),
            kin_mean_error=("kin_error", "mean"),
            rms_accel=("rms_angular_accel", "mean"),
            p95_accel=("p95_angular_accel", "mean"),
            max_accel=("max_angular_accel", "mean"),
            mean_speed=("mean_angular_speed", "mean"),
        )
    )

    group_path = OUT_DIR / "paired_acceleration_group_summary.csv"
    group_summary.to_csv(group_path)

    print("\nSaved:", out_csv)
    print("Saved:", group_path)


if __name__ == "__main__":
    main()
