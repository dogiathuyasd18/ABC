from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr

from src.data.dataset import load_segments


H5_PATH = Path("data/processed/head_pose_v2.h5")
PRED_PATH = Path("outputs/gru_val_seed42_predictions.npz")
OUT_DIR = Path("outputs")
DT = 0.01
CONTEXT = 100


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
    rel /= np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-12)

    # Choose the shortest quaternion representation.
    rel = np.where(rel[..., 3:4] < 0, -rel, rel)
    return Rotation.from_quat(rel).as_rotvec()


def geodesic_error_deg(q1, q2):
    rel = quat_mul(quat_inv(q1), q2)
    rel /= np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-12)
    angle = 2 * np.arctan2(
        np.linalg.norm(rel[..., :3], axis=-1),
        np.abs(rel[..., 3])
    )
    return np.degrees(angle)


def acceleration_metrics(q):
    # q: [100, 4], xyzw
    # Relative rotation vectors are expressed in the preceding body frame.
    rotvec = relative_rotvec(q[:-1], q[1:])
    omega_body = rotvec / DT

    # Express angular velocity in a common world frame.
    omega_world = Rotation.from_quat(q[:-1]).apply(omega_body)

    # Finite difference of world-frame angular velocity.
    alpha = np.diff(omega_world, axis=0) / DT
    alpha_mag = np.linalg.norm(alpha, axis=-1)

    speed = np.linalg.norm(omega_world, axis=-1)

    return {
        "mean_angular_speed_rad_s": float(np.mean(speed)),
        "rms_angular_accel_rad_s2": float(np.sqrt(np.mean(alpha_mag**2))),
        "p95_angular_accel_rad_s2": float(np.percentile(alpha_mag, 95)),
        "max_angular_accel_rad_s2": float(np.max(alpha_mag)),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)

    data = np.load(PRED_PATH, allow_pickle=True)

    pred = data["pred_quat"]
    true = data["true_quat"]

    errors = geodesic_error_deg(pred[:, 19], true[:, 19])

    users = data["user_id"].astype(str)
    sessions = data["session_id"].astype(str)
    sequences = data["sequence_name"].astype(str)
    starts = data["start_index"].astype(int)

    unique_users = sorted(set(users))
    segments = load_segments(H5_PATH, unique_users)

    segment_map = {
        (str(u), str(s), str(name)): q
        for u, s, name, q in segments
    }

    rows = []
    missing = 0

    for i in range(len(errors)):
        key = (users[i], sessions[i], sequences[i])
        quats = segment_map.get(key)

        if quats is None:
            missing += 1
            continue

        start = starts[i]
        context_q = quats[start:start + CONTEXT]

        if len(context_q) != CONTEXT:
            missing += 1
            continue

        metrics = acceleration_metrics(context_q)

        rows.append({
            "user_id": users[i],
            "session_id": sessions[i],
            "sequence_name": sequences[i],
            "start_index": starts[i],
            "error_deg": float(errors[i]),
            **metrics,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No windows matched. Check HDF5 path and segment metadata."
        )

    out_csv = OUT_DIR / "gru_acceleration_per_window.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== MATCHING ===")
    print(f"Prediction windows: {len(errors)}")
    print(f"Matched windows: {len(df)}")
    print(f"Unmatched windows: {missing}")

    print("\n=== GRU STEP-20 ERROR ===")
    print(f"Mean:   {df.error_deg.mean():.4f} deg")
    print(f"Median: {df.error_deg.median():.4f} deg")
    print(f"P95:    {df.error_deg.quantile(0.95):.4f} deg")

    print("\n=== CORRELATION WITH ANGULAR ACCELERATION ===")

    for metric in [
        "rms_angular_accel_rad_s2",
        "p95_angular_accel_rad_s2",
        "max_angular_accel_rad_s2",
        "mean_angular_speed_rad_s",
    ]:
        x = df[metric].to_numpy()
        y = df["error_deg"].to_numpy()

        if np.std(x) < 1e-12:
            continue

        r, p = pearsonr(x, y)
        rho, sp = spearmanr(x, y)

        print(f"\n{metric}")
        print(f"  Pearson r:  {r:.4f} (p={p:.3g})")
        print(f"  Spearman:   {rho:.4f} (p={sp:.3g})")

    print("\n=== ERROR BY USER / SESSION ===")
    grouped = (
        df.groupby(["user_id", "session_id"])
        .agg(
            windows=("error_deg", "size"),
            mean_error_deg=("error_deg", "mean"),
            median_error_deg=("error_deg", "median"),
            mean_accel=("rms_angular_accel_rad_s2", "mean"),
        )
        .sort_values("mean_error_deg", ascending=False)
    )

    print(grouped.head(20).to_string())

    print(f"\nSaved per-window results: {out_csv}")


if __name__ == "__main__":
    main()
