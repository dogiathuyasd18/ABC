
from pathlib import Path
import csv

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr


# Dataset được resample ở 100 Hz
DT = 0.01

INPUT = Path("outputs/gru_predictions.npz")
OUTPUT = Path("outputs/acceleration_analysis.csv")


def normalize_quaternions(q):
    """Normalize quaternion, convention xyzw."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-12)


def angular_errors_deg(pred, true):
    """Geodesic angular error for every prediction step."""
    pred = normalize_quaternions(pred)
    true = normalize_quaternions(true)

    dot = np.sum(pred * true, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    return np.degrees(2.0 * np.arccos(dot))


def angular_acceleration_deg_s2(quaternions, dt=DT):
    """
    Estimate angular acceleration from a quaternion sequence.

    Input:
        quaternions: [N, T, 4], quaternion order xyzw

    Output:
        angular acceleration magnitude: [N, T-2], in deg/s^2
    """
    q = normalize_quaternions(quaternions)

    n, t, _ = q.shape
    if t < 3:
        raise ValueError("Cần ít nhất 3 quaternion để tính gia tốc góc.")

    # Rotation at each time step
    rotations = Rotation.from_quat(q.reshape(-1, 4))
    rotations = Rotation.concatenate(
        [rotations[i * t:(i + 1) * t] for i in range(n)]
    ) if False else rotations

    # Reshape quaternion array into per-window rotations
    angular_acc = np.zeros((n, t - 2), dtype=np.float64)

    for i in range(n):
        r = Rotation.from_quat(q[i])

        # Relative rotation between consecutive samples
        relative = r[:-1].inv() * r[1:]

        # Express angular velocity in a common/world coordinate frame
        omega_local = relative.as_rotvec()
        omega_world = r[:-1].apply(omega_local)

        # Convert rad/s to deg/s
        omega_world = np.degrees(omega_world / dt)

        # Angular acceleration = change in angular velocity / time
        alpha = np.diff(omega_world, axis=0) / dt

        # Magnitude of angular acceleration
        angular_acc[i] = np.linalg.norm(alpha, axis=-1)

    return angular_acc


def correlation(x, y):
    """Return Pearson and Spearman correlation, handling invalid data."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, len(x)

    pearson = pearsonr(x, y).statistic
    spearman = spearmanr(x, y).statistic

    return float(pearson), float(spearman), len(x)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {INPUT}. "
            "Hãy xuất predictions từ evaluate.py trước."
        )

    data = np.load(INPUT, allow_pickle=True)

    required = {"pred_quat", "true_quat"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"File NPZ thiếu các keys: {missing}")

    pred = np.asarray(data["pred_quat"], dtype=np.float64)
    true = np.asarray(data["true_quat"], dtype=np.float64)

    if pred.shape != true.shape or pred.ndim != 3 or pred.shape[-1] != 4:
        raise ValueError(
            f"Shape không hợp lệ: pred={pred.shape}, true={true.shape}. "
            "Mong đợi [N, horizon, 4]."
        )

    n, horizon, _ = true.shape

    # 1. Prediction error at every future step
    errors = angular_errors_deg(pred, true)

    # Error at the 20th future step
    error_step20 = errors[:, -1]

    # 2. Angular acceleration along the ground-truth future motion
    acceleration = angular_acceleration_deg_s2(true)

    # Summarize acceleration for each prediction window
    mean_acc = np.mean(acceleration, axis=1)
    max_acc = np.max(acceleration, axis=1)
    p95_acc = np.percentile(acceleration, 95, axis=1)

    # Also compare average prediction error over the horizon
    mean_error = np.mean(errors, axis=1)

    # 3. Correlation analysis
    print(f"\nNumber of windows: {n}")
    print(f"Prediction horizon: {horizon} samples")
    print(f"Sampling interval: {DT:.3f} s")

    comparisons = [
        ("Mean acceleration vs step-20 error", mean_acc, error_step20),
        ("Max acceleration vs step-20 error", max_acc, error_step20),
        ("P95 acceleration vs step-20 error", p95_acc, error_step20),
        ("Mean acceleration vs mean error", mean_acc, mean_error),
    ]

    print("\n=== CORRELATION ===")
    for name, x, y in comparisons:
        p, s, count = correlation(x, y)
        print(
            f"{name}\n"
            f"  Pearson  = {p:.4f}\n"
            f"  Spearman = {s:.4f}\n"
            f"  Samples  = {count}\n"
        )

    # 4. Divide windows into acceleration quartiles
    boundaries = np.quantile(mean_acc, [0.25, 0.50, 0.75])
    groups = np.digitize(mean_acc, boundaries)

    print("=== ERROR BY ACCELERATION GROUP ===")
    print("Group | Windows | Mean acceleration | Mean step-20 error")
    for group in range(4):
        mask = groups == group
        if not np.any(mask):
            continue

        print(
            f"Q{group + 1}    | "
            f"{np.sum(mask):7d} | "
            f"{np.mean(mean_acc[mask]):17.3f} | "
            f"{np.mean(error_step20[mask]):18.3f} deg"
        )

    # 5. Export per-window results
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    metadata_keys = [
        "user_id",
        "session_id",
        "sequence_name",
        "start_index",
    ]

    metadata = {}
    for key in metadata_keys:
        if key in data.files:
            values = data[key]
            if len(values) == n:
                metadata[key] = values

    with OUTPUT.open("w", newline="") as f:
        writer = csv.writer(f)

        header = (
            ["window_index"]
            + list(metadata.keys())
            + [
                "error_step20_deg",
                "mean_error_deg",
                "mean_angular_acc_deg_s2",
                "max_angular_acc_deg_s2",
                "p95_angular_acc_deg_s2",
                "acceleration_quartile",
            ]
        )
        writer.writerow(header)

        for i in range(n):
            row = [i]

            for key in metadata:
                row.append(metadata[key][i])

            row += [
                error_step20[i],
                mean_error[i],
                mean_acc[i],
                max_acc[i],
                p95_acc[i],
                f"Q{groups[i] + 1}",
            ]
            writer.writerow(row)

    print(f"\nSaved analysis to: {OUTPUT}")


if __name__ == "__main__":
    main()
