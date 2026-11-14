
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr


INPUT = Path("outputs/gru_predictions.npz")
OUTPUT = Path("outputs/signed_error_analysis.csv")


def normalize_quat(q):
    return q / np.maximum(
        np.linalg.norm(q, axis=-1, keepdims=True), 1e-12
    )


def wrap_angle_deg(angle):
    """Wrap angular difference into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def quaternion_to_euler(q):
    """
    Quaternion order: xyzw.
    Euler convention: ZYX.
    Returns columns: yaw, pitch, roll.
    """
    q = normalize_quat(q)
    euler = Rotation.from_quat(
        q.reshape(-1, 4)
    ).as_euler("ZYX", degrees=True)

    return euler.reshape(*q.shape[:-1], 3)


def correlation(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    return (
        pearsonr(x, y).statistic,
        spearmanr(x, y).statistic,
    )


def main():
    data = np.load(INPUT, allow_pickle=True)

    pred = np.asarray(data["pred_quat"], dtype=float)
    true = np.asarray(data["true_quat"], dtype=float)

    # Euler angles: [N, horizon, 3]
    pred_euler = quaternion_to_euler(pred)
    true_euler = quaternion_to_euler(true)

    # Signed prediction error
    signed = wrap_angle_deg(pred_euler - true_euler)

    # Analyze the 20th future sample
    signed20 = signed[:, -1, :]

    # Geodesic error for reference
    dot = np.sum(
        normalize_quat(pred[:, -1]) *
        normalize_quat(true[:, -1]), axis=-1
    )
    geodesic = np.degrees(
        2 * np.arccos(np.clip(np.abs(dot), 0, 1))
    )

    # Load acceleration analysis
    acc_df = pd.read_csv("outputs/acceleration_analysis.csv")
    acc = acc_df["mean_angular_acc_deg_s2"].to_numpy()

    if len(acc) != len(signed20):
        raise ValueError(
            "Số windows trong CSV và NPZ không khớp."
        )

    axes = ["yaw", "pitch", "roll"]
    rows = []

    print("\n=== SIGNED ERROR ANALYSIS ===")

    for j, axis in enumerate(axes):
        err = signed20[:, j]
        pearson, spearman = correlation(acc, err)

        positive = np.mean(err > 0) * 100
        negative = np.mean(err < 0) * 100

        print(f"\n{axis.upper()}")
        print(f"Mean signed error: {np.mean(err):.3f} deg")
        print(f"Median signed error: {np.median(err):.3f} deg")
        print(f"Positive errors: {positive:.2f}%")
        print(f"Negative errors: {negative:.2f}%")
        print(f"Pearson(acc, error): {pearson:.4f}")
        print(f"Spearman(acc, error): {spearman:.4f}")

        for i in range(len(err)):
            rows.append({
                "window_index": i,
                "acceleration": acc[i],
                "axis": axis,
                "signed_error_deg": err[i],
                "absolute_error_deg": abs(err[i]),
                "geodesic_error_deg": geodesic[i],
            })

    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
