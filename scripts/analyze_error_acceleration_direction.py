
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


INPUT = Path("outputs/gru_predictions.npz")
OUTPUT = Path("outputs/error_acceleration_direction.csv")

DT = 0.01  # 100 Hz


def normalize_quat(q):
    return q / np.maximum(
        np.linalg.norm(q, axis=-1, keepdims=True), 1e-12
    )


def analyze():
    data = np.load(INPUT, allow_pickle=True)

    pred = normalize_quat(
        np.asarray(data["pred_quat"], dtype=float)
    )
    true = normalize_quat(
        np.asarray(data["true_quat"], dtype=float)
    )

    if pred.shape != true.shape or pred.ndim != 3:
        raise ValueError(
            f"Unexpected shapes: pred={pred.shape}, true={true.shape}"
        )

    n_windows, horizon, _ = true.shape

    if horizon < 3:
        raise ValueError("Need at least 3 future samples.")

    results = []

    for i in range(n_windows):
        # Convert future orientations to rotations.
        r_true = Rotation.from_quat(true[i])
        r_pred = Rotation.from_quat(pred[i])

        # 1. Prediction error at the final future sample.
        # Express the relative rotation in the true orientation's
        # local frame, then transform it into world coordinates.
        error_local = (
            r_true[-1].inv() * r_pred[-1]
        ).as_rotvec()

        error_world = r_true[-1].apply(error_local)

        # 2. Estimate world-frame angular velocity.
        relative_rotations = (
            r_true[:-1].inv() * r_true[1:]
        )

        omega_local = (
            relative_rotations.as_rotvec() / DT
        )

        # Express angular velocity in a common world frame.
        omega_world = r_true[:-1].apply(omega_local)

        # 3. Angular acceleration in world coordinates.
        alpha_world = np.diff(omega_world, axis=0) / DT

        # Average acceleration vector over the future horizon.
        mean_alpha = np.mean(alpha_world, axis=0)

        error_norm = np.linalg.norm(error_world)
        alpha_norm = np.linalg.norm(mean_alpha)

        if error_norm < 1e-8 or alpha_norm < 1e-8:
            cosine = np.nan
            projected_error = np.nan
        else:
            cosine = np.dot(
                error_world, mean_alpha
            ) / (error_norm * alpha_norm)

            cosine = np.clip(cosine, -1.0, 1.0)

            # Signed prediction error along acceleration direction.
            projected_error = np.dot(
                error_world, mean_alpha / alpha_norm
            )

        results.append({
            "window_index": i,
            "error_magnitude_deg": np.degrees(error_norm),
            "mean_acceleration_magnitude_deg_s2":
                np.degrees(alpha_norm),
            "cosine_similarity": cosine,
            "direction_angle_deg":
                np.degrees(np.arccos(cosine))
                if np.isfinite(cosine) else np.nan,
            "projected_error_deg":
                np.degrees(projected_error)
                if np.isfinite(projected_error) else np.nan,
        })

    df = pd.DataFrame(results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    valid = df.dropna(subset=["cosine_similarity"])
    cos = valid["cosine_similarity"]

    print("\n=== ERROR vs ACCELERATION DIRECTION ===")
    print(f"Valid windows: {len(valid)} / {n_windows}")

    print(f"Mean cosine similarity: {cos.mean():.4f}")
    print(f"Median cosine similarity: {cos.median():.4f}")

    print(
        f"Same direction (cos > 0.1): "
        f"{(cos > 0.1).mean() * 100:.2f}%"
    )
    print(
        f"Opposite direction (cos < -0.1): "
        f"{(cos < -0.1).mean() * 100:.2f}%"
    )
    print(
        f"Nearly perpendicular (-0.1 to 0.1): "
        f"{((cos >= -0.1) & (cos <= 0.1)).mean() * 100:.2f}%"
    )

    print("\nMean projected error (deg):",
          valid["projected_error_deg"].mean())

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    analyze()
