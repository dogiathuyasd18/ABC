
"""Geodesic losses on SO(3). q and -q represent the same rotation."""

from src.utils.quaternions import geodesic_degrees, quat_normalize


def geodesic_loss(prediction, future, step20_weight=1.0):
    """Optimize horizon-wide geodesic error with additional weight on step 20."""
    if prediction.shape != future.shape:
        raise ValueError(
            f"Prediction and target shapes must match: "
            f"{prediction.shape} != {future.shape}"
        )

    if prediction.ndim != 3 or prediction.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternion tensors with shape (batch, horizon, 4), "
            f"got {prediction.shape}"
        )

    if prediction.shape[1] < 1:
        raise ValueError("Prediction horizon must contain at least one timestep.")

    if step20_weight < 0:
        raise ValueError("step20_weight must be non-negative.")

    prediction = quat_normalize(prediction)
    future = quat_normalize(future)

    degrees = geodesic_degrees(prediction, future)

    horizon_loss = degrees.mean()
    final_step_loss = degrees[:, -1].mean()

    loss = horizon_loss + step20_weight * final_step_loss

    return loss, degrees
