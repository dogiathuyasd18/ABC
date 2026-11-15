"""Geodesic losses on SO(3). q and -q represent the same rotation."""

import torch

from src.utils.quaternions import geodesic_degrees, quat_normalize


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of unit quaternion in (x, y, z, w) order."""
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions in (x, y, z, w) order."""
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack((x, y, z, w), dim=-1)


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


def acceleration_weighted_geodesic_loss(
    prediction,
    future,
    acceleration,
    accel_lambda=0.5,
    max_weight=2.0,
    step20_weight=1.0,
    q_threshold=0.75,
):
    """
    Geodesic loss weighted by observed historical angular acceleration.

    Samples at or below the q_threshold quantile (Q1-Q3) keep a fixed weight of 1.0,
    while high-acceleration samples above the threshold (Q4) receive scaled weights
    up to max_weight.
    """
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

    if acceleration.ndim != 1 or acceleration.shape[0] != prediction.shape[0]:
        raise ValueError("Acceleration must have shape (batch,)")

    if accel_lambda < 0 or max_weight < 1.0:
        raise ValueError("Invalid acceleration weighting parameters")

    if step20_weight < 0:
        raise ValueError("step20_weight must be non-negative.")

    if not 0.0 <= q_threshold < 1.0:
        raise ValueError("q_threshold must be in [0, 1)")

    prediction = quat_normalize(prediction)
    future = quat_normalize(future)

    degrees = geodesic_degrees(prediction, future)

    # Keep weight = 1.0 for Q1-Q3 (a <= tau); scale weight > 1.0 only for Q4 (a > tau).
    accel = acceleration.detach().clamp_min(0.0)
    tau = torch.quantile(accel, q_threshold).clamp_min(1e-8)

    excess = (accel - tau).clamp_min(0.0) / tau
    weights = (1.0 + accel_lambda * excess).clamp(max=max_weight)

    per_window_loss = (
        degrees.mean(dim=1)
        + step20_weight * degrees[:, -1]
    )

    loss = (weights * per_window_loss).mean()

    return loss, degrees, weights


def acceleration_deviation_geodesic_loss(
    prediction,
    future,
    acceleration,
    q_last=None,
    accel_lambda=0.5,
    max_weight=1.75,
    step20_weight=1.0,
    dev_weight=0.25,
    kin_accel_weight=1.0,
    q_threshold=0.75,
):
    """
    Combined SO(3) loss targeting both rotational acceleration and trajectory deviation:
      1. Pose geodesic error + horizon error standard deviation (penalizes drift spikes).
      2. Second-order SO(3) kinematic acceleration deviation between predicted and true trajectories.
      3. Hinge weighting on Q4 high-acceleration windows using historical acceleration only.
    """
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

    if acceleration.ndim != 1 or acceleration.shape[0] != prediction.shape[0]:
        raise ValueError("Acceleration must have shape (batch,)")

    if accel_lambda < 0 or max_weight < 1.0:
        raise ValueError("Invalid acceleration weighting parameters")

    if step20_weight < 0 or dev_weight < 0 or kin_accel_weight < 0:
        raise ValueError("Loss weights must be non-negative")

    if not 0.0 <= q_threshold < 1.0:
        raise ValueError("q_threshold must be in [0, 1)")

    prediction = quat_normalize(prediction)
    future = quat_normalize(future)

    # 1. Pose geodesic error & across-horizon deviation (spread of error across steps)
    degrees = geodesic_degrees(prediction, future)
    horizon_deviation = degrees.std(dim=1, unbiased=False)

    per_window_loss = (
        degrees.mean(dim=1)
        + step20_weight * degrees[:, -1]
        + dev_weight * horizon_deviation
    )

    # 2. Kinematic angular acceleration deviation on SO(3)
    if kin_accel_weight > 0.0:
        if q_last is not None:
            q0 = quat_normalize(q_last).unsqueeze(1)
            pred_seq = torch.cat((q0, prediction), dim=1)
            true_seq = torch.cat((q0, future), dim=1)
        else:
            pred_seq = prediction
            true_seq = future

        if pred_seq.shape[1] >= 3:
            # First-order relative rotations (angular velocity increments)
            vel_pred = quat_normalize(
                _quat_multiply(_quat_conjugate(pred_seq[:, :-1]), pred_seq[:, 1:])
            )
            vel_true = quat_normalize(
                _quat_multiply(_quat_conjugate(true_seq[:, :-1]), true_seq[:, 1:])
            )

            # Second-order relative rotations (angular acceleration increments)
            acc_pred = quat_normalize(
                _quat_multiply(_quat_conjugate(vel_pred[:, :-1]), vel_pred[:, 1:])
            )
            acc_true = quat_normalize(
                _quat_multiply(_quat_conjugate(vel_true[:, :-1]), vel_true[:, 1:])
            )

            kin_accel_dev = geodesic_degrees(acc_pred, acc_true).mean(dim=1)
            per_window_loss = per_window_loss + kin_accel_weight * kin_accel_dev

    # 3. Hinge sample weighting for Q4 extreme acceleration windows (Q1-Q3 remain 1.0)
    accel = acceleration.detach().clamp_min(0.0)
    tau = torch.quantile(accel, q_threshold).clamp_min(1e-8)

    excess = (accel - tau).clamp_min(0.0) / tau
    weights = (1.0 + accel_lambda * excess).clamp(max=max_weight)

    loss = (weights * per_window_loss).mean()

    return loss, degrees, weights