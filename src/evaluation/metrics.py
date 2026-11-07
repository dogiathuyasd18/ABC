"""Official 9th ABC accuracy: mean geodesic degrees at predicted sample 20."""

from __future__ import annotations

import torch

from src.utils.quaternions import geodesic_degrees, quat_normalize


INVALID_DEGREES = 180.0


def geodesic_matrix(prediction, future):
    """(batch, horizon) geodesic error in degrees. Non-finite predictions score 180°."""
    prediction = quat_normalize(prediction)
    future = quat_normalize(future)
    degrees = geodesic_degrees(prediction, future)
    bad = ~torch.isfinite(degrees)
    if bad.any():
        degrees = degrees.clone()
        degrees[bad] = INVALID_DEGREES
    return degrees


@torch.no_grad()
def window_weighted_step20(prediction, future):
    degrees = geodesic_matrix(prediction, future)
    step20 = degrees[:, -1]
    return step20.mean().item(), step20


@torch.no_grad()
def horizon_means(prediction, future):
    degrees = geodesic_matrix(prediction, future)
    return degrees.mean(dim=0)


def summarise_errors(step20: torch.Tensor, horizon: torch.Tensor, user_ids: list[str]) -> dict:
    step20 = step20.detach().cpu()
    horizon = horizon.detach().cpu()
    per_user = {}
    for user in sorted(set(user_ids)):
        mask = torch.tensor([uid == user for uid in user_ids])
        if mask.any():
            per_user[user] = float(step20[mask].mean())
    user_macro = float(sum(per_user.values()) / len(per_user)) if per_user else float("nan")
    return {
        "n_windows": int(step20.numel()),
        "official_step20_mean_deg": float(step20.mean()),
        "step20_median_deg": float(step20.median()),
        "horizon_mean_deg": [float(v) for v in horizon],
        "user_macro_step20_mean_deg": user_macro,
        "per_user_step20_mean_deg": per_user,
        "aggregation": "official figure is window-weighted mean of sample-20 geodesic degrees",
        "units": "degrees",
    }
