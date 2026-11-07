"""Geodesic losses on SO(3). q and -q are the same rotation."""

from src.utils.quaternions import geodesic_degrees, quat_normalize


def geodesic_loss(prediction, future, step20_weight=1.0):
    """Mean geodesic over the horizon plus extra weight on the scored 20th sample."""
    prediction = quat_normalize(prediction)
    future = quat_normalize(future)
    degrees = geodesic_degrees(prediction, future)
    loss = degrees.mean() + step20_weight * degrees[:, -1].mean()
    return loss, degrees
