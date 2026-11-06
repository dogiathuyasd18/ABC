import torch

from src.utils.quaternions import geodesic_degrees


@torch.no_grad()
def window_weighted_step20(prediction, future):
    """Mean geodesic angle, in degrees, at the 20th predicted sample.

    Every evaluation window counts once. This is the challenge's accuracy figure.
    """
    degrees = geodesic_degrees(prediction, future)
    step20 = degrees[:, -1]
    return step20.mean().item(), step20
