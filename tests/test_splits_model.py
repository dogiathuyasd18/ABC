import numpy as np
import torch

from src.data.splits import split_users
from src.evaluation.baseline import constant_velocity_quaternions, last_pose_quaternions
from src.inference.predict import predict_future
from src.models.gru import HeadPoseGRU
from src.models.predict import decode_model
from src.training.loss import geodesic_loss
from src.utils.quaternions import geodesic_degrees, quat_normalize


def test_user_split_is_disjoint_and_seeded():
    users = [f"u{i:02d}" for i in range(20)]
    a = split_users(users, seed=0)
    b = split_users(users, seed=0)
    c = split_users(users, seed=1)
    assert a == b
    assert a != c
    train, val, test = a
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert len(train) + len(val) + len(test) == 20
    assert train and val and test


def test_geodesic_identifies_q_and_neg_q():
    q = quat_normalize(torch.tensor([[0.1, 0.2, 0.3, 0.9]]))
    assert geodesic_degrees(q, q).item() < 1e-4
    assert geodesic_degrees(q, -q).item() < 1e-4


def test_model_forward_unit_quaternions():
    model = HeadPoseGRU(hidden_size=8, horizon=20)
    deltas = torch.zeros(2, 10, 3)
    q_last = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.1, 0.0, 0.995]])
    q_last = quat_normalize(q_last)
    pred, _ = decode_model(model, deltas, q_last)
    assert pred.shape == (2, 20, 4)
    norms = pred.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_loss_zero_when_perfect():
    future = quat_normalize(torch.randn(4, 20, 4))
    loss, degrees = geodesic_loss(future, future)
    assert loss.item() < 1e-4
    assert degrees.max().item() < 1e-3


def test_baselines_on_constant_yaw():
    steps = 420
    angle = torch.arange(steps).float() * 0.01
    quat = torch.stack(
        (torch.zeros(steps), torch.sin(angle / 2), torch.zeros(steps), torch.cos(angle / 2)),
        -1,
    )
    windows = quat.view(1, steps, 4)
    const_vel = constant_velocity_quaternions(windows, 400, 20)
    last = last_pose_quaternions(windows, 400, 20)
    future = windows[:, 400:]
    err_v = geodesic_degrees(const_vel, future)[0, -1].item()
    err_p = geodesic_degrees(last, future)[0, -1].item()
    assert err_v < 0.05
    assert err_p > 1.0


def test_inference_emits_20_poses():
    model = HeadPoseGRU(hidden_size=8, horizon=20)
    context = torch.zeros(30, 4)
    context[:, 3] = 1.0
    future = predict_future(model, context)
    assert future.shape == (1, 20, 4)
