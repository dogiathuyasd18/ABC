"""So sánh chi tiết checkpoint dùng geodesic_loss (cũ) và acceleration_weighted_geodesic_loss (mới)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    CONTEXT_LENGTH,
    HORIZON,
    HeadPoseWindowDataset,
    SAMPLE_RATE_HZ,
    context_targets,
    list_users,
)
from src.data.splits import split_users
from src.models.gru import HeadPoseGRU
from src.models.jepa import HeadPoseJEPA
from src.models.kinematics import KinematicsGRU
from src.models.kinematics_tcn import KinematicsTCN
from src.models.msrtn import MultiScaleResidualTemporalNetwork
from src.models.predict import decode_model
from src.models.res_mlp import ResidualMLP
from src.models.tcn import TemporalCNN
from src.runtime import resolve_device
from src.utils.quaternions import geodesic_degrees, quat_normalize


def build_model_from_ckpt(ckpt: dict) -> torch.nn.Module:
    cfg = ckpt.get("training_config", {})
    model_name = ckpt.get("model_name") or cfg.get("model")
    context = ckpt.get("context") or cfg.get("context", CONTEXT_LENGTH)
    horizon = ckpt.get("horizon") or cfg.get("horizon", HORIZON)
    hidden = ckpt.get("hidden_size") or cfg.get("hidden", 64)
    layers = ckpt.get("num_layers") or cfg.get("layers", 1)
    d_model = ckpt.get("d_model") or cfg.get("d_model", 64)
    blocks = ckpt.get("num_blocks") or cfg.get("blocks", 3)
    dropout = ckpt.get("dropout") or cfg.get("dropout", 0.1)

    if model_name == "gru":
        return HeadPoseGRU(hidden_size=hidden, num_layers=layers, horizon=horizon)
    if model_name == "kinematics":
        return KinematicsGRU(
            hidden_size=hidden,
            num_layers=layers,
            horizon=horizon,
            dt=1.0 / SAMPLE_RATE_HZ,
            dropout=dropout,
        )
    if model_name == "msrtn":
        return MultiScaleResidualTemporalNetwork(
            d_model=d_model, num_blocks=blocks, horizon=horizon, dropout=dropout
        )
    if model_name == "kinematics_tcn":
        return KinematicsTCN(
            hidden_size=hidden,
            horizon=horizon,
            dt=1.0 / SAMPLE_RATE_HZ,
            dropout=dropout,
            residual_scale=0.1,
            num_blocks=7,
            kernel_size=3,
        )
    if model_name == "tcn":
        return TemporalCNN(
            input_dim=3,
            hidden_size=hidden,
            horizon=horizon,
            num_layers=layers,
            dropout=dropout,
        )
    if model_name == "res_mlp":
        return ResidualMLP(
            context_len=context,
            input_dim=3,
            hidden_size=hidden,
            horizon=horizon,
            num_blocks=blocks,
            dropout=dropout,
        )
    if model_name == "jepa":
        return HeadPoseJEPA(
            input_dim=3,
            latent_dim=hidden,
            horizon=horizon,
            hidden_size=hidden * 2,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model in checkpoint: {model_name}")


def compute_historical_acceleration(deltas: torch.Tensor, sample_rate_hz: float) -> torch.Tensor:
    omega = deltas * sample_rate_hz
    alpha = torch.diff(omega, dim=1) * sample_rate_hz
    alpha_mag = torch.linalg.norm(alpha, dim=-1)
    return torch.sqrt(torch.mean(alpha_mag ** 2, dim=1) + 1e-12)


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: Path, loader: DataLoader, context: int, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt).to(device)
    incomp = model.load_state_dict(ckpt["model_state"], strict=False)
    if incomp.missing_keys:
        raise RuntimeError(f"Missing keys: {incomp.missing_keys}")
    model.eval()

    cfg = ckpt.get("training_config", {})
    loss_type = cfg.get("loss_type", "geodesic (legacy)")
    model_name = ckpt.get("model_name") or cfg.get("model", "?")

    all_step20 = []
    all_horizon = []
    all_accel = []

    for windows in loader:
        windows = windows.to(device)
        deltas, q_last, future = context_targets(windows, context)
        pred, _ = decode_model(model, deltas, q_last)

        degrees = geodesic_degrees(quat_normalize(pred), quat_normalize(future))
        accel = compute_historical_acceleration(deltas, SAMPLE_RATE_HZ)

        all_step20.append(degrees[:, -1].cpu().numpy())
        all_horizon.append(degrees.mean(dim=1).cpu().numpy())
        all_accel.append(accel.cpu().numpy())

    step20 = np.concatenate(all_step20)
    horizon = np.concatenate(all_horizon)
    accel = np.concatenate(all_accel)

    return {
        "path": ckpt_path.parent.name,
        "model": model_name,
        "loss_type": loss_type,
        "step20": step20,
        "horizon": horizon,
        "accel": accel,
    }


def summarize_by_quartiles(res: dict, q_edges: tuple[float, float, float]) -> dict:
    s20 = res["step20"]
    acc = res["accel"]
    q25, q50, q75 = q_edges

    masks = {
        "Q1": acc <= q25,
        "Q2": (acc > q25) & (acc <= q50),
        "Q3": (acc > q50) & (acc <= q75),
        "Q4": acc > q75,
    }

    stats = {
        "Overall_Step20": float(np.mean(s20)),
        "Horizon_Mean": float(np.mean(res["horizon"])),
        "Median_Step20": float(np.median(s20)),
        "P95_Step20": float(np.percentile(s20, 95)),
    }
    for q_name, m in masks.items():
        stats[q_name] = float(np.mean(s20[m]))
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new loss checkpoints")
    parser.add_argument("--old", type=Path, required=True, help="Path to old best.pt (geodesic)")
    parser.add_argument("--new", type=Path, required=True, help="Path to new best.pt (accel_weighted)")
    parser.add_argument("--h5", type=Path, default=Path("data/processed/head_pose_full_v3.h5"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    device = resolve_device(args.device)
    old_ckpt = torch.load(args.old, map_location="cpu", weights_only=False)
    cfg = old_ckpt.get("training_config", {})

    context = old_ckpt.get("context") or cfg.get("context", CONTEXT_LENGTH)
    horizon = old_ckpt.get("horizon") or cfg.get("horizon", HORIZON)
    stride = cfg.get("stride", 20)
    seed = cfg.get("seed", 42)

    split_dict = old_ckpt.get("split")
    if split_dict:
        eval_users = split_dict["val_users"] if args.split == "val" else split_dict["test_users"]
    else:
        users = list_users(args.h5)
        _, val_u, test_u = split_users(users, val_fraction=0.15, test_fraction=0.15, seed=seed)
        eval_users = val_u if args.split == "val" else test_u

    dataset = HeadPoseWindowDataset(args.h5, eval_users, context, horizon, stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    res_old = evaluate_checkpoint(args.old, loader, context, device)
    res_new = evaluate_checkpoint(args.new, loader, context, device)

    # Cố định ngưỡng chia Q1-Q4 từ gia tốc của dataset để so sánh 1-1 trên cùng tập mẫu
    q_edges = tuple(np.quantile(res_old["accel"], [0.25, 0.50, 0.75]))

    st_old = summarize_by_quartiles(res_old, q_edges)
    st_new = summarize_by_quartiles(res_new, q_edges)

    print(f"\n=== SO SÁNH LOSS CŨ VS LOSS MỚI ({args.split.upper()} SET - {len(dataset)} windows) ===")
    print(f"Old Checkpoint: {res_old['path']} | Model: {res_old['model']} | Loss: {res_old['loss_type']}")
    print(f"New Checkpoint: {res_new['path']} | Model: {res_new['model']} | Loss: {res_new['loss_type']}")
    print(f"Ngưỡng phân vị gia tốc (rad/s^2): Q25={q_edges[0]:.2f}, Q50={q_edges[1]:.2f}, Q75={q_edges[2]:.2f}\n")

    header = f"{'Metric (deg)':<22} | {'Old Loss':<12} | {'New Loss':<12} | {'Delta (New - Old)':<18}"
    print(header)
    print("-" * len(header))

    metrics_order = [
        ("Overall_Step20", "Overall Step-20 Mean"),
        ("Horizon_Mean", "Horizon Mean (1..20)"),
        ("Median_Step20", "Median Step-20"),
        ("P95_Step20", "P95 Tail Error"),
        ("Q1", "Q1 (Low Accel)"),
        ("Q2", "Q2 (Moderate Accel)"),
        ("Q3", "Q3 (High Accel)"),
        ("Q4", "Q4 (Extreme Accel)"),
    ]

    for key, label in metrics_order:
        v_old = st_old[key]
        v_new = st_new[key]
        diff = v_new - v_old
        sign = "+" if diff > 0 else ""
        note = " (Tốt hơn)" if diff < -1e-4 else (" (Tăng nhẹ)" if diff > 1e-4 else "")
        print(f"{label:<22} | {v_old:<12.4f} | {v_new:<12.4f} | {sign}{diff:<8.4f}{note}")


if __name__ == "__main__":
    main()