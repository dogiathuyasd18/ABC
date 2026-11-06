"""Train the causal head-pose GRU and write a checkpoint plus an ONNX file."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    CONTEXT_LENGTH,
    HORIZON,
    HeadPoseWindowDataset,
    context_targets,
    list_users,
    official_loss,
    predict_quaternions,
    split_users,
)
from src.evaluation.baseline import constant_velocity_quaternions
from src.evaluation.metrics import window_weighted_step20
from src.models.gru import HeadPoseGRU
from src.utils.quaternions import apply_body_rotvec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H5 = PROJECT_ROOT / "data" / "processed" / "head_pose.h5"


def evaluate(model, loader, device, context):
    model.eval()
    model_sum = 0.0
    baseline_sum = 0.0
    count = 0
    with torch.no_grad():
        for windows in loader:
            windows = windows.to(device)
            prediction, future = predict_quaternions(model, windows, context)
            baseline = constant_velocity_quaternions(windows, context, future.shape[1])
            model_mean, model_angles = window_weighted_step20(prediction, future)
            baseline_mean, baseline_angles = window_weighted_step20(baseline, future)
            batch = model_angles.shape[0]
            model_sum += model_mean * batch
            baseline_sum += baseline_mean * batch
            count += batch
    if count == 0:
        raise RuntimeError("Validation loader produced no windows")
    return model_sum / count, baseline_sum / count, count


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_users, val_users = split_users(list_users(args.h5), args.val_fraction)
    print(f"Train users ({len(train_users)}): {train_users}")
    print(f"Val users ({len(val_users)}): {val_users}")

    train_set = HeadPoseWindowDataset(
        args.h5, train_users, context=args.context, horizon=args.horizon, stride=args.stride
    )
    val_set = HeadPoseWindowDataset(
        args.h5, val_users, context=args.context, horizon=args.horizon, stride=args.stride
    )
    print(f"Train windows: {len(train_set)} | Val windows: {len(val_set)}")
    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError("A split has no windows. Lower the stride or check the HDF5 file.")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = HeadPoseGRU(hidden_size=args.hidden, num_layers=args.layers, horizon=args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Trainable parameters: {model.count_parameters()}")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for step, windows in enumerate(train_loader, start=1):
            windows = windows.to(device)
            deltas, q_last, future = context_targets(windows, args.context)
            rotvecs, _hidden = model(deltas)
            prediction = apply_body_rotvec(q_last, rotvecs)
            loss, _degrees = official_loss(prediction, future)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * windows.shape[0]
            seen += windows.shape[0]
            if args.max_steps and step >= args.max_steps:
                break

        val_model, val_baseline, val_count = evaluate(model, val_loader, device, args.context)
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_geodesic_deg_step20": val_model,
            "val_constant_velocity_deg_step20": val_baseline,
            "val_windows": val_count,
        }
        history.append(row)
        print(
            f"Epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"val_step20={val_model:.4f} deg "
            f"constant_velocity={val_baseline:.4f} deg "
            f"windows={val_count}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "head_pose_gru.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden_size": args.hidden,
            "num_layers": args.layers,
            "context": args.context,
            "horizon": args.horizon,
            "quaternion_order": "x,y,z,w",
            "sample_rate_hz": 100,
            "train_users": train_users,
            "val_users": val_users,
            "history": history,
        },
        checkpoint_path,
    )

    report = {
        "parameters": model.count_parameters(),
        "metric": "mean geodesic degrees at predicted sample 20, window-weighted, user-held-out",
        "history": history,
    }
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2))

    model.eval()
    example = torch.zeros(1, args.context - 1, 3, device=device)
    onnx_path = output_dir / "head_pose_gru.onnx"

    class _RotvecOnly(torch.nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor

        def forward(self, delta_rotvec):
            rotvecs, _hidden = self.predictor(delta_rotvec)
            return rotvecs

    torch.onnx.export(
        _RotvecOnly(model),
        example,
        onnx_path,
        input_names=["delta_rotvec"],
        output_names=["future_rotvec"],
        dynamo=False,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"ONNX: {onnx_path}")
    print(f"Metrics: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the 9th ABC head-pose predictor")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--context", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop each epoch after this many batches. 0 means all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
