"""Train the causal head-pose GRU. Validation only; test is never used here."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CONTEXT_LENGTH, HORIZON, HeadPoseWindowDataset, context_targets, list_users
from src.data.splits import split_users
from src.evaluation.baseline import constant_velocity_quaternions, last_pose_quaternions
from src.evaluation.metrics import window_weighted_step20
from src.models.gru import HeadPoseGRU
from src.models.predict import decode_model
from src.prep.config import PROJECT_ROOT
from src.runtime import experiment_dir, experiment_manifest, resolve_device, seed_everything, write_json
from src.training.loss import geodesic_loss


DEFAULT_H5 = PROJECT_ROOT / "data" / "processed" / "head_pose_v2.h5"
LEGACY_H5 = PROJECT_ROOT / "data" / "processed" / "head_pose.h5"


def resolve_h5(path: Path) -> Path:
    if path.exists():
        return path
    if LEGACY_H5.exists():
        return LEGACY_H5
    raise FileNotFoundError(f"No HDF5 at {path} or {LEGACY_H5}")


@torch.no_grad()
def evaluate(model, loader, device, context, horizon):
    model.eval()
    totals = {"model": 0.0, "const_vel": 0.0, "last_pose": 0.0, "n": 0}
    for windows in loader:
        windows = windows.to(device)
        prediction, future = _predict(model, windows, context)
        const_vel = constant_velocity_quaternions(windows, context, horizon)
        last_pose = last_pose_quaternions(windows, context, horizon)
        batch = windows.shape[0]
        for key, pred in ("model", prediction), ("const_vel", const_vel), ("last_pose", last_pose):
            mean, _ = window_weighted_step20(pred, future)
            totals[key] += mean * batch
        totals["n"] += batch
    if totals["n"] == 0:
        raise RuntimeError("Loader produced no windows")
    n = totals["n"]
    return {key: totals[key] / n for key in ("model", "const_vel", "last_pose")}, n


def _predict(model, windows, context):
    deltas, q_last, future = context_targets(windows, context)
    prediction, _hidden = decode_model(model, deltas, q_last)
    return prediction, future


def export_onnx(model, path: Path, context: int, device: torch.device) -> None:
    class RotvecOnly(torch.nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor

        def forward(self, delta_rotvec):
            rotvecs, _hidden = self.predictor(delta_rotvec)
            return rotvecs

    model.eval()
    example = torch.zeros(1, context - 1, 3, device=device)
    torch.onnx.export(
        RotvecOnly(model),
        example,
        path,
        input_names=["delta_rotvec"],
        output_names=["future_rotvec"],
        dynamo=False,
    )


def train(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    h5_path = resolve_h5(args.h5)
    train_users, val_users, test_users = split_users(
        list_users(h5_path),
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    if not val_users:
        raise RuntimeError("Need at least one validation user. Add more users to the HDF5 file.")

    out = experiment_dir(args.output_dir, args.name, args.seed)
    split = {"train_users": train_users, "val_users": val_users, "test_users": test_users, "seed": args.seed}
    write_json(out / "split.json", split)
    print(f"Train users ({len(train_users)}): {train_users}")
    print(f"Val users ({len(val_users)}): {val_users}")
    print(f"Test users held out ({len(test_users)}): {test_users}")
    print("Test users are not used during training or model selection.")

    train_set = HeadPoseWindowDataset(h5_path, train_users, args.context, args.horizon, args.stride)
    val_set = HeadPoseWindowDataset(h5_path, val_users, args.context, args.horizon, args.stride)
    print(f"Train windows: {len(train_set)} | Val windows: {len(val_set)}")
    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError("A split has no windows. Check HDF5 length and stride.")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = HeadPoseGRU(hidden_size=args.hidden, num_layers=args.layers, horizon=args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    start_epoch = 1
    best_val = float("inf")
    bad_epochs = 0
    history = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt.get("optimizer_state", optimizer.state_dict()))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        history = list(ckpt.get("history", []))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print(f"Trainable parameters: {model.count_parameters()}")
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for step, windows in enumerate(train_loader, start=1):
            windows = windows.to(device)
            deltas, q_last, future = context_targets(windows, args.context)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                prediction, _hidden = decode_model(model, deltas, q_last)
                loss, _degrees = geodesic_loss(prediction, future)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * windows.shape[0]
            seen += windows.shape[0]
            if args.max_steps and step >= args.max_steps:
                break

        val_scores, val_count = evaluate(model, val_loader, device, args.context, args.horizon)
        scheduler.step(val_scores["model"])
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_step20_deg": val_scores["model"],
            "val_constant_velocity_step20_deg": val_scores["const_vel"],
            "val_last_pose_step20_deg": val_scores["last_pose"],
            "val_windows": val_count,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"val_step20={row['val_step20_deg']:.4f} "
            f"const_vel={row['val_constant_velocity_step20_deg']:.4f} "
            f"last_pose={row['val_last_pose_step20_deg']:.4f}"
        )

        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "hidden_size": args.hidden,
            "num_layers": args.layers,
            "context": args.context,
            "horizon": args.horizon,
            "quaternion_order": "x,y,z,w",
            "sample_rate_hz": 100,
            "parameters": model.count_parameters(),
            "split": split,
            "history": history,
            "epoch": epoch,
            "best_val": min(best_val, val_scores["model"]),
        }
        torch.save(payload, out / "last.pt")
        if val_scores["model"] < best_val - 1e-6:
            best_val = val_scores["model"]
            bad_epochs = 0
            torch.save(payload, out / "best.pt")
            print(f"  saved best.pt ({best_val:.4f} deg)")
        else:
            bad_epochs += 1
            if args.patience and bad_epochs >= args.patience:
                print(f"Early stop after {args.patience} epochs without val improvement")
                break

    write_json(out / "config.json", experiment_manifest({"train_args": vars(args), "h5": str(h5_path)}))
    write_json(out / "history.json", {"best_val_step20_deg": best_val, "history": history})
    best_path = out / "best.pt"
    if best_path.exists():
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state"])
    try:
        export_onnx(model, out / "head_pose_gru.onnx", args.context, device)
    except Exception as error:
        print(f"ONNX export skipped: {error}")
    print(f"Experiment: {out}")
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the 9th ABC head-pose predictor")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments")
    parser.add_argument("--name", default="gru")
    parser.add_argument("--context", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
