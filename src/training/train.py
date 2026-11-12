
"""Train head-pose forecasting models. Validation only; test is never used here."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    CONTEXT_LENGTH,
    HORIZON,
    HeadPoseWindowDataset,
    context_targets,
    list_users,
)
from src.data.splits import split_users
from src.evaluation.baseline import (
    constant_velocity_quaternions,
    last_pose_quaternions,
)
from src.evaluation.metrics import window_weighted_step20
from src.models.gru import HeadPoseGRU
from src.models.msrtn import MultiScaleResidualTemporalNetwork
from src.models.predict import decode_model
from src.prep.config import PROJECT_ROOT
from src.runtime import (
    experiment_dir,
    experiment_manifest,
    resolve_device,
    seed_everything,
    write_json,
)
from src.training.loss import geodesic_loss


DEFAULT_H5 = PROJECT_ROOT / "data" / "processed" / "head_pose_v2.h5"


def resolve_h5(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    return path


def validate_args(args):
    args.model = getattr(args, "model", "gru")
    args.d_model = getattr(args, "d_model", 64)
    args.blocks = getattr(args, "blocks", 3)
    args.dropout = getattr(args, "dropout", 0.1)

    if args.context < 2:
        raise ValueError("--context must be at least 2")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    if args.hidden < 1 or args.layers < 1:
        raise ValueError("--hidden and --layers must be positive")
    if args.d_model < 4:
        raise ValueError("--d-model must be at least 4")
    if args.blocks < 1:
        raise ValueError("--blocks must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.model not in ("gru", "msrtn"):
        raise ValueError(f"Unsupported model: {args.model}")
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("--batch-size and --epochs must be positive")
    if args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("--lr and --grad-clip must be positive")
    if args.patience < 0:
        raise ValueError("--patience cannot be negative")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    if not 0 <= args.test_fraction < 1:
        raise ValueError("--test-fraction must be between 0 and 1")
    if args.val_fraction + args.test_fraction >= 1:
        raise ValueError(
            "Validation and test fractions must sum to less than 1"
        )
    if args.max_steps < 0:
        raise ValueError("--max-steps cannot be negative")


def build_model(args):
    if args.model == "gru":
        return HeadPoseGRU(
            hidden_size=args.hidden,
            num_layers=args.layers,
            horizon=args.horizon,
        )

    if args.model == "msrtn":
        return MultiScaleResidualTemporalNetwork(
            d_model=args.d_model,
            num_blocks=args.blocks,
            horizon=args.horizon,
            dropout=args.dropout,
        )

    raise ValueError(f"Unsupported model: {args.model}")


def training_config(args, h5_path):
    stat = h5_path.stat()

    return {
        "model": args.model,
        "h5_path": str(h5_path),
        "h5_size": stat.st_size,
        "h5_mtime_ns": stat.st_mtime_ns,
        "context": args.context,
        "horizon": args.horizon,
        "stride": args.stride,
        "hidden": args.hidden,
        "layers": args.layers,
        "d_model": args.d_model,
        "blocks": args.blocks,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "amp": args.amp,
    }


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state):
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def evaluate(model, loader, device, context, horizon):
    model.eval()

    totals = {
        "model": 0.0,
        "const_vel": 0.0,
        "last_pose": 0.0,
        "n": 0,
    }

    for windows in loader:
        windows = windows.to(device)
        prediction, future = _predict(model, windows, context)

        const_vel = constant_velocity_quaternions(
            windows, context, horizon
        )
        last_pose = last_pose_quaternions(
            windows, context, horizon
        )

        batch_size = windows.shape[0]

        for key, pred in (
            ("model", prediction),
            ("const_vel", const_vel),
            ("last_pose", last_pose),
        ):
            mean, _ = window_weighted_step20(pred, future)
            totals[key] += float(mean) * batch_size

        totals["n"] += batch_size

    if totals["n"] == 0:
        raise RuntimeError("Evaluation loader produced no windows")

    count = totals["n"]

    scores = {
        key: totals[key] / count
        for key in ("model", "const_vel", "last_pose")
    }

    return scores, count


def _predict(model, windows, context):
    deltas, q_last, future = context_targets(windows, context)
    prediction, _hidden = decode_model(model, deltas, q_last)
    return prediction, future


def export_onnx(model, path: Path, context: int, device: torch.device):
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
    validate_args(args)
    seed_everything(args.seed)

    device = resolve_device(args.device)
    h5_path = resolve_h5(args.h5)
    run_config = training_config(args, h5_path)

    users = list_users(h5_path)

    if len(users) < 3:
        raise RuntimeError(
            f"Need at least 3 users for train/validation/test split; "
            f"found {len(users)}"
        )

    train_users, val_users, test_users = split_users(
        users,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    if not train_users or not val_users:
        raise RuntimeError(
            "Train or validation split is empty. Check the number of users "
            "and split fractions."
        )

    out = experiment_dir(args.output_dir, args.name, args.seed)

    split = {
        "train_users": train_users,
        "val_users": val_users,
        "test_users": test_users,
        "seed": args.seed,
    }

    print(f"HDF5: {h5_path}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Users: {len(users)}")
    print(f"Train users: {len(train_users)}")
    print(f"Validation users: {len(val_users)}")
    print(f"Held-out test users: {len(test_users)}")
    print("Test users are not used during training or model selection.")

    train_set = HeadPoseWindowDataset(
        h5_path,
        train_users,
        args.context,
        args.horizon,
        args.stride,
    )

    val_set = HeadPoseWindowDataset(
        h5_path,
        val_users,
        args.context,
        args.horizon,
        args.stride,
    )

    print(f"Train windows: {len(train_set)}")
    print(f"Validation windows: {len(val_set)}")

    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError(
            "Train or validation dataset has no windows. "
            "Check HDF5 segments, user split, context, horizon and stride."
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = build_model(args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_val = float("inf")
    bad_epochs = 0
    history = []

    if args.resume:
        resume_path = args.resume.expanduser().resolve()

        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        saved_config = checkpoint.get("training_config")

        if saved_config != run_config:
            raise ValueError(
                "Resume configuration does not match the current run. "
                "Use the same model, HDF5 file and training parameters. "
                "--epochs may be increased."
            )

        if checkpoint.get("split") != split:
            raise ValueError(
                "Resume user split does not match the current user split."
            )

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", float("inf")))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        history = list(checkpoint.get("history", []))

        restore_rng_state(checkpoint.get("rng_state"))

        print(f"Resumed from: {resume_path}")
        print(f"Next epoch: {start_epoch}")

    print(f"Trainable parameters: {model.count_parameters()}")
    print(f"AMP enabled: {use_amp}")
    print(f"Experiment directory: {out}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        running_loss = 0.0
        seen = 0

        for step, windows in enumerate(train_loader, start=1):
            windows = windows.to(device)

            deltas, q_last, future = context_targets(
                windows, args.context
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp,
            ):
                prediction, _hidden = decode_model(
                    model, deltas, q_last
                )
                loss, _degrees = geodesic_loss(prediction, future)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}, step {step}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            scaler.step(optimizer)
            scaler.update()

            batch_size = windows.shape[0]
            running_loss += loss.detach().item() * batch_size
            seen += batch_size

            if args.max_steps and step >= args.max_steps:
                break

        if seen == 0:
            raise RuntimeError("No training batches were processed")

        val_scores, val_count = evaluate(
            model,
            val_loader,
            device,
            args.context,
            args.horizon,
        )

        scheduler.step(val_scores["model"])

        train_loss = running_loss / seen
        improved = val_scores["model"] < best_val - 1e-6

        if improved:
            best_val = val_scores["model"]
            bad_epochs = 0
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_step20_deg": val_scores["model"],
            "val_constant_velocity_step20_deg": val_scores["const_vel"],
            "val_last_pose_step20_deg": val_scores["last_pose"],
            "val_windows": val_count,
            "lr": optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.6f} "
            f"val_step20={val_scores['model']:.4f} deg "
            f"const_vel={val_scores['const_vel']:.4f} deg "
            f"last_pose={val_scores['last_pose']:.4f} deg "
            f"lr={row['lr']:.2e}"
        )

        payload = {
            "model_name": args.model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "training_config": run_config,
            "hidden_size": args.hidden,
            "num_layers": args.layers,
            "d_model": args.d_model,
            "num_blocks": args.blocks,
            "dropout": args.dropout,
            "context": args.context,
            "horizon": args.horizon,
            "quaternion_order": "x,y,z,w",
            "sample_rate_hz": 100,
            "parameters": model.count_parameters(),
            "split": split,
            "history": history,
            "epoch": epoch,
            "best_val": best_val,
            "bad_epochs": bad_epochs,
            "rng_state": capture_rng_state(),
        }

        torch.save(payload, out / "last.pt")

        if improved:
            torch.save(payload, out / "best.pt")
            print(f"  Saved best.pt ({best_val:.4f} deg)")

        if args.patience and bad_epochs >= args.patience:
            print(
                f"Early stopping after {args.patience} epochs "
                "without validation improvement"
            )
            break

    write_json(
        out / "config.json",
        experiment_manifest({
            "train_args": vars(args),
            "h5": str(h5_path),
            "training_config": run_config,
        }),
    )

    write_json(
        out / "history.json",
        {
            "best_val_step20_deg": best_val,
            "history": history,
        },
    )

    best_path = out / "best.pt"

    if best_path.is_file():
        best = torch.load(
            best_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(best["model_state"])

        onnx_path = out / f"head_pose_{args.model}_rotvec.onnx"

        try:
            export_onnx(
                model,
                onnx_path,
                args.context,
                device,
            )
            print(f"Exported ONNX: {onnx_path}")
        except Exception as error:
            print(f"Internal rotation-vector ONNX export skipped: {error}")
    else:
        print("No best checkpoint was produced; ONNX export skipped.")

    print(f"Best validation step-20 error: {best_val:.4f} deg")
    print(f"Experiment: {out}")

    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the 9th ABC head-pose predictor"
    )

    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments",
    )
    parser.add_argument("--name", default="head_pose_pilot")

    parser.add_argument(
        "--model",
        choices=["gru", "msrtn"],
        default="gru",
    )

    parser.add_argument("--context", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--stride", type=int, default=20)

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)

    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)

    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)

    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
