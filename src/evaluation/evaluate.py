"""Held-out evaluation. Do not tune hyperparameters on the split you pass here."""

from __future__ import annotations
from src.models.kinematics import KinematicsGRU

import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import HeadPoseWindowDataset, context_targets, list_users
from src.data.splits import split_users
from src.evaluation.baseline import constant_velocity_quaternions, last_pose_quaternions
from src.evaluation.metrics import geodesic_matrix, summarise_errors
from src.models.gru import HeadPoseGRU
from src.models.predict import decode_model
from src.prep.config import PROJECT_ROOT
from src.runtime import resolve_device, write_json
from src.training.train import resolve_h5


def load_model(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    model_name = ckpt.get("model_name", "gru")

    if model_name == "gru":
        model = HeadPoseGRU(
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            horizon=ckpt["horizon"],
        )

    elif model_name == "kinematics":
        model = KinematicsGRU(
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            horizon=ckpt["horizon"],
            dt=1.0 / ckpt.get("sample_rate_hz", 100),
            dropout=ckpt.get("dropout", 0.0),
        )

    else:
        raise ValueError(f"Unsupported checkpoint model: {model_name}")

    model = model.to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt


@torch.no_grad()

@torch.no_grad()
def run_split(
    model, h5_path, user_ids, context, horizon,
    batch_size, device, predictions_out=None
) -> dict:
    dataset = HeadPoseWindowDataset(
        h5_path,
        user_ids,
        context=context,
        horizon=horizon,
        stride=20,
        return_meta=True,
    )

    if len(dataset) == 0:
        return {"n_windows": 0, "note": "no windows in this split"}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    store = {
        "model": [],
        "const_vel": [],
        "last_pose": [],
        "horizon_model": [],
        "users": [],
        "pred_quat": [],
        "true_quat": [],
        "sessions": [],
        "names": [],
        "starts": [],
    }

    for batch in loader:
        windows, users, sessions, names, starts = batch
        windows = windows.to(device)

        deltas, q_last, future = context_targets(windows, context)
        pred, _ = decode_model(model, deltas, q_last)

        const_vel = constant_velocity_quaternions(
            windows, context, horizon
        )
        last_pose = last_pose_quaternions(
            windows, context, horizon
        )

        # Giữ nguyên cách tính metric cũ
        model_errors = geodesic_matrix(pred, future)
        const_errors = geodesic_matrix(const_vel, future)
        last_errors = geodesic_matrix(last_pose, future)

        store["model"].append(model_errors[:, -1].cpu())
        store["const_vel"].append(const_errors[:, -1].cpu())
        store["last_pose"].append(last_errors[:, -1].cpu())
        store["horizon_model"].append(model_errors.cpu())

        # Lưu quaternion dự đoán và ground truth
        store["pred_quat"].append(pred.detach().cpu().numpy())
        store["true_quat"].append(future.detach().cpu().numpy())

        # Lưu metadata của từng window
        store["users"].extend([str(x) for x in users])
        store["sessions"].extend([str(x) for x in sessions])
        store["names"].extend([str(x) for x in names])

        if torch.is_tensor(starts):
            store["starts"].extend(starts.cpu().numpy().tolist())
        else:
            store["starts"].extend(list(starts))

    # Ghép dữ liệu của tất cả batch
    step20 = torch.cat(store["model"])
    horizon_mean = torch.cat(store["horizon_model"]).mean(dim=0)

    # Xuất dữ liệu để phân tích gia tốc
    if predictions_out is not None:
        predictions_out = Path(predictions_out)
        predictions_out.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            predictions_out,
            pred_quat=np.concatenate(store["pred_quat"], axis=0),
            true_quat=np.concatenate(store["true_quat"], axis=0),
            user_id=np.asarray(store["users"], dtype=str),
            session_id=np.asarray(store["sessions"], dtype=str),
            sequence_name=np.asarray(store["names"], dtype=str),
            start_index=np.asarray(store["starts"]),
        )

        print(f"Saved predictions to {predictions_out}")
        print(
            f"Prediction shape: "
            f"{np.concatenate(store['pred_quat'], axis=0).shape}"
        )

    report = {
        "model": summarise_errors(
            step20, horizon_mean, store["users"]
        ),
        "constant_velocity": {
            "official_step20_mean_deg": float(
                torch.cat(store["const_vel"]).mean()
            ),
        },
        "last_pose": {
            "official_step20_mean_deg": float(
                torch.cat(store["last_pose"]).mean()
            ),
        },
        "beats_constant_velocity": (
            float(step20.mean())
            < float(torch.cat(store["const_vel"]).mean())
        ),
        "beats_last_pose": (
            float(step20.mean())
            < float(torch.cat(store["last_pose"]).mean())
        ),
        "n_users": len(set(store["users"])),
        "parameters": model.count_parameters(),
    }

    return report



def evaluate(args):
    device = resolve_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    h5_path = resolve_h5(args.h5)
    context = int(ckpt.get("context", 400))
    horizon = int(ckpt.get("horizon", 20))
    split = ckpt.get("split") or {}
    if args.split == "from-checkpoint" and split.get("test_users"):
        users = split["test_users"]
        label = "test"
    elif args.split == "val" and split.get("val_users"):
        users = split["val_users"]
        label = "val"
    else:
        train, val, test = split_users(list_users(h5_path), seed=args.seed)
        users = {"train": train, "val": val, "test": test}[args.split if args.split != "from-checkpoint" else "test"]
        label = args.split if args.split != "from-checkpoint" else "test"

    report = run_split(
        model,
        h5_path,
        users,
        context,
        horizon,
        args.batch_size,
        device,
        predictions_out=args.predictions_out,
    )
    report["split_name"] = label
    report["users"] = users
    report["checkpoint"] = str(args.checkpoint)
    report["h5"] = str(h5_path)
    report["note"] = (
        "This is a local user-held-out split, not the unreleased ST test set. "
        "The official accuracy is mean geodesic degrees at sample 20."
    )
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / f"eval_{label}.json"
    write_json(out, report)
    print(f"split={label} windows={report.get('model', {}).get('n_windows')} "
          f"step20={report.get('model', {}).get('official_step20_mean_deg')} "
          f"const_vel={report.get('constant_velocity', {}).get('official_step20_mean_deg')} "
          f"last_pose={report.get('last_pose', {}).get('official_step20_mean_deg')}")
    print(f"Wrote {out}")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained head-pose checkpoint"
    )

    parser.add_argument("--checkpoint", type=Path, required=True)

    parser.add_argument(
        "--h5",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "head_pose_v2.h5"
    )

    parser.add_argument(
        "--split",
        default="from-checkpoint",
        choices=("from-checkpoint", "val", "test", "train")
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)

    # Save predictions, ground truth, user IDs and session IDs
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="Path to save prediction data for error analysis"
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
