"""Run a trained model on a context window and emit 20 future xyzw quaternions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.evaluation.evaluate import load_model
from src.models.predict import decode_model
from src.runtime import resolve_device
from src.utils.quaternions import quat_inv, quat_mul, quat_normalize, quat_to_rotvec


def context_to_deltas(context_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    context_quat = quat_normalize(context_quat)
    if context_quat.ndim == 2:
        context_quat = context_quat.unsqueeze(0)
    if context_quat.shape[-1] != 4:
        raise ValueError(f"Expected xyzw quaternions, got {tuple(context_quat.shape)}")
    if context_quat.shape[1] < 2:
        raise ValueError("Need at least 2 context samples")
    deltas = quat_to_rotvec(quat_mul(quat_inv(context_quat[:, :-1]), context_quat[:, 1:]))
    return deltas, context_quat[:, -1]


@torch.no_grad()
def predict_future(model, context_quat: torch.Tensor) -> torch.Tensor:
    deltas, q_last = context_to_deltas(context_quat)
    quaternions, _hidden = decode_model(model, deltas, q_last)
    norms = quaternions.norm(dim=-1)
    if not torch.isfinite(quaternions).all() or (norms < 0.5).any():
        raise RuntimeError("Model produced invalid quaternions")
    return quat_normalize(quaternions)


def load_context(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        array = np.load(path)
    elif path.suffix == ".npz":
        packed = np.load(path)
        array = packed["quaternions"] if "quaternions" in packed else packed[packed.files[0]]
    elif path.suffix == ".json":
        array = np.asarray(json.loads(path.read_text())["quaternions"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported input: {path.suffix}")
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"Context must be (T, 4) xyzw, got {array.shape}")
    return array.astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict 20 future head poses from a context window")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="(T, 4) xyzw context, T<=400")
    parser.add_argument("--out", type=Path, default=Path("predictions.json"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    context = load_context(args.input)
    max_context = int(ckpt.get("context", 400))
    if len(context) > max_context:
        context = context[-max_context:]
    tensor = torch.from_numpy(context).to(device)
    future = predict_future(model, tensor).cpu().numpy()[0]
    payload = {
        "quaternion_order": "x,y,z,w",
        "sample_rate_hz": 100,
        "n_context": int(len(context)),
        "n_horizon": int(future.shape[0]),
        "horizon_seconds": future.shape[0] / 100.0,
        "quaternions": future.tolist(),
        "checkpoint": str(args.checkpoint),
        "parameters": int(ckpt.get("parameters", model.count_parameters())),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    np.savez(args.out.with_suffix(".npz"), quaternions=future.astype(np.float32))
    print(f"Wrote {args.out} shape={future.shape}")
    return payload


if __name__ == "__main__":
    main()
