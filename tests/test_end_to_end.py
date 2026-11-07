from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import HeadPoseWindowDataset
from src.evaluation.evaluate import run_split
from src.inference.predict import main as infer_main
from src.models.gru import HeadPoseGRU
from src.prep.hdf5 import init_file, save_session
from src.prep.resampling import ResampledSegment
from src.training.train import train


def _yaw_segment(n=500, dt=0.01):
    t = np.arange(n) * dt
    ang = 0.002 * np.arange(n)
    q = np.stack([np.zeros(n), np.sin(ang / 2), np.zeros(n), np.cos(ang / 2)], axis=1).astype(np.float32)
    return ResampledSegment(t, q, n, n, 0, float(t[-1] - t[0]))


def _tiny_h5(path: Path, n_users=4):
    init_file(path, resampled=True)
    for i in range(n_users):
        save_session(
            path,
            user_id=f"user-{i}",
            session_id="sess",
            member_name="m.xror",
            source_archive="a.tar",
            segments=[_yaw_segment(500)],
            attrs={"resampled": True, "sampling_rate_hz": 100.0},
        )


def test_windows_stay_inside_one_user(tmp_path):
    h5 = tmp_path / "d.h5"
    _tiny_h5(h5, 2)
    data = HeadPoseWindowDataset(h5, ["user-0"], context=400, horizon=20, stride=20, return_meta=True)
    assert len(data) > 0
    _, user, *_ = data[0]
    assert user == "user-0"


def test_train_eval_infer_smoke(tmp_path):
    h5 = tmp_path / "d.h5"
    _tiny_h5(h5, 4)
    args = Namespace(
        h5=h5,
        output_dir=tmp_path / "exp",
        name="smoke",
        context=400,
        horizon=20,
        stride=50,
        hidden=8,
        layers=1,
        batch_size=8,
        epochs=1,
        lr=1e-3,
        grad_clip=1.0,
        patience=2,
        val_fraction=0.25,
        test_fraction=0.25,
        max_steps=2,
        seed=0,
        device="cpu",
        amp=False,
        resume=None,
    )
    out = train(args)
    best = out / "best.pt"
    assert best.exists() or (out / "last.pt").exists()
    ckpt = best if best.exists() else out / "last.pt"
    model = HeadPoseGRU(hidden_size=8, horizon=20)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    report = run_split(model, h5, ["user-0"], 400, 20, 8, torch.device("cpu"))
    assert report["model"]["n_windows"] > 0
    context = np.zeros((40, 4), dtype=np.float32)
    context[:, 3] = 1.0
    inp = tmp_path / "ctx.npy"
    np.save(inp, context)
    pred = tmp_path / "pred.json"
    payload = infer_main(
        ["--checkpoint", str(ckpt), "--input", str(inp), "--out", str(pred), "--device", "cpu"]
    )
    assert payload["n_horizon"] == 20
    assert Path(str(pred)[:-5] + ".npz").exists() or pred.with_suffix(".npz").exists()
