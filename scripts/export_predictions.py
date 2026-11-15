import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

# Tự động thêm thư mục gốc vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import HeadPoseWindowDataset, context_targets
from src.models.predict import decode_model
from src.runtime import resolve_device

# Import các mô hình để rebuild
from src.models.gru import HeadPoseGRU
from src.models.kinematics import KinematicsGRU
from src.models.kinematics_tcn import KinematicsTCN
from src.models.msrtn import MultiScaleResidualTemporalNetwork
from src.models.tcn import TemporalCNN
from src.models.res_mlp import ResidualMLP
from src.models.jepa import HeadPoseJEPA


def rebuild_model(config, device, state_dict=None):
    model_name = config["model"]
    hidden = config.get("hidden", 64)
    layers = config.get("layers", 1)
    horizon = config.get("horizon", 20)
    dropout = config.get("dropout", 0.1)
    d_model = config.get("d_model", 64)
    blocks = config.get("blocks", 3)

    if model_name == "gru":
        model = HeadPoseGRU(hidden_size=hidden, num_layers=layers, horizon=horizon)
    elif model_name == "res_mlp":
        model = ResidualMLP(hidden_size=hidden, horizon=horizon, num_blocks=blocks, dropout=dropout)
        if state_dict is not None and "head.weight" in state_dict:
            out_feat, in_feat = state_dict["head.weight"].shape
            model.head = torch.nn.Linear(in_feat, out_feat)
    elif model_name == "tcn":
        model = TemporalCNN(hidden_size=hidden, horizon=horizon, num_layers=layers, dropout=dropout)
    elif model_name == "jepa":
        model = HeadPoseJEPA(latent_dim=hidden, horizon=horizon, hidden_size=hidden*2, dropout=dropout)
    elif model_name == "msrtn":
        model = MultiScaleResidualTemporalNetwork(d_model=d_model, num_blocks=blocks, horizon=horizon, dropout=dropout)
    elif model_name == "kinematics":
        model = KinematicsGRU(hidden_size=hidden, num_layers=layers, horizon=horizon, dropout=dropout)
    elif model_name == "kinematics_tcn":
        model = KinematicsTCN(hidden_size=hidden, horizon=horizon, dropout=dropout)
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_name}")

    return model.to(device)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Xuất file dự đoán .npz từ checkpoint best.pt")
    parser.add_argument("--model-name", type=str, default="res_mlp", help="Tên model (vd: res_mlp, tcn, jepa)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Đường dẫn trực tiếp tới file best.pt")
    parser.add_argument("--experiment-dir", type=Path, default=PROJECT_ROOT / "experiments", help="Thư mục chứa experiments")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    device = resolve_device(args.device)
    
    if args.checkpoint and args.checkpoint.exists():
        best_pt = args.checkpoint
    else:
        matching_dirs = []
        for d in sorted(args.experiment_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if d.is_dir():
                pt_path = d / "best.pt"
                if pt_path.exists():
                    try:
                        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
                        if "training_config" in ckpt and ckpt["training_config"].get("model") == args.model_name:
                            matching_dirs.append(d)
                    except Exception:
                        pass
        
        if not matching_dirs:
            raise FileNotFoundError(f"Không tìm thấy thư mục experiment nào chứa checkpoint của model '{args.model_name}'.")
        
        latest_exp = matching_dirs[0]
        best_pt = latest_exp / "best.pt"
        print(f"🔍 Đã tự động nhận diện thư mục experiment phù hợp nhất: {latest_exp.name}")

    if not best_pt.exists():
        raise FileNotFoundError(f"Không tìm thấy file best.pt tại {best_pt}")

    print(f"📦 Đang load checkpoint từ: {best_pt}")
    checkpoint = torch.load(best_pt, map_location=device, weights_only=False)
    config = checkpoint["training_config"]
    split = checkpoint["split"]
    state_dict = checkpoint["model_state"]

    h5_path = Path(config["h5_path"])
    val_users = split["val_users"]
    context = config.get("context", 400)
    horizon = config.get("horizon", 20)
    stride = config.get("stride", 1)

    model = rebuild_model(config, device, state_dict=state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    val_set = HeadPoseWindowDataset(h5_path, val_users, context, horizon, stride)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)

    print(f"🚀 Đang chạy inference trên tập Validation ({len(val_set)} windows)...")

    pred_quats = []
    true_quats = []

    for windows in val_loader:
        windows = windows.to(device)
        deltas, q_last, future = context_targets(windows, context)
        prediction, _ = decode_model(model, deltas, q_last)

        pred_quats.append(prediction.cpu().numpy())
        true_quats.append(future.cpu().numpy())

    pred_quat = np.concatenate(pred_quats, axis=0)
    true_quat = np.concatenate(true_quats, axis=0)

    # Đồng bộ metadata chuẩn xác từ file npz mẫu của GRU (vì dùng chung tập val)
    gru_npz_path = Path("outputs/gru_val_seed42_predictions.npz")
    if gru_npz_path.exists():
        print("📦 Đang đồng bộ metadata chuẩn từ file dự đoán mẫu của GRU...")
        gru_data = np.load(gru_npz_path, allow_pickle=True)
        user_ids = gru_data["user_id"]
        session_ids = gru_data["session_id"]
        sequence_names = gru_data["sequence_name"] if "sequence_name" in gru_data.files else np.array(["0000"] * len(pred_quat))
        start_indices = gru_data["start_index"] if "start_index" in gru_data.files else np.zeros(len(pred_quat), dtype=int)
        
        # Khớp chính xác số lượng window nếu có chênh lệch nhỏ
        if len(user_ids) != len(pred_quat):
            min_len = min(len(user_ids), len(pred_quat))
            user_ids = user_ids[:min_len]
            session_ids = session_ids[:min_len]
            sequence_names = sequence_names[:min_len]
            start_indices = start_indices[:min_len]
            pred_quat = pred_quat[:min_len]
            true_quat = true_quat[:min_len]
    else:
        user_ids = np.array(["unknown"] * len(pred_quat))
        session_ids = np.array(["unknown"] * len(pred_quat))
        sequence_names = np.array(["0000"] * len(pred_quat))
        start_indices = np.zeros(len(pred_quat), dtype=int)

    out_npz = Path("outputs") / f"{args.model_name}_val_seed42_predictions.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_npz,
        pred_quat=pred_quat,
        true_quat=true_quat,
        user_id=user_ids,
        session_id=session_ids,
        sequence_name=sequence_names,
        start_index=start_indices,
    )

    print(f"✅ Đã xuất file dự đoán thành công tại: {out_npz}")


if __name__ == "__main__":
    main()