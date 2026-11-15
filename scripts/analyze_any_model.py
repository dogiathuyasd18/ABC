from pathlib import Path
import argparse
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

H5_PATH = Path("data/processed/head_pose_full_v3.h5")
OUT_DIR = Path("outputs")
DT = 0.01
CONTEXT = 400


def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], axis=-1)


def quat_inv(q):
    result = q.copy()
    result[..., :3] *= -1
    return result


def relative_rotvec(q0, q1):
    rel = quat_mul(quat_inv(q0), q1)
    rel /= np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-12)
    rel = np.where(rel[..., 3:4] < 0, -rel, rel)
    return Rotation.from_quat(rel).as_rotvec()


def geodesic_error_deg(q1, q2):
    rel = quat_mul(quat_inv(q1), q2)
    rel /= np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-12)
    angle = 2 * np.arctan2(
        np.linalg.norm(rel[..., :3], axis=-1),
        np.abs(rel[..., 3])
    )
    return np.degrees(angle)


def acceleration_metrics(q):
    rotvec = relative_rotvec(q[:-1], q[1:])
    omega_body = rotvec / DT
    omega_world = Rotation.from_quat(q[:-1]).apply(omega_body)

    alpha = np.diff(omega_world, axis=0) / DT
    alpha_mag = np.linalg.norm(alpha, axis=-1)
    speed = np.linalg.norm(omega_world, axis=-1)

    return {
        "mean_angular_speed_rad_s": float(np.mean(speed)),
        "rms_angular_accel_rad_s2": float(np.sqrt(np.mean(alpha_mag**2))),
        "p95_angular_accel_rad_s2": float(np.percentile(alpha_mag, 95)),
        "max_angular_accel_rad_s2": float(np.max(alpha_mag)),
    }


def load_segments_v3(h5_path: Path):
    segment_map = {}
    with h5py.File(h5_path, "r") as f:
        if 'users' not in f:
            return segment_map
        users_grp = f['users']
        for user_id in users_grp.keys():
            user_sub = users_grp[user_id]
            if 'sessions' not in user_sub:
                continue
            sessions_grp = user_sub['sessions']
            for session_id in sessions_grp.keys():
                session_item = sessions_grp[session_id]
                if 'segments' not in session_item:
                    continue
                segments_grp = session_item['segments']
                for seg_id in segments_grp.keys():
                    seg_item = segments_grp[seg_id]
                    if 'quaternions' in seg_item:
                        quats = seg_item['quaternions'][:]
                        segment_map[(str(user_id), str(session_id), str(seg_id))] = quats
    return segment_map


def main():
    parser = argparse.ArgumentParser(description="Phân tích tương quan phi tuyến gia tốc vs sai số cho mọi mô hình.")
    parser.add_argument("--model", type=str, required=True, help="Tên mô hình (vd: gru, res_mlp, jepa, tcn)")
    args = parser.parse_args()

    pred_path = OUT_DIR / f"{args.model}_val_seed42_predictions.npz"
    if not pred_path.exists():
        # Thử tìm các tên file dự đoán phổ biến khác nếu có
        alt_path = OUT_DIR / f"{args.model}_predictions.npz"
        if alt_path.exists():
            pred_path = alt_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dự đoán cho model '{args.model}' tại {pred_path}. Hãy đảm bảo bạn đã xuất file .npz dự đoán của mô hình này.")

    OUT_DIR.mkdir(exist_ok=True)
    data = np.load(pred_path, allow_pickle=True)
    pred = data["pred_quat"]
    true = data["true_quat"]

    errors = geodesic_error_deg(pred[:, -1], true[:, -1])

    users = data["user_id"].astype(str)
    sessions = data["session_id"].astype(str)
    sequences = data["sequence_name"].astype(str) if "sequence_name" in data.files else np.array(["0000"] * len(users))
    starts = data["start_index"].astype(int)

    print(f"📦 Đang nạp segment từ HDF5 v3 cho mô hình: [{args.model.upper()}]")
    segment_map = load_segments_v3(H5_PATH)

    rows = []
    missing = 0

    for i in range(len(errors)):
        key = (users[i], sessions[i], sequences[i])
        quats = segment_map.get(key)

        if quats is None:
            matched = False
            for (u, s, seg), q_data in segment_map.items():
                if u == users[i] and s == sessions[i]:
                    quats = q_data
                    matched = True
                    break
            if not matched:
                missing += 1
                continue

        start = starts[i]
        context_q = quats[start:start + CONTEXT]

        if len(context_q) != CONTEXT:
            missing += 1
            continue

        metrics = acceleration_metrics(context_q)
        rows.append({
            "user_id": users[i],
            "session_id": sessions[i],
            "sequence_name": sequences[i],
            "start_index": starts[i],
            "error_deg": float(errors[i]),
            **metrics,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Không khớp được window nào.")

    out_csv = OUT_DIR / f"{args.model}_nonlinear_acceleration_analysis.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "="*60)
    print(f"📊 BÁO CÁO PHÂN TÍCH TƯƠNG QUAN PHI TUYẾN - MODEL: [{args.model.upper()}]")
    print("="*60)
    print(f"Tổng số window khớp: {len(df)} | Bị bỏ qua: {missing}")

    print("\n--- THỐNG KÊ SAI SỐ STEP-20 ---")
    print(f"  • Mean error   : {df.error_deg.mean():.4f} deg")
    print(f"  • Median error : {df.error_deg.median():.4f} deg")
    print(f"  • P95 error    : {df.error_deg.quantile(0.95):.4f} deg")

    print("\n--- ĐÁNH GIÁ ĐA CHỈ SỐ (Pearson, Spearman & Mutual Information) ---")
    metrics_to_test = [
        "rms_angular_accel_rad_s2",
        "p95_angular_accel_rad_s2",
        "max_angular_accel_rad_s2",
        "mean_angular_speed_rad_s",
    ]

    y = df["error_deg"].to_numpy()

    for m in metrics_to_test:
        x = df[m].to_numpy()
        if np.std(x) < 1e-12:
            continue

        r, p_val = pearsonr(x, y)
        rho, sp_val = spearmanr(x, y)
        mi = mutual_info_regression(x[:, None], y, random_state=42)[0]

        print(f"\nMetric: {m}")
        print(f"  • Pearson r  = {r:.4f} (p = {p_val:.3g})")
        print(f"  • Spearman   = {rho:.4f} (p = {sp_val:.3g})")
        print(f"  • Mutual Inf = {mi:.4f}")

    print("\n--- PHÂN TÍCH SAI SỐ THEO NHÓM GIA TỐC (QUARTILE BINNING) ---")
    boundaries = np.quantile(df["rms_angular_accel_rad_s2"], [0.25, 0.50, 0.75])
    df["quartile"] = pd.cut(
        df["rms_angular_accel_rad_s2"], 
        bins=[-np.inf, boundaries[0], boundaries[1], boundaries[2], np.inf],
        labels=["Q1 (Thấp)", "Q2 (Trung bình thấp)", "Q3 (Trung bình cao)", "Q4 (Rất cao)"]
    )

    q_summary = df.groupby("quartile", observed=False).agg(
        windows=("error_deg", "size"),
        mean_accel=("rms_angular_accel_rad_s2", "mean"),
        mean_error=("error_deg", "mean"),
        median_error=("error_deg", "median")
    )
    print(q_summary.to_string())
    print(f"\n💾 Đã lưu kết quả chi tiết tại: {out_csv}")


if __name__ == "__main__":
    main()