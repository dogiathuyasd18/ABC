
import numpy as np
import pandas as pd

THRESHOLD = 4.84
HORIZON_INDEX = 19  # Bước dự đoán thứ 20, index bắt đầu từ 0

# Load kết quả dự đoán
data = np.load("outputs/gru_predictions.npz", allow_pickle=True)

pred = data["pred_quat"]
true = data["true_quat"]
user_ids = data["user_id"]

session_ids = (
    data["session_id"]
    if "session_id" in data.files
    else np.array(["unknown"] * len(pred))
)

assert pred.shape == true.shape
assert pred.ndim == 3 and pred.shape[1] >= 20
assert len(user_ids) == len(pred)

# Chuẩn hóa quaternion
def normalize_quat(q):
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-12, None)

pred = normalize_quat(pred[:, HORIZON_INDEX, :])
true = normalize_quat(true[:, HORIZON_INDEX, :])

# Sai số góc geodesic (độ)
# Công thức dùng trị tuyệt đối dot product để xử lý q và -q
dot = np.sum(pred * true, axis=-1)
dot = np.clip(np.abs(dot), 0.0, 1.0)

errors = 2 * np.arccos(dot) * 180 / np.pi

# Tạo bảng kết quả
df = pd.DataFrame({
    "sample_index": np.arange(len(errors)),
    "user_id": user_ids,
    "session_id": session_ids,
    "error_deg": errors
})

# Lọc các mẫu sai số > 4.84 độ
bad = df[df["error_deg"] > THRESHOLD].copy()
bad = bad.sort_values("error_deg", ascending=False)

print(f"Tổng số mẫu: {len(df)}")
print(f"Số mẫu lỗi > {THRESHOLD}°: {len(bad)}")
print(f"Tỷ lệ: {len(bad) / len(df) * 100:.2f}%")

# Thống kê lỗi theo từng người
user_stats = (
    bad.groupby("user_id")
    .agg(
        bad_samples=("error_deg", "count"),
        mean_error=("error_deg", "mean"),
        max_error=("error_deg", "max"),
        sessions=("session_id", "nunique")
    )
    .sort_values("bad_samples", ascending=False)
)

print("\n=== THỐNG KÊ THEO USER ===")
print(user_stats.to_string())

# So sánh tỷ lệ lỗi của từng người trên toàn bộ mẫu
all_stats = (
    df.groupby("user_id")
    .agg(
        total_samples=("error_deg", "count"),
        bad_samples=("error_deg", lambda x: (x > THRESHOLD).sum()),
        mean_error=("error_deg", "mean")
    )
)

all_stats["bad_rate_percent"] = (
    all_stats["bad_samples"] /
    all_stats["total_samples"] * 100
)

print("\n=== TỶ LỆ LỖI THEO USER ===")
print(
    all_stats.sort_values(
        "bad_rate_percent", ascending=False
    ).to_string()
)

# Xuất kết quả
bad.to_csv("gru_bad_samples.csv", index=False)
user_stats.to_csv("gru_bad_by_user.csv")
all_stats.to_csv("gru_all_user_stats.csv")

print("\nĐã xuất:")
print("- gru_bad_samples.csv")
print("- gru_bad_by_user.csv")
print("- gru_all_user_stats.csv")
