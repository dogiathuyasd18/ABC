# 9th ABC Challenge — head-pose prediction

Dự đoán **20 quaternion** tư thế đầu trong tương lai (100 ms tại 100 Hz) từ cửa sổ ngữ cảnh tối đa 4 giây. Model phải nhân quả, không fine-tune theo từng người, và chạy được trên nền tảng XR.

Metric chính là **góc geodesic trung bình tại mẫu thứ 20**, tính bằng độ:

`θ = (180/π) * 2 * arccos(|⟨q̂, q⟩|)`

Mỗi cửa sổ đánh giá tính một lần (window-weighted). Tách dữ liệu theo **user**, không theo cửa sổ.

## Dữ liệu đang có

`data/processed/head_pose.h5` là lát cắt đã lọc từ BOXRR-23 (quaternion `x,y,z,w`, đã chuẩn hoá và chỉnh hemisphere). Đây chưa phải corpus ~40 GB của toàn bộ challenge. Loader đưa mỗi phiên lên lưới đều **100 Hz** bằng SLERP và cắt đoạn nếu khoảng trống dài hơn 0,5 giây.

## Chạy

Dùng môi trường đã có PyTorch, h5py, scipy, numpy:

```bash
python -m src.preprocessing.validate_dataset
python -m src.training.train --epochs 3 --stride 40 --context 100
```

Kết quả ghi vào `checkpoints/`:

- `head_pose_gru.pt` — trọng số và danh sách user train/val
- `head_pose_gru.onnx` — đầu vào `delta_rotvec` dạng `(batch, context-1, 3)`, đầu ra 20 rotation vector trong hệ thân của quaternion cuối
- `metrics.json` — sai số validation tại mẫu 20, cạnh baseline vận tốc không đổi

Baseline lặp bước góc cuối cùng: mẫu thứ `k` là bước đó nhân `k`. Model chỉ đáng giữ nếu sai số mẫu 20 thấp hơn baseline này trên user chưa gặp.

Suy luận: quaternion ngữ cảnh `x,y,z,w` → rotation vector tương đối giữa các mẫu liên tiếp → GRU → 20 rotation vector → nhân phải vào quaternion cuối.
