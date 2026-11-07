# 9th ABC Challenge — head-pose prediction

Dự đoán **20 quaternion** tương lai (0,2 s tại 100 Hz) từ tối đa **400 mẫu** ngữ cảnh (4 s). Model phải nhân quả, không fine-tune theo user, và xuất được ONNX/PyTorch.

Metric chính thức (accuracy): **trung bình góc geodesic tại mẫu thứ 20**, đơn vị độ, mỗi cửa sổ một lần:

`θ = (180/π) · 2 · arccos(|⟨q̂, q⟩|)`

`q` và `-q` cùng một phép quay. Tập test của ST không phát hành; số local là user-held-out, không phải điểm challenge.

## Pipeline dữ liệu

Session filtering (rate 85–110 Hz, sweep > 17°, velocity > 15°/s) **không** dùng ngưỡng 420 mẫu. Training usability chỉ sau clean + cắt gap + resample 100 Hz: segment ≥ 420 mẫu.

```bash
python -m src.prep.pipeline download --start 0
python -m src.prep.pipeline filter-local --raw-dir data/raw
python -m src.prep.pipeline migrate-legacy --h5 data/processed/head_pose.h5
python -m src.prep.pipeline report
python -m src.prep.pipeline compare
pytest
```

HDF5: `data/processed/head_pose_v2.h5` theo `users/{id}/sessions/{id}/segments/{n}`. Không lưu window. Split theo user, seed cố định.

## Train / eval / infer

```bash
python -m src.training.train --h5 data/processed/head_pose_v2.h5 --epochs 8 --context 400 --horizon 20
python -m src.evaluation.evaluate --checkpoint experiments/.../best.pt --split from-checkpoint
python -m src.inference.predict --checkpoint experiments/.../best.pt --input context.npy --out predictions.json
```

Baseline bắt buộc: last-pose (persistence) và constant angular velocity. Model học (GRU) chỉ được giữ nếu bước 20 trên **validation users** thấp hơn constant-velocity. Test users trong `split.json` không dùng để chọn hyperparameter.

## Nộp bài challenge

1. `best.pt` + `head_pose_gru.onnx` (batch 1, input `delta_rotvec` `(1, context-1, 3)`).
2. Ghi chú load/inference trên Linux+CUDA (script `src/inference/predict.py`).
3. Bài báo IEEE 4–6 trang hoặc IJABC 8–10 trang — chưa có trong repo.
4. Link Google Drive — do team nộp.

ST đo latency trên phần cứng XR. Repo không giả lập số đó.
