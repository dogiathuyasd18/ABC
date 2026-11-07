# 9th ABC Challenge — head-pose prediction

Dự đoán **20 quaternion** tư thế đầu (100 ms tại 100 Hz) từ tối đa 400 mẫu ngữ cảnh. Metric chính là góc geodesic trung bình **tại mẫu thứ 20**.

## Tách hai loại điều kiện

| Bước | Quyết định | Không dùng |
|---|---|---|
| **Session filtering** | sampling rate 85–110 Hz, sweep > 17°, velocity > 15°/s (spec `9th-abc-challenge-data-page`) | số sample 420 |
| **Training usability** | sau clean + cắt gap + resample 100 Hz, segment có ≥ 420 mẫu (400 + 20) | resampling để “cứu” session sai tần số |

Ngưỡng gap nội suy là **0.04 s** (không phải 0.5 s): chỉ SLERP các dropout ngắn; gap dài hơn tách thành segment độc lập. Cửa sổ train không đi xuyên session hay xuyên gap.

## Pipeline

```bash
python -m src.prep.pipeline download --start 0
python -m src.prep.pipeline filter-local --raw-dir data/raw
python -m src.prep.pipeline report
python -m src.prep.pipeline compare --h5 data/processed/head_pose.h5
pytest
```

Download ghi checkpoint vào `data/raw/download_state.json`, retry, resume. Session PASS được ghi HDF5 và đọc lại checksum **trước** khi xóa TAR.

## HDF5

- Trung gian / kết quả: `data/processed/head_pose_v2.h5` theo `users/{id}/sessions/{id}/segments/{n}`
- Không lưu bản sao window. Dataset tự lập chỉ số cửa sổ.
- Legacy `data/processed/head_pose.h5` vẫn đọc được; loader sẽ resample theo ngưỡng gap mới.

```bash
python -m src.preprocessing.validate_dataset
python -m src.training.train --epochs 3 --context 400 --horizon 20 --stride 20
```

Báo cáo: `reports/data_quality_report.md`, `reports/prep_before_after.md`.
