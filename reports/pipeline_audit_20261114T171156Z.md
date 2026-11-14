# ABC preprocessing pipeline audit

- Generated (UTC): 2026-11-14T17:11:56.891893+00:00
- PASS: 9 | WARN: 6 | FAIL: 0 | NOT_CHECKED: 0

> PASS means directly verified by this script. WARN/NOT_CHECKED items are not proof of correctness. Review all FAIL items before using the dataset for training.

| Status | Check | Result |
|---|---|---|
| **PASS** | Required preprocessing files | All expected preprocessing files exist. |
| **PASS** | Python compilation | compileall completed without syntax errors. |
| **PASS** | Preprocessing imports | All expected preprocessing modules imported. |
| **WARN** | Challenge filter configuration | Threshold-like values were found in config.py, but this does not prove that the filtering code applies them correctly. |
| **PASS** | Session CSV logs | Recognized CSV logs are readable and have consistent row widths. |
| **PASS** | PASS session key integrity | 3055 rows; duplicate (user, session) keys: 0. |
| **WARN** | Rejected-session accounting | No recognized reject log found. PASS-only CSV cannot establish how many sessions were rejected or why. |
| **PASS** | HDF5 structural/data checks: head_pose.h5 | head_pose.h5: 386 datasets; 0 failure(s), 0 warning(s). |
| **PASS** | HDF5 structural/data checks: head_pose_v2.h5 | head_pose_v2.h5: 658 datasets; 0 failure(s), 0 warning(s). |
| **WARN** | Canonical HDF5 selection | Both head_pose.h5 and head_pose_v2.h5 exist. Select one canonical training artifact after comparing schemas and contents. |
| **PASS** | Pipeline storage cap | Tracked pipeline-managed files are within the configured cap. |
| **PASS** | Free disk reserve | Free disk space meets the configured safety reserve. |
| **WARN** | Source markers: src/prep/resampling.py | Expected implementation markers not found; inspect manually. |
| **WARN** | Source markers: src/prep/session.py | Markers found, but source-text checks do not prove correctness. |
| **WARN** | Source markers: src/prep/windows.py | Markers found, but source-text checks do not prove correctness. |

## Evidence

### PASS — Required preprocessing files

All expected preprocessing files exist.

```json
[
  "src/prep/session.py",
  "src/prep/resampling.py",
  "src/prep/windows.py",
  "src/prep/store_log.py",
  "src/prep/download.py"
]
```

### PASS — Python compilation

compileall completed without syntax errors.

### PASS — Preprocessing imports

All expected preprocessing modules imported.

### WARN — Challenge filter configuration

Threshold-like values were found in config.py, but this does not prove that the filtering code applies them correctly.

```json
{
  "detected": {
    "sampling-rate lower bound 85 Hz": true,
    "sampling-rate upper bound 110 Hz": true,
    "minimum sweep 17 degrees": true,
    "minimum velocity 15 degrees/second": true
  },
  "default_configs": {
    "FilterConfig": {
      "spec_version": "9th-abc-challenge-data-page",
      "trim_seconds": 5.0,
      "motion_window_seconds": 5.0,
      "min_samples_after_trim": 20,
      "min_valid_samples_per_window": 10,
      "min_sampling_rate_hz": 85.0,
      "max_sampling_rate_hz": 110.0,
      "min_sweep_degrees": 17.0,
      "min_velocity_deg_s": 15.0,
      "sweep_comparison": "greater_than",
      "velocity_comparison": "greater_than",
      "euler_order": "xyz",
      "quaternion_axes_xror": [
        "i",
        "j",
        "k",
        "1"
      ]
    },
    "CleanConfig": {
      "preprocessing_version": "2.0",
      "near_zero_norm": 1e-08,
      "unit_norm_atol": 1e-05,
      "duplicate_quat_atol": 1e-06,
      "timestamp_eps": 1e-09,
      "dropout_interval_seconds": 0.023529411764705882,
      "discontinuity_degrees": 30.0,
      "max_interpolate_gap_seconds": 0.04,
      "gap_threshold_rationale": "Challenge: SLERP short dropouts; split longer ones; do not invent half a second of motion. Native band 85–110 Hz ⇒ tick 9.1–11.8 ms. 0.04 s ≤ three missed ticks at 85 Hz. Gaps above this start a new segment.",
      "target_sample_rate_hz": 100.0,
      "do_not_sort_timestamps": true,
      "do_not_extrapolate": true,
      "do_not_delete_fast_motion": true
    },
    "WindowConfig": {
      "context_samples": 400,
      "horizon_samples": 20,
      "stride": 20
    }
  }
}
```

### PASS — Session CSV logs

Recognized CSV logs are readable and have consistent row widths.

```json
{
  "data/processed/passed_sessions.csv": {
    "rows": 3055,
    "columns": [
      "archive",
      "user_id",
      "session_id",
      "member_name",
      "frames_after_trim",
      "sampling_rate",
      "sampling_rate_valid",
      "num_valid_windows",
      "sweep_x",
      "sweep_y",
      "sweep_z",
      "velocity_x",
      "velocity_y",
      "velocity_z",
      "session_sweep",
      "session_velocity",
      "sweep_valid",
      "velocity_valid",
      "session_valid",
      "status",
      "reject_reason"
    ]
  },
  "data/processed/hdf5_errors.csv": {
    "rows": 19,
    "columns": [
      "user_id",
      "session_id",
      "member_name",
      "error"
    ]
  }
}
```

### PASS — PASS session key integrity

3055 rows; duplicate (user, session) keys: 0.

```json
{
  "columns": [
    "archive",
    "user_id",
    "session_id",
    "member_name",
    "frames_after_trim",
    "sampling_rate",
    "sampling_rate_valid",
    "num_valid_windows",
    "sweep_x",
    "sweep_y",
    "sweep_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "session_sweep",
    "session_velocity",
    "sweep_valid",
    "velocity_valid",
    "session_valid",
    "status",
    "reject_reason"
  ],
  "duplicate_keys": 0
}
```

### WARN — Rejected-session accounting

No recognized reject log found. PASS-only CSV cannot establish how many sessions were rejected or why.

### PASS — HDF5 structural/data checks: head_pose.h5

head_pose.h5: 386 datasets; 0 failure(s), 0 warning(s).

```json
{
  "evidence": {
    "size_bytes": 35210991,
    "datasets": [
      {
        "path": "users/00743d56-a912-4ec7-a9d4-3e6e22756488/sessions/005b994e-5330-48af-859a-ae8beaed8b91/quaternions",
        "shape": [
          14269,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00743d56-a912-4ec7-a9d4-3e6e22756488/sessions/005b994e-5330-48af-859a-ae8beaed8b91/timestamps",
        "shape": [
          14269
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007a16e6-6862-40f1-b7ee-07c959e4ac92/sessions/832329ed-e4e8-4ba4-a3a0-529a3fd31636/quaternions",
        "shape": [
          19325,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007a16e6-6862-40f1-b7ee-07c959e4ac92/sessions/832329ed-e4e8-4ba4-a3a0-529a3fd31636/timestamps",
        "shape": [
          19325
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007a16e6-6862-40f1-b7ee-07c959e4ac92/sessions/a410b074-9b79-4335-a0bf-a42e9af944b0/quaternions",
        "shape": [
          509,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007a16e6-6862-40f1-b7ee-07c959e4ac92/sessions/a410b074-9b79-4335-a0bf-a42e9af944b0/timestamps",
        "shape": [
          509
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/0e515c8e-3241-4e7d-ad91-516c9aef7f6c/quaternions",
        "shape": [
          913,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/0e515c8e-3241-4e7d-ad91-516c9aef7f6c/timestamps",
        "shape": [
          913
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/11bd715b-e6ab-40cb-aff1-81d2160d2e5e/quaternions",
        "shape": [
          15585,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/11bd715b-e6ab-40cb-aff1-81d2160d2e5e/timestamps",
        "shape": [
          15585
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/18f668eb-fa6b-475b-aefc-76bf2b91a863/quaternions",
        "shape": [
          17325,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/18f668eb-fa6b-475b-aefc-76bf2b91a863/timestamps",
        "shape": [
          17325
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/1c3cca2d-af0f-4fa3-9c1c-584bbc39f441/quaternions",
        "shape": [
          1836,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/1c3cca2d-af0f-4fa3-9c1c-584bbc39f441/timestamps",
        "shape": [
          1836
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/1c7c66e0-a1b5-4eb3-8c1b-51068921f348/quaternions",
        "shape": [
          12445,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/1c7c66e0-a1b5-4eb3-8c1b-51068921f348/timestamps",
        "shape": [
          12445
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/22d3d598-4957-4958-9051-a74ffae55b50/quaternions",
        "shape": [
          18797,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/22d3d598-4957-4958-9051-a74ffae55b50/timestamps",
        "shape": [
          18797
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/27eadf41-b990-4a33-9f07-74672d5cf8d8/quaternions",
        "shape": [
          4827,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/27eadf41-b990-4a33-9f07-74672d5cf8d8/timestamps",
        "shape": [
          4827
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/389a2476-4c0a-4ab4-8222-e505399252f0/quaternions",
        "shape": [
          7345,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/389a2476-4c0a-4ab4-8222-e505399252f0/timestamps",
        "shape": [
          7345
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/42cb613e-5607-49cd-8c96-ead309561748/quaternions",
        "shape": [
          799,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/42cb613e-5607-49cd-8c96-ead309561748/timestamps",
        "shape": [
          799
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/4f219cbb-6c08-413e-aa17-f57e09dd2f86/quaternions",
        "shape": [
          20968,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/4f219cbb-6c08-413e-aa17-f57e09dd2f86/timestamps",
        "shape": [
          20968
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/540f99ba-f6e2-43ef-b0bc-7391c9f89823/quaternions",
        "shape": [
          5015,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/540f99ba-f6e2-43ef-b0bc-7391c9f89823/timestamps",
        "shape": [
          5015
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/54965afb-f0f2-4d6b-b4c0-cb7fc9de5988/quaternions",
        "shape": [
          18285,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/54965afb-f0f2-4d6b-b4c0-cb7fc9de5988/timestamps",
        "shape": [
          18285
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/55ba83aa-de1d-42fa-b5ad-0d543df5d3f8/quaternions",
        "shape": [
          1757,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/55ba83aa-de1d-42fa-b5ad-0d543df5d3f8/timestamps",
        "shape": [
          1757
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/638377f0-3c4c-494d-8b40-99b15260c5ea/quaternions",
        "shape": [
          16129,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/638377f0-3c4c-494d-8b40-99b15260c5ea/timestamps",
        "shape": [
          16129
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/681fdec1-df69-40ed-a9f9-e09aaa13ab6c/quaternions",
        "shape": [
          22255,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/681fdec1-df69-40ed-a9f9-e09aaa13ab6c/timestamps",
        "shape": [
          22255
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/701fd8c5-fc3f-4d53-b7bc-a211131a6a28/quaternions",
        "shape": [
          10476,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/701fd8c5-fc3f-4d53-b7bc-a211131a6a28/timestamps",
        "shape": [
          10476
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/77f3d6b8-0931-4c9c-a1c1-77b148a13f9a/quaternions",
        "shape": [
          6683,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/77f3d6b8-0931-4c9c-a1c1-77b148a13f9a/timestamps",
        "shape": [
          6683
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/7b8adb4e-c563-4d80-b939-1b9acb665d97/quaternions",
        "shape": [
          12679,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/7b8adb4e-c563-4d80-b939-1b9acb665d97/timestamps",
        "shape": [
          12679
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/7ef0137d-b728-492b-86a9-7b7c49997d2d/quaternions",
        "shape": [
          4547,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/7ef0137d-b728-492b-86a9-7b7c49997d2d/timestamps",
        "shape": [
          4547
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/88495478-a9b5-4413-bd5b-805f7c008841/quaternions",
        "shape": [
          979,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/88495478-a9b5-4413-bd5b-805f7c008841/timestamps",
        "shape": [
          979
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/887d9762-3b4b-4f82-8a4a-92bec34b6103/quaternions",
        "shape": [
          17979,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/887d9762-3b4b-4f82-8a4a-92bec34b6103/timestamps",
        "shape": [
          17979
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/8a55947d-4d17-4e49-bb52-9722c42e8299/quaternions",
        "shape": [
          16943,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/8a55947d-4d17-4e49-bb52-9722c42e8299/timestamps",
        "shape": [
          16943
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/902aaaea-1c56-47b9-a02e-aa04c6215390/quaternions",
        "shape": [
          18702,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/902aaaea-1c56-47b9-a02e-aa04c6215390/timestamps",
        "shape": [
          18702
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/92fa43d7-2f2f-4d6f-8b59-15864fadc81b/quaternions",
        "shape": [
          15730,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/92fa43d7-2f2f-4d6f-8b59-15864fadc81b/timestamps",
        "shape": [
          15730
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/93aae0b0-7424-49a2-990b-6dcaead0e1d4/quaternions",
        "shape": [
          1055,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/93aae0b0-7424-49a2-990b-6dcaead0e1d4/timestamps",
        "shape": [
          1055
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/97b75f77-deb7-42ec-8bb3-35671edcb98a/quaternions",
        "shape": [
          16917,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/97b75f77-deb7-42ec-8bb3-35671edcb98a/timestamps",
        "shape": [
          16917
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/9c8d3e9e-c3cf-47cd-b9ff-7bfcef0ae4ab/quaternions",
        "shape": [
          16167,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/9c8d3e9e-c3cf-47cd-b9ff-7bfcef0ae4ab/timestamps",
        "shape": [
          16167
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/9dbc71c7-7355-4305-89fd-e9e758e2d893/quaternions",
        "shape": [
          15489,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/9dbc71c7-7355-4305-89fd-e9e758e2d893/timestamps",
        "shape": [
          15489
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/a6ad17ca-d396-4ec3-8a68-9bdf6857c6be/quaternions",
        "shape": [
          14927,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/a6ad17ca-d396-4ec3-8a68-9bdf6857c6be/timestamps",
        "shape": [
          14927
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/a9481103-32ec-4afa-918d-8d5fffd295ce/quaternions",
        "shape": [
          686,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/a9481103-32ec-4afa-918d-8d5fffd295ce/timestamps",
        "shape": [
          686
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/aab7ae2e-7f78-468e-88b6-6808e04b4791/quaternions",
        "shape": [
          2705,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/aab7ae2e-7f78-468e-88b6-6808e04b4791/timestamps",
        "shape": [
          2705
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/b559863c-d366-4b4c-af59-9b07fdfc033a/quaternions",
        "shape": [
          25714,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/b559863c-d366-4b4c-af59-9b07fdfc033a/timestamps",
        "shape": [
          25714
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/cfe696ff-e750-4796-992c-8bd950994a04/quaternions",
        "shape": [
          20876,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/cfe696ff-e750-4796-992c-8bd950994a04/timestamps",
        "shape": [
          20876
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/d50b7ab6-643a-4aae-9d4e-bc4b779dee14/quaternions",
        "shape": [
          10965,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/d50b7ab6-643a-4aae-9d4e-bc4b779dee14/timestamps",
        "shape": [
          10965
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/d80cb020-b6a1-4db5-a2f7-ccebb6c44697/quaternions",
        "shape": [
          14563,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/d80cb020-b6a1-4db5-a2f7-ccebb6c44697/timestamps",
        "shape": [
          14563
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/dc002a1f-ec5b-415b-a842-2bd6d66deabd/quaternions",
        "shape": [
          4459,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/dc002a1f-ec5b-415b-a842-2bd6d66deabd/timestamps",
        "shape": [
          4459
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/df4959aa-4196-43e4-9bff-ef036af8abbe/quaternions",
        "shape": [
          22430,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/df4959aa-4196-43e4-9bff-ef036af8abbe/timestamps",
        "shape": [
          22430
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/e15379f1-f0bd-48e7-a887-5b3783b06784/quaternions",
        "shape": [
          17587,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/e15379f1-f0bd-48e7-a887-5b3783b06784/timestamps",
        "shape": [
          17587
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/e46471bb-b56a-4fa9-83d2-032069f83728/quaternions",
        "shape": [
          14243,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/e46471bb-b56a-4fa9-83d2-032069f83728/timestamps",
        "shape": [
          14243
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/efcdfedc-d978-48ea-9e57-d92385d47549/quaternions",
        "shape": [
          2842,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/efcdfedc-d978-48ea-9e57-d92385d47549/timestamps",
        "shape": [
          2842
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/fcacc95f-8f06-4aec-b49d-827cc5e75ee4/quaternions",
        "shape": [
          10910,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/fcacc95f-8f06-4aec-b49d-827cc5e75ee4/timestamps",
        "shape": [
          10910
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/454f5fcd-c194-45d2-90b2-d6a7bba9bc04/quaternions",
        "shape": [
          8585,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/454f5fcd-c194-45d2-90b2-d6a7bba9bc04/timestamps",
        "shape": [
          8585
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/52875a99-9f93-4a6d-bd75-bc0048288f65/quaternions",
        "shape": [
          12533,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/52875a99-9f93-4a6d-bd75-bc0048288f65/timestamps",
        "shape": [
          12533
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/bd2265bc-5e0b-47ba-8d7d-3cc958245f22/quaternions",
        "shape": [
          1624,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/bd2265bc-5e0b-47ba-8d7d-3cc958245f22/timestamps",
        "shape": [
          1624
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/c8544adc-51e8-41a6-ba11-4323a46df959/quaternions",
        "shape": [
          12007,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/c8544adc-51e8-41a6-ba11-4323a46df959/timestamps",
        "shape": [
          12007
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/e4f34f34-2e8b-4a72-802d-d835d1a8ffd8/quaternions",
        "shape": [
          1575,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/e4f34f34-2e8b-4a72-802d-d835d1a8ffd8/timestamps",
        "shape": [
          1575
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/f9a55a06-88d0-4c51-8b43-4b60cc900e33/quaternions",
        "shape": [
          1839,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/f9a55a06-88d0-4c51-8b43-4b60cc900e33/timestamps",
        "shape": [
          1839
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/quaternions",
        "shape": [
          20466,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/timestamps",
        "shape": [
          20466
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/0a7c6042-c152-4df7-8db1-cbba1033e4c6/quaternions",
        "shape": [
          16261,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/0a7c6042-c152-4df7-8db1-cbba1033e4c6/timestamps",
        "shape": [
          16261
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/0ba5d8f0-562f-41f3-8e15-8b8dba147e3d/quaternions",
        "shape": [
          8325,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/0ba5d8f0-562f-41f3-8e15-8b8dba147e3d/timestamps",
        "shape": [
          8325
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/1437c2d3-70fc-49a3-9349-63bfa3d19c41/quaternions",
        "shape": [
          15121,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/1437c2d3-70fc-49a3-9349-63bfa3d19c41/timestamps",
        "shape": [
          15121
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/286ae5b6-27e5-477e-ad5c-a41590f09729/quaternions",
        "shape": [
          7703,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/286ae5b6-27e5-477e-ad5c-a41590f09729/timestamps",
        "shape": [
          7703
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/44e4c72a-90c6-46cb-82b6-8bb009af0ce6/quaternions",
        "shape": [
          15506,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/44e4c72a-90c6-46cb-82b6-8bb009af0ce6/timestamps",
        "shape": [
          15506
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/58ad7b03-7ed7-43c3-94ec-a8416a6f64d1/quaternions",
        "shape": [
          7670,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/58ad7b03-7ed7-43c3-94ec-a8416a6f64d1/timestamps",
        "shape": [
          7670
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/af5b5890-bc28-47b9-bbb2-cc76da0e95b9/quaternions",
        "shape": [
          18287,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/af5b5890-bc28-47b9-bbb2-cc76da0e95b9/timestamps",
        "shape": [
          18287
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/d3de1c53-2e9c-4986-a2ff-c63e4f04f9d6/quaternions",
        "shape": [
          16551,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/d3de1c53-2e9c-4986-a2ff-c63e4f04f9d6/timestamps",
        "shape": [
          16551
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/df175928-a3df-4ce0-9d61-efe26e26a90d/quaternions",
        "shape": [
          7709,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/df175928-a3df-4ce0-9d61-efe26e26a90d/timestamps",
        "shape": [
          7709
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/e3096cd3-55d0-4be0-9cc9-08ad473f84c0/quaternions",
        "shape": [
          7802,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/e3096cd3-55d0-4be0-9cc9-08ad473f84c0/timestamps",
        "shape": [
          7802
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/e579faa9-b3f4-4010-80d0-f90cc4e847fc/quaternions",
        "shape": [
          14974,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/e579faa9-b3f4-4010-80d0-f90cc4e847fc/timestamps",
        "shape": [
          14974
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/f83ea77a-45d0-47fc-bd72-aa5089cf8a01/quaternions",
        "shape": [
          21545,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/f83ea77a-45d0-47fc-bd72-aa5089cf8a01/timestamps",
        "shape": [
          21545
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/ffea28fc-ad4f-43c1-9a5c-438e824d9ca7/quaternions",
        "shape": [
          21525,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/ffea28fc-ad4f-43c1-9a5c-438e824d9ca7/timestamps",
        "shape": [
          21525
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/036e81b9-ae98-404e-90b0-44b76921d3c0/quaternions",
        "shape": [
          1833,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/036e81b9-ae98-404e-90b0-44b76921d3c0/timestamps",
        "shape": [
          1833
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/06665804-68bf-4952-a510-3be1689063eb/quaternions",
        "shape": [
          19544,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/06665804-68bf-4952-a510-3be1689063eb/timestamps",
        "shape": [
          19544
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/08bd7993-dc8e-442b-acc3-03488fae1bc6/quaternions",
        "shape": [
          18541,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/08bd7993-dc8e-442b-acc3-03488fae1bc6/timestamps",
        "shape": [
          18541
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/18d518e7-6fa8-4018-8f03-2007509ab065/quaternions",
        "shape": [
          22577,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/18d518e7-6fa8-4018-8f03-2007509ab065/timestamps",
        "shape": [
          22577
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/1ab0c277-768c-4a35-8404-c39be48f8dba/quaternions",
        "shape": [
          20729,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/1ab0c277-768c-4a35-8404-c39be48f8dba/timestamps",
        "shape": [
          20729
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/1beb2a2b-4a92-471e-b0db-908e2fa0835e/quaternions",
        "shape": [
          18179,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/1beb2a2b-4a92-471e-b0db-908e2fa0835e/timestamps",
        "shape": [
          18179
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/36a3b584-9e70-4c04-8a8b-bb6835af4b7e/quaternions",
        "shape": [
          17505,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/36a3b584-9e70-4c04-8a8b-bb6835af4b7e/timestamps",
        "shape": [
          17505
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/3720d7ec-0910-49c8-b79c-b87f5b0e9912/quaternions",
        "shape": [
          21887,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/3720d7ec-0910-49c8-b79c-b87f5b0e9912/timestamps",
        "shape": [
          21887
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/3c41b90d-7ea3-4a14-ade4-29d4d22702da/quaternions",
        "shape": [
          10633,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/3c41b90d-7ea3-4a14-ade4-29d4d22702da/timestamps",
        "shape": [
          10633
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/42bc72f5-90ae-4a02-bbc1-ba501eb585fb/quaternions",
        "shape": [
          17456,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/42bc72f5-90ae-4a02-bbc1-ba501eb585fb/timestamps",
        "shape": [
          17456
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/43e3b45d-a05d-452b-a03a-59288b57699a/quaternions",
        "shape": [
          12070,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/43e3b45d-a05d-452b-a03a-59288b57699a/timestamps",
        "shape": [
          12070
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/449e141d-97bc-4f00-823b-7861c10d53b1/quaternions",
        "shape": [
          17039,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/449e141d-97bc-4f00-823b-7861c10d53b1/timestamps",
        "shape": [
          17039
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4a368f4d-833a-44d2-82b4-8cde3d3ebed5/quaternions",
        "shape": [
          15398,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4a368f4d-833a-44d2-82b4-8cde3d3ebed5/timestamps",
        "shape": [
          15398
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4be4c058-43ed-4009-be4d-9b0b336ccdfb/quaternions",
        "shape": [
          15939,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4be4c058-43ed-4009-be4d-9b0b336ccdfb/timestamps",
        "shape": [
          15939
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4dff667e-0d36-4a54-8ca9-a4e3e7b00a77/quaternions",
        "shape": [
          10224,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4dff667e-0d36-4a54-8ca9-a4e3e7b00a77/timestamps",
        "shape": [
          10224
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/52dfa1fa-5f4c-42ce-8060-d6e35c625f7a/quaternions",
        "shape": [
          13448,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/52dfa1fa-5f4c-42ce-8060-d6e35c625f7a/timestamps",
        "shape": [
          13448
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/53fe906f-0550-4272-b861-69d3bf10f283/quaternions",
        "shape": [
          7612,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/53fe906f-0550-4272-b861-69d3bf10f283/timestamps",
        "shape": [
          7612
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/5a5b5fb7-a0ba-4d9a-8660-598b5a3014b0/quaternions",
        "shape": [
          9356,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/5a5b5fb7-a0ba-4d9a-8660-598b5a3014b0/timestamps",
        "shape": [
          9356
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/5d3b10d1-b3d9-4e69-af5a-4c6646e00637/quaternions",
        "shape": [
          16504,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/5d3b10d1-b3d9-4e69-af5a-4c6646e00637/timestamps",
        "shape": [
          16504
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/61ea04db-03e8-44d3-b244-ac2bf7c4c47e/quaternions",
        "shape": [
          15667,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/61ea04db-03e8-44d3-b244-ac2bf7c4c47e/timestamps",
        "shape": [
          15667
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/63f3aacf-bbe1-4276-9dda-e8aa3c208911/quaternions",
        "shape": [
          12392,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/63f3aacf-bbe1-4276-9dda-e8aa3c208911/timestamps",
        "shape": [
          12392
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/65552dea-be7f-4aca-9a83-88be40325fc6/quaternions",
        "shape": [
          15299,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/65552dea-be7f-4aca-9a83-88be40325fc6/timestamps",
        "shape": [
          15299
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6911cb00-494a-420e-9766-2b0292b16435/quaternions",
        "shape": [
          15833,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6911cb00-494a-420e-9766-2b0292b16435/timestamps",
        "shape": [
          15833
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6a9fcd63-7ef5-47d5-82ad-60ebcf605229/quaternions",
        "shape": [
          10415,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6a9fcd63-7ef5-47d5-82ad-60ebcf605229/timestamps",
        "shape": [
          10415
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6f056bc8-96cc-4250-b222-4948206f3db8/quaternions",
        "shape": [
          12100,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6f056bc8-96cc-4250-b222-4948206f3db8/timestamps",
        "shape": [
          12100
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/72a634db-7e7c-4f5a-a214-2772847cdf3a/quaternions",
        "shape": [
          21328,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/72a634db-7e7c-4f5a-a214-2772847cdf3a/timestamps",
        "shape": [
          21328
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/74146ba6-d47f-4869-bba7-028fc558334a/quaternions",
        "shape": [
          11876,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/74146ba6-d47f-4869-bba7-028fc558334a/timestamps",
        "shape": [
          11876
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/7484b559-f93f-476c-822d-c4523d8d2056/quaternions",
        "shape": [
          15914,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/7484b559-f93f-476c-822d-c4523d8d2056/timestamps",
        "shape": [
          15914
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/7ea6e9eb-8188-47c4-9346-52e41d4350e3/quaternions",
        "shape": [
          20900,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/7ea6e9eb-8188-47c4-9346-52e41d4350e3/timestamps",
        "shape": [
          20900
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/quaternions",
        "shape": [
          19383,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/timestamps",
        "shape": [
          19383
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/80e7dc8e-2c2c-4810-bc72-d5b894a2ed51/quaternions",
        "shape": [
          13041,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/80e7dc8e-2c2c-4810-bc72-d5b894a2ed51/timestamps",
        "shape": [
          13041
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/845189e5-da4c-4a30-813d-8326f6cb73dc/quaternions",
        "shape": [
          10668,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/845189e5-da4c-4a30-813d-8326f6cb73dc/timestamps",
        "shape": [
          10668
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/88724b28-641d-40b5-8c56-117740831111/quaternions",
        "shape": [
          24315,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/88724b28-641d-40b5-8c56-117740831111/timestamps",
        "shape": [
          24315
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/quaternions",
        "shape": [
          16980,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/timestamps",
        "shape": [
          16980
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/quaternions",
        "shape": [
          21305,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/timestamps",
        "shape": [
          21305
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8d8aeadf-b4a0-4783-8aef-4cbcb4b63628/quaternions",
        "shape": [
          16505,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8d8aeadf-b4a0-4783-8aef-4cbcb4b63628/timestamps",
        "shape": [
          16505
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8e364e69-c51c-46ad-ae9c-5976b3e5d72d/quaternions",
        "shape": [
          20431,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8e364e69-c51c-46ad-ae9c-5976b3e5d72d/timestamps",
        "shape": [
          20431
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8e9573ce-d378-4708-9c81-18c5871d66b6/quaternions",
        "shape": [
          15673,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8e9573ce-d378-4708-9c81-18c5871d66b6/timestamps",
        "shape": [
          15673
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8ed9dae8-e94c-4dba-8780-48077210bcba/quaternions",
        "shape": [
          19872,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8ed9dae8-e94c-4dba-8780-48077210bcba/timestamps",
        "shape": [
          19872
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/97683149-ae91-4b47-aa6a-4a38dc2e3bc8/quaternions",
        "shape": [
          21797,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/97683149-ae91-4b47-aa6a-4a38dc2e3bc8/timestamps",
        "shape": [
          21797
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/9d5fa1a5-9f1e-4d96-91ff-95a9859e6710/quaternions",
        "shape": [
          20624,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/9d5fa1a5-9f1e-4d96-91ff-95a9859e6710/timestamps",
        "shape": [
          20624
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/9de2a636-f5a2-4714-963e-a5027eecfb92/quaternions",
        "shape": [
          15529,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/9de2a636-f5a2-4714-963e-a5027eecfb92/timestamps",
        "shape": [
          15529
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a1e6bdea-4da7-4d8a-8f56-817c5ff88b58/quaternions",
        "shape": [
          18102,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a1e6bdea-4da7-4d8a-8f56-817c5ff88b58/timestamps",
        "shape": [
          18102
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a2250eb3-f762-467b-908a-f10e796fea71/quaternions",
        "shape": [
          14653,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a2250eb3-f762-467b-908a-f10e796fea71/timestamps",
        "shape": [
          14653
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a95945dc-57e1-43d0-8137-23cf6600aacb/quaternions",
        "shape": [
          15186,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a95945dc-57e1-43d0-8137-23cf6600aacb/timestamps",
        "shape": [
          15186
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/abba7a78-fdd8-48ef-90fe-c371c9573a7b/quaternions",
        "shape": [
          2234,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/abba7a78-fdd8-48ef-90fe-c371c9573a7b/timestamps",
        "shape": [
          2234
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6a538a9-c2bf-4c37-a0c0-9bbd48dfe1a9/quaternions",
        "shape": [
          17823,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6a538a9-c2bf-4c37-a0c0-9bbd48dfe1a9/timestamps",
        "shape": [
          17823
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6c797c5-a454-49db-b08b-fa6cdb37e393/quaternions",
        "shape": [
          11698,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6c797c5-a454-49db-b08b-fa6cdb37e393/timestamps",
        "shape": [
          11698
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/be2b5237-0525-40f1-8820-6c9132492e8b/quaternions",
        "shape": [
          3176,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/be2b5237-0525-40f1-8820-6c9132492e8b/timestamps",
        "shape": [
          3176
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/bfdde7ba-77d8-4363-a742-0d0d8c7e11ea/quaternions",
        "shape": [
          16423,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/bfdde7ba-77d8-4363-a742-0d0d8c7e11ea/timestamps",
        "shape": [
          16423
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/cd3684f2-4d5a-41c8-8233-a134407ccdac/quaternions",
        "shape": [
          1799,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/cd3684f2-4d5a-41c8-8233-a134407ccdac/timestamps",
        "shape": [
          1799
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d4f62f36-957e-45fe-ab57-737d79cd7789/quaternions",
        "shape": [
          8741,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d4f62f36-957e-45fe-ab57-737d79cd7789/timestamps",
        "shape": [
          8741
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d77c03be-a31d-414e-82d7-92e4eb33067e/quaternions",
        "shape": [
          15747,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d77c03be-a31d-414e-82d7-92e4eb33067e/timestamps",
        "shape": [
          15747
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/dc59d6aa-0f42-4412-8ad8-31111e85b2f1/quaternions",
        "shape": [
          15693,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/dc59d6aa-0f42-4412-8ad8-31111e85b2f1/timestamps",
        "shape": [
          15693
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/quaternions",
        "shape": [
          10138,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/timestamps",
        "shape": [
          10138
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ee3891da-55f0-4b93-9e84-efec2ff4274c/quaternions",
        "shape": [
          17429,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ee3891da-55f0-4b93-9e84-efec2ff4274c/timestamps",
        "shape": [
          17429
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ef8df4e6-9a8a-409d-afec-d859dd382be1/quaternions",
        "shape": [
          13520,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ef8df4e6-9a8a-409d-afec-d859dd382be1/timestamps",
        "shape": [
          13520
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f15514d9-fdeb-40c1-877b-736740aef5f3/quaternions",
        "shape": [
          14765,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f15514d9-fdeb-40c1-877b-736740aef5f3/timestamps",
        "shape": [
          14765
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f1a3875f-b56a-4290-8d57-5cbbc7248e1c/quaternions",
        "shape": [
          20729,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f1a3875f-b56a-4290-8d57-5cbbc7248e1c/timestamps",
        "shape": [
          20729
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f59fee7e-bf1d-4e63-9776-dc84a2c964e5/quaternions",
        "shape": [
          23121,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f59fee7e-bf1d-4e63-9776-dc84a2c964e5/timestamps",
        "shape": [
          23121
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/fb974c1d-0403-4917-b326-3a69d96c335e/quaternions",
        "shape": [
          7605,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/fb974c1d-0403-4917-b326-3a69d96c335e/timestamps",
        "shape": [
          7605
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/fc76d986-54b1-46c2-80f2-a918f5333e25/quaternions",
        "shape": [
          10350,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/fc76d986-54b1-46c2-80f2-a918f5333e25/timestamps",
        "shape": [
          10350
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ff61c263-8591-4c82-8255-9eaad43e0928/quaternions",
        "shape": [
          1671,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ff61c263-8591-4c82-8255-9eaad43e0928/timestamps",
        "shape": [
          1671
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ffb4d561-67c7-4716-a194-3f0ff80d749e/quaternions",
        "shape": [
          15794,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ffb4d561-67c7-4716-a194-3f0ff80d749e/timestamps",
        "shape": [
          15794
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/1cd90ba4-a27a-4d0c-8df5-13c35ff95cec/quaternions",
        "shape": [
          19549,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/1cd90ba4-a27a-4d0c-8df5-13c35ff95cec/timestamps",
        "shape": [
          19549
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/2348afb9-8b30-477e-a901-802b9e8d3581/quaternions",
        "shape": [
          20720,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/2348afb9-8b30-477e-a901-802b9e8d3581/timestamps",
        "shape": [
          20720
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/29ae9b37-2206-4c0b-8344-3af75daa0ca4/quaternions",
        "shape": [
          17154,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/29ae9b37-2206-4c0b-8344-3af75daa0ca4/timestamps",
        "shape": [
          17154
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/2a06f0ef-3ffb-4d2b-8d6f-8c2a222152ce/quaternions",
        "shape": [
          20418,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/2a06f0ef-3ffb-4d2b-8d6f-8c2a222152ce/timestamps",
        "shape": [
          20418
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/46eee1cb-4799-4b31-b23a-02bfccbfc124/quaternions",
        "shape": [
          17151,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/46eee1cb-4799-4b31-b23a-02bfccbfc124/timestamps",
        "shape": [
          17151
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/6fb13ce8-b7f9-4be8-b09b-104e2f617cc6/quaternions",
        "shape": [
          17812,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/6fb13ce8-b7f9-4be8-b09b-104e2f617cc6/timestamps",
        "shape": [
          17812
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/86067a02-cd48-4a26-8caa-442aebfd147a/quaternions",
        "shape": [
          7694,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/86067a02-cd48-4a26-8caa-442aebfd147a/timestamps",
        "shape": [
          7694
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/9a169cda-4f88-4ae6-8607-1c2c38961b44/quaternions",
        "shape": [
          21418,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/9a169cda-4f88-4ae6-8607-1c2c38961b44/timestamps",
        "shape": [
          21418
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/b0cde597-0d9e-44a7-af96-8ed88c2db1f9/quaternions",
        "shape": [
          15014,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/b0cde597-0d9e-44a7-af96-8ed88c2db1f9/timestamps",
        "shape": [
          15014
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/bb39a4c2-6cd7-4b15-8613-7c1ffd3e985c/quaternions",
        "shape": [
          20410,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/bb39a4c2-6cd7-4b15-8613-7c1ffd3e985c/timestamps",
        "shape": [
          20410
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/c6fa3515-d3a2-46b9-b906-0d047be96222/quaternions",
        "shape": [
          20238,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/c6fa3515-d3a2-46b9-b906-0d047be96222/timestamps",
        "shape": [
          20238
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00926ea3-8296-4f72-959d-a9b289bb541e/sessions/5e51e407-7c50-4284-a48f-aaf4019b7071/quaternions",
        "shape": [
          14514,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00926ea3-8296-4f72-959d-a9b289bb541e/sessions/5e51e407-7c50-4284-a48f-aaf4019b7071/timestamps",
        "shape": [
          14514
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00a82f96-9871-4d8d-a1c7-c739e34f928f/sessions/321a63e9-cc19-4faa-abc2-63c249d0d89b/quaternions",
        "shape": [
          20554,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00a82f96-9871-4d8d-a1c7-c739e34f928f/sessions/321a63e9-cc19-4faa-abc2-63c249d0d89b/timestamps",
        "shape": [
          20554
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00a82f96-9871-4d8d-a1c7-c739e34f928f/sessions/c1256ad6-6e63-4669-ac1b-7158206e9a84/quaternions",
        "shape": [
          7736,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00a82f96-9871-4d8d-a1c7-c739e34f928f/sessions/c1256ad6-6e63-4669-ac1b-7158206e9a84/timestamps",
        "shape": [
          7736
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00a85435-a5d4-4ce3-b452-3d99b6f8b399/sessions/aa152f3e-67e8-4220-8d0a-7c919ee0c112/quaternions",
        "shape": [
          18262,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00a85435-a5d4-4ce3-b452-3d99b6f8b399/sessions/aa152f3e-67e8-4220-8d0a-7c919ee0c112/timestamps",
        "shape": [
          18262
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0494f6f8-8266-4dfd-8bff-039cfaa85167/quaternions",
        "shape": [
          25591,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0494f6f8-8266-4dfd-8bff-039cfaa85167/timestamps",
        "shape": [
          25591
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b2d7047-6244-482e-8a69-37736a0d1f7f/quaternions",
        "shape": [
          20137,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b2d7047-6244-482e-8a69-37736a0d1f7f/timestamps",
        "shape": [
          20137
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b506391-efca-47e2-81c1-a30506cc7b1c/quaternions",
        "shape": [
          12889,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b506391-efca-47e2-81c1-a30506cc7b1c/timestamps",
        "shape": [
          12889
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1007760b-ffd9-45f7-8af7-f4de77f3cd04/quaternions",
        "shape": [
          7832,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1007760b-ffd9-45f7-8af7-f4de77f3cd04/timestamps",
        "shape": [
          7832
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/146c1920-8262-4e24-9e90-d699c0025dd2/quaternions",
        "shape": [
          9761,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/146c1920-8262-4e24-9e90-d699c0025dd2/timestamps",
        "shape": [
          9761
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1bb847f8-8ecd-4d79-aa99-242a9a3d2386/quaternions",
        "shape": [
          8019,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1bb847f8-8ecd-4d79-aa99-242a9a3d2386/timestamps",
        "shape": [
          8019
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1c49274b-4ab9-456a-9c7e-d16f30929dfb/quaternions",
        "shape": [
          9729,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1c49274b-4ab9-456a-9c7e-d16f30929dfb/timestamps",
        "shape": [
          9729
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1e9346ed-9f96-4cfa-a0c7-dabd0121e711/quaternions",
        "shape": [
          5090,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1e9346ed-9f96-4cfa-a0c7-dabd0121e711/timestamps",
        "shape": [
          5090
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2474682a-06c7-4218-afe5-b8739cd65c4e/quaternions",
        "shape": [
          30204,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2474682a-06c7-4218-afe5-b8739cd65c4e/timestamps",
        "shape": [
          30204
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2647c161-532a-4113-8941-00380f20f3a7/quaternions",
        "shape": [
          15603,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2647c161-532a-4113-8941-00380f20f3a7/timestamps",
        "shape": [
          15603
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/29ae3c69-ce66-4ff0-97cc-ae367e4cc00c/quaternions",
        "shape": [
          18511,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/29ae3c69-ce66-4ff0-97cc-ae367e4cc00c/timestamps",
        "shape": [
          18511
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/356cedd6-23c9-428a-b32b-ab34d596292a/quaternions",
        "shape": [
          7739,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/356cedd6-23c9-428a-b32b-ab34d596292a/timestamps",
        "shape": [
          7739
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/3644ca0f-7ace-41dd-aa36-d3feec124a0c/quaternions",
        "shape": [
          19691,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/3644ca0f-7ace-41dd-aa36-d3feec124a0c/timestamps",
        "shape": [
          19691
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/48d802ce-674f-49f7-93db-0e29890873e2/quaternions",
        "shape": [
          1149,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/48d802ce-674f-49f7-93db-0e29890873e2/timestamps",
        "shape": [
          1149
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/50450f94-7dd3-461c-8ae4-10e76a8653bb/quaternions",
        "shape": [
          6012,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/50450f94-7dd3-461c-8ae4-10e76a8653bb/timestamps",
        "shape": [
          6012
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/5941ae62-022f-4649-b9e1-c62eaa2b00a3/quaternions",
        "shape": [
          22564,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/5941ae62-022f-4649-b9e1-c62eaa2b00a3/timestamps",
        "shape": [
          22564
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/60d421cb-78b1-405b-8fda-3423910764d7/quaternions",
        "shape": [
          6965,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/60d421cb-78b1-405b-8fda-3423910764d7/timestamps",
        "shape": [
          6965
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/61c78e4e-77d5-4eb3-9198-7e9440e171fd/quaternions",
        "shape": [
          5500,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/61c78e4e-77d5-4eb3-9198-7e9440e171fd/timestamps",
        "shape": [
          5500
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/quaternions",
        "shape": [
          4067,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/timestamps",
        "shape": [
          4067
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/698713e4-d18e-49d7-9935-8a569b5ff2a5/quaternions",
        "shape": [
          16467,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/698713e4-d18e-49d7-9935-8a569b5ff2a5/timestamps",
        "shape": [
          16467
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/6c8a9d0d-a00c-450b-a4e2-d1d8fa998d50/quaternions",
        "shape": [
          716,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/6c8a9d0d-a00c-450b-a4e2-d1d8fa998d50/timestamps",
        "shape": [
          716
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/707c9ea7-ebba-4f82-bdab-c14e9f0288f0/quaternions",
        "shape": [
          11954,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/707c9ea7-ebba-4f82-bdab-c14e9f0288f0/timestamps",
        "shape": [
          11954
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/70b7d934-b91e-4209-b09a-789ad7595f4b/quaternions",
        "shape": [
          2429,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/70b7d934-b91e-4209-b09a-789ad7595f4b/timestamps",
        "shape": [
          2429
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/7f5569b8-a641-4a71-b15e-08569310f1ef/quaternions",
        "shape": [
          10431,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/7f5569b8-a641-4a71-b15e-08569310f1ef/timestamps",
        "shape": [
          10431
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/831d7362-9709-4089-b0ef-873a1ab95bf2/quaternions",
        "shape": [
          26035,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/831d7362-9709-4089-b0ef-873a1ab95bf2/timestamps",
        "shape": [
          26035
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/877c649e-5d91-4474-bfe3-8496c203f252/quaternions",
        "shape": [
          5430,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/877c649e-5d91-4474-bfe3-8496c203f252/timestamps",
        "shape": [
          5430
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/8aeec78a-02ca-4306-88d9-045dc9a6a9e5/quaternions",
        "shape": [
          5088,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/8aeec78a-02ca-4306-88d9-045dc9a6a9e5/timestamps",
        "shape": [
          5088
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/9447d4b0-d424-4bc7-bcad-483373d55fe1/quaternions",
        "shape": [
          5083,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/9447d4b0-d424-4bc7-bcad-483373d55fe1/timestamps",
        "shape": [
          5083
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/a2f5ec43-8903-49f6-ab58-47a792b5f52d/quaternions",
        "shape": [
          7741,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/a2f5ec43-8903-49f6-ab58-47a792b5f52d/timestamps",
        "shape": [
          7741
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/a51f84cf-cc2b-4ea7-8c66-106fc1ce6e3f/quaternions",
        "shape": [
          19367,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/a51f84cf-cc2b-4ea7-8c66-106fc1ce6e3f/timestamps",
        "shape": [
          19367
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/ae7b3e97-ba8c-416e-bcba-36b47c854b32/quaternions",
        "shape": [
          285,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/ae7b3e97-ba8c-416e-bcba-36b47c854b32/timestamps",
        "shape": [
          285
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/afa74301-5d4f-4e1b-9153-bff088e815c7/quaternions",
        "shape": [
          14823,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/afa74301-5d4f-4e1b-9153-bff088e815c7/timestamps",
        "shape": [
          14823
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/b03cfadd-7255-4d53-9fa5-6e68fdfc98db/quaternions",
        "shape": [
          11990,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/b03cfadd-7255-4d53-9fa5-6e68fdfc98db/timestamps",
        "shape": [
          11990
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/quaternions",
        "shape": [
          13278,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/timestamps",
        "shape": [
          13278
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bf0de83e-ccc4-46d3-b4e2-cf2bea6ea89e/quaternions",
        "shape": [
          13844,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bf0de83e-ccc4-46d3-b4e2-cf2bea6ea89e/timestamps",
        "shape": [
          13844
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/quaternions",
        "shape": [
          10124,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/timestamps",
        "shape": [
          10124
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/cd85cf88-620f-4107-878e-98acf956844e/quaternions",
        "shape": [
          7001,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/cd85cf88-620f-4107-878e-98acf956844e/timestamps",
        "shape": [
          7001
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/d3b5807d-3816-4049-bce1-0f0231e559da/quaternions",
        "shape": [
          26022,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/d3b5807d-3816-4049-bce1-0f0231e559da/timestamps",
        "shape": [
          26022
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/d5a3260a-1b43-4657-9480-c969d1e8845f/quaternions",
        "shape": [
          2635,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/d5a3260a-1b43-4657-9480-c969d1e8845f/timestamps",
        "shape": [
          2635
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/e4c14c73-14a7-4821-aec3-c21dddeef590/quaternions",
        "shape": [
          11961,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/e4c14c73-14a7-4821-aec3-c21dddeef590/timestamps",
        "shape": [
          11961
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00d2ae65-e526-422e-8418-a0505a16187c/sessions/137f1f67-7154-4b98-bb00-6f053eb5ffa1/quaternions",
        "shape": [
          15987,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00d2ae65-e526-422e-8418-a0505a16187c/sessions/137f1f67-7154-4b98-bb00-6f053eb5ffa1/timestamps",
        "shape": [
          15987
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00dbd9ee-71c8-4bad-ad21-8eee56a4c200/sessions/d7abd210-1bf3-47a0-83e7-2e089c156181/quaternions",
        "shape": [
          22132,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00dbd9ee-71c8-4bad-ad21-8eee56a4c200/sessions/d7abd210-1bf3-47a0-83e7-2e089c156181/timestamps",
        "shape": [
          22132
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/quaternions",
        "shape": [
          12770,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/timestamps",
        "shape": [
          12770
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00e95b89-2b88-4c29-974d-cf9e1934ec5b/sessions/353bbf02-559a-48c8-922f-b1d541a1c413/quaternions",
        "shape": [
          2482,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00e95b89-2b88-4c29-974d-cf9e1934ec5b/sessions/353bbf02-559a-48c8-922f-b1d541a1c413/timestamps",
        "shape": [
          2482
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f13eb1-6bc7-4e38-8525-5b412af4b9b7/sessions/1e7826cd-88d0-46d6-abd1-43b3e88f9b31/quaternions",
        "shape": [
          14241,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f13eb1-6bc7-4e38-8525-5b412af4b9b7/sessions/1e7826cd-88d0-46d6-abd1-43b3e88f9b31/timestamps",
        "shape": [
          14241
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f13eb1-6bc7-4e38-8525-5b412af4b9b7/sessions/d6000cf8-d5e3-411d-887d-187155ac9e2e/quaternions",
        "shape": [
          14873,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f13eb1-6bc7-4e38-8525-5b412af4b9b7/sessions/d6000cf8-d5e3-411d-887d-187155ac9e2e/timestamps",
        "shape": [
          14873
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/quaternions",
        "shape": [
          13083,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/timestamps",
        "shape": [
          13083
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f73e02-eb53-4387-bd0f-5c701474261f/sessions/1c07646c-2301-4412-aac2-0effc55bf9c2/quaternions",
        "shape": [
          1659,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f73e02-eb53-4387-bd0f-5c701474261f/sessions/1c07646c-2301-4412-aac2-0effc55bf9c2/timestamps",
        "shape": [
          1659
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/04231a70-0286-4084-817f-e3fb8984cff0/quaternions",
        "shape": [
          15230,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/04231a70-0286-4084-817f-e3fb8984cff0/timestamps",
        "shape": [
          15230
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/1268d39c-458a-46cf-b727-d6157564405f/quaternions",
        "shape": [
          8722,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/1268d39c-458a-46cf-b727-d6157564405f/timestamps",
        "shape": [
          8722
        ],
        "dtype": "float64"
      }
    ]
  },
  "failures": [],
  "warnings": []
}
```

### PASS — HDF5 structural/data checks: head_pose_v2.h5

head_pose_v2.h5: 658 datasets; 0 failure(s), 0 warning(s).

```json
{
  "evidence": {
    "size_bytes": 40156128,
    "datasets": [
      {
        "path": "users/00743d56-a912-4ec7-a9d4-3e6e22756488/sessions/005b994e-5330-48af-859a-ae8beaed8b91/segments/0000/quaternions",
        "shape": [
          15896,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00743d56-a912-4ec7-a9d4-3e6e22756488/sessions/005b994e-5330-48af-859a-ae8beaed8b91/segments/0000/timestamps",
        "shape": [
          15896
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007a16e6-6862-40f1-b7ee-07c959e4ac92/sessions/832329ed-e4e8-4ba4-a3a0-529a3fd31636/segments/0000/quaternions",
        "shape": [
          21462,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007a16e6-6862-40f1-b7ee-07c959e4ac92/sessions/832329ed-e4e8-4ba4-a3a0-529a3fd31636/segments/0000/timestamps",
        "shape": [
          21462
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007a16e6-6862-40f1-b7ee-07c959e4ac92/sessions/a410b074-9b79-4335-a0bf-a42e9af944b0/segments/0000/quaternions",
        "shape": [
          601,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007a16e6-6862-40f1-b7ee-07c959e4ac92/sessions/a410b074-9b79-4335-a0bf-a42e9af944b0/segments/0000/timestamps",
        "shape": [
          601
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/0e515c8e-3241-4e7d-ad91-516c9aef7f6c/segments/0000/quaternions",
        "shape": [
          1014,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/0e515c8e-3241-4e7d-ad91-516c9aef7f6c/segments/0000/timestamps",
        "shape": [
          1014
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/11bd715b-e6ab-40cb-aff1-81d2160d2e5e/segments/0000/quaternions",
        "shape": [
          17314,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/11bd715b-e6ab-40cb-aff1-81d2160d2e5e/segments/0000/timestamps",
        "shape": [
          17314
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/18f668eb-fa6b-475b-aefc-76bf2b91a863/segments/0000/quaternions",
        "shape": [
          19240,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/18f668eb-fa6b-475b-aefc-76bf2b91a863/segments/0000/timestamps",
        "shape": [
          19240
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/1c3cca2d-af0f-4fa3-9c1c-584bbc39f441/segments/0000/quaternions",
        "shape": [
          2041,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/1c3cca2d-af0f-4fa3-9c1c-584bbc39f441/segments/0000/timestamps",
        "shape": [
          2041
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/1c7c66e0-a1b5-4eb3-8c1b-51068921f348/segments/0000/quaternions",
        "shape": [
          13862,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/1c7c66e0-a1b5-4eb3-8c1b-51068921f348/segments/0000/timestamps",
        "shape": [
          13862
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/22d3d598-4957-4958-9051-a74ffae55b50/segments/0000/quaternions",
        "shape": [
          20885,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/22d3d598-4957-4958-9051-a74ffae55b50/segments/0000/timestamps",
        "shape": [
          20885
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/27eadf41-b990-4a33-9f07-74672d5cf8d8/segments/0000/quaternions",
        "shape": [
          5376,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/27eadf41-b990-4a33-9f07-74672d5cf8d8/segments/0000/timestamps",
        "shape": [
          5376
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/389a2476-4c0a-4ab4-8222-e505399252f0/segments/0000/quaternions",
        "shape": [
          10698,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/389a2476-4c0a-4ab4-8222-e505399252f0/segments/0000/timestamps",
        "shape": [
          10698
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/42cb613e-5607-49cd-8c96-ead309561748/segments/0000/quaternions",
        "shape": [
          887,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/42cb613e-5607-49cd-8c96-ead309561748/segments/0000/timestamps",
        "shape": [
          887
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/4f219cbb-6c08-413e-aa17-f57e09dd2f86/segments/0000/quaternions",
        "shape": [
          23298,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/4f219cbb-6c08-413e-aa17-f57e09dd2f86/segments/0000/timestamps",
        "shape": [
          23298
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/540f99ba-f6e2-43ef-b0bc-7391c9f89823/segments/0000/quaternions",
        "shape": [
          5573,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/540f99ba-f6e2-43ef-b0bc-7391c9f89823/segments/0000/timestamps",
        "shape": [
          5573
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/54965afb-f0f2-4d6b-b4c0-cb7fc9de5988/segments/0000/quaternions",
        "shape": [
          20189,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/54965afb-f0f2-4d6b-b4c0-cb7fc9de5988/segments/0000/timestamps",
        "shape": [
          20189
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/54965afb-f0f2-4d6b-b4c0-cb7fc9de5988/segments/0001/quaternions",
        "shape": [
          132,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/54965afb-f0f2-4d6b-b4c0-cb7fc9de5988/segments/0001/timestamps",
        "shape": [
          132
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/55ba83aa-de1d-42fa-b5ad-0d543df5d3f8/segments/0000/quaternions",
        "shape": [
          1952,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/55ba83aa-de1d-42fa-b5ad-0d543df5d3f8/segments/0000/timestamps",
        "shape": [
          1952
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/638377f0-3c4c-494d-8b40-99b15260c5ea/segments/0000/quaternions",
        "shape": [
          17924,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/638377f0-3c4c-494d-8b40-99b15260c5ea/segments/0000/timestamps",
        "shape": [
          17924
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/681fdec1-df69-40ed-a9f9-e09aaa13ab6c/segments/0000/quaternions",
        "shape": [
          1236,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/681fdec1-df69-40ed-a9f9-e09aaa13ab6c/segments/0000/timestamps",
        "shape": [
          1236
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/681fdec1-df69-40ed-a9f9-e09aaa13ab6c/segments/0001/quaternions",
        "shape": [
          23554,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/681fdec1-df69-40ed-a9f9-e09aaa13ab6c/segments/0001/timestamps",
        "shape": [
          23554
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/701fd8c5-fc3f-4d53-b7bc-a211131a6a28/segments/0000/quaternions",
        "shape": [
          11639,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/701fd8c5-fc3f-4d53-b7bc-a211131a6a28/segments/0000/timestamps",
        "shape": [
          11639
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/77f3d6b8-0931-4c9c-a1c1-77b148a13f9a/segments/0000/quaternions",
        "shape": [
          7422,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/77f3d6b8-0931-4c9c-a1c1-77b148a13f9a/segments/0000/timestamps",
        "shape": [
          7422
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/7b8adb4e-c563-4d80-b939-1b9acb665d97/segments/0000/quaternions",
        "shape": [
          14082,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/7b8adb4e-c563-4d80-b939-1b9acb665d97/segments/0000/timestamps",
        "shape": [
          14082
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/7ef0137d-b728-492b-86a9-7b7c49997d2d/segments/0000/quaternions",
        "shape": [
          5064,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/7ef0137d-b728-492b-86a9-7b7c49997d2d/segments/0000/timestamps",
        "shape": [
          5064
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/88495478-a9b5-4413-bd5b-805f7c008841/segments/0000/quaternions",
        "shape": [
          1086,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/88495478-a9b5-4413-bd5b-805f7c008841/segments/0000/timestamps",
        "shape": [
          1086
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/887d9762-3b4b-4f82-8a4a-92bec34b6103/segments/0000/quaternions",
        "shape": [
          20033,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/887d9762-3b4b-4f82-8a4a-92bec34b6103/segments/0000/timestamps",
        "shape": [
          20033
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/8a55947d-4d17-4e49-bb52-9722c42e8299/segments/0000/quaternions",
        "shape": [
          18829,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/8a55947d-4d17-4e49-bb52-9722c42e8299/segments/0000/timestamps",
        "shape": [
          18829
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/902aaaea-1c56-47b9-a02e-aa04c6215390/segments/0000/quaternions",
        "shape": [
          9060,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/902aaaea-1c56-47b9-a02e-aa04c6215390/segments/0000/timestamps",
        "shape": [
          9060
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/902aaaea-1c56-47b9-a02e-aa04c6215390/segments/0001/quaternions",
        "shape": [
          11710,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/902aaaea-1c56-47b9-a02e-aa04c6215390/segments/0001/timestamps",
        "shape": [
          11710
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/92fa43d7-2f2f-4d6f-8b59-15864fadc81b/segments/0000/quaternions",
        "shape": [
          17478,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/92fa43d7-2f2f-4d6f-8b59-15864fadc81b/segments/0000/timestamps",
        "shape": [
          17478
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/93aae0b0-7424-49a2-990b-6dcaead0e1d4/segments/0000/quaternions",
        "shape": [
          1172,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/93aae0b0-7424-49a2-990b-6dcaead0e1d4/segments/0000/timestamps",
        "shape": [
          1172
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/97b75f77-deb7-42ec-8bb3-35671edcb98a/segments/0000/quaternions",
        "shape": [
          18801,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/97b75f77-deb7-42ec-8bb3-35671edcb98a/segments/0000/timestamps",
        "shape": [
          18801
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/9c8d3e9e-c3cf-47cd-b9ff-7bfcef0ae4ab/segments/0000/quaternions",
        "shape": [
          18006,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/9c8d3e9e-c3cf-47cd-b9ff-7bfcef0ae4ab/segments/0000/timestamps",
        "shape": [
          18006
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/9dbc71c7-7355-4305-89fd-e9e758e2d893/segments/0000/quaternions",
        "shape": [
          17213,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/9dbc71c7-7355-4305-89fd-e9e758e2d893/segments/0000/timestamps",
        "shape": [
          17213
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/a6ad17ca-d396-4ec3-8a68-9bdf6857c6be/segments/0000/quaternions",
        "shape": [
          16585,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/a6ad17ca-d396-4ec3-8a68-9bdf6857c6be/segments/0000/timestamps",
        "shape": [
          16585
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/a9481103-32ec-4afa-918d-8d5fffd295ce/segments/0000/quaternions",
        "shape": [
          762,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/a9481103-32ec-4afa-918d-8d5fffd295ce/segments/0000/timestamps",
        "shape": [
          762
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/aab7ae2e-7f78-468e-88b6-6808e04b4791/segments/0000/quaternions",
        "shape": [
          3005,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/aab7ae2e-7f78-468e-88b6-6808e04b4791/segments/0000/timestamps",
        "shape": [
          3005
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/b559863c-d366-4b4c-af59-9b07fdfc033a/segments/0000/quaternions",
        "shape": [
          28573,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/b559863c-d366-4b4c-af59-9b07fdfc033a/segments/0000/timestamps",
        "shape": [
          28573
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/cfe696ff-e750-4796-992c-8bd950994a04/segments/0000/quaternions",
        "shape": [
          23250,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/cfe696ff-e750-4796-992c-8bd950994a04/segments/0000/timestamps",
        "shape": [
          23250
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/d50b7ab6-643a-4aae-9d4e-bc4b779dee14/segments/0000/quaternions",
        "shape": [
          12211,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/d50b7ab6-643a-4aae-9d4e-bc4b779dee14/segments/0000/timestamps",
        "shape": [
          12211
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/d80cb020-b6a1-4db5-a2f7-ccebb6c44697/segments/0000/quaternions",
        "shape": [
          16180,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/d80cb020-b6a1-4db5-a2f7-ccebb6c44697/segments/0000/timestamps",
        "shape": [
          16180
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/dc002a1f-ec5b-415b-a842-2bd6d66deabd/segments/0000/quaternions",
        "shape": [
          4966,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/dc002a1f-ec5b-415b-a842-2bd6d66deabd/segments/0000/timestamps",
        "shape": [
          4966
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/df4959aa-4196-43e4-9bff-ef036af8abbe/segments/0000/quaternions",
        "shape": [
          24932,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/df4959aa-4196-43e4-9bff-ef036af8abbe/segments/0000/timestamps",
        "shape": [
          24932
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/e15379f1-f0bd-48e7-a887-5b3783b06784/segments/0000/quaternions",
        "shape": [
          19587,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/e15379f1-f0bd-48e7-a887-5b3783b06784/segments/0000/timestamps",
        "shape": [
          19587
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/e46471bb-b56a-4fa9-83d2-032069f83728/segments/0000/quaternions",
        "shape": [
          15835,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/e46471bb-b56a-4fa9-83d2-032069f83728/segments/0000/timestamps",
        "shape": [
          15835
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/efcdfedc-d978-48ea-9e57-d92385d47549/segments/0000/quaternions",
        "shape": [
          3157,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/efcdfedc-d978-48ea-9e57-d92385d47549/segments/0000/timestamps",
        "shape": [
          3157
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/fcacc95f-8f06-4aec-b49d-827cc5e75ee4/segments/0000/quaternions",
        "shape": [
          12155,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007d3ac4-8443-4430-94e2-25b9474e2ce6/sessions/fcacc95f-8f06-4aec-b49d-827cc5e75ee4/segments/0000/timestamps",
        "shape": [
          12155
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/454f5fcd-c194-45d2-90b2-d6a7bba9bc04/segments/0000/quaternions",
        "shape": [
          9546,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/454f5fcd-c194-45d2-90b2-d6a7bba9bc04/segments/0000/timestamps",
        "shape": [
          9546
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/52875a99-9f93-4a6d-bd75-bc0048288f65/segments/0000/quaternions",
        "shape": [
          10946,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/52875a99-9f93-4a6d-bd75-bc0048288f65/segments/0000/timestamps",
        "shape": [
          10946
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/52875a99-9f93-4a6d-bd75-bc0048288f65/segments/0001/quaternions",
        "shape": [
          3021,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/52875a99-9f93-4a6d-bd75-bc0048288f65/segments/0001/timestamps",
        "shape": [
          3021
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/bd2265bc-5e0b-47ba-8d7d-3cc958245f22/segments/0000/quaternions",
        "shape": [
          1813,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/bd2265bc-5e0b-47ba-8d7d-3cc958245f22/segments/0000/timestamps",
        "shape": [
          1813
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/c8544adc-51e8-41a6-ba11-4323a46df959/segments/0000/quaternions",
        "shape": [
          11373,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/c8544adc-51e8-41a6-ba11-4323a46df959/segments/0000/timestamps",
        "shape": [
          11373
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/e4f34f34-2e8b-4a72-802d-d835d1a8ffd8/segments/0000/quaternions",
        "shape": [
          1754,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/e4f34f34-2e8b-4a72-802d-d835d1a8ffd8/segments/0000/timestamps",
        "shape": [
          1754
        ],
        "dtype": "float64"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/f9a55a06-88d0-4c51-8b43-4b60cc900e33/segments/0000/quaternions",
        "shape": [
          2048,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/007e46f7-4f73-4dfa-8e4f-8258563abd57/sessions/f9a55a06-88d0-4c51-8b43-4b60cc900e33/segments/0000/timestamps",
        "shape": [
          2048
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0000/quaternions",
        "shape": [
          922,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0000/timestamps",
        "shape": [
          922
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0001/quaternions",
        "shape": [
          5996,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0001/timestamps",
        "shape": [
          5996
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0002/quaternions",
        "shape": [
          1994,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0002/timestamps",
        "shape": [
          1994
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0003/quaternions",
        "shape": [
          5996,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0003/timestamps",
        "shape": [
          5996
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0004/quaternions",
        "shape": [
          2298,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0004/timestamps",
        "shape": [
          2298
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0005/quaternions",
        "shape": [
          1690,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0005/timestamps",
        "shape": [
          1690
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0006/quaternions",
        "shape": [
          3917,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00803645-7423-4083-9637-627a4c32759f/sessions/06523a17-a4c5-48a1-b8e7-7160277e1c09/segments/0006/timestamps",
        "shape": [
          3917
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/0a7c6042-c152-4df7-8db1-cbba1033e4c6/segments/0000/quaternions",
        "shape": [
          18122,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/0a7c6042-c152-4df7-8db1-cbba1033e4c6/segments/0000/timestamps",
        "shape": [
          18122
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/0ba5d8f0-562f-41f3-8e15-8b8dba147e3d/segments/0000/quaternions",
        "shape": [
          9286,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/0ba5d8f0-562f-41f3-8e15-8b8dba147e3d/segments/0000/timestamps",
        "shape": [
          9286
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/1437c2d3-70fc-49a3-9349-63bfa3d19c41/segments/0000/quaternions",
        "shape": [
          16857,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/1437c2d3-70fc-49a3-9349-63bfa3d19c41/segments/0000/timestamps",
        "shape": [
          16857
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/286ae5b6-27e5-477e-ad5c-a41590f09729/segments/0000/quaternions",
        "shape": [
          8581,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/286ae5b6-27e5-477e-ad5c-a41590f09729/segments/0000/timestamps",
        "shape": [
          8581
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/44e4c72a-90c6-46cb-82b6-8bb009af0ce6/segments/0000/quaternions",
        "shape": [
          16076,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/44e4c72a-90c6-46cb-82b6-8bb009af0ce6/segments/0000/timestamps",
        "shape": [
          16076
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/44e4c72a-90c6-46cb-82b6-8bb009af0ce6/segments/0001/quaternions",
        "shape": [
          1217,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/44e4c72a-90c6-46cb-82b6-8bb009af0ce6/segments/0001/timestamps",
        "shape": [
          1217
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/58ad7b03-7ed7-43c3-94ec-a8416a6f64d1/segments/0000/quaternions",
        "shape": [
          5481,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/58ad7b03-7ed7-43c3-94ec-a8416a6f64d1/segments/0000/timestamps",
        "shape": [
          5481
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/58ad7b03-7ed7-43c3-94ec-a8416a6f64d1/segments/0001/quaternions",
        "shape": [
          3081,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/58ad7b03-7ed7-43c3-94ec-a8416a6f64d1/segments/0001/timestamps",
        "shape": [
          3081
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/af5b5890-bc28-47b9-bbb2-cc76da0e95b9/segments/0000/quaternions",
        "shape": [
          20390,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/af5b5890-bc28-47b9-bbb2-cc76da0e95b9/segments/0000/timestamps",
        "shape": [
          20390
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/d3de1c53-2e9c-4986-a2ff-c63e4f04f9d6/segments/0000/quaternions",
        "shape": [
          9525,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/d3de1c53-2e9c-4986-a2ff-c63e4f04f9d6/segments/0000/timestamps",
        "shape": [
          9525
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/d3de1c53-2e9c-4986-a2ff-c63e4f04f9d6/segments/0001/quaternions",
        "shape": [
          8933,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/d3de1c53-2e9c-4986-a2ff-c63e4f04f9d6/segments/0001/timestamps",
        "shape": [
          8933
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/df175928-a3df-4ce0-9d61-efe26e26a90d/segments/0000/quaternions",
        "shape": [
          553,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/df175928-a3df-4ce0-9d61-efe26e26a90d/segments/0000/timestamps",
        "shape": [
          553
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/df175928-a3df-4ce0-9d61-efe26e26a90d/segments/0001/quaternions",
        "shape": [
          1705,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/df175928-a3df-4ce0-9d61-efe26e26a90d/segments/0001/timestamps",
        "shape": [
          1705
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/df175928-a3df-4ce0-9d61-efe26e26a90d/segments/0002/quaternions",
        "shape": [
          6360,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/df175928-a3df-4ce0-9d61-efe26e26a90d/segments/0002/timestamps",
        "shape": [
          6360
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/e3096cd3-55d0-4be0-9cc9-08ad473f84c0/segments/0000/quaternions",
        "shape": [
          8701,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/e3096cd3-55d0-4be0-9cc9-08ad473f84c0/segments/0000/timestamps",
        "shape": [
          8701
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/e579faa9-b3f4-4010-80d0-f90cc4e847fc/segments/0000/quaternions",
        "shape": [
          16686,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/e579faa9-b3f4-4010-80d0-f90cc4e847fc/segments/0000/timestamps",
        "shape": [
          16686
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/f83ea77a-45d0-47fc-bd72-aa5089cf8a01/segments/0000/quaternions",
        "shape": [
          24019,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/f83ea77a-45d0-47fc-bd72-aa5089cf8a01/segments/0000/timestamps",
        "shape": [
          24019
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/ffea28fc-ad4f-43c1-9a5c-438e824d9ca7/segments/0000/quaternions",
        "shape": [
          13323,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/ffea28fc-ad4f-43c1-9a5c-438e824d9ca7/segments/0000/timestamps",
        "shape": [
          13323
        ],
        "dtype": "float64"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/ffea28fc-ad4f-43c1-9a5c-438e824d9ca7/segments/0001/quaternions",
        "shape": [
          10693,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/0081a211-d913-4ee8-b71a-9fd1ed7d3e2d/sessions/ffea28fc-ad4f-43c1-9a5c-438e824d9ca7/segments/0001/timestamps",
        "shape": [
          10693
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/036e81b9-ae98-404e-90b0-44b76921d3c0/segments/0000/quaternions",
        "shape": [
          2041,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/036e81b9-ae98-404e-90b0-44b76921d3c0/segments/0000/timestamps",
        "shape": [
          2041
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/06665804-68bf-4952-a510-3be1689063eb/segments/0000/quaternions",
        "shape": [
          21705,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/06665804-68bf-4952-a510-3be1689063eb/segments/0000/timestamps",
        "shape": [
          21705
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/08bd7993-dc8e-442b-acc3-03488fae1bc6/segments/0000/quaternions",
        "shape": [
          20650,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/08bd7993-dc8e-442b-acc3-03488fae1bc6/segments/0000/timestamps",
        "shape": [
          20650
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/18d518e7-6fa8-4018-8f03-2007509ab065/segments/0000/quaternions",
        "shape": [
          25081,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/18d518e7-6fa8-4018-8f03-2007509ab065/segments/0000/timestamps",
        "shape": [
          25081
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/1ab0c277-768c-4a35-8404-c39be48f8dba/segments/0000/quaternions",
        "shape": [
          23025,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/1ab0c277-768c-4a35-8404-c39be48f8dba/segments/0000/timestamps",
        "shape": [
          23025
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/1beb2a2b-4a92-471e-b0db-908e2fa0835e/segments/0000/quaternions",
        "shape": [
          20246,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/1beb2a2b-4a92-471e-b0db-908e2fa0835e/segments/0000/timestamps",
        "shape": [
          20246
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/36a3b584-9e70-4c04-8a8b-bb6835af4b7e/segments/0000/quaternions",
        "shape": [
          19443,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/36a3b584-9e70-4c04-8a8b-bb6835af4b7e/segments/0000/timestamps",
        "shape": [
          19443
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/3720d7ec-0910-49c8-b79c-b87f5b0e9912/segments/0000/quaternions",
        "shape": [
          24309,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/3720d7ec-0910-49c8-b79c-b87f5b0e9912/segments/0000/timestamps",
        "shape": [
          24309
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/3c41b90d-7ea3-4a14-ade4-29d4d22702da/segments/0000/quaternions",
        "shape": [
          11841,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/3c41b90d-7ea3-4a14-ade4-29d4d22702da/segments/0000/timestamps",
        "shape": [
          11841
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/42bc72f5-90ae-4a02-bbc1-ba501eb585fb/segments/0000/quaternions",
        "shape": [
          19443,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/42bc72f5-90ae-4a02-bbc1-ba501eb585fb/segments/0000/timestamps",
        "shape": [
          19443
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/43e3b45d-a05d-452b-a03a-59288b57699a/segments/0000/quaternions",
        "shape": [
          13411,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/43e3b45d-a05d-452b-a03a-59288b57699a/segments/0000/timestamps",
        "shape": [
          13411
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/449e141d-97bc-4f00-823b-7861c10d53b1/segments/0000/quaternions",
        "shape": [
          18923,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/449e141d-97bc-4f00-823b-7861c10d53b1/segments/0000/timestamps",
        "shape": [
          18923
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4a368f4d-833a-44d2-82b4-8cde3d3ebed5/segments/0000/quaternions",
        "shape": [
          17150,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4a368f4d-833a-44d2-82b4-8cde3d3ebed5/segments/0000/timestamps",
        "shape": [
          17150
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4be4c058-43ed-4009-be4d-9b0b336ccdfb/segments/0000/quaternions",
        "shape": [
          17622,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4be4c058-43ed-4009-be4d-9b0b336ccdfb/segments/0000/timestamps",
        "shape": [
          17622
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4be4c058-43ed-4009-be4d-9b0b336ccdfb/segments/0001/quaternions",
        "shape": [
          132,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4be4c058-43ed-4009-be4d-9b0b336ccdfb/segments/0001/timestamps",
        "shape": [
          132
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4dff667e-0d36-4a54-8ca9-a4e3e7b00a77/segments/0000/quaternions",
        "shape": [
          11361,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/4dff667e-0d36-4a54-8ca9-a4e3e7b00a77/segments/0000/timestamps",
        "shape": [
          11361
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/52dfa1fa-5f4c-42ce-8060-d6e35c625f7a/segments/0000/quaternions",
        "shape": [
          14977,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/52dfa1fa-5f4c-42ce-8060-d6e35c625f7a/segments/0000/timestamps",
        "shape": [
          14977
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/53fe906f-0550-4272-b861-69d3bf10f283/segments/0000/quaternions",
        "shape": [
          8480,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/53fe906f-0550-4272-b861-69d3bf10f283/segments/0000/timestamps",
        "shape": [
          8480
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/5a5b5fb7-a0ba-4d9a-8660-598b5a3014b0/segments/0000/quaternions",
        "shape": [
          10290,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/5a5b5fb7-a0ba-4d9a-8660-598b5a3014b0/segments/0000/timestamps",
        "shape": [
          10290
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/5a5b5fb7-a0ba-4d9a-8660-598b5a3014b0/segments/0001/quaternions",
        "shape": [
          131,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/5a5b5fb7-a0ba-4d9a-8660-598b5a3014b0/segments/0001/timestamps",
        "shape": [
          131
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/5d3b10d1-b3d9-4e69-af5a-4c6646e00637/segments/0000/quaternions",
        "shape": [
          18338,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/5d3b10d1-b3d9-4e69-af5a-4c6646e00637/segments/0000/timestamps",
        "shape": [
          18338
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/61ea04db-03e8-44d3-b244-ac2bf7c4c47e/segments/0000/quaternions",
        "shape": [
          17317,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/61ea04db-03e8-44d3-b244-ac2bf7c4c47e/segments/0000/timestamps",
        "shape": [
          17317
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/61ea04db-03e8-44d3-b244-ac2bf7c4c47e/segments/0001/quaternions",
        "shape": [
          131,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/61ea04db-03e8-44d3-b244-ac2bf7c4c47e/segments/0001/timestamps",
        "shape": [
          131
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/63f3aacf-bbe1-4276-9dda-e8aa3c208911/segments/0000/quaternions",
        "shape": [
          13802,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/63f3aacf-bbe1-4276-9dda-e8aa3c208911/segments/0000/timestamps",
        "shape": [
          13802
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/65552dea-be7f-4aca-9a83-88be40325fc6/segments/0000/quaternions",
        "shape": [
          16994,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/65552dea-be7f-4aca-9a83-88be40325fc6/segments/0000/timestamps",
        "shape": [
          16994
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6911cb00-494a-420e-9766-2b0292b16435/segments/0000/quaternions",
        "shape": [
          17585,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6911cb00-494a-420e-9766-2b0292b16435/segments/0000/timestamps",
        "shape": [
          17585
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6a9fcd63-7ef5-47d5-82ad-60ebcf605229/segments/0000/quaternions",
        "shape": [
          11568,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6a9fcd63-7ef5-47d5-82ad-60ebcf605229/segments/0000/timestamps",
        "shape": [
          11568
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6f056bc8-96cc-4250-b222-4948206f3db8/segments/0000/quaternions",
        "shape": [
          13478,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/6f056bc8-96cc-4250-b222-4948206f3db8/segments/0000/timestamps",
        "shape": [
          13478
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/72a634db-7e7c-4f5a-a214-2772847cdf3a/segments/0000/quaternions",
        "shape": [
          23621,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/72a634db-7e7c-4f5a-a214-2772847cdf3a/segments/0000/timestamps",
        "shape": [
          23621
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/72a634db-7e7c-4f5a-a214-2772847cdf3a/segments/0001/quaternions",
        "shape": [
          132,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/72a634db-7e7c-4f5a-a214-2772847cdf3a/segments/0001/timestamps",
        "shape": [
          132
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/74146ba6-d47f-4869-bba7-028fc558334a/segments/0000/quaternions",
        "shape": [
          13093,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/74146ba6-d47f-4869-bba7-028fc558334a/segments/0000/timestamps",
        "shape": [
          13093
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/74146ba6-d47f-4869-bba7-028fc558334a/segments/0001/quaternions",
        "shape": [
          132,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/74146ba6-d47f-4869-bba7-028fc558334a/segments/0001/timestamps",
        "shape": [
          132
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/7484b559-f93f-476c-822d-c4523d8d2056/segments/0000/quaternions",
        "shape": [
          17676,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/7484b559-f93f-476c-822d-c4523d8d2056/segments/0000/timestamps",
        "shape": [
          17676
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/7ea6e9eb-8188-47c4-9346-52e41d4350e3/segments/0000/quaternions",
        "shape": [
          84,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/7ea6e9eb-8188-47c4-9346-52e41d4350e3/segments/0000/timestamps",
        "shape": [
          84
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/7ea6e9eb-8188-47c4-9346-52e41d4350e3/segments/0001/quaternions",
        "shape": [
          23137,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/7ea6e9eb-8188-47c4-9346-52e41d4350e3/segments/0001/timestamps",
        "shape": [
          23137
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/segments/0000/quaternions",
        "shape": [
          8206,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/segments/0000/timestamps",
        "shape": [
          8206
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/segments/0001/quaternions",
        "shape": [
          6,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/segments/0001/timestamps",
        "shape": [
          6
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/segments/0002/quaternions",
        "shape": [
          10012,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/segments/0002/timestamps",
        "shape": [
          10012
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/segments/0003/quaternions",
        "shape": [
          184,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/segments/0003/timestamps",
        "shape": [
          184
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/segments/0004/quaternions",
        "shape": [
          3244,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/800260fd-f59e-43d4-9395-b97acf598c18/segments/0004/timestamps",
        "shape": [
          3244
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/80e7dc8e-2c2c-4810-bc72-d5b894a2ed51/segments/0000/quaternions",
        "shape": [
          14483,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/80e7dc8e-2c2c-4810-bc72-d5b894a2ed51/segments/0000/timestamps",
        "shape": [
          14483
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/845189e5-da4c-4a30-813d-8326f6cb73dc/segments/0000/quaternions",
        "shape": [
          11847,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/845189e5-da4c-4a30-813d-8326f6cb73dc/segments/0000/timestamps",
        "shape": [
          11847
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/88724b28-641d-40b5-8c56-117740831111/segments/0000/quaternions",
        "shape": [
          26944,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/88724b28-641d-40b5-8c56-117740831111/segments/0000/timestamps",
        "shape": [
          26944
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/88724b28-641d-40b5-8c56-117740831111/segments/0001/quaternions",
        "shape": [
          132,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/88724b28-641d-40b5-8c56-117740831111/segments/0001/timestamps",
        "shape": [
          132
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0000/quaternions",
        "shape": [
          844,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0000/timestamps",
        "shape": [
          844
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0001/quaternions",
        "shape": [
          1151,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0001/timestamps",
        "shape": [
          1151
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0002/quaternions",
        "shape": [
          3642,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0002/timestamps",
        "shape": [
          3642
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0003/quaternions",
        "shape": [
          4280,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0003/timestamps",
        "shape": [
          4280
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0004/quaternions",
        "shape": [
          2004,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0004/timestamps",
        "shape": [
          2004
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0005/quaternions",
        "shape": [
          7036,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8a700867-ee0c-4514-ba66-eeab54c44936/segments/0005/timestamps",
        "shape": [
          7036
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0000/quaternions",
        "shape": [
          71,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0000/timestamps",
        "shape": [
          71
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0001/quaternions",
        "shape": [
          9742,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0001/timestamps",
        "shape": [
          9742
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0002/quaternions",
        "shape": [
          656,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0002/timestamps",
        "shape": [
          656
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0003/quaternions",
        "shape": [
          885,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0003/timestamps",
        "shape": [
          885
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0004/quaternions",
        "shape": [
          4848,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0004/timestamps",
        "shape": [
          4848
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0005/quaternions",
        "shape": [
          6536,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0005/timestamps",
        "shape": [
          6536
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0006/quaternions",
        "shape": [
          929,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0006/timestamps",
        "shape": [
          929
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0007/quaternions",
        "shape": [
          131,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8c2a3678-36d3-438c-95ba-d39a08fbb443/segments/0007/timestamps",
        "shape": [
          131
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8d8aeadf-b4a0-4783-8aef-4cbcb4b63628/segments/0000/quaternions",
        "shape": [
          18338,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8d8aeadf-b4a0-4783-8aef-4cbcb4b63628/segments/0000/timestamps",
        "shape": [
          18338
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8e364e69-c51c-46ad-ae9c-5976b3e5d72d/segments/0000/quaternions",
        "shape": [
          22692,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8e364e69-c51c-46ad-ae9c-5976b3e5d72d/segments/0000/timestamps",
        "shape": [
          22692
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8e9573ce-d378-4708-9c81-18c5871d66b6/segments/0000/quaternions",
        "shape": [
          17407,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8e9573ce-d378-4708-9c81-18c5871d66b6/segments/0000/timestamps",
        "shape": [
          17407
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8ed9dae8-e94c-4dba-8780-48077210bcba/segments/0000/quaternions",
        "shape": [
          21940,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8ed9dae8-e94c-4dba-8780-48077210bcba/segments/0000/timestamps",
        "shape": [
          21940
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8ed9dae8-e94c-4dba-8780-48077210bcba/segments/0001/quaternions",
        "shape": [
          131,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/8ed9dae8-e94c-4dba-8780-48077210bcba/segments/0001/timestamps",
        "shape": [
          131
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/97683149-ae91-4b47-aa6a-4a38dc2e3bc8/segments/0000/quaternions",
        "shape": [
          21204,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/97683149-ae91-4b47-aa6a-4a38dc2e3bc8/segments/0000/timestamps",
        "shape": [
          21204
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/97683149-ae91-4b47-aa6a-4a38dc2e3bc8/segments/0001/quaternions",
        "shape": [
          3088,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/97683149-ae91-4b47-aa6a-4a38dc2e3bc8/segments/0001/timestamps",
        "shape": [
          3088
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/9d5fa1a5-9f1e-4d96-91ff-95a9859e6710/segments/0000/quaternions",
        "shape": [
          16986,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/9d5fa1a5-9f1e-4d96-91ff-95a9859e6710/segments/0000/timestamps",
        "shape": [
          16986
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/9d5fa1a5-9f1e-4d96-91ff-95a9859e6710/segments/0001/quaternions",
        "shape": [
          5924,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/9d5fa1a5-9f1e-4d96-91ff-95a9859e6710/segments/0001/timestamps",
        "shape": [
          5924
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/9de2a636-f5a2-4714-963e-a5027eecfb92/segments/0000/quaternions",
        "shape": [
          17295,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/9de2a636-f5a2-4714-963e-a5027eecfb92/segments/0000/timestamps",
        "shape": [
          17295
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a1e6bdea-4da7-4d8a-8f56-817c5ff88b58/segments/0000/quaternions",
        "shape": [
          20105,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a1e6bdea-4da7-4d8a-8f56-817c5ff88b58/segments/0000/timestamps",
        "shape": [
          20105
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a2250eb3-f762-467b-908a-f10e796fea71/segments/0000/quaternions",
        "shape": [
          16145,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a2250eb3-f762-467b-908a-f10e796fea71/segments/0000/timestamps",
        "shape": [
          16145
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a2250eb3-f762-467b-908a-f10e796fea71/segments/0001/quaternions",
        "shape": [
          131,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a2250eb3-f762-467b-908a-f10e796fea71/segments/0001/timestamps",
        "shape": [
          131
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a95945dc-57e1-43d0-8137-23cf6600aacb/segments/0000/quaternions",
        "shape": [
          16736,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a95945dc-57e1-43d0-8137-23cf6600aacb/segments/0000/timestamps",
        "shape": [
          16736
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a95945dc-57e1-43d0-8137-23cf6600aacb/segments/0001/quaternions",
        "shape": [
          131,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/a95945dc-57e1-43d0-8137-23cf6600aacb/segments/0001/timestamps",
        "shape": [
          131
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/abba7a78-fdd8-48ef-90fe-c371c9573a7b/segments/0000/quaternions",
        "shape": [
          2481,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/abba7a78-fdd8-48ef-90fe-c371c9573a7b/segments/0000/timestamps",
        "shape": [
          2481
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6a538a9-c2bf-4c37-a0c0-9bbd48dfe1a9/segments/0000/quaternions",
        "shape": [
          12027,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6a538a9-c2bf-4c37-a0c0-9bbd48dfe1a9/segments/0000/timestamps",
        "shape": [
          12027
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6a538a9-c2bf-4c37-a0c0-9bbd48dfe1a9/segments/0001/quaternions",
        "shape": [
          7694,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6a538a9-c2bf-4c37-a0c0-9bbd48dfe1a9/segments/0001/timestamps",
        "shape": [
          7694
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6a538a9-c2bf-4c37-a0c0-9bbd48dfe1a9/segments/0002/quaternions",
        "shape": [
          131,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6a538a9-c2bf-4c37-a0c0-9bbd48dfe1a9/segments/0002/timestamps",
        "shape": [
          131
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6c797c5-a454-49db-b08b-fa6cdb37e393/segments/0000/quaternions",
        "shape": [
          1325,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6c797c5-a454-49db-b08b-fa6cdb37e393/segments/0000/timestamps",
        "shape": [
          1325
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6c797c5-a454-49db-b08b-fa6cdb37e393/segments/0001/quaternions",
        "shape": [
          11706,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/b6c797c5-a454-49db-b08b-fa6cdb37e393/segments/0001/timestamps",
        "shape": [
          11706
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/be2b5237-0525-40f1-8820-6c9132492e8b/segments/0000/quaternions",
        "shape": [
          3527,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/be2b5237-0525-40f1-8820-6c9132492e8b/segments/0000/timestamps",
        "shape": [
          3527
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/bfdde7ba-77d8-4363-a742-0d0d8c7e11ea/segments/0000/quaternions",
        "shape": [
          18239,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/bfdde7ba-77d8-4363-a742-0d0d8c7e11ea/segments/0000/timestamps",
        "shape": [
          18239
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/cd3684f2-4d5a-41c8-8233-a134407ccdac/segments/0000/quaternions",
        "shape": [
          1997,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/cd3684f2-4d5a-41c8-8233-a134407ccdac/segments/0000/timestamps",
        "shape": [
          1997
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d4f62f36-957e-45fe-ab57-737d79cd7789/segments/0000/quaternions",
        "shape": [
          6053,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d4f62f36-957e-45fe-ab57-737d79cd7789/segments/0000/timestamps",
        "shape": [
          6053
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d4f62f36-957e-45fe-ab57-737d79cd7789/segments/0001/quaternions",
        "shape": [
          3659,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d4f62f36-957e-45fe-ab57-737d79cd7789/segments/0001/timestamps",
        "shape": [
          3659
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d77c03be-a31d-414e-82d7-92e4eb33067e/segments/0000/quaternions",
        "shape": [
          17368,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d77c03be-a31d-414e-82d7-92e4eb33067e/segments/0000/timestamps",
        "shape": [
          17368
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d77c03be-a31d-414e-82d7-92e4eb33067e/segments/0001/quaternions",
        "shape": [
          131,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/d77c03be-a31d-414e-82d7-92e4eb33067e/segments/0001/timestamps",
        "shape": [
          131
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/dc59d6aa-0f42-4412-8ad8-31111e85b2f1/segments/0000/quaternions",
        "shape": [
          17431,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/dc59d6aa-0f42-4412-8ad8-31111e85b2f1/segments/0000/timestamps",
        "shape": [
          17431
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/segments/0000/quaternions",
        "shape": [
          208,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/segments/0000/timestamps",
        "shape": [
          208
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/segments/0001/quaternions",
        "shape": [
          1598,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/segments/0001/timestamps",
        "shape": [
          1598
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/segments/0002/quaternions",
        "shape": [
          3428,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/segments/0002/timestamps",
        "shape": [
          3428
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/segments/0003/quaternions",
        "shape": [
          1872,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/segments/0003/timestamps",
        "shape": [
          1872
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/segments/0004/quaternions",
        "shape": [
          4183,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/e3905b28-3fe3-4f81-9ad3-e0de6744ea62/segments/0004/timestamps",
        "shape": [
          4183
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ee3891da-55f0-4b93-9e84-efec2ff4274c/segments/0000/quaternions",
        "shape": [
          19356,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ee3891da-55f0-4b93-9e84-efec2ff4274c/segments/0000/timestamps",
        "shape": [
          19356
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ef8df4e6-9a8a-409d-afec-d859dd382be1/segments/0000/quaternions",
        "shape": [
          15055,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ef8df4e6-9a8a-409d-afec-d859dd382be1/segments/0000/timestamps",
        "shape": [
          15055
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f15514d9-fdeb-40c1-877b-736740aef5f3/segments/0000/quaternions",
        "shape": [
          16406,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f15514d9-fdeb-40c1-877b-736740aef5f3/segments/0000/timestamps",
        "shape": [
          16406
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f1a3875f-b56a-4290-8d57-5cbbc7248e1c/segments/0000/quaternions",
        "shape": [
          23030,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f1a3875f-b56a-4290-8d57-5cbbc7248e1c/segments/0000/timestamps",
        "shape": [
          23030
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f59fee7e-bf1d-4e63-9776-dc84a2c964e5/segments/0000/quaternions",
        "shape": [
          25678,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/f59fee7e-bf1d-4e63-9776-dc84a2c964e5/segments/0000/timestamps",
        "shape": [
          25678
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/fb974c1d-0403-4917-b326-3a69d96c335e/segments/0000/quaternions",
        "shape": [
          8469,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/fb974c1d-0403-4917-b326-3a69d96c335e/segments/0000/timestamps",
        "shape": [
          8469
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/fc76d986-54b1-46c2-80f2-a918f5333e25/segments/0000/quaternions",
        "shape": [
          11530,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/fc76d986-54b1-46c2-80f2-a918f5333e25/segments/0000/timestamps",
        "shape": [
          11530
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ff61c263-8591-4c82-8255-9eaad43e0928/segments/0000/quaternions",
        "shape": [
          1860,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ff61c263-8591-4c82-8255-9eaad43e0928/segments/0000/timestamps",
        "shape": [
          1860
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ffb4d561-67c7-4716-a194-3f0ff80d749e/segments/0000/quaternions",
        "shape": [
          17544,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008a4099-eadc-4013-b4d6-69ea848c6949/sessions/ffb4d561-67c7-4716-a194-3f0ff80d749e/segments/0000/timestamps",
        "shape": [
          17544
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/1cd90ba4-a27a-4d0c-8df5-13c35ff95cec/segments/0000/quaternions",
        "shape": [
          21777,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/1cd90ba4-a27a-4d0c-8df5-13c35ff95cec/segments/0000/timestamps",
        "shape": [
          21777
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/2348afb9-8b30-477e-a901-802b9e8d3581/segments/0000/quaternions",
        "shape": [
          1242,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/2348afb9-8b30-477e-a901-802b9e8d3581/segments/0000/timestamps",
        "shape": [
          1242
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/2348afb9-8b30-477e-a901-802b9e8d3581/segments/0001/quaternions",
        "shape": [
          21843,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/2348afb9-8b30-477e-a901-802b9e8d3581/segments/0001/timestamps",
        "shape": [
          21843
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/29ae9b37-2206-4c0b-8344-3af75daa0ca4/segments/0000/quaternions",
        "shape": [
          19106,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/29ae9b37-2206-4c0b-8344-3af75daa0ca4/segments/0000/timestamps",
        "shape": [
          19106
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/2a06f0ef-3ffb-4d2b-8d6f-8c2a222152ce/segments/0000/quaternions",
        "shape": [
          22745,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/2a06f0ef-3ffb-4d2b-8d6f-8c2a222152ce/segments/0000/timestamps",
        "shape": [
          22745
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/46eee1cb-4799-4b31-b23a-02bfccbfc124/segments/0000/quaternions",
        "shape": [
          19106,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/46eee1cb-4799-4b31-b23a-02bfccbfc124/segments/0000/timestamps",
        "shape": [
          19106
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/6fb13ce8-b7f9-4be8-b09b-104e2f617cc6/segments/0000/quaternions",
        "shape": [
          19844,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/6fb13ce8-b7f9-4be8-b09b-104e2f617cc6/segments/0000/timestamps",
        "shape": [
          19844
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/86067a02-cd48-4a26-8caa-442aebfd147a/segments/0000/quaternions",
        "shape": [
          8571,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/86067a02-cd48-4a26-8caa-442aebfd147a/segments/0000/timestamps",
        "shape": [
          8571
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/9a169cda-4f88-4ae6-8607-1c2c38961b44/segments/0000/quaternions",
        "shape": [
          23873,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/9a169cda-4f88-4ae6-8607-1c2c38961b44/segments/0000/timestamps",
        "shape": [
          23873
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/b0cde597-0d9e-44a7-af96-8ed88c2db1f9/segments/0000/quaternions",
        "shape": [
          16725,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/b0cde597-0d9e-44a7-af96-8ed88c2db1f9/segments/0000/timestamps",
        "shape": [
          16725
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/bb39a4c2-6cd7-4b15-8613-7c1ffd3e985c/segments/0000/quaternions",
        "shape": [
          17829,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/bb39a4c2-6cd7-4b15-8613-7c1ffd3e985c/segments/0000/timestamps",
        "shape": [
          17829
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/bb39a4c2-6cd7-4b15-8613-7c1ffd3e985c/segments/0001/quaternions",
        "shape": [
          4910,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/bb39a4c2-6cd7-4b15-8613-7c1ffd3e985c/segments/0001/timestamps",
        "shape": [
          4910
        ],
        "dtype": "float64"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/c6fa3515-d3a2-46b9-b906-0d047be96222/segments/0000/quaternions",
        "shape": [
          22551,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/008c5ce1-7db3-416f-ab82-21b22c6266ee/sessions/c6fa3515-d3a2-46b9-b906-0d047be96222/segments/0000/timestamps",
        "shape": [
          22551
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00926ea3-8296-4f72-959d-a9b289bb541e/sessions/5e51e407-7c50-4284-a48f-aaf4019b7071/segments/0000/quaternions",
        "shape": [
          9147,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00926ea3-8296-4f72-959d-a9b289bb541e/sessions/5e51e407-7c50-4284-a48f-aaf4019b7071/segments/0000/timestamps",
        "shape": [
          9147
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00926ea3-8296-4f72-959d-a9b289bb541e/sessions/5e51e407-7c50-4284-a48f-aaf4019b7071/segments/0001/quaternions",
        "shape": [
          7606,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00926ea3-8296-4f72-959d-a9b289bb541e/sessions/5e51e407-7c50-4284-a48f-aaf4019b7071/segments/0001/timestamps",
        "shape": [
          7606
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00a82f96-9871-4d8d-a1c7-c739e34f928f/sessions/321a63e9-cc19-4faa-abc2-63c249d0d89b/segments/0000/quaternions",
        "shape": [
          22829,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00a82f96-9871-4d8d-a1c7-c739e34f928f/sessions/321a63e9-cc19-4faa-abc2-63c249d0d89b/segments/0000/timestamps",
        "shape": [
          22829
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00a82f96-9871-4d8d-a1c7-c739e34f928f/sessions/c1256ad6-6e63-4669-ac1b-7158206e9a84/segments/0000/quaternions",
        "shape": [
          8597,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00a82f96-9871-4d8d-a1c7-c739e34f928f/sessions/c1256ad6-6e63-4669-ac1b-7158206e9a84/segments/0000/timestamps",
        "shape": [
          8597
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00a85435-a5d4-4ce3-b452-3d99b6f8b399/sessions/aa152f3e-67e8-4220-8d0a-7c919ee0c112/segments/0000/quaternions",
        "shape": [
          10378,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00a85435-a5d4-4ce3-b452-3d99b6f8b399/sessions/aa152f3e-67e8-4220-8d0a-7c919ee0c112/segments/0000/timestamps",
        "shape": [
          10378
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00a85435-a5d4-4ce3-b452-3d99b6f8b399/sessions/aa152f3e-67e8-4220-8d0a-7c919ee0c112/segments/0001/quaternions",
        "shape": [
          10033,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00a85435-a5d4-4ce3-b452-3d99b6f8b399/sessions/aa152f3e-67e8-4220-8d0a-7c919ee0c112/segments/0001/timestamps",
        "shape": [
          10033
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0494f6f8-8266-4dfd-8bff-039cfaa85167/segments/0000/quaternions",
        "shape": [
          28400,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0494f6f8-8266-4dfd-8bff-039cfaa85167/segments/0000/timestamps",
        "shape": [
          28400
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0494f6f8-8266-4dfd-8bff-039cfaa85167/segments/0001/quaternions",
        "shape": [
          30,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0494f6f8-8266-4dfd-8bff-039cfaa85167/segments/0001/timestamps",
        "shape": [
          30
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b2d7047-6244-482e-8a69-37736a0d1f7f/segments/0000/quaternions",
        "shape": [
          157,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b2d7047-6244-482e-8a69-37736a0d1f7f/segments/0000/timestamps",
        "shape": [
          157
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b2d7047-6244-482e-8a69-37736a0d1f7f/segments/0001/quaternions",
        "shape": [
          20946,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b2d7047-6244-482e-8a69-37736a0d1f7f/segments/0001/timestamps",
        "shape": [
          20946
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b2d7047-6244-482e-8a69-37736a0d1f7f/segments/0002/quaternions",
        "shape": [
          1304,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b2d7047-6244-482e-8a69-37736a0d1f7f/segments/0002/timestamps",
        "shape": [
          1304
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b2d7047-6244-482e-8a69-37736a0d1f7f/segments/0003/quaternions",
        "shape": [
          32,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b2d7047-6244-482e-8a69-37736a0d1f7f/segments/0003/timestamps",
        "shape": [
          32
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b506391-efca-47e2-81c1-a30506cc7b1c/segments/0000/quaternions",
        "shape": [
          14284,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b506391-efca-47e2-81c1-a30506cc7b1c/segments/0000/timestamps",
        "shape": [
          14284
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b506391-efca-47e2-81c1-a30506cc7b1c/segments/0001/quaternions",
        "shape": [
          31,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/0b506391-efca-47e2-81c1-a30506cc7b1c/segments/0001/timestamps",
        "shape": [
          31
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1007760b-ffd9-45f7-8af7-f4de77f3cd04/segments/0000/quaternions",
        "shape": [
          1308,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1007760b-ffd9-45f7-8af7-f4de77f3cd04/segments/0000/timestamps",
        "shape": [
          1308
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1007760b-ffd9-45f7-8af7-f4de77f3cd04/segments/0001/quaternions",
        "shape": [
          7426,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1007760b-ffd9-45f7-8af7-f4de77f3cd04/segments/0001/timestamps",
        "shape": [
          7426
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/146c1920-8262-4e24-9e90-d699c0025dd2/segments/0000/quaternions",
        "shape": [
          10846,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/146c1920-8262-4e24-9e90-d699c0025dd2/segments/0000/timestamps",
        "shape": [
          10846
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1bb847f8-8ecd-4d79-aa99-242a9a3d2386/segments/0000/quaternions",
        "shape": [
          5497,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1bb847f8-8ecd-4d79-aa99-242a9a3d2386/segments/0000/timestamps",
        "shape": [
          5497
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1bb847f8-8ecd-4d79-aa99-242a9a3d2386/segments/0001/quaternions",
        "shape": [
          3438,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1bb847f8-8ecd-4d79-aa99-242a9a3d2386/segments/0001/timestamps",
        "shape": [
          3438
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1c49274b-4ab9-456a-9c7e-d16f30929dfb/segments/0000/quaternions",
        "shape": [
          9384,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1c49274b-4ab9-456a-9c7e-d16f30929dfb/segments/0000/timestamps",
        "shape": [
          9384
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1c49274b-4ab9-456a-9c7e-d16f30929dfb/segments/0001/quaternions",
        "shape": [
          1458,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1c49274b-4ab9-456a-9c7e-d16f30929dfb/segments/0001/timestamps",
        "shape": [
          1458
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1e9346ed-9f96-4cfa-a0c7-dabd0121e711/segments/0000/quaternions",
        "shape": [
          5661,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/1e9346ed-9f96-4cfa-a0c7-dabd0121e711/segments/0000/timestamps",
        "shape": [
          5661
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2474682a-06c7-4218-afe5-b8739cd65c4e/segments/0000/quaternions",
        "shape": [
          11825,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2474682a-06c7-4218-afe5-b8739cd65c4e/segments/0000/timestamps",
        "shape": [
          11825
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2474682a-06c7-4218-afe5-b8739cd65c4e/segments/0001/quaternions",
        "shape": [
          21924,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2474682a-06c7-4218-afe5-b8739cd65c4e/segments/0001/timestamps",
        "shape": [
          21924
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2647c161-532a-4113-8941-00380f20f3a7/segments/0000/quaternions",
        "shape": [
          17340,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2647c161-532a-4113-8941-00380f20f3a7/segments/0000/timestamps",
        "shape": [
          17340
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2647c161-532a-4113-8941-00380f20f3a7/segments/0001/quaternions",
        "shape": [
          31,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/2647c161-532a-4113-8941-00380f20f3a7/segments/0001/timestamps",
        "shape": [
          31
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/29ae3c69-ce66-4ff0-97cc-ae367e4cc00c/segments/0000/quaternions",
        "shape": [
          20646,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/29ae3c69-ce66-4ff0-97cc-ae367e4cc00c/segments/0000/timestamps",
        "shape": [
          20646
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/356cedd6-23c9-428a-b32b-ab34d596292a/segments/0000/quaternions",
        "shape": [
          8627,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/356cedd6-23c9-428a-b32b-ab34d596292a/segments/0000/timestamps",
        "shape": [
          8627
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/3644ca0f-7ace-41dd-aa36-d3feec124a0c/segments/0000/quaternions",
        "shape": [
          21872,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/3644ca0f-7ace-41dd-aa36-d3feec124a0c/segments/0000/timestamps",
        "shape": [
          21872
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/48d802ce-674f-49f7-93db-0e29890873e2/segments/0000/quaternions",
        "shape": [
          1276,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/48d802ce-674f-49f7-93db-0e29890873e2/segments/0000/timestamps",
        "shape": [
          1276
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/50450f94-7dd3-461c-8ae4-10e76a8653bb/segments/0000/quaternions",
        "shape": [
          6679,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/50450f94-7dd3-461c-8ae4-10e76a8653bb/segments/0000/timestamps",
        "shape": [
          6679
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/5941ae62-022f-4649-b9e1-c62eaa2b00a3/segments/0000/quaternions",
        "shape": [
          25136,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/5941ae62-022f-4649-b9e1-c62eaa2b00a3/segments/0000/timestamps",
        "shape": [
          25136
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/60d421cb-78b1-405b-8fda-3423910764d7/segments/0000/quaternions",
        "shape": [
          7762,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/60d421cb-78b1-405b-8fda-3423910764d7/segments/0000/timestamps",
        "shape": [
          7762
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/61c78e4e-77d5-4eb3-9198-7e9440e171fd/segments/0000/quaternions",
        "shape": [
          6172,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/61c78e4e-77d5-4eb3-9198-7e9440e171fd/segments/0000/timestamps",
        "shape": [
          6172
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0000/quaternions",
        "shape": [
          689,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0000/timestamps",
        "shape": [
          689
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0001/quaternions",
        "shape": [
          3548,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0001/timestamps",
        "shape": [
          3548
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0002/quaternions",
        "shape": [
          7,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0002/timestamps",
        "shape": [
          7
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0003/quaternions",
        "shape": [
          12,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0003/timestamps",
        "shape": [
          12
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0004/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0004/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0005/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0005/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0006/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0006/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0007/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0007/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0008/quaternions",
        "shape": [
          7,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0008/timestamps",
        "shape": [
          7
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0009/quaternions",
        "shape": [
          8,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0009/timestamps",
        "shape": [
          8
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0010/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0010/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0011/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0011/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0012/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0012/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0013/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0013/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0014/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0014/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0015/quaternions",
        "shape": [
          5,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0015/timestamps",
        "shape": [
          5
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0016/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0016/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0017/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0017/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0018/quaternions",
        "shape": [
          12,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0018/timestamps",
        "shape": [
          12
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0019/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0019/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0020/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0020/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0021/quaternions",
        "shape": [
          12,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0021/timestamps",
        "shape": [
          12
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0022/quaternions",
        "shape": [
          23,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0022/timestamps",
        "shape": [
          23
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0023/quaternions",
        "shape": [
          11,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0023/timestamps",
        "shape": [
          11
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0024/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0024/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0025/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0025/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0026/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0026/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0027/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0027/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0028/quaternions",
        "shape": [
          12,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0028/timestamps",
        "shape": [
          12
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0029/quaternions",
        "shape": [
          12,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0029/timestamps",
        "shape": [
          12
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0030/quaternions",
        "shape": [
          15,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0030/timestamps",
        "shape": [
          15
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0031/quaternions",
        "shape": [
          4,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0031/timestamps",
        "shape": [
          4
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0032/quaternions",
        "shape": [
          406,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/651093e8-2815-4303-83a3-ce6ca61bf8fb/segments/0032/timestamps",
        "shape": [
          406
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/698713e4-d18e-49d7-9935-8a569b5ff2a5/segments/0000/quaternions",
        "shape": [
          18343,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/698713e4-d18e-49d7-9935-8a569b5ff2a5/segments/0000/timestamps",
        "shape": [
          18343
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/6c8a9d0d-a00c-450b-a4e2-d1d8fa998d50/segments/0000/quaternions",
        "shape": [
          798,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/6c8a9d0d-a00c-450b-a4e2-d1d8fa998d50/segments/0000/timestamps",
        "shape": [
          798
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/707c9ea7-ebba-4f82-bdab-c14e9f0288f0/segments/0000/quaternions",
        "shape": [
          13322,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/707c9ea7-ebba-4f82-bdab-c14e9f0288f0/segments/0000/timestamps",
        "shape": [
          13322
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/70b7d934-b91e-4209-b09a-789ad7595f4b/segments/0000/quaternions",
        "shape": [
          2697,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/70b7d934-b91e-4209-b09a-789ad7595f4b/segments/0000/timestamps",
        "shape": [
          2697
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/7f5569b8-a641-4a71-b15e-08569310f1ef/segments/0000/quaternions",
        "shape": [
          11639,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/7f5569b8-a641-4a71-b15e-08569310f1ef/segments/0000/timestamps",
        "shape": [
          11639
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/831d7362-9709-4089-b0ef-873a1ab95bf2/segments/0000/quaternions",
        "shape": [
          29025,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/831d7362-9709-4089-b0ef-873a1ab95bf2/segments/0000/timestamps",
        "shape": [
          29025
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/877c649e-5d91-4474-bfe3-8496c203f252/segments/0000/quaternions",
        "shape": [
          6033,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/877c649e-5d91-4474-bfe3-8496c203f252/segments/0000/timestamps",
        "shape": [
          6033
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/8aeec78a-02ca-4306-88d9-045dc9a6a9e5/segments/0000/quaternions",
        "shape": [
          5661,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/8aeec78a-02ca-4306-88d9-045dc9a6a9e5/segments/0000/timestamps",
        "shape": [
          5661
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/9447d4b0-d424-4bc7-bcad-483373d55fe1/segments/0000/quaternions",
        "shape": [
          5661,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/9447d4b0-d424-4bc7-bcad-483373d55fe1/segments/0000/timestamps",
        "shape": [
          5661
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/a2f5ec43-8903-49f6-ab58-47a792b5f52d/segments/0000/quaternions",
        "shape": [
          8627,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/a2f5ec43-8903-49f6-ab58-47a792b5f52d/segments/0000/timestamps",
        "shape": [
          8627
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/a51f84cf-cc2b-4ea7-8c66-106fc1ce6e3f/segments/0000/quaternions",
        "shape": [
          21536,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/a51f84cf-cc2b-4ea7-8c66-106fc1ce6e3f/segments/0000/timestamps",
        "shape": [
          21536
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/ae7b3e97-ba8c-416e-bcba-36b47c854b32/segments/0000/quaternions",
        "shape": [
          286,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/ae7b3e97-ba8c-416e-bcba-36b47c854b32/segments/0000/timestamps",
        "shape": [
          286
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/ae7b3e97-ba8c-416e-bcba-36b47c854b32/segments/0001/quaternions",
        "shape": [
          31,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/ae7b3e97-ba8c-416e-bcba-36b47c854b32/segments/0001/timestamps",
        "shape": [
          31
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/afa74301-5d4f-4e1b-9153-bff088e815c7/segments/0000/quaternions",
        "shape": [
          1116,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/afa74301-5d4f-4e1b-9153-bff088e815c7/segments/0000/timestamps",
        "shape": [
          1116
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/afa74301-5d4f-4e1b-9153-bff088e815c7/segments/0001/quaternions",
        "shape": [
          15392,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/afa74301-5d4f-4e1b-9153-bff088e815c7/segments/0001/timestamps",
        "shape": [
          15392
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/b03cfadd-7255-4d53-9fa5-6e68fdfc98db/segments/0000/quaternions",
        "shape": [
          10306,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/b03cfadd-7255-4d53-9fa5-6e68fdfc98db/segments/0000/timestamps",
        "shape": [
          10306
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/b03cfadd-7255-4d53-9fa5-6e68fdfc98db/segments/0001/quaternions",
        "shape": [
          3011,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/b03cfadd-7255-4d53-9fa5-6e68fdfc98db/segments/0001/timestamps",
        "shape": [
          3011
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0000/quaternions",
        "shape": [
          5285,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0000/timestamps",
        "shape": [
          5285
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0001/quaternions",
        "shape": [
          55,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0001/timestamps",
        "shape": [
          55
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0002/quaternions",
        "shape": [
          1213,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0002/timestamps",
        "shape": [
          1213
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0003/quaternions",
        "shape": [
          1002,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0003/timestamps",
        "shape": [
          1002
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0004/quaternions",
        "shape": [
          2369,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0004/timestamps",
        "shape": [
          2369
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0005/quaternions",
        "shape": [
          10,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0005/timestamps",
        "shape": [
          10
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0006/quaternions",
        "shape": [
          1458,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0006/timestamps",
        "shape": [
          1458
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0007/quaternions",
        "shape": [
          5,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0007/timestamps",
        "shape": [
          5
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0008/quaternions",
        "shape": [
          3457,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bad3209b-c963-4614-a0cd-093777c9faee/segments/0008/timestamps",
        "shape": [
          3457
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bf0de83e-ccc4-46d3-b4e2-cf2bea6ea89e/segments/0000/quaternions",
        "shape": [
          15431,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/bf0de83e-ccc4-46d3-b4e2-cf2bea6ea89e/segments/0000/timestamps",
        "shape": [
          15431
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0000/quaternions",
        "shape": [
          3867,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0000/timestamps",
        "shape": [
          3867
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0001/quaternions",
        "shape": [
          6275,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0001/timestamps",
        "shape": [
          6275
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0002/quaternions",
        "shape": [
          39,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0002/timestamps",
        "shape": [
          39
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0003/quaternions",
        "shape": [
          132,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0003/timestamps",
        "shape": [
          132
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0004/quaternions",
        "shape": [
          11,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0004/timestamps",
        "shape": [
          11
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0005/quaternions",
        "shape": [
          5,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0005/timestamps",
        "shape": [
          5
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0006/quaternions",
        "shape": [
          9,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0006/timestamps",
        "shape": [
          9
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0007/quaternions",
        "shape": [
          40,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0007/timestamps",
        "shape": [
          40
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0008/quaternions",
        "shape": [
          12,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0008/timestamps",
        "shape": [
          12
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0009/quaternions",
        "shape": [
          1092,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/c0d3ae3e-c1cb-4e3e-b161-38f167facf33/segments/0009/timestamps",
        "shape": [
          1092
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/cd85cf88-620f-4107-878e-98acf956844e/segments/0000/quaternions",
        "shape": [
          7782,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/cd85cf88-620f-4107-878e-98acf956844e/segments/0000/timestamps",
        "shape": [
          7782
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/d3b5807d-3816-4049-bce1-0f0231e559da/segments/0000/quaternions",
        "shape": [
          313,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/d3b5807d-3816-4049-bce1-0f0231e559da/segments/0000/timestamps",
        "shape": [
          313
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/d3b5807d-3816-4049-bce1-0f0231e559da/segments/0001/quaternions",
        "shape": [
          28704,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/d3b5807d-3816-4049-bce1-0f0231e559da/segments/0001/timestamps",
        "shape": [
          28704
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/d5a3260a-1b43-4657-9480-c969d1e8845f/segments/0000/quaternions",
        "shape": [
          2941,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/d5a3260a-1b43-4657-9480-c969d1e8845f/segments/0000/timestamps",
        "shape": [
          2941
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/e4c14c73-14a7-4821-aec3-c21dddeef590/segments/0000/quaternions",
        "shape": [
          13326,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00c6dd85-707d-4e77-9bac-fcd00cd58cd8/sessions/e4c14c73-14a7-4821-aec3-c21dddeef590/segments/0000/timestamps",
        "shape": [
          13326
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00d2ae65-e526-422e-8418-a0505a16187c/sessions/137f1f67-7154-4b98-bb00-6f053eb5ffa1/segments/0000/quaternions",
        "shape": [
          17757,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00d2ae65-e526-422e-8418-a0505a16187c/sessions/137f1f67-7154-4b98-bb00-6f053eb5ffa1/segments/0000/timestamps",
        "shape": [
          17757
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00dbd9ee-71c8-4bad-ad21-8eee56a4c200/sessions/d7abd210-1bf3-47a0-83e7-2e089c156181/segments/0000/quaternions",
        "shape": [
          23515,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00dbd9ee-71c8-4bad-ad21-8eee56a4c200/sessions/d7abd210-1bf3-47a0-83e7-2e089c156181/segments/0000/timestamps",
        "shape": [
          23515
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00dbd9ee-71c8-4bad-ad21-8eee56a4c200/sessions/d7abd210-1bf3-47a0-83e7-2e089c156181/segments/0001/quaternions",
        "shape": [
          46,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00dbd9ee-71c8-4bad-ad21-8eee56a4c200/sessions/d7abd210-1bf3-47a0-83e7-2e089c156181/segments/0001/timestamps",
        "shape": [
          46
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00dbd9ee-71c8-4bad-ad21-8eee56a4c200/sessions/d7abd210-1bf3-47a0-83e7-2e089c156181/segments/0002/quaternions",
        "shape": [
          524,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00dbd9ee-71c8-4bad-ad21-8eee56a4c200/sessions/d7abd210-1bf3-47a0-83e7-2e089c156181/segments/0002/timestamps",
        "shape": [
          524
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00dbd9ee-71c8-4bad-ad21-8eee56a4c200/sessions/d7abd210-1bf3-47a0-83e7-2e089c156181/segments/0003/quaternions",
        "shape": [
          827,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00dbd9ee-71c8-4bad-ad21-8eee56a4c200/sessions/d7abd210-1bf3-47a0-83e7-2e089c156181/segments/0003/timestamps",
        "shape": [
          827
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/segments/0000/quaternions",
        "shape": [
          4530,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/segments/0000/timestamps",
        "shape": [
          4530
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/segments/0001/quaternions",
        "shape": [
          7885,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/segments/0001/timestamps",
        "shape": [
          7885
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/segments/0002/quaternions",
        "shape": [
          29,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/segments/0002/timestamps",
        "shape": [
          29
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/segments/0003/quaternions",
        "shape": [
          45,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/segments/0003/timestamps",
        "shape": [
          45
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/segments/0004/quaternions",
        "shape": [
          3321,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00e5a085-dbaf-46c4-abbc-b45edde3d662/sessions/170effa3-f258-4a26-8093-2c21bb905b7f/segments/0004/timestamps",
        "shape": [
          3321
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00e95b89-2b88-4c29-974d-cf9e1934ec5b/sessions/353bbf02-559a-48c8-922f-b1d541a1c413/segments/0000/quaternions",
        "shape": [
          2768,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00e95b89-2b88-4c29-974d-cf9e1934ec5b/sessions/353bbf02-559a-48c8-922f-b1d541a1c413/segments/0000/timestamps",
        "shape": [
          2768
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f13eb1-6bc7-4e38-8525-5b412af4b9b7/sessions/1e7826cd-88d0-46d6-abd1-43b3e88f9b31/segments/0000/quaternions",
        "shape": [
          15835,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f13eb1-6bc7-4e38-8525-5b412af4b9b7/sessions/1e7826cd-88d0-46d6-abd1-43b3e88f9b31/segments/0000/timestamps",
        "shape": [
          15835
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f13eb1-6bc7-4e38-8525-5b412af4b9b7/sessions/d6000cf8-d5e3-411d-887d-187155ac9e2e/segments/0000/quaternions",
        "shape": [
          16540,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f13eb1-6bc7-4e38-8525-5b412af4b9b7/sessions/d6000cf8-d5e3-411d-887d-187155ac9e2e/segments/0000/timestamps",
        "shape": [
          16540
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0000/quaternions",
        "shape": [
          3175,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0000/timestamps",
        "shape": [
          3175
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0001/quaternions",
        "shape": [
          1287,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0001/timestamps",
        "shape": [
          1287
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0002/quaternions",
        "shape": [
          7838,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0002/timestamps",
        "shape": [
          7838
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0003/quaternions",
        "shape": [
          1086,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0003/timestamps",
        "shape": [
          1086
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0004/quaternions",
        "shape": [
          79,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0004/timestamps",
        "shape": [
          79
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0005/quaternions",
        "shape": [
          1763,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f4ddc2-a595-47a4-8883-16284807a850/sessions/77a2ab15-26b5-4797-b0a9-8ad0ca2e0cb8/segments/0005/timestamps",
        "shape": [
          1763
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00f73e02-eb53-4387-bd0f-5c701474261f/sessions/1c07646c-2301-4412-aac2-0effc55bf9c2/segments/0000/quaternions",
        "shape": [
          1842,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00f73e02-eb53-4387-bd0f-5c701474261f/sessions/1c07646c-2301-4412-aac2-0effc55bf9c2/segments/0000/timestamps",
        "shape": [
          1842
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/04231a70-0286-4084-817f-e3fb8984cff0/segments/0000/quaternions",
        "shape": [
          12970,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/04231a70-0286-4084-817f-e3fb8984cff0/segments/0000/timestamps",
        "shape": [
          12970
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/04231a70-0286-4084-817f-e3fb8984cff0/segments/0001/quaternions",
        "shape": [
          4026,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/04231a70-0286-4084-817f-e3fb8984cff0/segments/0001/timestamps",
        "shape": [
          4026
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/1268d39c-458a-46cf-b727-d6157564405f/segments/0000/quaternions",
        "shape": [
          2772,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/1268d39c-458a-46cf-b727-d6157564405f/segments/0000/timestamps",
        "shape": [
          2772
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/1268d39c-458a-46cf-b727-d6157564405f/segments/0001/quaternions",
        "shape": [
          3636,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/1268d39c-458a-46cf-b727-d6157564405f/segments/0001/timestamps",
        "shape": [
          3636
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/1268d39c-458a-46cf-b727-d6157564405f/segments/0002/quaternions",
        "shape": [
          581,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/1268d39c-458a-46cf-b727-d6157564405f/segments/0002/timestamps",
        "shape": [
          581
        ],
        "dtype": "float64"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/1268d39c-458a-46cf-b727-d6157564405f/segments/0003/quaternions",
        "shape": [
          2749,
          4
        ],
        "dtype": "float32"
      },
      {
        "path": "users/00fa7124-4838-4275-a473-1424cb52f42f/sessions/1268d39c-458a-46cf-b727-d6157564405f/segments/0003/timestamps",
        "shape": [
          2749
        ],
        "dtype": "float64"
      }
    ]
  },
  "failures": [],
  "warnings": []
}
```

### WARN — Canonical HDF5 selection

Both head_pose.h5 and head_pose_v2.h5 exist. Select one canonical training artifact after comparing schemas and contents.

```json
[
  "head_pose.h5",
  "head_pose_v2.h5"
]
```

### PASS — Pipeline storage cap

Tracked pipeline-managed files are within the configured cap.

```json
{
  "managed_bytes": 978166064,
  "managed_gib": 0.911,
  "limit_gib": 10.0,
  "free_bytes": 17486106624,
  "free_gib": 16.285,
  "reserve_gib": 5.0,
  "breakdown_bytes": {
    "data": 923039342,
    "checkpoints": 147925,
    "outputs": 10689913,
    "reports": 613653,
    "experiments": 43675231
  }
}
```

### PASS — Free disk reserve

Free disk space meets the configured safety reserve.

```json
{
  "managed_bytes": 978166064,
  "managed_gib": 0.911,
  "limit_gib": 10.0,
  "free_bytes": 17486106624,
  "free_gib": 16.285,
  "reserve_gib": 5.0,
  "breakdown_bytes": {
    "data": 923039342,
    "checkpoints": 147925,
    "outputs": 10689913,
    "reports": 613653,
    "experiments": 43675231
  }
}
```

### WARN — Source markers: src/prep/resampling.py

Expected implementation markers not found; inspect manually.

```json
[
  "Efficient nearest-timestamp lookup marker"
]
```

### WARN — Source markers: src/prep/session.py

Markers found, but source-text checks do not prove correctness.

```json
[
  "Empty resampling result handling",
  "Window count handling"
]
```

### WARN — Source markers: src/prep/windows.py

Markers found, but source-text checks do not prove correctness.

```json
[
  "Context-window logic marker"
]
```
