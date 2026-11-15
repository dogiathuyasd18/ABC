# Data quality report

- spec: `9th-abc-challenge-data-page`
- preprocessing: `2.1`
- schema: `2.0`

## Session filtering

- users inspected: 37
- sessions inspected: 4074
- PASS: 400
- FAIL: 3674
- reject reasons: `{'PASS': 400, 'duplicate_timestamps': 505, 'sweep_below_threshold': 2243, 'sampling_rate_out_of_range': 732, 'too_few_samples_after_trim': 34, 'no_valid_motion_windows': 15, 'non_monotonic_timestamps': 142, 'velocity_below_threshold': 3}`

## Training usability (after clean / segment / 100 Hz)

- sessions with ≥1 window of 400+20: 272
- sessions with no window: 1
- usable windows: 189443
- segments: 513
- resampled samples: 3943844
- interpolated samples: 3941051
