
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


INPUT = Path("outputs/acceleration_analysis.csv")
OUT_USER = Path("outputs/acceleration_by_user.csv")
OUT_SESSION = Path("outputs/acceleration_by_session.csv")


def safe_corr(x, y):
    """Calculate Pearson and Spearman correlations."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 10:
        return np.nan, np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    return (
        float(pearsonr(x, y).statistic),
        float(spearmanr(x, y).statistic),
    )


def analyze_group(df, group_columns):
    results = []

    for keys, group in df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        acceleration = group["mean_angular_acc_deg_s2"]
        error = group["error_step20_deg"]

        pearson, spearman = safe_corr(acceleration, error)

        row = {
            column: value
            for column, value in zip(group_columns, keys)
        }

        row.update({
            "n_windows": len(group),
            "mean_acceleration": acceleration.mean(),
            "mean_error_deg": error.mean(),
            "median_error_deg": error.median(),
            "pearson": pearson,
            "spearman": spearman,
        })

        results.append(row)

    return pd.DataFrame(results)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = [
        "mean_angular_acc_deg_s2",
        "error_step20_deg",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Available columns:", list(df.columns))

    # Convert numeric columns
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    # ------------------------------------
    # 1. Analysis per user
    # ------------------------------------
    user_col = next(
        (c for c in ["user_id", "user", "users"] if c in df.columns),
        None
    )

    if user_col is None:
        raise ValueError(
            "Không tìm thấy user_id trong CSV. "
            "Cần bổ sung user metadata khi export predictions."
        )

    user_result = analyze_group(df, [user_col])
    user_result = user_result.sort_values(
        "spearman", ascending=False, na_position="last"
    )

    OUT_USER.parent.mkdir(parents=True, exist_ok=True)
    user_result.to_csv(OUT_USER, index=False)

    print("\n=== CORRELATION PER USER ===")
    print(user_result.to_string(index=False))
    print(f"\nSaved: {OUT_USER}")

    # ------------------------------------
    # 2. Analysis per session
    # ------------------------------------
    session_col = next(
        (c for c in ["session_id", "session"] if c in df.columns),
        None
    )

    if session_col is None:
        print("\nKhông tìm thấy session_id. Bỏ qua phân tích session.")
    else:
        session_result = analyze_group(
            df, [user_col, session_col]
        )

        session_result = session_result.sort_values(
            "spearman", ascending=False, na_position="last"
        )

        session_result.to_csv(OUT_SESSION, index=False)

        print("\n=== CORRELATION PER USER / SESSION ===")
        print(session_result.to_string(index=False))
        print(f"\nSaved: {OUT_SESSION}")

    # ------------------------------------
    # 3. Compare users with high errors
    # ------------------------------------
    print("\n=== USERS RANKED BY MEAN ERROR ===")

    summary = (
        df.groupby(user_col)
        .agg(
            n_windows=("error_step20_deg", "size"),
            mean_error=("error_step20_deg", "mean"),
            mean_acceleration=(
                "mean_angular_acc_deg_s2", "mean"
            ),
        )
        .sort_values("mean_error", ascending=False)
    )

    print(summary.to_string())


if __name__ == "__main__":
    main()
