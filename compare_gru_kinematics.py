import numpy as np
import pandas as pd

GRU_FILE = "outputs/gru_val_seed42_predictions.npz"
KIN_FILE = "outputs/kinematics_val_seed42_predictions.npz"

GRU_CONTEXT = 100
KIN_CONTEXT = 400
HORIZON = 20

def load_predictions(path, context):
    data = np.load(path)

    result = {}

    for i in range(len(data["start_index"])):
        key = (
            str(data["user_id"][i]),
            str(data["session_id"][i]),
            str(data["sequence_name"][i]),
            int(data["start_index"][i]) + context + HORIZON - 1,
        )

        result[key] = {
            "pred": data["pred_quat"][i, -1],
            "true": data["true_quat"][i, -1],
        }

    return result


def angular_error_deg(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    pred = pred / np.linalg.norm(pred)
    true = true / np.linalg.norm(true)

    dot = np.abs(np.sum(pred * true))
    dot = np.clip(dot, 0.0, 1.0)

    return np.degrees(2.0 * np.arccos(dot))


gru = load_predictions(GRU_FILE, GRU_CONTEXT)
kin = load_predictions(KIN_FILE, KIN_CONTEXT)

common = sorted(set(gru) & set(kin))

rows = []

for key in common:
    g = gru[key]
    k = kin[key]

    truth_difference = angular_error_deg(g["true"], k["true"])

    if truth_difference > 0.01:
        raise ValueError(
            f"Ground truth mismatch at {key}: {truth_difference:.6f}"
        )

    rows.append({
        "user_id": key[0],
        "session_id": key[1],
        "sequence_name": key[2],
        "target_index": key[3],
        "gru_error": angular_error_deg(g["pred"], g["true"]),
        "kin_error": angular_error_deg(k["pred"], k["true"]),
    })

df = pd.DataFrame(rows)

print("\n=== MATCHING ===")
print("GRU windows:", len(gru))
print("Kinematics windows:", len(kin))
print("Matched targets:", len(df))
print("Unmatched GRU:", len(set(gru) - set(kin)))
print("Unmatched Kinematics:", len(set(kin) - set(gru)))

if df.empty:
    raise RuntimeError("No matching targets found")

print("\n=== PAIRED ERROR ===")
print("GRU mean:", round(df.gru_error.mean(), 4))
print("Kinematics mean:", round(df.kin_error.mean(), 4))
print("Mean improvement:", round(
    (df.gru_error.mean() - df.kin_error.mean()), 4
), "degrees")

for threshold in [3, 5, 10]:
    g_good = df.gru_error <= threshold
    k_good = df.kin_error <= threshold

    both_good = g_good & k_good
    gru_only = g_good & ~k_good
    kin_only = ~g_good & k_good
    both_bad = ~g_good & ~k_good

    n = len(df)

    print(f"\n=== THRESHOLD: {threshold} degrees ===")

    categories = {
        "Both good": both_good,
        "GRU good, Kinematics bad": gru_only,
        "Kinematics good, GRU bad": kin_only,
        "Both bad": both_bad,
    }

    for name, mask in categories.items():
        print(
            f"{name}: {mask.sum()} "
            f"({mask.mean() * 100:.2f}%)"
        )

    gru_bad_n = (~g_good).sum()
    kin_bad_n = (~k_good).sum()

    p_kin_good_given_gru_bad = (
        kin_only.sum() / gru_bad_n * 100
        if gru_bad_n else float("nan")
    )

    p_gru_good_given_kin_bad = (
        gru_only.sum() / kin_bad_n * 100
        if kin_bad_n else float("nan")
    )

    print(
        "P(Kinematics good | GRU bad):",
        round(p_kin_good_given_gru_bad, 2), "%"
    )

    print(
        "P(GRU good | Kinematics bad):",
        round(p_gru_good_given_kin_bad, 2), "%"
    )

df["error_difference"] = df["gru_error"] - df["kin_error"]
df.to_csv("outputs/gru_vs_kinematics_paired.csv", index=False)

print("\nSaved: outputs/gru_vs_kinematics_paired.csv")

print("\n=== SESSION SUMMARY (5-degree threshold) ===")
df["gru_good"] = df.gru_error <= 5
df["kin_good"] = df.kin_error <= 5

summary = df.groupby(
    ["user_id", "session_id"]
).agg(
    windows=("gru_error", "size"),
    gru_mean=("gru_error", "mean"),
    kin_mean=("kin_error", "mean"),
    gru_only_good=("gru_good", lambda x: (
        x & ~df.loc[x.index, "kin_good"]
    ).sum()),
    kin_only_good=("kin_good", lambda x: (
        x & ~df.loc[x.index, "gru_good"]
    ).sum()),
).reset_index()

print(summary.to_string(index=False))
