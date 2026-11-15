import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Cấu hình giao diện biểu đồ chuyên nghiệp
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

def compute_rms_angular_acceleration(quat_sequence):
    """Tính gia tốc góc RMS từ chuỗi quaternion."""
    q = quat_sequence / np.linalg.norm(quat_sequence, axis=-1, keepdims=True)
    q_curr = q[:, 1:]
    q_prev = q[:, :-1]
    dot = np.sum(q_curr * q_prev, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = 2.0 * np.arccos(dot)
    omega = theta 
    if omega.shape[1] > 1:
        alpha = np.abs(np.diff(omega, axis=1))
        rms_accel = np.sqrt(np.mean(alpha ** 2, axis=1)) * (180.0 / np.pi)
    else:
        rms_accel = np.zeros(omega.shape[0])
    return rms_accel

def calculate_step20_error(pred_quat, true_quat):
    """Tính sai số góc (độ) tại bước dự đoán cuối cùng (Step-20)."""
    q_pred = pred_quat[:, -1, :]
    q_true = true_quat[:, -1, :]
    q_pred = q_pred / np.linalg.norm(q_pred, axis=-1, keepdims=True)
    q_true = q_true / np.linalg.norm(q_true, axis=-1, keepdims=True)
    dot = np.sum(q_pred * q_true, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    error_deg = 2.0 * np.arccos(dot) * (180.0 / np.pi)
    return error_deg

def main():
    outputs_dir = Path("outputs")
    models = ["res_mlp", "tcn", "jepa", "kinematics_tcn"]
    model_labels = {
        "res_mlp": "ResMLP",
        "tcn": "TCN",
        "jepa": "JEPA",
        "kinematics_tcn": "Kinematics TCN"
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    quartile_data = {m: [] for m in models}
    
    print("📈 Processing and visualizing data from .npz files...")
    
    for model_name in models:
        npz_path = outputs_dir / f"{model_name}_val_seed42_predictions.npz"
        if not npz_path.exists():
            print(f"⚠️ File not found: {npz_path}. Skipping.")
            continue
            
        data = np.load(npz_path, allow_pickle=True)
        pred_quat = data["pred_quat"]
        true_quat = data["true_quat"]
        
        errors = calculate_step20_error(pred_quat, true_quat)
        accels = compute_rms_angular_acceleration(true_quat)
        
        valid_idx = ~np.isnan(errors) & ~np.isnan(accels)
        errors = errors[valid_idx]
        accels = accels[valid_idx]
        
        # --- PLOT 1: Scatter + Trendline ---
        sns.regplot(
            x=accels, y=errors,
            ax=axes[0],
            scatter_kws={'alpha': 0.15, 's': 10},
            line_kws={'linewidth': 2},
            label=model_labels.get(model_name, model_name)
        )
        
        # --- Quartile Binning for Plot 2 (Dùng duplicates='drop' để tránh lỗi trùng biên) ---
        try:
            quartiles = pd.qcut(accels, 4, labels=["Q1 (Low)", "Q2 (Mid-Low)", "Q3 (Mid-High)", "Q4 (High)"], duplicates='drop')
            for q_label in ["Q1 (Low)", "Q2 (Mid-Low)", "Q3 (Mid-High)", "Q4 (High)"]:
                subset_err = errors[quartiles == q_label]
                mean_err_q = np.mean(subset_err) if len(subset_err) > 0 else 0.0
                quartile_data[model_name].append(mean_err_q)
        except Exception as e:
            print(f"⚠️ Quartile calculation warning for {model_name}: {e}")
            quartile_data[model_name] = [0.0, 0.0, 0.0, 0.0]

    # --- Tinh chỉnh Plot 1 (English Labels) ---
    axes[0].set_title("Non-linear Correlation: Angular Acceleration vs Prediction Error", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("RMS Angular Acceleration (deg/s²)", fontsize=11)
    axes[0].set_ylabel("Step-20 Angular Error (degrees)", fontsize=11)
    axes[0].set_ylim(0, 25)
    axes[0].legend(title="Models", loc="upper left")

    # --- PLOT 2: Bar Chart Comparison (English Labels) ---
    plot2_rows = []
    for m in models:
        if len(quartile_data[m]) == 4:
            for q_idx, q_name in enumerate(["Q1 (Low)", "Q2 (Mid-Low)", "Q3 (Mid-High)", "Q4 (High)"]):
                plot2_rows.append({
                    "Model": model_labels.get(m, m),
                    "Quartile": q_name,
                    "Mean Error (deg)": quartile_data[m][q_idx]
                })
                
    if plot2_rows:
        df_q = pd.DataFrame(plot2_rows)
        sns.barplot(
            data=df_q, x="Quartile", y="Mean Error (deg)", hue="Model",
            ax=axes[1], palette="muted"
        )
        axes[1].set_title("Mean Error Comparison Across Acceleration Quartiles", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Acceleration Quartiles", fontsize=11)
        axes[1].set_ylabel("Mean Angular Error (deg)", fontsize=11)
        axes[1].legend(title="Models")
        axes[1].set_ylim(0, 8)

    plt.tight_layout()
    out_img = outputs_dir / "acceleration_vs_error_comparison.png"
    plt.savefig(out_img, dpi=300)
    print(f"✅ Successfully saved English visualization chart to: {out_img}")
    plt.show()

if __name__ == "__main__":
    main()