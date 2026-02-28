import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import json
from datetime import datetime

from data.physionet_loader import load_physionet


# ============================
# CONFIG
# ============================

EXPERIMENT_ID = "exp_20260223_084850"

BASE_EVAL_DIR = "evaluation_results"
SAVE_DIR = os.path.join(BASE_EVAL_DIR, EXPERIMENT_ID)
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================
# LOAD REAL DATA
# ============================

REAL_PATH = "Rawdata/Physionet"
SEQ_LEN = 1248

real_ecg = load_physionet(
    physionet_dir=REAL_PATH,
    seq_len=SEQ_LEN,
    lowcut=0.5,
    highcut=40.0,
    lead_index=0
)

print("Real ECG shape:", real_ecg.shape)


# ============================
# LOAD FAKE DATA
# ============================

fake_ecg = np.load("generated_data/generated_ecg_2000.npy")
print("Fake ECG shape:", fake_ecg.shape)

real_ecg = real_ecg[:, :, 0]
fake_ecg = fake_ecg[:, 0, :]


# ============================
# STATISTICAL COMPARISON
# ============================

print("\n--- Statistical Comparison ---")

real_mean = np.mean(real_ecg)
fake_mean = np.mean(fake_ecg)

real_std = np.std(real_ecg)
fake_std = np.std(fake_ecg)

print("Real Mean:", real_mean)
print("Fake Mean:", fake_mean)
print("Real Std:", real_std)
print("Fake Std:", fake_std)


# ============================
# SAVE WAVEFORM COMPARISON
# ============================

plt.figure(figsize=(10,4))
plt.plot(real_ecg[0], label="Real")
plt.plot(fake_ecg[0], label="Fake")
plt.title("Real vs Fake ECG (Sample 1)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "waveform_comparison.png"))
plt.close()


# ============================
# SAVE MULTIPLE SAMPLE PLOTS
# ============================

for i in range(5):
    plt.figure(figsize=(10,4))
    plt.plot(real_ecg[i], label="Real")
    plt.plot(fake_ecg[i], label="Fake")
    plt.title(f"Sample Comparison {i+1}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"sample_comparison_{i+1}.png"))
    plt.close()


# ============================
# POWER SPECTRAL DENSITY
# ============================

fs = 360
N_ANALYZE = 200

real_subset = real_ecg[:N_ANALYZE]
fake_subset = fake_ecg[:N_ANALYZE]

real_psd_all = []
fake_psd_all = []

for i in range(N_ANALYZE):
    f_r, P_r = welch(real_subset[i], fs)
    f_f, P_f = welch(fake_subset[i], fs)

    real_psd_all.append(P_r)
    fake_psd_all.append(P_f)

real_psd_mean = np.mean(real_psd_all, axis=0)
fake_psd_mean = np.mean(fake_psd_all, axis=0)

plt.figure(figsize=(8,5))
plt.semilogy(f_r, real_psd_mean, label="Real (Mean)")
plt.semilogy(f_f, fake_psd_mean, label="Fake (Mean)")
plt.title("Mean Power Spectral Density")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mean_psd_comparison.png"))
plt.close()

spectral_mse = np.mean((real_psd_mean - fake_psd_mean)**2)

hf_band = f_r > 60
real_hf_energy = np.mean(real_psd_mean[hf_band])
fake_hf_energy = np.mean(fake_psd_mean[hf_band])

print("Spectral MSE:", spectral_mse)
print("High-Frequency Energy (Real):", real_hf_energy)
print("High-Frequency Energy (Fake):", fake_hf_energy)


# ============================
# SAVE METRICS
# ============================

metrics = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiment_id": EXPERIMENT_ID,
    "real_mean": float(real_mean),
    "fake_mean": float(fake_mean),
    "real_std": float(real_std),
    "fake_std": float(fake_std),
    "spectral_mse": float(spectral_mse),
    "real_high_freq_energy": float(real_hf_energy),
    "fake_high_freq_energy": float(fake_hf_energy)
}

with open(os.path.join(SAVE_DIR, "evaluation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

with open(os.path.join(SAVE_DIR, "evaluation_summary.txt"), "w") as f:
    f.write("ECG WGAN Evaluation Report\n")
    f.write("=" * 40 + "\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

print("\nEvaluation results saved to:", SAVE_DIR)