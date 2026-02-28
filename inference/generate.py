import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from models.generator import Generator


# ============================
# CONFIG
# ============================

EXPERIMENT_ID = "exp_20260223_084850"
BASE_DIR = "experiments"

LATENT_DIM = 100
SEQ_LEN = 1248
N_SAMPLES = 10

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    EXPERIMENT_ID,
    "checkpoints",
    "best_model.pt"
)


# ============================
# LOAD GENERATOR
# ============================

device = torch.device("cpu")

generator = Generator(
    seq_len=SEQ_LEN,
    latent_dim=LATENT_DIM
).to(device)


checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

if "generator" in checkpoint:
    state_dict = checkpoint["generator"]
else:
    state_dict = checkpoint

generator.load_state_dict(state_dict)

generator.eval()

print("Model loaded successfully.")


# ============================
# GENERATE ECG
# ============================

z = torch.randn(N_SAMPLES, LATENT_DIM).to(device)

with torch.no_grad():
    fake_ecg = generator(z).cpu().numpy()

print("Generated shape:", fake_ecg.shape)


# ============================
# VISUALIZE
# ============================

for i in range(N_SAMPLES):
    plt.figure(figsize=(10,4))
    plt.plot(fake_ecg[i][0])
    plt.title(f"Generated ECG {i+1}")
    plt.show()


# ============================
# GENERATE LARGE DATASET
# ============================

N_SAMPLES = 2000

z = torch.randn(N_SAMPLES, LATENT_DIM).to(device)

with torch.no_grad():
    fake_ecg = generator(z).cpu().numpy()

os.makedirs("generated_data", exist_ok=True)

np.save(
    "generated_data/generated_ecg_2000.npy",
    fake_ecg
)

print("Saved 2000 fake ECG samples.")