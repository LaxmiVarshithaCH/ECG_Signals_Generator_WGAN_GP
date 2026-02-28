import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.generator import Generator


MODEL_PATH = "experiments/exp_20260223_084850/checkpoints/best_model.pt"

LATENT_DIM = 100
SEQ_LEN = 1248

DEVICE = torch.device("cpu")

st.set_page_config(page_title="ECG Generator", layout="wide")

st.title("🫀 Synthetic ECG Generator (WGAN-GP)")


# ------------------------
# Load Model
# ------------------------

@st.cache_resource
def load_model():

    model = Generator(
        seq_len=SEQ_LEN,
        latent_dim=LATENT_DIM
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    return model


generator = load_model()


# ------------------------
# UI
# ------------------------

n_samples = st.slider(
    "Number of Samples",
    1,
    10,
    1
)

if st.button("Generate ECG"):

    z = torch.randn(
        n_samples,
        LATENT_DIM
    ).to(DEVICE)

    with torch.no_grad():
        samples = generator(z).cpu().numpy()

    for i in range(n_samples):

        st.subheader(f"Sample {i+1}")

        fig, ax = plt.subplots()

        ax.plot(samples[i][0])
        ax.set_title("Synthetic ECG")

        st.pyplot(fig)

    st.download_button(
        label="Download ECG (.npy)",
        data=samples.tobytes(),
        file_name="generated_ecg.npy",
        mime="application/octet-stream"
    )