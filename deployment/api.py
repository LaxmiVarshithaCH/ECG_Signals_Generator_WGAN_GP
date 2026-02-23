import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from fastapi import FastAPI
import torch
import numpy as np
from models.generator import Generator

app = FastAPI()

MODEL_PATH = "experiments/exp_20260222_163138/checkpoints/best_model.pt"
LATENT_DIM = 100
SEQ_LEN = 1248

DEVICE = torch.device("cpu")

generator = Generator(seq_len=SEQ_LEN, latent_dim=LATENT_DIM)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
generator.load_state_dict(checkpoint)
generator.eval()


@app.get("/")
def home():
    return {"message": "ECG WGAN API Running"}


@app.get("/generate")
def generate_ecg(n_samples: int = 1):

    z = torch.randn(n_samples, LATENT_DIM).to(DEVICE)

    with torch.no_grad():
        samples = generator(z).cpu().numpy()

    return {"generated_ecg": samples.tolist()}