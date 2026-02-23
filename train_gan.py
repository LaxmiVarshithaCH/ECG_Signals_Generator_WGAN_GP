# train_gan.py

import yaml
from monitoring.experiment_manager import ExperimentManager
from monitoring.logger import setup_logger
from data.physionet_loader import load_physionet
from data.dataset import ECGDataset
from torch.utils.data import DataLoader
import torch
from models.generator import Generator
from models.critic import Critic
from training.wgan_trainer import WGANTrainer

def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():

    # ----------------------------
    # Load Configuration
    # ----------------------------
    config = load_config()

    # ----------------------------
    # Initialize Experiment
    # ----------------------------
    exp_manager = ExperimentManager(config=config)

    # ----------------------------
    # Initialize Logger
    # ----------------------------
    logger = setup_logger(exp_manager.paths["logs"])

    logger.info("Experiment started")
    logger.info(f"Experiment ID: {exp_manager.exp_id}")
    logger.info(f"Full config:\n{config}")

    print("\nExperiment created at:")
    print(exp_manager.paths["root"])


    logger.info("Loading PhysioNet data...")

    ecg_array = load_physionet(
        physionet_dir=config["data"]["physionet_path"],
        seq_len=config["data"]["seq_len"],
        lowcut=config["data"]["lowcut"],
        highcut=config["data"]["highcut"],
        lead_index=config["data"]["lead_index"]
    )

    logger.info(f"Loaded ECG shape: {ecg_array.shape}")

    dataset = ECGDataset(ecg_array)

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    logger.info(f"Total batches per epoch: {len(dataloader)}")

    sample_batch = next(iter(dataloader))
    logger.info(f"Sample batch shape: {sample_batch.shape}")


    device = torch.device("cpu")

    generator = Generator(
        seq_len=config["data"]["seq_len"],
        latent_dim=config["model"]["latent_dim"]
    ).to(device)

    critic = Critic(
        seq_len=config["data"]["seq_len"]
    ).to(device)

    logger.info("Models initialized successfully.")

    # Test forward pass
    z = torch.randn(4, config["model"]["latent_dim"]).to(device)
    fake = generator(z)

    logger.info(f"Generator output shape: {fake.shape}")

    score = critic(fake)
    logger.info(f"Critic output shape: {score.shape}")
    

    trainer = WGANTrainer(
        generator,
        critic,
        config,
        device,
        logger,
        exp_manager
    )

    logger.info("Starting WGAN-GP training...")

    trainer.train(
        dataloader,
        epochs=config["training"]["epochs"]
    )

    logger.info("Training finished.")
if __name__ == "__main__":
    main()