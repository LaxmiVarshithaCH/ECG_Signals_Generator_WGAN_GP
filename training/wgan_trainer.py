# training/wgan_trainer.py

import torch
from torch.optim import Adam
from training.gradient_penalty import compute_gradient_penalty
from tqdm import tqdm

class WGANTrainer:

    def __init__(self, generator, critic, config, device, logger, exp_manager):

        self.G = generator
        self.C = critic
        self.device = device
        self.logger = logger
        self.exp_manager = exp_manager

        self.latent_dim = config["model"]["latent_dim"]

        self.lr = config["training"]["lr"]
        self.n_critic = config["training"]["n_critic"]
        self.gp_lambda = config["training"]["gp_lambda"]

        self.g_optimizer = Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.c_optimizer = Adam(self.C.parameters(), lr=self.lr, betas=(0.5, 0.9))

        self.best_g_loss = float("inf")

    def train(self, dataloader, epochs):

        for epoch in range(epochs):

            g_losses = []
            c_losses = []

            for real in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):

                real = real.to(self.device)

                # ---------------------
                # Train Critic
                # ---------------------
                for _ in range(self.n_critic):

                    z = torch.randn(real.size(0), self.latent_dim).to(self.device)
                    fake = self.G(z)

                    real_score = self.C(real)
                    fake_score = self.C(fake.detach())

                    gp = compute_gradient_penalty(
                        self.C, real, fake.detach(), self.device
                    )

                    c_loss = (
                        fake_score.mean()
                        - real_score.mean()
                        + self.gp_lambda * gp
                    )

                    self.c_optimizer.zero_grad()
                    c_loss.backward()
                    self.c_optimizer.step()

                # ---------------------
                # Train Generator
                # ---------------------
                z = torch.randn(real.size(0), self.latent_dim).to(self.device)
                fake = self.G(z)
                g_loss = -self.C(fake).mean()

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                g_losses.append(g_loss.item())
                c_losses.append(c_loss.item())

            mean_g = sum(g_losses) / len(g_losses)
            mean_c = sum(c_losses) / len(c_losses)

            print(
                f"\nEpoch [{epoch+1}/{epochs}] "
                f"G Loss: {mean_g:.4f} | "
                f"C Loss: {mean_c:.4f}"
            )

            self.logger.info(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"G Loss: {mean_g:.4f} | "
                f"C Loss: {mean_c:.4f}"
            )

            self.exp_manager.update_metrics(mean_g, mean_c)

            self._save_checkpoint(mean_g)

    def _save_checkpoint(self, g_loss):

        paths = self.exp_manager.get_checkpoint_paths()

        # Save last
        torch.save({
            "generator": self.G.state_dict(),
            "critic": self.C.state_dict()
        }, paths["last"])

        # Save best
        if g_loss < self.best_g_loss:
            self.best_g_loss = g_loss
            torch.save({
                "generator": self.G.state_dict(),
                "critic": self.C.state_dict()
            }, paths["best"])

            self.logger.info("Best model updated.")