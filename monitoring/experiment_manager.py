# monitoring/experiment_manager.py

import os
import yaml
import json
from datetime import datetime

class ExperimentManager:

    def __init__(self, base_dir="experiments", config=None):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self.exp_id = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.exp_path = os.path.join(base_dir, self.exp_id)

        self.paths = {}
        self._create_structure()

        if config:
            self.save_config(config)

        self.metrics = {
            "generator_loss": [],
            "critic_loss": []
        }

    def _create_structure(self):
        self.paths["root"] = self.exp_path
        self.paths["logs"] = os.path.join(self.exp_path, "logs")
        self.paths["checkpoints"] = os.path.join(self.exp_path, "checkpoints")
        self.paths["tensorboard"] = os.path.join(self.exp_path, "tensorboard")
        self.paths["samples"] = os.path.join(self.exp_path, "samples")

        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    def save_config(self, config):
        config_path = os.path.join(self.exp_path, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def update_metrics(self, g_loss, c_loss):
        self.metrics["generator_loss"].append(float(g_loss))
        self.metrics["critic_loss"].append(float(c_loss))

        metrics_path = os.path.join(self.exp_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def get_checkpoint_paths(self):
        return {
            "best": os.path.join(self.paths["checkpoints"], "best_model.pt"),
            "last": os.path.join(self.paths["checkpoints"], "last_model.pt")
        }

    def get_tensorboard_path(self):
        return self.paths["tensorboard"]

    def get_sample_path(self, epoch):
        return os.path.join(self.paths["samples"], f"epoch_{epoch}.png")

    def summary(self):
        return {
            "Experiment ID": self.exp_id,
            "Root Path": self.exp_path
        }