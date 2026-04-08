"""Entry point for VLM training."""

from training.config import TrainingConfig
from training.train import train

if __name__ == "__main__":
    config = TrainingConfig()
    train(config)