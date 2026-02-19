
from pathlib import Path
import datetime
import sys

import torch
import torchinfo

from torchinfo import summary

sys.path.append(str(Path(__file__).parent.parent))

from src import model
from src.dataset_util import make_dataloader
from src.model import DeGhostUNet
from src.train_util import train_deghost_residual, TrainConfig


def train(config):
    """
    Train a model based on the provided configuration.

    Args:
        config (str): Path to the configuration file.
    """
    # Load the configuration
    import yaml

    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    # Set up logging
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting training with configuration: %s", cfg)


    cfg_train = cfg["data"].get("train", {})
    cfg_val = cfg["data"].get("val", {})

   

    train_loader = make_dataloader(
            **cfg_train 
        )

    val_loader = make_dataloader(
        **cfg_val,
    )

    logger.info(f"created data loaders: train size {len(train_loader)}, validation size {len(val_loader)}")


    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    logger.info("Using device: %s", device)

    cfg_model = cfg.get("model", {})

    model = DeGhostUNet(**cfg_model)
    model.to(device)

    logger.info("Model created with config: %s", cfg_model)

    summary(model, input_size=(1, cfg_model.get("in_ch", 1), 256, 256))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    
    identifier = f"lvl-{model.levels}_b-{model.base}_{timestamp}"

    file_out = Path("checkpoints") / f"deghost_cnn_{identifier}.pt"


    logger.info(f"Model will be saved to: {file_out}")

    cfg_train = cfg.get("train", {})

    cfg_train_obj = TrainConfig(**cfg_train)
    ## Training loop
    history = train_deghost_residual(model, train_loader,val_loader, device, cfg_train_obj)

    # Optional: save model + history
    torch.save({"model": model.state_dict(), "history": history}, file_out)
    logger.info(f"Saved checkpoint: {file_out}")


if __name__ == "__main__": 
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    train(args.config)