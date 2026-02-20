
from typing import Any
from pathlib import Path
import datetime
import sys

import torch
import torchinfo
import numpy as np

from PIL import Image

from torchinfo import summary

sys.path.append(str(Path(__file__).parent.parent))


from deghost.dataset_util import make_dataloader
from deghost.model import DeGhostUNet
from deghost.train_util import train_deghost_residual, TrainConfig


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

    def safe_len(ds: Any) -> int | None:
        try:
            return len(ds)
        except TypeError:
            return None

    train_n = safe_len(train_loader.dataset)
    val_n   = safe_len(val_loader.dataset)

    logger.info(
        "Created data loaders | "
        f"Train: {train_n if train_n is not None else '?'} samples "
        f"({len(train_loader)} batches, batch_size={train_loader.batch_size}) | "
        f"Val: {val_n if val_n is not None else '?'} samples "
        f"({len(val_loader)} batches, batch_size={val_loader.batch_size})"
    )
    cfg_test = cfg["data"].get("test")


    if cfg_test["data"] is not None:
        # load image
        img = Image.open(cfg_test["data"]).convert("L")  # force grayscale

        # to numpy float32 in [0,1]
        img_np = np.array(img, dtype=np.float32) / 255.0

        # to torch tensor (1,1,H,W)
        test_image = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

    else:
        test_image = None


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
    
    identifier = f"lvl-{cfg_model.get('levels',0)}_b-{cfg_model.get('base',0)}_{timestamp}"

    file_out = Path("checkpoints") / f"deghost_cnn_{identifier}.pt"


    logger.info(f"Model will be saved to: {file_out}")

    cfg_train = cfg.get("train", {})

    cfg_train_obj = TrainConfig(**cfg_train)
    ## Training loop
    history = train_deghost_residual(model, train_loader,val_loader, test_image, device, cfg_train_obj)

    # Optional: save model + history
    torch.save({"model": model.state_dict(), "history": history}, file_out)
    logger.info(f"Saved checkpoint: {file_out}")


if __name__ == "__main__": 
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    train(args.config)