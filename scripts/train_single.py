"""Train a single TopK SAE on one layer's embeddings.

Usage:
    python -m scripts.train_single configs/default.toml
"""
import json
import sys
from pathlib import Path

import toml
import torch
import wandb

from sae.data import make_dataloader
from sae.model import TopKSAE
from sae.train import train

def main():
    # Fetch config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.toml"
    config = toml.load(config_path)

    # Build a run id
    run_id = '_'.join([
        config['data']['layer_name'], 
        'k' + config['model']['k'], 
        'n' + config['model']['n_latent']
    ])

    # Create output directory
    out_dir = Path(config['output_dir']) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Set torch metaparameters
    torch.manual_seed(config["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data and model
    loader, n_channels, _ = make_dataloader(**config["data"])
    model = TopKSAE(n_channels, **config["model"]).to(device)

    # WandB logger
    logger = wandb.init(
        project="cae-sae",
        name=run_id,
        config=config,
    ) if config.get("use_wandb", False) else None

    # Train
    train(model, loader, **config["training"], out_dir=out_dir, logger=logger)

    if logger is not None: logger.finish()

if __name__ == "__main__":
    main()