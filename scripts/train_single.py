"""Train a single TopK SAE on one layer's embeddings.

Usage:
    python -m scripts.train_single configs/default.toml
"""
from __future__ import annotations

import sys

import toml

from sae.train import train


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.toml"
    cfg = toml.load(config_path)["sae"]
    print(f"Config: {cfg}")
    train(cfg)


if __name__ == "__main__":
    main()
