import json
import time
from pathlib import Path

import torch

from sae.data import make_dataloader
from sae.losses import sae_loss
from sae.model import TopKSAE


def train(cfg):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    loader, n_channels, dataset = make_dataloader(
        cfg["data_dir"], cfg["layer_name"], cfg["batch_size"], cfg["seed"]
    )
    print(f"Layer {cfg['layer_name']}: {len(dataset.data)} samples, {n_channels} channels")

    # Model
    model = TopKSAE(n_channels, cfg["n_latent"], cfg["k"], cfg["aux_k"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Output directory
    out_dir = Path(cfg["output_dir"]) / cfg["layer_name"] / f"k{cfg['k']}_n{cfg['n_latent']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Wandb
    if cfg.get("use_wandb", False):
        import wandb
        wandb.init(project="cae-sae", name=f"{cfg['layer_name']}_k{cfg['k']}_n{cfg['n_latent']}", config=cfg)

    best_mse = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(cfg["num_epochs"]):
        model.train()
        epoch_mse = 0.0
        epoch_steps = 0

        for (batch,) in loader:
            batch = batch.to(device)

            # Forward
            x_hat, info = model(batch)
            loss, metrics = sae_loss(batch, x_hat, info["aux_x_hat"], beta=cfg["aux_beta"])

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient projection: remove component parallel to decoder columns
            model.project_decoder_grads()

            optimizer.step()

            # Post-step: re-normalize decoder columns
            model._normalize_decoder()

            # Update dead feature tracking
            model.update_dead_mask(info["fired_mask"])

            epoch_mse += metrics["mse"]
            epoch_steps += 1
            global_step += 1

            if global_step % cfg["log_every"] == 0:
                n_dead = (model.miss_counts >= cfg["dead_threshold"]).sum().item()
                n_active = (info["fired_mask"].any(dim=0)).sum().item()
                elapsed = time.time() - t0
                log_msg = (
                    f"[step {global_step}] "
                    f"mse={metrics['mse']:.6f} aux={metrics['aux_mse']:.6f} "
                    f"active={n_active}/{cfg['n_latent']} dead={n_dead} "
                    f"time={elapsed:.1f}s"
                )
                print(log_msg)
                if cfg.get("use_wandb", False):
                    import wandb
                    wandb.log({
                        "mse": metrics["mse"],
                        "aux_mse": metrics["aux_mse"],
                        "total_loss": metrics["total_loss"],
                        "n_active": n_active,
                        "n_dead": n_dead,
                        "epoch": epoch,
                    }, step=global_step)

        avg_mse = epoch_mse / epoch_steps
        print(f"Epoch {epoch+1}/{cfg['num_epochs']} — avg MSE: {avg_mse:.6f}")

        # Save best checkpoint
        if avg_mse < best_mse:
            best_mse = avg_mse
            _save_checkpoint(model, dataset, cfg, out_dir / "best.pt")

    # Save final checkpoint
    _save_checkpoint(model, dataset, cfg, out_dir / "final.pt")
    print(f"Training complete. Best MSE: {best_mse:.6f}")
    print(f"Checkpoints saved to {out_dir}")

    if cfg.get("use_wandb", False):
        import wandb
        wandb.finish()


def _save_checkpoint(model, dataset, cfg, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_input": model.n_input,
        "n_latent": model.n_latent,
        "k": model.k,
        "aux_k": model.aux_k,
        "norm_mean": dataset.mean,
        "norm_std": dataset.std,
        "config": cfg,
    }, path)
