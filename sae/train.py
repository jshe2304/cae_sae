import time

import torch

from sae.losses import sae_loss


def train(model, loader, num_epochs, lr, aux_beta, log_every, dead_threshold, out_dir, logger=None):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mse = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_mse = 0.0
        epoch_steps = 0

        for (batch,) in loader:
            batch = batch.to(device)

            # Forward
            x_hat, info = model(batch)
            loss, metrics = sae_loss(batch, x_hat, info["aux_x_hat"], beta=aux_beta)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            model.project_decoder_grads()
            optimizer.step()
            model._normalize_decoder()
            model.update_dead_mask(info["fired_mask"])

            epoch_mse += metrics["mse"]
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                n_dead = (model.miss_counts >= dead_threshold).sum().item()
                n_active = (info["fired_mask"].any(dim=0)).sum().item()
                elapsed = time.time() - t0
                print(
                    f"[step {global_step}] "
                    f"mse={metrics['mse']:.6f} aux={metrics['aux_mse']:.6f} "
                    f"active={n_active}/{model.n_latent} dead={n_dead} "
                    f"time={elapsed:.1f}s"
                )
                if logger is not None:
                    logger.log({
                        "mse": metrics["mse"],
                        "aux_mse": metrics["aux_mse"],
                        "total_loss": metrics["total_loss"],
                        "n_active": n_active,
                        "n_dead": n_dead,
                        "epoch": epoch,
                    }, step=global_step)

        avg_mse = epoch_mse / epoch_steps
        print(f"Epoch {epoch+1}/{num_epochs} — avg MSE: {avg_mse:.6f}")

        if avg_mse < best_mse:
            best_mse = avg_mse
            _save_checkpoint(model, out_dir / "best.pt")

    _save_checkpoint(model, out_dir / "final.pt")
    print(f"Training complete. Best MSE: {best_mse:.6f}")
    print(f"Checkpoints saved to {out_dir}")
    return best_mse


def _save_checkpoint(model, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_input": model.n_input,
        "n_latent": model.n_latent,
        "k": model.k,
        "aux_k": model.aux_k,
    }, path)
