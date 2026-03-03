from pathlib import Path

import torch

from sae.model import TopKSAE


def load_sae(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = TopKSAE(
        n_input=ckpt["n_input"],
        n_latent=ckpt["n_latent"],
        k=ckpt["k"],
        aux_k=ckpt["aux_k"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def compute_metrics(model, data, batch_size=8192, device="cpu"):
    model.eval()
    model.to(device)

    total_mse = 0.0
    total_var = 0.0
    feature_counts = torch.zeros(model.n_latent, device=device)
    n_samples = 0

    data_mean = data.mean(dim=0).to(device)

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size].to(device)
        x_hat, info = model(batch)

        total_mse += (batch - x_hat).pow(2).sum().item()
        total_var += (batch - data_mean).pow(2).sum().item()
        feature_counts += info["fired_mask"].sum(dim=0).float()
        n_samples += batch.shape[0]

    mse = total_mse / (n_samples * model.n_input)
    variance_explained = 1.0 - total_mse / total_var
    freq = feature_counts / n_samples

    return {
        "mse": mse,
        "variance_explained": variance_explained,
        "feature_freq": freq.cpu(),
        "n_dead": (freq == 0).sum().item(),
        "n_samples": n_samples,
    }


@torch.no_grad()
def extract_features(model, data, batch_size=8192, device="cpu"):
    """Get sparse activations for all samples. Returns (n_samples, n_latent)."""
    model.eval()
    model.to(device)

    alphas = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size].to(device)
        _, info = model(batch)
        alphas.append(info["alpha"].cpu())

    return torch.cat(alphas, dim=0)
