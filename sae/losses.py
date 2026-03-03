from __future__ import annotations

import torch


def sae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    aux_x_hat: torch.Tensor | None,
    beta: float = 1 / 32,
) -> tuple[torch.Tensor, dict]:
    mse = (x - x_hat).pow(2).mean()

    aux_mse = torch.tensor(0.0, device=x.device)
    if aux_x_hat is not None:
        residual = x - x_hat.detach()
        aux_mse = (residual - aux_x_hat).pow(2).mean()

    total = mse + beta * aux_mse

    metrics = {
        "mse": mse.item(),
        "aux_mse": aux_mse.item(),
        "total_loss": total.item(),
    }
    return total, metrics
