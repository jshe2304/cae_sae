import torch


def sae_loss(x, x_hat, aux_x_hat, beta=1/32):
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
