import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

LAYER_NAMES = ["IN", "E1", "E2", "E3", "E4", "E5", "D1", "D2", "D3", "D4", "D5", "OUT"]


class EmbeddingDataset:
    """Load a CAE layer embedding tensor and reshape for SAE training."""

    def __init__(self, path):
        tensor = torch.load(path, weights_only=True)  # (N, C, Ny, Nx)
        N, C, Ny, Nx = tensor.shape
        # Reshape: (N, C, Ny, Nx) -> (N, Ny, Nx, C) -> (N*Ny*Nx, C)
        self.data = tensor.permute(0, 2, 3, 1).reshape(-1, C).float()
        self.n_channels = C

        # Per-channel normalization
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0).clamp(min=1e-8)
        self.data = (self.data - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


def make_dataloader(data_dir, layer_name, batch_size, seed=42):
    path = Path(data_dir) / f"{layer_name}.pt"
    ds = EmbeddingDataset(path)
    tensor_ds = TensorDataset(ds.data)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        tensor_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    return loader, ds.n_channels, ds
