# TopK Sparse Autoencoders for CAE Layer Embeddings

Train TopK Sparse Autoencoders (SAEs) on embeddings from each layer of a pre-trained Convolutional Autoencoder (CAE) for interpretability analysis. Follows the approach from the GraphCast interpretability paper ([arXiv:2512.24440](https://arxiv.org/abs/2512.24440)).

## Project Structure

```
cae_vae/
├── pyproject.toml              # Dependencies: torch, toml, wandb
├── configs/
│   ├── default.toml            # Default hyperparameters for single training
│   └── sweep.toml              # Sweep grid and SLURM settings
├── sae/
│   ├── model.py                # TopKSAE module
│   ├── data.py                 # Load .pt embeddings, reshape, normalize
│   ├── losses.py               # MSE + auxiliary dead-feature loss
│   ├── train.py                # Training loop
│   └── eval.py                 # Checkpoint loading, metrics, feature extraction
└── scripts/
    ├── train_single.py         # Train one SAE (one layer, one config)
    └── sweep.py                # Grid search across layers and hyperparams
```

## Data Format

Each layer's embeddings are stored as a `.pt` file containing a tensor of shape `(N, C, Ny, Nx)`:

- **N** = 10,000 snapshots
- **C** = number of channels (varies per layer)
- **Ny, Nx** = spatial dimensions (16, 32, 64, 128, or 256)

Layer files should be named `IN.pt`, `E1.pt`, ..., `E5.pt`, `D1.pt`, ..., `D5.pt`, `OUT.pt` and placed in a single directory.

During loading, each tensor is reshaped to `(N*Ny*Nx, C)` so that each spatial location is an independent training sample. Per-channel normalization (zero mean, unit variance) is applied automatically.

## Architecture

The TopK SAE uses:

1. **Encoder**: `W_enc @ (x - b) + b_enc` maps input to a latent space
2. **TopK activation**: zeros out all but the top-k pre-activations
3. **Decoder**: `W_dec @ alpha + b` reconstructs the input (unit-norm columns)
4. **Loss**: `MSE(x, x_hat) + (1/32) * auxiliary_dead_feature_loss`

Dead features (latents that haven't fired recently) are revived via an auxiliary loss that uses only dead features to reconstruct the residual.

After each optimizer step, decoder column norms are projected back to unit norm and gradients are projected to maintain this constraint.

## Setup

```bash
pip install -e .
```

## Configuration

All scripts are configured via TOML files — no command-line arguments. Edit a `.toml` file and pass its path as the single positional argument.

## Training a Single Layer

Edit `configs/default.toml` (or create a new `.toml` file) to set your parameters:

```toml
[sae]
n_latent = 4096
k = 64
batch_size = 8192
lr = 3e-4
num_epochs = 50
aux_k = 512
aux_beta = 0.03125
dead_threshold = 50000
data_dir = "/path/to/embeddings"
layer_name = "E1"
output_dir = "./outputs"
use_wandb = false
seed = 42
log_every = 100
```

Then run:

```bash
python -m scripts.train_single configs/default.toml
```

Checkpoints are saved to `{output_dir}/{layer_name}/k{k}_n{n_latent}/` as `best.pt` (lowest epoch-average MSE) and `final.pt`.

## Hyperparameter Sweep

Grid search over `k` and `n_latent` for all (or a subset of) layers. Configure via `configs/sweep.toml`:

```toml
[sweep]
mode = "local"  # "local" or "slurm"
data_dir = "/path/to/embeddings"
output_dir = "./outputs"
num_epochs = 50
use_wandb = false
layers = ["IN", "E1", "E2", "E3", "E4", "E5", "D1", "D2", "D3", "D4", "D5", "OUT"]
k_values = [16, 32, 64, 128]
n_latent_values = [2048, 4096, 8192]

[slurm]
partition = "gpu"
time = "04:00:00"
```

Then run:

```bash
python -m scripts.sweep configs/sweep.toml
```

The sweep generates per-job `.toml` configs in `{output_dir}/sweep_configs/` and either runs them sequentially (local) or submits each as a SLURM job.

## Evaluation

```python
from sae.eval import load_sae, compute_metrics, extract_features
from sae.data import EmbeddingDataset

# Load a trained SAE
model, ckpt = load_sae("outputs/E1/k64_n4096/best.pt")

# Compute metrics on the training data
ds = EmbeddingDataset("/path/to/embeddings/E1.pt")
metrics = compute_metrics(model, ds.data)
print(f"Variance explained: {metrics['variance_explained']:.4f}")
print(f"Dead features: {metrics['n_dead']}")

# Extract sparse feature activations (n_samples, n_latent)
features = extract_features(model, ds.data)
```

## Sweep Analysis

After the sweep completes, evaluate all checkpoints and save metrics to a CSV:

1. Edit paths at the top of `scripts/eval_sweep.py` (`SWEEP_DIR`, `DATA_DIR`), then run:

```bash
python -m scripts.eval_sweep
```

This produces `{SWEEP_DIR}/sweep_metrics.csv` with variance explained, MSE, dead feature counts, and feature frequency statistics for every run.

2. Open `notebooks/analyze_sweep.ipynb`, set `CSV_PATH` to the generated CSV, and run all cells. The notebook produces:
   - Variance explained vs. k (per layer)
   - Dead feature % vs. k (per layer)
   - Reconstruction–sparsity trade-off plots
   - Summary heatmaps across all layers
   - Recommended configs per layer (sparsest k achieving ≥90% variance explained)

## Default Hyperparameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `k` | 64 | Mid-range sparsity |
| `n_latent` | 4096 | 4-8x overcomplete |
| `batch_size` | 8192 | Per GraphCast paper |
| `lr` | 3e-4 | Per GraphCast paper |
| `num_epochs` | 50 | Multiple passes for smaller dataset |
| `aux_beta` | 1/32 | Per GraphCast paper |
| `dead_threshold` | 50000 | ~10% of smallest layer samples/epoch |
