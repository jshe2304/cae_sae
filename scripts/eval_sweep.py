"""Evaluate all sweep checkpoints and save metrics to CSV.

Usage:
    python -m scripts.eval_sweep

Produces: {SWEEP_DIR}/sweep_metrics.csv
"""

from pathlib import Path

import torch
import pandas as pd

from sae.data import EmbeddingDataset, LAYER_NAMES
from sae.eval import load_sae, compute_metrics

# ---------- Configuration ----------
SWEEP_DIR = Path("/ocean/projects/atm170004p/jshen6/cae_sae/")
DATA_DIR = Path("/ocean/projects/atm170004p/lxu5/ConvAE/EVAL_DATA/SNAPSHOTS/CAEwCP_TrP_A_TrS_10000_KS_5_LN_5_LD_256_BS_2_LR_0.001_WD_1e-06_DP_0.0_EN_1500_SD_42/ep_best_pi_0/TeP_A_10000_TD_0.2_SAE/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = LAYER_NAMES
K_VALUES = [16, 32, 64, 128]
N_LATENT_VALUES = [2048, 4096, 8192]
# ------------------------------------

OUT_PATH = SWEEP_DIR / "sweep_metrics.csv"

rows = []
missing = []

for layer in LAYERS:
    print(f"\n=== Layer {layer} ===")
    ds = EmbeddingDataset(DATA_DIR / f"{layer}.pt")
    data = ds.data

    for k in K_VALUES:
        for n_latent in N_LATENT_VALUES:
            run_name = f"{layer}_k{k}_n{n_latent}"
            ckpt_path = SWEEP_DIR / run_name / "best.pt"

            if not ckpt_path.exists():
                missing.append(run_name)
                continue

            model, _ = load_sae(ckpt_path, device=DEVICE)
            metrics = compute_metrics(model, data, device=DEVICE)

            freq = metrics["feature_freq"]
            alive_freq = freq[freq > 0]

            rows.append({
                "layer": layer,
                "k": k,
                "n_latent": n_latent,
                "mse": metrics["mse"],
                "variance_explained": metrics["variance_explained"],
                "n_dead": metrics["n_dead"],
                "dead_pct": metrics["n_dead"] / n_latent * 100,
                "n_samples": metrics["n_samples"],
                "freq_mean": alive_freq.mean().item() if len(alive_freq) > 0 else 0,
                "freq_median": alive_freq.median().item() if len(alive_freq) > 0 else 0,
                "freq_max": alive_freq.max().item() if len(alive_freq) > 0 else 0,
                "freq_min": alive_freq.min().item() if len(alive_freq) > 0 else 0,
            })
            print(f"  {run_name}: VE={metrics['variance_explained']:.4f}, "
                  f"dead={metrics['n_dead']}/{n_latent} ({metrics['n_dead']/n_latent*100:.1f}%)")

            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    del data, ds

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(df)} results to {OUT_PATH}")

if missing:
    print(f"\n{len(missing)} runs missing:")
    for m in missing:
        print(m)
