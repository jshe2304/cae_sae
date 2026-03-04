"""Grid search over layers and hyperparameters.

Usage:
    python -m scripts.sweep configs/sweep.toml
"""
import subprocess
import sys
import textwrap
from itertools import product
from pathlib import Path

import toml

from sae.data import LAYER_NAMES


def build_job_config(config, layer, k, n_latent):
    """
    Build a config dict for a single training job.
    """
    return {
        "output_dir": config["output_dir"],
        "use_wandb": config["use_wandb"],
        "data": {**config["data"], "layer_name": layer},
        "model": {**config["model"], "n_latent": n_latent, "k": k},
        "training": config["training"],
    }


def build_jobs(config):
    """
    Build (job_name, config_path) pairs for all grid combinations.
    """
    grid = config["grid"]
    layers = grid.get("layers", LAYER_NAMES)
    k_values = grid.get("k_values", [16, 32, 64, 128])
    n_latent_values = grid.get("n_latent_values", [2048, 4096, 8192])

    configs_dir = Path(config["output_dir"]) / "sweep_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for layer, k, n_latent in product(layers, k_values, n_latent_values):
        name = f"{layer}_k{k}_n{n_latent}"
        cfg_path = configs_dir / f"{name}.toml"
        cfg_path.write_text(toml.dumps(build_job_config(config, layer, k, n_latent)))
        jobs.append((name, cfg_path))

    return jobs

def make_slurm_script(name, config_path, log_dir, slurm):
    slurm = {
        **slurm,
        "job-name": name,
        "output": f"{log_dir}/{name}_%j.out",
        "error": f"{log_dir}/{name}_%j.err",
    }
    slurm_args = "\n".join(f"#SBATCH --{k}={v}" for k, v in slurm.items())

    cmd = f"{sys.executable} -m scripts.train_single {config_path}"
    return "#!/bin/bash" + f"\n{slurm_args}\n\n" + cmd


def run_slurm(jobs, config):
    slurm = config.get("slurm", {})
    scripts_dir = Path(config["output_dir"]) / "slurm_scripts"
    log_dir = Path(config["output_dir"]) / "slurm_logs"

    scripts_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    for name, config_path in jobs:
        script_path = scripts_dir / f"{name}.sh"
        script_path.write_text(make_slurm_script(name, config_path, log_dir, slurm))

        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        print(f"Submitted {name}: {result.stdout.strip()}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/sweep.toml"
    config = toml.load(config_path)

    jobs = build_jobs(config)
    print(f"Total jobs: {len(jobs)}")

    run_slurm(jobs, config)

if __name__ == "__main__":
    main()
