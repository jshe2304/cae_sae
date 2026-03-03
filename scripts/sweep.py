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

DEFAULTS = {
    "batch_size": 8192,
    "lr": 3e-4,
    "aux_k": 512,
    "aux_beta": 0.03125,
    "dead_threshold": 50000,
    "seed": 42,
    "log_every": 100,
}


def build_job_config(layer, k, n_latent, sweep):
    """Build a config dict for a single training job."""
    return {
        "sae": {
            **DEFAULTS,
            "n_latent": n_latent,
            "k": k,
            "num_epochs": sweep["num_epochs"],
            "data_dir": sweep["data_dir"],
            "layer_name": layer,
            "output_dir": sweep["output_dir"],
            "use_wandb": sweep["use_wandb"],
        }
    }


def build_jobs(sweep):
    """Build (job_name, config_path) pairs for all grid combinations."""
    layers = sweep.get("layers", LAYER_NAMES)
    k_values = sweep.get("k_values", [16, 32, 64, 128])
    n_latent_values = sweep.get("n_latent_values", [2048, 4096, 8192])

    configs_dir = Path(sweep["output_dir"]) / "sweep_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for layer, k, n_latent in product(layers, k_values, n_latent_values):
        name = f"{layer}_k{k}_n{n_latent}"
        cfg_path = configs_dir / f"{name}.toml"
        cfg_path.write_text(toml.dumps(build_job_config(layer, k, n_latent, sweep)))
        jobs.append((name, cfg_path))

    return jobs


def run_local(jobs):
    for name, cfg_path in jobs:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        result = subprocess.run([sys.executable, "-m", "scripts.train_single", str(cfg_path)])
        if result.returncode != 0:
            print(f"FAILED: {name} (exit code {result.returncode})")


def make_slurm_script(name, cfg_path, log_dir, slurm):
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={name}
        #SBATCH --output={log_dir}/{name}_%j.out
        #SBATCH --error={log_dir}/{name}_%j.err
        #SBATCH --partition={slurm.get('partition', 'gpu')}
        #SBATCH --gres=gpu:1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem=32G
        #SBATCH --time={slurm.get('time', '04:00:00')}

        {sys.executable} -m scripts.train_single {cfg_path}
    """)


def run_slurm(jobs, sweep, slurm):
    scripts_dir = Path(sweep["output_dir"]) / "slurm_scripts"
    log_dir = Path(sweep["output_dir"]) / "slurm_logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg_path in jobs:
        script_path = scripts_dir / f"{name}.sh"
        script_path.write_text(make_slurm_script(name, cfg_path, log_dir, slurm))

        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        print(f"Submitted {name}: {result.stdout.strip()}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/sweep.toml"
    raw = toml.load(config_path)
    sweep = raw["sweep"]
    slurm = raw.get("slurm", {})

    jobs = build_jobs(sweep)
    print(f"Total jobs: {len(jobs)}")

    if sweep["mode"] == "local":
        run_local(jobs)
    else:
        run_slurm(jobs, sweep, slurm)


if __name__ == "__main__":
    main()
