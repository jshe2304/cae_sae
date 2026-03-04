#!/bin/bash
#SBATCH --job-name=eval_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/atm170004p/jshen6/cae_sae/slurm_logs/eval_sweep_%j.out
#SBATCH --error=/ocean/projects/atm170004p/jshen6/cae_sae/slurm_logs/eval_sweep_%j.err

python -m scripts.eval_sweep
