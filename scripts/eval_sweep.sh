#!/bin/bash
#SBATCH -A atm170004p
#SBATCH -t 04:00:00
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-16:1

python -m scripts.eval_sweep
