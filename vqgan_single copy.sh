#!/bin/bash
#SBATCH --job-name=vqgan_mono
#SBATCH --output=vqgan_mono_%j.log
#SBATCH --error=vqgan_mono_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00
# Activate virtual environment
srun /home/spruce/.venv/bin/python /home/spruce/ndp/cli_single.py fit