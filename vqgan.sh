#!/bin/bash
#SBATCH --job-name=vqgan
#SBATCH --output=vqgan_%j.log
#SBATCH --error=vqgan_%j.err
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00

# Activate virtual environment
srun /home/spruce/.venv/bin/python /home/spruce/ndp/cli_main.py fit