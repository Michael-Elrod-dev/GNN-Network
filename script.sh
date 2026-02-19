#!/bin/bash
#SBATCH --job-name=gnn
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --gpus=a100:1
#SBATCH --partition=work1

# Run the main script
python main.py
