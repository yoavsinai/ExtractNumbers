#!/bin/bash
#SBATCH --job-name=Download_SVHN
#SBATCH --partition=generic
#SBATCH --output=slurm_scripts/logs/download_%j.out
#SBATCH --error=slurm_scripts/logs/download_%j.err
#SBATCH --mail-user=mati.halamish@gmail.com
#SBATCH --mail-type=FAIL,END

# Activate the virtual environment
# source .venv/bin/activate

# Run data preparation
echo "Starting data download..."
python src/data_prep.py

# Deactivate the environment
# deactivate
