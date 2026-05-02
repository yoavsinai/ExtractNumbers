#!/bin/bash
#SBATCH --job-name=extract_numbers
#SBATCH --output=/home/dsi/halamim1/ExtractNumbers/slurm_logs/pipeline_%j.out
#SBATCH --error=/home/dsi/halamim1/ExtractNumbers/slurm_logs/pipeline_%j.err
#SBATCH --partition=L4-4h        # Partition name
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=4        # Request 4 CPU cores
#SBATCH --mem=16G                # Request 16GB of RAM
#SBATCH --time=04:00:00          # Time limit (hh:mm:ss)
#SBATCH --mail-user=mati.halamish@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# ==============================================================================
# PREREQUISITES BEFORE RUNNING THIS SCRIPT:
# 1. Make sure to download the required data.
# 2. Update the directory paths in this script (like --output, --error, and 'cd')
#    to match the location where the project is saved on your machine.
# ==============================================================================

echo "Starting pipeline job on $(hostname) at $(date)"

# Navigate to the project root
cd /home/dsi/halamim1/ExtractNumbers

# Activate the Python virtual environment
source .venv/bin/activate

# Run the pipeline script
python src/training/train_pipeline.py

echo "Job completed at $(date)"