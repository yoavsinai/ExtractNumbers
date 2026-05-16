#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --output=/home/dsi/halamim1/ExtractNumbers/slurm_logs/eval_all_%j.out
#SBATCH --error=/home/dsi/halamim1/ExtractNumbers/slurm_logs/eval_all_%j.err
#SBATCH --partition=L4-4h        # Partition name
#SBATCH --cpus-per-task=4        # Request 4 CPU cores
#SBATCH --mem=16G                # Request 16GB of RAM
#SBATCH --time=02:00:00          # Time limit (hh:mm:ss)
#SBATCH --mail-user=mati.halamish@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# ======================================================================
# SLURM SCRIPT FOR RUNNING THE FULL EVALUATION PIPELINE
# This will run src/evaluation/eval_all.py and keep all outputs
# under the project directories and Slurm log files in /slurm_logs.
# ======================================================================

set -e

echo "Starting evaluation job on $(hostname) at $(date)"

cd /home/dsi/halamim1/ExtractNumbers
source .venv/bin/activate

# Optional: pass a maximum sample count via SBATCH --export or environment
MAX_SAMPLES=${MAX_SAMPLES:-2000}

python src/evaluation/eval_all.py --max-samples "$MAX_SAMPLES"

echo "Evaluation job completed at $(date)"
