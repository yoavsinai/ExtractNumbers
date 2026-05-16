#!/bin/bash
#SBATCH --job-name=generic_job
#SBATCH --output=/home/dsi/halamim1/ExtractNumbers/slurm_logs/generic_%j.out
#SBATCH --error=/home/dsi/halamim1/ExtractNumbers/slurm_logs/generic_%j.err
#SBATCH --partition=L4-4h        # Partition name
#SBATCH --cpus-per-task=4        # Request 4 CPU cores
#SBATCH --mem=16G                # Request 16GB of RAM
#SBATCH --time=04:00:00          # Time limit (hh:mm:ss)
#SBATCH --mail-user=mati.halamish@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# ======================================================================
# GENERIC SLURM SCRIPT
# Usage: sbatch slarm_code/run_generic.slurm <script_path> [args...]
# ======================================================================

set -e

if [ $# -eq 0 ]; then
    echo "❌ Error: No python script provided."
    echo "Usage: sbatch run_generic.slurm <script_path> [arguments...]"
    exit 1
fi

SCRIPT_PATH=$1
shift # Remove the script path from arguments list, leaving the rest

echo "🚀 Starting job: $SCRIPT_PATH"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Arguments: $@"

cd /home/dsi/halamim1/ExtractNumbers
source .venv/bin/activate

python "$SCRIPT_PATH" "$@"

echo "✅ Job completed at $(date)"
