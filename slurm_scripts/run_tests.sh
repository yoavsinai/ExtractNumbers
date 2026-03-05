#!/bin/bash
#SBATCH --job-name=Test_OCR
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_scripts/logs/test_%j.out
#SBATCH --error=slurm_scripts/logs/test_%j.err
#SBATCH --mail-user=mati.halamish@gmail.com
#SBATCH --mail-type=ALL

# עבור לתיקיית הפרויקט
cd $SLURM_SUBMIT_DIR

# יצירת/הפעלת venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# (אופציונלי אבל מומלץ בפעם הראשונה)
# if [ -f "requirements.txt" ]; then
#     pip install --upgrade pip
#     pip install -r requirements.txt
# fi

# חשוב מאוד כדי ש-src יזוהה כמודול
export PYTHONPATH=$SLURM_SUBMIT_DIR
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "Running tests..."
python tests/test_ocr.py