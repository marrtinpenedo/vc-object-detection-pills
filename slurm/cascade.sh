#!/usr/bin/env bash
#SBATCH --job-name=pills-cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source "${CASCADE_ENV:?Set CASCADE_ENV to your virtual environment}/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
exec python -m scripts.train_baseline cascade "$@"
