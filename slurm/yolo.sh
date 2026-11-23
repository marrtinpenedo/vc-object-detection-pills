#!/bin/bash -l
#SBATCH --job-name=pills-yolo
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --open-mode=append
set -eo pipefail

module --force purge
module load cesga/2022
module load miniconda3/22.11.1-1
module load cuda/12.1.1
set -u

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f scripts/train_baseline.py || { printf '%s\n' 'Envía el trabajo desde la raíz del repositorio.' >&2; exit 1; }
baseline_work="${STORE:?Falta STORE en el entorno de CESGA}/pills-baseline"
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$baseline_work/cache/ultralytics"
mkdir -p "$YOLO_CONFIG_DIR"
exec "$baseline_work/envs/yolo/bin/python" -m scripts.train_baseline yolo "$@"
