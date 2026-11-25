#!/bin/bash -l
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --open-mode=append
set -eo pipefail

mode="${1:?Uso: bash campaign.sh launch|status|collect|reconcile}"
shift
module --force purge
module load cesga/2022
module load miniconda3/22.11.1-1
module load cuda/12.1.1
set -u
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
if [[ "$mode" == job ]]; then
    test -n "${SLURM_JOB_ID:-}" || { printf '%s\n' 'Usa sbatch para tareas GPU.' >&2; exit 1; }
    cd -- "${SLURM_SUBMIT_DIR:?Falta el directorio de envío}/.."
    exec "$1/envs/$2/bin/python" controller.py job "$@"
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
work="${PILLS_CAMPAIGN_WORK:-${STORE:?Define STORE o PILLS_CAMPAIGN_WORK}/pills-baseline}"
backend=$(python3 -c 'import json; print(next(iter(json.load(open("plan.json"))["packages"])))')
exec "$work/envs/$backend/bin/python" controller.py "$mode" --work "$work" "$@"
