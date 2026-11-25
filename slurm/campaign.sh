#!/bin/bash -l
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --open-mode=append
set -eo pipefail

# The submitter passes absolute paths as script arguments. The jobs can outlive the session.
campaign=()
login_backend=yolo
if [[ ${1:-} == exp008 ]]; then
    campaign=(--exp008)
    login_backend=cascade
    shift
fi
mode="${1:?Uso: bash slurm/campaign.sh launch|status|collect|reconcile ...}"
shift
module --force purge
module load cesga/2022
module load miniconda3/22.11.1-1
module load cuda/12.1.1
set -u
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
# The launcher submits from the source root. BASH_SOURCE in a batch job points into the spool.
if [[ "$mode" == job ]]; then
    cd -- "${SLURM_SUBMIT_DIR:?Falta el directorio de envío}"
    test -n "${SLURM_JOB_ID:-}" || { printf '%s\n' 'Usa sbatch para tareas GPU.' >&2; exit 1; }
    exec "$1/envs/$2/bin/python" -m scripts.campaign "${campaign[@]}" job "$@"
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
work="${PILLS_CAMPAIGN_WORK:-${STORE:?Define STORE o PILLS_CAMPAIGN_WORK}/pills-baseline}"
exec "$work/envs/$login_backend/bin/python" -m scripts.campaign "${campaign[@]}" "$mode" --work "$work" "$@"
