#!/bin/bash -l
#SBATCH --job-name=pills-evaluate
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --open-mode=append
set -eo pipefail

export PILLS_EVAL_ENTRYPOINT
PILLS_EVAL_ENTRYPOINT=$(realpath -e -- "${BASH_SOURCE[0]}")
if [[ ${1-} == --help || ${1-} == -h ]]; then
    launcher_dir="${PILLS_EVAL_LAUNCHER_DIR:-${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/slurm}}"
    exec python3 "${launcher_dir:-${PILLS_EVAL_ENTRYPOINT%/*}}/evaluate.py" "$@"
fi

cd "${SLURM_SUBMIT_DIR:?Envía el trabajo desde la raíz del repositorio}"
test -f scripts/evaluate_baseline.py || { printf '%s\n' 'Envía el trabajo desde la raíz del repositorio.' >&2; exit 1; }
launcher_dir="${PILLS_EVAL_LAUNCHER_DIR:-$PWD/slurm}"

module --force purge
module load cesga/2022
module load miniconda3/22.11.1-1
module load cuda/12.1.1
set -u

envs="${STORE:?Falta STORE en el entorno de CESGA}/pills-baseline/envs"
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
for backend in cascade yolo; do
    python="$envs/$backend/bin/python"
    if [[ -x "$python" ]]; then
        exec "$python" "$launcher_dir/evaluate.py" "$@"
    fi
done
printf '%s\n' "No se encontró un intérprete Cascade o YOLO en $envs." >&2
exit 1
