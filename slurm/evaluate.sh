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

if [[ $# -ne 4 ]]; then
    printf '%s\n' 'Uso: sbatch slurm/evaluate.sh DATOS RUN_CASCADE RUN_YOLO SALIDA_NUEVA' >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:?Envía el trabajo desde la raíz del repositorio}"
test -f scripts/evaluate_baseline.py || { printf '%s\n' 'Envía el trabajo desde la raíz del repositorio.' >&2; exit 1; }
data=$(realpath -e -- "$1")
cascade_run=$(realpath -e -- "$2")
yolo_run=$(realpath -e -- "$3")
test ! -L "$4" || { printf '%s\n' 'La salida debe ser un directorio nuevo.' >&2; exit 1; }
output=$(realpath -m -- "$4")
for input in "$data" "$cascade_run" "$yolo_run"; do
    if [[ ! -d "$input" || "$output/" == "$input/"* ]]; then
        printf '%s\n' 'La salida debe quedar fuera de los directorios de datos y entrenamiento.' >&2
        exit 1
    fi
done

module --force purge
module load cesga/2022
module load miniconda3/22.11.1-1
module load cuda/12.1.1
set -u

baseline_work="${STORE:?Falta STORE en el entorno de CESGA}/pills-baseline"
cascade_python="$baseline_work/envs/cascade/bin/python"
yolo_python="$baseline_work/envs/yolo/bin/python"
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$baseline_work/cache/ultralytics"
mkdir -p -- "$YOLO_CONFIG_DIR"
mkdir -- "$output"

common=(--data "$data" --split valid --checkpoint best --benchmark --warmup 20 --iterations 100)
"$cascade_python" -m scripts.evaluate_baseline "${common[@]}" --run "$cascade_run" --output "$output/cascade"
"$yolo_python" -m scripts.evaluate_baseline "${common[@]}" --run "$yolo_run" --output "$output/yolo"
exec "$cascade_python" -m scripts.verify_baseline --data "$data" --split valid \
    --runs "$cascade_run" "$yolo_run" --evaluations "$output/cascade" "$output/yolo"
