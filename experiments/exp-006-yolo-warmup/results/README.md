# Artefactos de EXP-006

Pendientes de ejecución. Cada tanda conserva el commit y las huellas de código, datos y pesos iniciales. El nombre `sSEMILLA/BRAZO` identifica un entrenamiento. `eval-RONDA-sSEMILLA/BRAZO` identifica la reevaluación, que puede reutilizarse para varias comparaciones en la misma asignación. Los JPEG muestran las mismas imágenes de validación en todos los brazos.

## Recuperación selectiva

Desde la raíz del clon en el portátil, iniciar `bash` si el shell habitual es Zsh. Definir `CESGA_SSH` conforme a la guía y asignar a `REMOTE_EXP` la ruta absoluta mostrada al preparar la tanda. El ejemplo recupera A. Para incluir B, agregar `b:2026:control,warmup-5,warmup-3`. Para incluir C, agregar `c:2027:control,BRAZO` y `c:2028:control,BRAZO` con el brazo elegido. Cada descarga usa una carpeta nueva bajo el directorio local ignorado `results/`. [Acceso a CESGA](../../../docs/cesga.md#acceso-y-almacenamiento)

```bash
rounds=(a:2026:control,warmup-5)
mkdir -p results
export LOCAL_EXP
LOCAL_EXP=$(mktemp -d "$PWD/results/exp-006-XXXXXX")
(
  set -euo pipefail
  : "${CESGA_SSH:?}" "${LOCAL_EXP:?}"
  [[ "${REMOTE_EXP:?}" == /*/exp-006-yolo-warmup-* ]]
  {
    printf '%s\n' source-commit code-config.sha256 inputs.sha256 data/manifest.json
    for round in "${rounds[@]}"; do
      IFS=: read -r phase seed arms <<< "$round"
      IFS=, read -r -a selected <<< "$arms"
      eval_dir="eval-$phase-s$seed"
      printf '%s\n' "$eval_dir"/{eval.job,verified.ok}
      for arm in "${selected[@]}"; do
        slot="s$seed/$arm"
        printf '%s\n' "$slot"/{config.json,train.job,train.path}
        printf '%s\n' "$slot/train"/{run.json,status.json,history.csv,ultralytics_runtime.json,training.png}
        printf '%s\n' "$eval_dir/$arm.train.sacct"
        printf '%s\n' "$eval_dir/$arm"/{metrics.json,predictions.json,pr_curves.json,pr_curves.png}
        printf '%s\n' "$eval_dir/$arm/examples"/image-{57,61,80}.jpg
      done
    done
  } | LC_ALL=C sort -u > "$LOCAL_EXP/download.files"
  rsync -a --files-from="$LOCAL_EXP/download.files" -- "${CESGA_SSH}:${REMOTE_EXP}/" "$LOCAL_EXP/"
  cd "$LOCAL_EXP"
  while IFS= read -r file; do sha256sum "$file"; done < download.files > downloaded.sha256
) && printf '%s\n' "$LOCAL_EXP"
```

La lista incluye configuración y registro efectivo, IDs, estados, métricas, predicciones, curvas y ejemplos emparejados. Los checkpoints, logs y datos de entrenamiento permanecen en CESGA. Antes del análisis, verificar la descarga con `(cd "$LOCAL_EXP" && sha256sum --check --quiet downloaded.sha256)`. [Reproducción del análisis](../README.md#recuperación-y-reproducción-del-análisis)

## Evidencia para versionar

Al cerrar el experimento, guardar cada tanda utilizada en una carpeta nueva de este directorio y mantener las rutas del inventario. Conservar los archivos de procedencia (`source-commit`, `code-config.sha256`, `inputs.sha256` y `data/manifest.json`), las configuraciones y estados de SLURM, los historiales y metadatos de entrenamiento, y `metrics.json`, `predictions.json`, `pr_curves.json` y `pr_curves.png` de cada evaluación utilizada. El PNG de PR forma parte de la comprobación del verificador común. El manifiesto registra huellas y recuentos del dataset.

El siguiente bloque Bash archiva la evidencia mínima de una descarga comprobada. Elegir en `evidence` una ruta nueva. Para conservar un JPEG citado o una figura original de entrenamiento, incluir su entrada en `evidence.sha256` desde el inventario de la descarga antes de la copia.

```bash
(
  set -euo pipefail
  source=$(realpath "${LOCAL_EXP:?}")
  evidence="$PWD/experiments/exp-006-yolo-warmup/results/tanda-$(basename "$source")"
  (cd "$source" && sha256sum --check --quiet downloaded.sha256)
  mkdir "$evidence"
  awk '$2 !~ /\/train\/training\.png$|\/examples\//' "$source/downloaded.sha256" > "$evidence/evidence.sha256"
  awk '{print $2}' "$evidence/evidence.sha256" |
    rsync -a --files-from=- -- "$source/" "$evidence/"
  (cd "$evidence" && sha256sum --check --quiet evidence.sha256)
)
```

Guardar en `results/analysis/` los `summary.json` verificados y las figuras citadas en los hallazgos, separados por comparación. Tras versionar las conclusiones, definir `LOCAL_EXP` con la ruta de la carpeta de evidencia y repetir el análisis con las mismas parejas. El procedimiento consulta `source-commit` y escribe las salidas regeneradas bajo el `results/` ignorado de la raíz. Las descargas completas, los pesos, los logs masivos y las anotaciones originales permanecen fuera de Git. [Reproducción del análisis](../README.md#recuperación-y-reproducción-del-análisis)

Los JPEG proceden de *pills-sxdht*, de Mohamed Attia, distribuido en Roboflow 100 bajo licencia CC BY 4.0. Las imágenes de ejemplo incorporan cajas y etiquetas TP, FP y FN. [Dataset original](https://universe.roboflow.com/roboflow-100/pills-sxdht) · [Licencia](https://creativecommons.org/licenses/by/4.0/)
