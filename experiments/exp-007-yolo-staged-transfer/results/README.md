# Evidencias de EXP-007

Pendientes de ejecución. Archivar la selección de resultados en `phase-a/`, `phase-b/` y `analysis/`. Conservar la descarga en el `results/` ignorado de la raíz. Los pesos, datos brutos, cachés y logs masivos permanecen fuera de Git.

## Recuperación selectiva

Desde el clon local en Bash, definir `CESGA_SSH` y `REMOTE_EXP` con la ruta absoluta de la tanda. El bloque descarga A. Para B, sustituir `rounds` por `b:2027:control,BRAZO` y `b:2028:control,BRAZO`, usando el brazo seleccionado. Verificar `COMPLETED|0:0` y `verified.ok` antes de recuperar una ronda.

```bash
rounds=(a:2026:control,freeze-5,freeze-10,freeze-15)
mkdir -p results
export LOCAL_EXP
LOCAL_EXP=$(mktemp -d "$PWD/results/exp-007-XXXXXX")
(
  set -euo pipefail
  : "${CESGA_SSH:?}" "${LOCAL_EXP:?}"
  [[ "${REMOTE_EXP:?}" == /*/exp-007-yolo-staged-* ]]
  {
    printf '%s\n' source-commit code-config.sha256 inputs.sha256 data/manifest.json
    for round in "${rounds[@]}"; do
      IFS=: read -r phase seed arms <<< "$round"
      IFS=, read -r -a selected <<< "$arms"
      ev="eval-$phase-s$seed"
      printf '%s\n' "$ev"/{eval.job,launch.json,verified.ok}
      for arm in "${selected[@]}"; do
        slot="s$seed/$arm"
        printf '%s\n' "$slot"/{config.json,train.job,train.path}
        printf '%s\n' "$slot/train"/{run.json,status.json,history.csv,ultralytics_runtime.json,parameters.json,weight_transfer.json}
        if [[ $arm == control ]]; then
          printf '%s\n' "$slot/train/parameter_updates.json"
        else
          printf '%s\n' "$slot/train/staged_transfer.json"
        fi
        printf '%s\n' "$ev/$arm.train.sacct"
        printf '%s\n' "$ev/$arm"/{metrics.json,predictions.json,pr_curves.json,pr_curves.png}
        printf '%s\n' "$ev/$arm/examples"/image-{57,61,80}.jpg
      done
    done
  } | LC_ALL=C sort -u > "$LOCAL_EXP/download.files"
  rsync -a --files-from="$LOCAL_EXP/download.files" -- "$CESGA_SSH:$REMOTE_EXP/" "$LOCAL_EXP/"
  cd "$LOCAL_EXP"
  while IFS= read -r file; do sha256sum "$file"; done < download.files > downloaded.sha256
) && printf '%s\n' "$LOCAL_EXP"
```

Para publicar una fase, copiar los archivos listados sin alterar su estructura y conservar sus huellas bajo `evidence.sha256`. Guardar en `analysis/` el resumen y las figuras citadas. Al reproducir, verificar primero las huellas y ejecutar el analizador desde el commit indicado por `source-commit`. Los binarios de `best` se verifican en CESGA al evaluar. La evidencia local conserva sus huellas y permite recalcular COCO a partir de las predicciones. [Reproducción](../README.md#recuperación-y-reproducción-local)

Los ejemplos del dataset *pills-sxdht* (Mohamed Attia, Roboflow 100) llevan sobreimpresos los resultados al punto operativo. [Dataset](https://universe.roboflow.com/roboflow-100/pills-sxdht) · [Licencia CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
