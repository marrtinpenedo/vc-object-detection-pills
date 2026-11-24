# Evidencias de EXP-007

[`phase-a/`](phase-a/evidence.sha256) contiene las recetas y auditorías de entrenamiento y las métricas, predicciones y curvas PR (JSON y PNG) de la evaluación `12005631`. Las cuatro gráficas individuales se regeneraron a partir de los CSV originales porque la descarga de A solo incluía los historiales. `analysis/` contiene el [resumen](analysis/phase-a.json) y la [comparación](analysis/phase-a-history-s2026.png) reproducidos con el protocolo original. B y C tendrán archivos separados. Los entrenamientos reutilizados permanecen en su fase de origen.

## Ejemplos emparejados

Se conservan las escenas 57, 61 y 80 para el control y `freeze-15`. Sus JPEG se copiaron con el nombre derivado del punto operativo. Se añadió la escena 10, dibujada a partir de las predicciones existentes y de la imagen verificada contra el manifiesto. Se aplica el protocolo común de score ≥ 0,25 e IoU ≥ 0,50. Los contornos verdes indican TP, los rojos FP y los amarillos FN. Un `correct-ID.jpg` contiene al menos un TP y cero FP y FN. Un `failure-ID.jpg` contiene algún FP o FN. `empty-ID.jpg` indica que no hay TP, FP ni FN.

Los ejemplos del dataset *pills-sxdht* (Mohamed Attia, Roboflow 100) llevan sobreimpresos los resultados al punto operativo. [Dataset](https://universe.roboflow.com/roboflow-100/pills-sxdht) · [Licencia CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Recuperación selectiva de B y C

Ejecutar en Bash desde el clon local. Definir `CESGA_SSH` con la cuenta de CESGA y `REMOTE_HISTORICAL` con la ruta absoluta de la campaña original, terminada en `exp-007-yolo-staged-HhuHld`. El sufijo `-extension` identifica C. La descarga de C contiene solo los 13 entrenamientos nuevos y las tres reevaluaciones de siete brazos. Los otros ocho entrenamientos se leen directamente de A y B. Conservar la descarga en el `results/` ignorado de la raíz.

```bash
(
  set -euo pipefail
  : "${CESGA_SSH:?}" "${REMOTE_HISTORICAL:?}"
  mkdir -p results
  for phase in b c; do
    remote=$REMOTE_HISTORICAL
    if [[ $phase == c ]]; then remote="${REMOTE_HISTORICAL}-extension"; fi
    local_dir=$(mktemp -d "$PWD/results/exp-007-$phase-XXXXXX")
    uv run --offline --no-project --python 3.11 --with pillow==10.4.0 --with numpy==1.26.4 \
      python -m experiments.exp-007-yolo-staged-transfer.evidence inventory "$phase" > "$local_dir/download.files"
    rsync -a --files-from="$local_dir/download.files" -- "$CESGA_SSH:$remote/" "$local_dir/"
    (cd "$local_dir"; while IFS= read -r file; do sha256sum "$file"; done < download.files > downloaded.sha256)
    printf '%s %s\n' "$phase" "$local_dir"
  done
)
```

Quitar `--offline` si las dependencias faltan en caché. Conservar la descarga bruta con `downloaded.sha256` para repetir el análisis histórico y comprobar los archivos de trabajo. Al publicar, el archivo selectivo recibe su propio `evidence.sha256`. Conserva los datos numéricos, las figuras, el coste de entrenamiento y la procedencia. El analizador cruza `launch.json`, `run.json`, `status.json` y el entorno efectivo, y recalcula COCO y las curvas PR. Los estados auxiliares de SLURM y sus marcadores quedan en la descarga bruta y en CESGA. [Reproducción](../README.md#recuperación-y-reproducción-local)

## Publicación

Ejecutar en Bash desde el clon local. Ejemplo con B sin añadir JPEG. Para aportar una comparación visual, pasar un `--examples EVAL/BRAZO` por cada brazo elegido de una misma evaluación, con al menos dos brazos. Las tres escenas se mantienen emparejadas.

```bash
: "${PHASE_B:?}"
uv run --offline --no-project --python 3.11 --with pillow==10.4.0 --with numpy==1.26.4 \
  --with pycocotools==2.0.8 --with matplotlib==3.9.4 \
  python -m experiments.exp-007-yolo-staged-transfer.evidence archive b "$PHASE_B" \
  experiments/exp-007-yolo-staged-transfer/results/phase-b
```

Para C, utilizar `archive c`, su descarga de origen y `phase-c/`. Guardar los informes y comparaciones en `analysis/`. La fase A se exportó con `--examples eval-a-s2026/control --examples eval-a-s2026/freeze-15 --extra-image-id 10`.
