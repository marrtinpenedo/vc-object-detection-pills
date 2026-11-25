# Evidencias de EXP-007

- [Fase A](phase-a/evidence.sha256). Cuatro entrenamientos y una evaluación conjunta (`12005631`). Las gráficas individuales se regeneraron a partir de los CSV de la descarga. [Informe](analysis/phase-a.json) · [Historial 2026](analysis/phase-a-history-s2026.png)
- [Fase B](phase-b/evidence.sha256). Cuatro entrenamientos y dos evaluaciones de parejas (`12006695`, `12006698`). [Informe A/B](analysis/phase-b.json) · [Historial 2027](analysis/phase-b-history-s2027.png) · [Historial 2028](analysis/phase-b-history-s2028.png)
- [Fase C](phase-c/evidence.sha256). 13 entrenamientos, el smoke `12006692` y tres evaluaciones de siete checkpoints (`12006702`, `12006708`, `12006714`). Los ocho entrenamientos reutilizados permanecen en A y B. [Informe](analysis/phase-c/summary.json) · [Duración](analysis/phase-c/duracion.png) · [Prefijos](analysis/phase-c/prefijo-congelado.png) · [Historial 2026](analysis/phase-c/history-s2026.png) · [Historial 2027](analysis/phase-c/history-s2027.png) · [Historial 2028](analysis/phase-c/history-s2028.png)

Se conservan las 29 evaluaciones con métricas, predicciones y curvas PR (JSON y PNG) y los 21 historiales y gráficas individuales. A y B registran el commit `ed7f9a6ef75afa9c66673bf8f03d34be2fa0c228`. C registra `7862c0e77cdbf7f6b9bd5c9673b30002cfc9ac9b` para sus nuevos entrenamientos y el commit original para el evaluador.

## Ejemplos emparejados

Se conservan las escenas 57, 61 y 80 para el control y `freeze-15` en A/2026 y B/2027, y para el control, `freeze-20`, `freeze-25` y `freeze-30` en C/2028. A incluye además la escena 10 y C la 86, dibujadas desde las predicciones y las imágenes verificadas contra el manifiesto. Se aplica score ≥ 0,25 e IoU ≥ 0,50. Los contornos verdes indican TP, los rojos FP y los amarillos FN. Un `correct-ID.jpg` contiene al menos un TP y cero FP y FN. Un `failure-ID.jpg` contiene algún FP o FN. `empty-ID.jpg` indica que no hay TP, FP ni FN.

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

## Repetir el archivado

Ejecutar en Bash desde el clon local con las rutas de las descargas brutas en `PHASE_B` y `PHASE_C`. Los archivos de salida se crean en un directorio nuevo bajo el `results/` local. Compararlos con las fases ya archivadas.

```bash
(
  set -euo pipefail
  : "${PHASE_B:?}" "${PHASE_C:?}"
  archive=$(mktemp -d "$PWD/results/exp-007-archive-XXXXXX")
  evidence=(uv run --offline --no-project --python 3.11 --with pillow==10.4.0 --with numpy==1.26.4
            --with pycocotools==2.0.8 --with matplotlib==3.9.4
            python -m experiments.exp-007-yolo-staged-transfer.evidence)
  "${evidence[@]}" archive b "$PHASE_B" "$archive/phase-b" \
    --examples eval-b-s2027/control --examples eval-b-s2027/freeze-15
  "${evidence[@]}" archive c "$PHASE_C" "$archive/phase-c" \
    --examples eval-c-s2028/control --examples eval-c-s2028/freeze-20 \
    --examples eval-c-s2028/freeze-25 --examples eval-c-s2028/freeze-30 --extra-image-id 86
  for phase in b c; do
    diff -rq "$archive/phase-$phase" "experiments/exp-007-yolo-staged-transfer/results/phase-$phase"
  done
)
```
