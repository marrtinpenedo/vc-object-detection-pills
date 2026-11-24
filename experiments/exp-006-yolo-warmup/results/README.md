# Artefactos de EXP-006

## Recuperación selectiva

La carpeta estable [`phase-a/`](phase-a/) archiva 37 de los 39 archivos de la descarga verificada. [`evidence.sha256`](phase-a/evidence.sha256) conserva sus SHA-256 originales. Se seleccionaron los tres pares de JPEG revisados en los hallazgos y se omitieron los PNG originales de entrenamiento. La descarga, los checkpoints, las cachés, los logs y los datos brutos permanecen fuera de Git.

- **Procedencia.** [Commit de entrenamiento](phase-a/source-commit), [huellas de código y configuración](phase-a/code-config.sha256), [huellas de entrada](phase-a/inputs.sha256) y [manifiesto de datos](phase-a/data/manifest.json). Las huellas del lanzador extraídas de [`launch.json`](phase-a/eval-a-s2026/launch.json) coinciden con `slurm/evaluate.sh` y `slurm/evaluate.py` de `11b223b9a494e97886e715c0b043702ebdd24ba5`.
- **Entrenamiento.** [Control](phase-a/s2026/control/) y [warm-up 5](phase-a/s2026/warmup-5/) conservan el ID del trabajo, la ruta de origen, la configuración solicitada y efectiva, `run.json`, `status.json` e `history.csv`.
- **Evaluación.** [`eval-a-s2026/`](phase-a/eval-a-s2026/) contiene el ID de evaluación, los estados SLURM de entrenamiento, `verified.ok` y la procedencia del lanzamiento. Cada brazo conserva métricas, predicciones, curvas PR en JSON y PNG y los ejemplos emparejados 57, 61 y 80. Los SHA-256 declarados de `best` coinciden entre el lanzamiento y las métricas de cada brazo.

## Análisis derivado

El [resumen JSON](analysis/summary.json) y la [figura comparativa](analysis/history-s2026.png) proceden del analizador del commit de entrenamiento. El [procedimiento de reproducción](../README.md#recuperación-y-reproducción-del-análisis) usa `phase-a/` por defecto, valida sus huellas, reconstruye el COCO desde las anotaciones locales y genera nuevas salidas bajo el `results/` ignorado de la raíz.

Los JPEG proceden de *pills-sxdht*, de Mohamed Attia, distribuido en Roboflow 100 bajo licencia CC BY 4.0. Las imágenes de ejemplo incorporan cajas y etiquetas TP, FP y FN. [Dataset original](https://universe.roboflow.com/roboflow-100/pills-sxdht) · [Licencia](https://creativecommons.org/licenses/by/4.0/)
