# Artefactos de los baselines

Archivos exportados del entrenamiento y la evaluación en `valid` de Cascade R-CNN y YOLOv10-S en CESGA FinisTerrae III. La comparación y los diagnósticos están en los [hallazgos](../findings.md).

| Detector | Entrenamiento | Evaluación |
| --- | --- | --- |
| Cascade R-CNN | [Procedencia](cascade/training/run.json) · [Estado](cascade/training/status.json) · [Historial](cascade/training/history.csv) · [Figura](cascade/training/training.png) | [Métricas](cascade/evaluation/metrics.json) · [Predicciones](cascade/evaluation/predictions.json) · [Datos PR](cascade/evaluation/pr_curves.json) · [Curvas PR](cascade/evaluation/pr_curves.png) · [Ejemplos](cascade/evaluation/examples/) |
| YOLOv10-S | [Procedencia](yolo/training/run.json) · [Estado](yolo/training/status.json) · [Historial](yolo/training/history.csv) · [Figura](yolo/training/training.png) | [Métricas](yolo/evaluation/metrics.json) · [Predicciones](yolo/evaluation/predictions.json) · [Datos PR](yolo/evaluation/pr_curves.json) · [Curvas PR](yolo/evaluation/pr_curves.png) · [Ejemplos](yolo/evaluation/examples/) |

Los checkpoints entrenados se conservan fuera del repositorio. Los informes de evaluación identifican los checkpoints, el manifiesto de datos y el COCO de validación mediante SHA-256. Los registros de entrenamiento y evaluación tienen `git_head: null` y guardan `code_sha256` por script. La huella archivada de `scripts/baseline_metrics.py` difiere de las versiones del historial de `main`. La revisión Git exacta de las ejecuciones queda sin confirmar.

## Atribución de las imágenes

Las imágenes de `*/evaluation/examples/` proceden de `pills-sxdht`, creado por Mohamed Attia y distribuido en Roboflow 100. Se añadieron cajas y etiquetas TP, FP y FN para visualizar la evaluación. [Dataset original](https://universe.roboflow.com/roboflow-100/pills-sxdht) · [Copia utilizada](https://huggingface.co/datasets/Francesco/pills-sxdht) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Historial de YOLOv10-S

La figura de YOLO se regeneró a partir del `history.csv` original y conserva sus datos. Muestra las pérdidas en escala logarítmica y señala el pico de `val/cls_loss` en la curva. El panel de AP muestra la métrica con la que se seleccionó el checkpoint.

Para regenerarla desde la raíz del repositorio, con Matplotlib instalado, indique una ruta de salida nueva:

```bash
python experiments/baseline/plot_yolo_history.py \
  --history experiments/baseline/results/yolo/training/history.csv \
  --output /tmp/yolo-training.png
```
