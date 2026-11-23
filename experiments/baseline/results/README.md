# Resultados de las baselines

Resultados de entrenamiento y evaluación de Cascade R-CNN y YOLOv10-S en CESGA FinisTerrae III.

Las imágenes de `*/evaluation/examples/` proceden del dataset `pills-sxdht`, creado originalmente por Mohamed Attia y distribuido en Roboflow 100 bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Dataset: https://universe.roboflow.com/roboflow-100/pills-sxdht
- Copia utilizada: https://huggingface.co/datasets/Francesco/pills-sxdht
- Modificaciones: superposición de cajas y etiquetas TP, FP y FN para visualizar las evaluaciones.

Los checkpoints entrenados se conservan fuera del repositorio.

## Historial de YOLOv10-S

La figura [`training.png`](yolo/training/training.png) se regeneró a partir del [`history.csv`](yolo/training/history.csv) original, sin alterar los datos. Se representan las pérdidas en escala logarítmica y el AP de validación. Se señala el pico de `val/cls_loss` de 50 317 300 en la época 2; la selección del checkpoint se basó en el mAP de validación.

Para regenerarla desde la raíz del repositorio, con Matplotlib instalado, indique una ruta de salida nueva:

```bash
python experiments/baseline/plot_yolo_history.py \
  --history experiments/baseline/results/yolo/training/history.csv \
  --output /tmp/yolo-training.png
```
