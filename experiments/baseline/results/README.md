# Resultados de las baselines

Resultados de entrenamiento y evaluación de Cascade R-CNN y YOLOv10-S en CESGA FinisTerrae III.

Las imágenes de `*/evaluation/examples/` proceden del dataset `pills-sxdht`, creado originalmente por Mohamed Attia y distribuido en Roboflow 100 bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Dataset: https://universe.roboflow.com/roboflow-100/pills-sxdht
- Copia utilizada: https://huggingface.co/datasets/Francesco/pills-sxdht
- Modificaciones: superposición de cajas y etiquetas TP, FP y FN para visualizar las evaluaciones.

Los checkpoints entrenados se conservan fuera del repositorio.

## Historial de YOLOv10-S

Se conserva [la figura original](yolo/training/training.png) como registro de la ejecución. La [figura de lectura](yolo/training/training-readable.png) usa dos paneles logarítmicos de pérdidas. Un panel reúne `val/cls_loss` y `val/dfl_loss` para mostrar sus picos sin comprimir las demás curvas. La figura conserva todas las épocas y las curvas de AP de validación.

En la segunda época, `val/cls_loss` alcanzó 50 317 300. Este valor condiciona la lectura de la pérdida en escala lineal. La selección del checkpoint se basó en mAP de validación.

Para regenerar la figura, ejecute desde la raíz del repositorio con Matplotlib disponible, por ejemplo en el entorno Cascade. Indique un PNG de salida nuevo.

```bash
python experiments/baseline/plot_yolo_history.py \
  --history experiments/baseline/results/yolo/training/history.csv \
  --output /tmp/yolo-training-readable.png
```
