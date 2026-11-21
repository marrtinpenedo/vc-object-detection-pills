# Detección de pastillas

Práctica de Visión por Computador del Grado en Inteligencia Artificial de la USC, curso 2026–2027. El grupo I estudiará experimentalmente dos detectores preentrenados en MS COCO sobre el dataset asignado.

**Autores:** Martín Rodríguez Penedo y Santiago Pereira Olmeda.  
**Grupo interactivo:** I.

## Datos y modelos

- [pills-sxdht](https://huggingface.co/datasets/Francesco/pills-sxdht): imágenes de pastillas de siete clases. Se utilizarán las particiones de entrenamiento, validación y test proporcionadas.
- **Cascade R-CNN**, con ResNet-50 + FPN, mediante Detectron2: configuración `Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml`, desactivando las máscaras para detectar únicamente cajas.
- **YOLOv10-S**, mediante Ultralytics: pesos preentrenados `yolov10s.pt`.

## Objetivos

1. Ajustar ambos detectores al dataset y establecer un baseline para cada uno.
2. Proponer, implementar y evaluar mejoras técnicamente justificadas.
3. Comparar la evolución y el rendimiento de ambos modelos, cuantitativa y cualitativamente. La comparación incluirá curvas precisión-recall, mAP@0.5, mAP@0.5:0.95, tiempos de inferencia y ejemplos de aciertos y fallos.

