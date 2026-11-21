# Detección de pastillas

Práctica de Visión por Computador del Grado en Inteligencia Artificial de la USC, curso 2026–2027. El equipo I estudiará experimentalmente dos detectores preentrenados en MS COCO sobre el dataset asignado.

**Autores:** Martín Rodríguez Penedo y Santiago Pereira Olmeda.  
**Equipo:** I.

## Datos y modelos

- El archivo original contiene 451 imágenes JPEG y un JSON de anotaciones COCO por partición. Las carpetas originales son `train` (316 imágenes), `valid` (90) y `test` (45). [pills-sxdht en Hugging Face](https://huggingface.co/datasets/Francesco/pills-sxdht)
- **Cascade R-CNN**, con ResNet-50 + FPN, mediante Detectron2: configuración `Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml`, desactivando las máscaras para detectar únicamente cajas.
- **YOLOv10-S**, mediante Ultralytics: pesos preentrenados `yolov10s.pt`.

Los JSON declaran nueve categorías (ID y nombre): 0 `pills`, 1 `Cipro 500`, 2 `Ibuphil 600 mg`, 3 `Ibuphil Cold 400-60`, 4 `Xyzall 5mg`, 5 `blue`, 6 `pink`, 7 `red` y 8 `white`. Hay ocho categorías con anotaciones en el conjunto completo (IDs 1–8), frente a las siete clases del guion. El ID 0 no aparece en las anotaciones y el ID 2 aparece una vez en `train` y una vez en `valid`.

Desde la raíz del repositorio, con Python y pip disponibles:

```bash
python -m pip install huggingface_hub
python scripts/download_dataset.py
```

El script descarga `dataset.tar.gz` de una revisión fija y extrae las particiones originales en `data/pills-sxdht/`.

## Objetivos

1. Ajustar ambos detectores al dataset y establecer un baseline para cada uno.
2. Proponer, implementar y evaluar mejoras técnicamente justificadas.
3. Comparar la evolución y el rendimiento de ambos modelos, cuantitativa y cualitativamente. La comparación incluirá curvas precisión-recall, mAP@0.5, mAP@0.5:0.95, tiempos de inferencia y ejemplos de aciertos y fallos.

