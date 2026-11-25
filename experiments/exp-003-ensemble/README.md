# EXP-003 · Ensemble YOLO y Cascade

Hipótesis. Las cajas y scores complementarios de YOLOv10-S podrían mejorar el mAP COCO de Cascade control en validación. Los dos controles proceden de EXP-002. El diagnóstico anterior encontró un solo TP exclusivo de YOLO al punto operativo.

## Regla fijada antes del recálculo

En cada imagen y clase COCO, recorrer las cajas de Cascade por score descendente. Emparejar cada una con la caja libre de YOLO de mayor IoU, a partir de 0,55. Usar el score medio y promediar los cuatro componentes de la caja en píxeles con pesos proporcionales a los scores. Desempatar por orden de exportación. Una caja Cascade sin pareja conserva su score. Una caja YOLO sin pareja entra con medio score si su IoU frente a todas las cajas Cascade de su clase es inferior a 0,55. El descuento de una evidencia exclusiva reduce el efecto de añadir detecciones de YOLO sin corroboración. Se mantiene el score mínimo de 0,001 y el máximo de 100 cajas por imagen. Las clases siguen siendo los IDs COCO originales.

Criterio de éxito exploratorio. Superar el mAP de Cascade y comprobar si el signo se mantiene al promediar siete clases, con especial atención a AP por clase y al recall en score 0,25 e IoU 0,50. Se ejecuta una única configuración. Los scores de los modelos no están calibrados conjuntamente. La latencia real de servir ambos modelos requiere una medición con los dos checkpoints.

## Reproducción

Desde la raíz, con el dataset original y el entorno local de análisis, ejecutar:

```bash
MPLBACKEND=Agg uv run --no-project --python 3.11 \
  --with pycocotools==2.0.8 --with numpy==1.26.4 --with pillow==10.4.0 \
  --with matplotlib==3.9.4 --with pyyaml==6.0.4 \
  python -m experiments.exp-003-ensemble.analyze --output results/exp-003-ensemble-reproduction
```

Añadir `--offline` si todas las dependencias figuran en la caché de `uv`. También se puede usar el intérprete local que ya las tenga. Se verifican el inventario, la procedencia y el COCO reconstruido. Se publican las métricas, las predicciones fusionadas, las PR, las tablas y el recibo de la regla. Los resultados se computan en CPU y usan exclusivamente `valid`. [Hallazgos](findings.md)
