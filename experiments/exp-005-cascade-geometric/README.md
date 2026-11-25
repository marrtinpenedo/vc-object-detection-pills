# EXP-005 · Épocas × augmentación en Cascade R-CNN

## Hipótesis y diseño

Alargar el entrenamiento de Cascade puede cambiar el rendimiento y la época de saturación. La política afín puede modificar ese efecto. El resultado a 30 épocas de una tanda anterior deja abierta la respuesta a 100.

| Condición | Épocas | Política |
| --- | ---: | --- |
| A | 30 | Control |
| B | 100 | Control |
| C | 30 | Solo afín |
| D | 100 | Solo afín |

Las cuatro condiciones se lanzaron desde el snapshot original, con Detectron2 0.6, semilla 2026, ResNet50-FPN, pesos COCO, entrada 640, batch 4, FP32, SGD y LR 0,0025. Los controles históricos quedan como referencia descriptiva. El contraste usa el A contemporáneo. [Procedencia de los dos snapshots](../../docs/campaigns/exp-003-005/yolo-recovery.md)

Se aplica, solo en `train`, la afín compartida con probabilidad 0,5, rotación ±8°, traslación ±4 %, escala 0,90–1,10 y flip horizontal con probabilidad condicional 0,5. Se rechaza la transformación completa si alguna caja resulta inválida o sale del lienzo. Detectron2 transforma las cajas con el mismo mapa afín antes de `ResizeShortestEdge`. Cascade procesará imágenes individuales. Incorporar Mosaic exigiría un mapper multiimagen, semántica de recorte y pruebas adicionales, por lo que la campaña lo reserva para YOLO.

Cada época lógica equivale a 79 iteraciones. Las condiciones de 30 épocas usan **2 370 iteraciones** y las de 100 usan **7 900**. La validación y publicación de checkpoints ocurren cada 79 iteraciones. Se conserva el warm-up de 79 iteraciones y el horizonte de `WarmupCosineLR` se adapta a `MAX_ITER`. El mejor checkpoint maximiza el mAP COCO de `valid`, con preferencia por el primero en caso de empate. Detectron2 mantiene el modo CUDA no determinista necesario para la implementación de ROIAlign usada por el baseline.

## Lectura de resultados

Se informan los contrastes `B − A`, `D − C`, `C − A`, `D − B` y la interacción `(D − B) − (C − A)`, además de AP por clase, sensibilidad a Ibuphil 600 mg, precisión/recall, pérdidas, LR y evolución de `best`. Una ganancia oficial se interpreta junto al promedio de siete clases y el punto operativo antes de proponer adopción. La comparación cualitativa de las dos matrices debe tener en cuenta la diferencia de políticas. A 100 épocas, `E − B` en YOLO y `D − B` en Cascade aíslan la intención afín compartida dentro de cada detector. Los preprocesados y optimizadores siguen siendo propios de cada uno.

[Manifiesto](../../configs/campaign/exp-003-005.json) · [Operación y análisis](../../docs/campaigns/exp-003-005/README.md) · [Política geométrica](../../docs/campaigns/exp-003-005/geometric-policy.md) · [Hallazgos](findings.md)
