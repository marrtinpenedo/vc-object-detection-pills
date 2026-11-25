# EXP-004 · Épocas × augmentación en YOLOv10-S

## Hipótesis y diseño

Una receta de 100 épocas y una política geométrica pueden mejorar el mAP de YOLOv10-S por separado o cuando se combinan. RF100 aporta un antecedente de entrenamiento más largo y augmentación online con YOLOv5. Las diferencias de modelo, receta y protocolo impiden usar su cifra como objetivo comparable. [Código oficial de RF100](https://github.com/roboflow/roboflow-100-benchmark)

| Condición | Épocas | Política |
| --- | ---: | --- |
| A | 30 | Control |
| B | 100 | Control |
| C | 30 | Afín + Mosaic |
| D | 100 | Afín + Mosaic |
| E | 100 | Solo afín |

Se entrena un A nuevo junto con B–E. Los 30 ciclos históricos son una referencia descriptiva. La campaña cambia los cargadores y la revisión, y el nuevo control permite verificar la receta efectiva y las entradas bajo el mismo snapshot. Las cinco condiciones usan Ultralytics 8.3.161, semilla 2026, pesos COCO fijos, imagen 640, batch 8, FP32, AdamW y LR 0,001. Se evalúa en `valid` cada época y se reevaluará el checkpoint `best`.

## Transformaciones y presupuesto

La afín compartida se intenta con probabilidad 0,5 por muestra. Muestrea rotación ±8°, traslación ±4 % por eje y escala uniforme entre 0,90 y 1,10. El flip horizontal tiene probabilidad condicional 0,5. Si alguna caja transformada queda fuera del lienzo o degenera, se conserva la muestra previa completa. La imagen usa interpolación lineal y un borde neutro de valor 114. Las cajas se recalculan a partir de sus cuatro esquinas y mantienen su clase.

C y D anteponen Mosaic nativo de cuatro teselas con probabilidad 0,30. Las teselas adicionales se muestrean de todo `train`, con reemplazo, porque el cargador base no mantiene el buffer nativo de augmentación. Mosaic recorta en su lienzo de 2 × 640 y actualiza las etiquetas. Se conserva el lienzo completo y `LetterBox` lo reduce a 640. Si se acepta el sorteo afín, se aplica entre Mosaic y el resize. El probador nativo comprueba composición de píxeles, cajas, clases y colación con muestras simples. Toda augmentación se desactiva durante validación, inferencia y test.

Las condiciones de 30 y 100 épocas consumen 1 200 y 4 000 pasos nominales. El mínimo de warm-up de 100 índices de minibatch se conserva. Ultralytics ajusta el horizonte del coseno al número de épocas y `close_mosaic=0` mantiene Mosaic hasta la última época de C y D. La búsqueda del mejor checkpoint también usa más validaciones con 100 épocas. [Scheduler y cierre de Mosaic, Ultralytics 8.3.161](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/trainer.py)

## Lectura de resultados

La métrica principal es mAP COCO@0,50:0,95 de ocho clases en `valid`. La interacción es `(D − B) − (C − A)`. Se informan `B − A`, `D − C`, `C − A`, `D − B`, `D − E` para habilitar Mosaic en la receta y `E − B` para la afín sola a 100 épocas. El efecto de habilitar Mosaic incluye la reducción de escala del lienzo y un cambio de frecuencia afín efectiva. La simulación de cajas estima 42,9 % de afín efectiva en C/D y 49,5 % en E. También se informan AP50/AP75, AP por clase, sensibilidad sin Ibuphil 600 mg, precisión/recall, pérdidas, LR y diferencia entre mejor y última época. Una ganancia oficial es evidencia exploratoria. Para proponer adopción se revisarán su signo a siete clases, el punto operativo y las regresiones por clase. Una interacción observada en semilla 2026 no estima variabilidad entre semillas. [Auditoría geométrica](../../docs/campaigns/exp-003-005/geometric-policy.md#auditoría-de-la-receta-sobre-train)

[Manifiesto](../../configs/campaign/exp-003-005.json) · [Operación y análisis](../../docs/campaigns/exp-003-005/README.md) · [Hallazgos](findings.md)
