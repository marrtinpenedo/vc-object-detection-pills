# EXP-007 · Hallazgos en validación

## Fase A

Se comprobó la descarga por SHA-256 y se ejecutó el analizador de `ed7f9a6ef75afa9c66673bf8f03d34be2fa0c228`. El recálculo COCO reproduce el AP y las curvas PR de los cuatro checkpoints `best` evaluados en la misma asignación `12005631`. El protocolo anterior incluyó el smoke en A100 con transición, completado según el registro de la campaña. La evidencia local de A contiene las auditorías de los tres brazos congelados. [Resumen verificado](results/analysis/phase-a.json) · [Inventario](results/README.md)

| Brazo | mAP oficial | mAP de siete clases | Recall operativo | Mejor época | Último mAP del historial |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0,70209365 | 0,688107 | 0,693878 | 23 | 0,653203 |
| freeze-5 | 0,73572803 | 0,712261 | 0,673469 | 29 | 0,730831 |
| freeze-10 | 0,76169139 | 0,741933 | 0,714286 | 27 | 0,678327 |
| freeze-15 | 0,80732573 | 0,794087 | 0,816327 | 15 | 0,688505 |

La regla original seleccionó `freeze-15`. `freeze-10` también supera los márgenes de exploración. `freeze-5` pierde 0,020408 de recall operativo y rebasa el margen de 0,02. El mejor checkpoint de `freeze-15` se obtiene en la última época congelada. El orden observado y la evolución del historial motivan C con una sola semilla disponible. No establecen un perjuicio general de la descongelación. [Criterios](README.md#fases-y-criterio-de-decisión) · [Historial comparativo](results/analysis/phase-a-history-s2026.png)

En la escena 10 ambos modelos detectan el objeto. El control añade un falso positivo y `freeze-15` conserva únicamente el acierto. Ambos presentan fallos en las escenas 57, 61 y 80. [Escena 10, control](results/phase-a/eval-a-s2026/control/examples/failure-10.jpg) · [Escena 10, freeze-15](results/phase-a/eval-a-s2026/freeze-15/examples/correct-10.jpg)

## Fase B · Confirmación de freeze-15

Las dos semillas de confirmación mejoran el mAP oficial, el mAP de siete clases y el recall operativo frente a sus respectivos controles. Los deltas son variante menos control.

| Semilla de confirmación | mAP control | mAP freeze-15 | Δ mAP | Δ mAP sin Ibuphil | Δ recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2027 | 0,771630 | 0,798991 | +0,027361 | +0,045555 | +0,173469 |
| 2028 | 0,761310 | 0,804803 | +0,043494 | +0,078278 | +0,112245 |

El delta medio de mAP en B es +0,035427 (DE muestral 0,011408). En las tres parejas de A/B es +0,058695 (DE 0,041101), con tres mejoras, Δ medio de siete clases de +0,076604 y Δ medio de recall de +0,136054. Se cumple el criterio preestablecido y B confirma `freeze-15`. En las tres semillas su mejor checkpoint corresponde a la época 15, con el backbone congelado. En la escena 57 de la semilla 2027, el control omite el objeto y produce un falso positivo. `freeze-15` lo detecta sin falsos positivos en esa escena. [Informe A/B](results/analysis/phase-b.json) · [Escena 57, control](results/phase-b/eval-b-s2027/control/examples/failure-57.jpg) · [Freeze-15](results/phase-b/eval-b-s2027/freeze-15/examples/correct-57.jpg)

## Fase C · Duración de la congelación

Se reevaluaron los siete checkpoints `best` de cada semilla juntos. El recálculo local reproduce mAP, AP por clase, métricas del punto operativo y curvas PR. Las reevaluaciones de los checkpoints reutilizados coinciden con sus resultados previos de A/B, con el mismo SHA-256 de los pesos. Las latencias se midieron de nuevo en C. [Informe C y procedencia](results/analysis/phase-c/summary.json) · [Rendimiento por duración](results/analysis/phase-c/duracion.png)

| Brazo | 2026 | 2027 | 2028 | mAP medio ± DE | Δ emparejado vs. control ± DE | Δ medio vs. freeze-15 | Mejoras vs. control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 0,702094 | 0,771630 | 0,761310 | 0,745011 ± 0,037524 | — | −0,058695 | — |
| freeze-5 | 0,735728 | 0,683315 | 0,636579 | 0,685207 ± 0,049602 | −0,059804 ± 0,082943 | −0,118499 | 1/3 |
| freeze-10 | 0,761691 | 0,744554 | 0,823371 | 0,776539 ± 0,041453 | +0,031528 ± 0,050767 | −0,027168 | 2/3 |
| freeze-15 | 0,807326 | 0,798991 | 0,804803 | 0,803707 ± 0,004274 | +0,058695 ± 0,041101 | — | 3/3 |
| freeze-20 | 0,812451 | 0,844575 | 0,854798 | **0,837275 ± 0,022097** | **+0,092264 ± 0,018736** | +0,033568 | 3/3 |
| freeze-25 | 0,823811 | 0,818360 | 0,782325 | 0,808165 ± 0,022544 | +0,063154 ± 0,052321 | +0,004459 | 3/3 |
| freeze-30 | 0,823811 | 0,834355 | 0,842227 | 0,833464 ± 0,009240 | +0,088453 ± 0,030209 | +0,029758 | 3/3 |

`freeze-20` cumple los márgenes de C y encabeza el orden previsto, seguido de `freeze-30`, `freeze-25`, `freeze-15` y `freeze-10`. Frente al control, suma +0,105444 de mAP de siete clases y +0,149660 de recall. Frente a `freeze-30`, gana en 2027 y 2028 y la ventaja media es de 0,003811. `freeze-30` empata con `freeze-25` en el mejor mAP de 2026, tiene menor dispersión del mAP que `freeze-20` y alcanza la mayor precisión operativa. B conserva su decisión confirmatoria. La elección de 20 épocas proviene de la búsqueda ampliada sobre estas mismas semillas.

### Clases, tamaños y punto operativo

Media entre semillas. El punto operativo es score ≥ 0,25 e IoU ≥ 0,50. Los tamaños COCO corresponden a 2 objetos pequeños, 22 medianos y 74 grandes en `valid`.

| Brazo | mAP de 7 clases | AP50 | AP75 | AP pequeño | AP mediano | AP grande | Precisión | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 0,7181 | 0,8567 | 0,8443 | 0,0000 | 0,6378 | 0,7731 | 0,8207 | 0,7177 |
| freeze-5 | 0,7188 | 0,7945 | 0,7782 | 0,0000 | 0,6821 | 0,7219 | 0,7837 | 0,6973 |
| freeze-10 | 0,7613 | 0,8892 | 0,8802 | 0,0000 | 0,6806 | 0,8557 | 0,8179 | 0,7653 |
| freeze-15 | 0,7947 | 0,9340 | 0,9285 | 0,1167 | 0,7813 | 0,8574 | 0,8316 | 0,8537 |
| freeze-20 | 0,8236 | 0,9482 | 0,9402 | 0,1417 | 0,7957 | 0,8811 | 0,8722 | 0,8673 |
| freeze-25 | 0,8117 | 0,9158 | 0,9094 | 0,1167 | 0,7986 | 0,8405 | 0,8815 | 0,8605 |
| freeze-30 | 0,8192 | 0,9408 | 0,9332 | 0,1000 | 0,8238 | 0,8741 | 0,8981 | 0,8571 |

| Clase | Soporte | control | freeze-5 | freeze-10 | freeze-15 | freeze-20 | freeze-25 | freeze-30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cipro 500 | 26 | 0,662 | 0,682 | 0,718 | 0,747 | 0,779 | 0,782 | 0,790 |
| Ibuphil 600 mg | 1 | 0,933 | 0,450 | 0,883 | 0,867 | 0,933 | 0,783 | 0,933 |
| Ibuphil Cold 400-60 | 10 | 0,774 | 0,753 | 0,745 | 0,738 | 0,786 | 0,762 | 0,752 |
| Xyzall 5mg | 10 | 0,373 | 0,361 | 0,562 | 0,682 | 0,720 | 0,655 | 0,706 |
| blue | 11 | 0,849 | 0,853 | 0,879 | 0,874 | 0,888 | 0,906 | 0,910 |
| pink | 15 | 0,931 | 0,899 | 0,922 | 0,878 | 0,938 | 0,930 | 0,927 |
| red | 15 | 0,664 | 0,727 | 0,786 | 0,827 | 0,829 | 0,834 | 0,835 |
| white | 10 | 0,773 | 0,757 | 0,717 | 0,816 | 0,825 | 0,814 | 0,814 |

`freeze-20` mejora en promedio las siete clases restantes respecto al control. Los aumentos más grandes son Xyzall (+0,346), red (+0,164) y Cipro (+0,117). Su ventaja se mantiene al excluir Ibuphil 600 mg, cuyo AP medio es igual al del control. Para `freeze-5`, excluir esa clase deja casi empatado el promedio con el control (+0,0007), frente a −0,0598 en el mAP oficial. Ibuphil cuenta con una sola instancia en `train` y una en `valid`, pero aporta un octavo del mAP. El AP por tamaño mejora en medianos y grandes para `freeze-20` y `freeze-30`; dos objetos pequeños ofrecen poco soporte para interpretar diferencias en ese rango.

`freeze-20` promedia 85 TP, 12,67 FP y 13 FN frente a 70,33, 15,33 y 27,67 del control. `freeze-30` reduce los FP a 9,67 y promedia 84 TP y 14 FN. En la semilla 2028, los tres brazos de congelación larga aciertan en la escena 57, que el control omite. La escena 61 sigue siendo un fallo para los cuatro, y en la 80 se conservan omisiones en los cuatro. [Ejemplos de C](results/phase-c/eval-c-s2028/)

### Trayectorias y fase del mejor checkpoint

| Brazo | 2026 | 2027 | 2028 |
| --- | --- | --- | --- |
| control | 23, ajuste completo | 30, ajuste completo | 23, ajuste completo |
| freeze-5 | 29, descongelado | 21, descongelado | 27, descongelado |
| freeze-10 | 27, descongelado | 21, descongelado | 26, descongelado |
| freeze-15 | 15, congelado | 15, congelado | 15, congelado |
| freeze-20 | 30, descongelado | 29, descongelado | 29, descongelado |
| freeze-25 | 24, congelado | 29, descongelado | 30, descongelado |
| freeze-30 | 24, congelado | 26, congelado | 30, congelado |

En las 15 ejecuciones con transición el mAP del historial cae en la primera época descongelada, entre 0,018289 y 0,804477 absolutos. Los máximos posteriores superan el máximo congelado en las tres semillas de `freeze-5`, `freeze-10` y `freeze-20`, en dos de `freeze-25` según el historial y en ninguna de `freeze-15`. En `freeze-20` la brecha media entre su máximo y la última época es 0,004417. `freeze-30` evita la transición y aun así termina por debajo de su máximo en dos semillas, con brecha media de 0,033862. La caída inmediata no determina por sí sola el rendimiento final. [Historial 2026](results/analysis/phase-c/history-s2026.png) · [Historial 2027](results/analysis/phase-c/history-s2027.png) · [Historial 2028](results/analysis/phase-c/history-s2028.png)

En cada semilla, las seis variantes comparten valores exactamente iguales en las 175 comparaciones de pareja y época de sus prefijos congelados, excluyendo la columna de tiempo. Coinciden pérdidas, AP, mAP, métricas nativas y LR en los CSV. Las auditorías conservan la misma huella de los parámetros y buffers del backbone durante ese tramo. Esto acredita la identidad de los valores registrados, sin extenderla a todos los tensores de la red. En las tres ejecuciones de `freeze-30`, el backbone conserva la huella inicial y cero pasos AdamW durante las 30 épocas, con Batch Normalization y DFL congeladas y actualizaciones en neck y ambas ramas. [Prefijos congelados](results/analysis/phase-c/prefijo-congelado.png) · [Auditoría de freeze-30, semilla 2028](results/phase-c/s2028/freeze-30/train/staged_transfer.json)

En las primeras cinco épocas, el mAP medio de historial es 0,0892 en el control y 0,2689 en todas las variantes congeladas. Los controles registran 2, 1 y 0 épocas con mAP cero, frente a ninguna en las variantes. Los picos de pérdida de clasificación en validación pasan de 50 317 300, 1 085 070 y 7 662,46 a 835,126, 642,332 y 111,716 para las respectivas semillas. Esta estabilidad inicial también aparece en `freeze-5`, cuyo mAP final reevaluado es inferior al del control en promedio.

### Auditoría de la discrepancia en freeze-25, semilla 2028

El historial muestra 0,838575 redondeado y `status.json` registra **0,8385745663302676** como el mAP COCO de la validación durante el entrenamiento en la época 30. La columna nativa `metrics/mAP50-95(B)` de Ultralytics para esa época es 0,84775. El lanzamiento seleccionó `weights/best.pt` y su SHA-256 coincide con la métrica publicada. Recalcular las predicciones exportadas reproduce exactamente **0,7823245663302676** (Δ −0,05625) y AP50 de 0,8895219863. En las otras 20 ejecuciones, la mayor diferencia absoluta entre el historial y la reevaluación es inferior a 0,001. [Historial](results/phase-c/s2028/freeze-25/train/history.csv) · [Estado](results/phase-c/s2028/freeze-25/train/status.json) · [Lanzamiento](results/phase-c/eval-c-s2028/launch.json) · [Predicciones](results/phase-c/eval-c-s2028/freeze-25/predictions.json)

En la exportación, el falso positivo de Ibuphil en la imagen 86 tiene score 0,00573907 y precede a la caja que acierta el único Ibuphil real en la imagen 61 (score 0,00546521, IoU 0,943855). Esa pareja deja AP de clase 0,45. En un recálculo diagnóstico **solo en memoria**, retirar la predicción de clase 2 e imagen 86 de mayor score lleva su AP a 0,90, reproduce exactamente el mAP del estado y da AP50 de 0,9520219863, que redondea al 0,952022 del historial. La prueba demuestra que esa caja basta para explicar numéricamente la brecha. Ambas predicciones puntúan por encima del mínimo común de 0,001 para calcular AP y por debajo del umbral operativo de 0,25. Así, la imagen 86 sigue marcada como acierto en el ejemplo operativo. Aunque se sustituyera hipotéticamente ese mAP conjunto por el del estado, la media de `freeze-25` sería 0,826915, inferior a las de `freeze-20` y `freeze-30`. [Escena 86 al punto operativo](results/phase-c/eval-c-s2028/freeze-25/examples/correct-86.jpg)

Las dos vías usan el COCO de las mismas 90 imágenes, todas de 640 × 640, y el mismo cálculo de AP. El código de Ultralytics valida la EMA en FP32 con AMP desactivado, guarda la EMA en FP16 y el evaluador común recarga ese archivo en FP32. La validación durante el entrenamiento usa lotes de 16 y el predictor común procesa una imagen por vez. Son diferencias comprobadas de ejecución y posibles fuentes de variación en las cajas o scores. Las evidencias locales contienen agregados del historial y predicciones de la reevaluación, pero carecen de las predicciones de la época 30 y de la EMA previa a su conversión. Por ello no identifican cuál de esas diferencias generó el falso positivo ni si la discrepancia real incluyó otros cambios compensados. La decisión de C usa la reevaluación común sin corregirla. [Adaptador YOLO](../../scripts/baseline_yolo.py) · [Ultralytics 8.3.161, validación](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/validator.py) · [Guardado del checkpoint](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/trainer.py)

### Coste y contexto con Cascade R-CNN

| Brazo | Entrenamiento medio ± DE (s) | Pico asignado GPU (GiB) | Latencia media (ms/imagen) | FPS medios |
| --- | ---: | ---: | ---: | ---: |
| control | 303,5 ± 4,2 | 4,759 | 10,73 | 93,21 |
| freeze-5 | 289,5 ± 4,5 | 4,761 | 10,73 | 93,22 |
| freeze-10 | 273,1 ± 5,0 | 4,761 | 10,80 | 92,64 |
| freeze-15 | 255,7 ± 4,3 | 4,761 | 10,82 | 92,43 |
| freeze-20 | 236,4 ± 0,7 | 4,761 | 10,90 | 91,72 |
| freeze-25 | 222,3 ± 1,5 | 4,761 | 11,00 | 90,97 |
| freeze-30 | 206,1 ± 2,7 | 2,647 | 10,92 | 91,57 |

Los tiempos internos de entrenamiento incluyen la auditoría. `freeze-30` usa un 32 % menos de tiempo y un 44 % menos de memoria pico asignada que el control. `freeze-20` reduce el tiempo un 22 % y alcanza un pico parecido al control. Las latencias de YOLO están próximas entre brazos y el modelo de inferencia conserva la arquitectura.

Como referencia, Cascade R-CNN histórico obtuvo 0,802064 de mAP, recall 0,949 y latencia de 29,39 ms (34,02 FPS) bajo el protocolo común. Los promedios de `freeze-20` y `freeze-30` superan ese mAP histórico y tienen menor latencia, mientras que Cascade retiene mejor recall al punto operativo. Es una comparación contextual con una ejecución histórica de otro detector y otra asignación, sobre el mismo `valid`. No permite establecer superioridad general. [Baseline Cascade](../baseline/findings.md#comparación-global)

## Decisión y alcance

B confirma `freeze-15` según el protocolo original. C selecciona `freeze-20` como receta candidata y sitúa `freeze-30` como alternativa de menor coste y mayor precisión operativa. Las tres semillas comparten las mismas imágenes de validación y C las emplea para seleccionar la duración. Los resultados son descriptivos, sin prueba de significación ni evaluación en `test`. [Protocolo](README.md#fases-y-criterio-de-decisión)
