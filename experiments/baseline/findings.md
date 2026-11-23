# Hallazgos de los baselines en validación

## Resultados en validación

### Comparación global

En las 90 imágenes y 98 instancias de `valid`, Cascade R-CNN obtiene mayor AP y recall. YOLOv10-S procesa 2,7 veces más imágenes por segundo.

| Métrica de la evaluación independiente en `valid` | YOLOv10-S | Cascade R-CNN |
| --- | ---: | ---: |
| mAP@0,50:0,95 | 0,70209365 | 0,80206393 |
| AP50 / AP75 | 0,83319268 / 0,82607634 | 0,88029961 / 0,88029961 |
| Precisión / recall micro | 81,9 % / 69,4 % | 90,3 % / 94,9 % |
| TP / FP / FN | 68 / 15 / 30 | 93 / 10 / 5 |
| Latencia media / p95 | 10,83 / 10,86 ms | 29,39 / 29,76 ms |
| FPS | 92,32 | 34,02 |

La evaluación independiente del checkpoint `best` corresponde al trabajo `11945016` con [`slurm/evaluate.sh`](../../slurm/evaluate.sh), estado `COMPLETED` y código `0:0`. Las tres métricas de AP coincidieron con el trabajo `11944826` a ocho decimales, y también coincidieron los SHA-256 de cada checkpoint, del manifiesto de datos y del COCO de validación. Precisión, recall y recuentos corresponden a score ≥ 0,25 e IoU ≥ 0,50. La latencia se midió en la misma A100 siguiendo el protocolo común. [Métricas YOLO](results/yolo/evaluation/metrics.json) · [Métricas Cascade](results/cascade/evaluation/metrics.json) · [Protocolo](README.md#g-métricas-y-protocolo-de-inferencia)

### Comparación por clase

| Clase | Instancias en `valid` | AP YOLO / Cascade | TP YOLO / Cascade |
| --- | ---: | ---: | ---: |
| Cipro 500 | 26 | 0,590 / 0,888 | 14 / 23 |
| Ibuphil 600 mg | 1 | 0,800 / 0,180 | 0 / 0 |
| Ibuphil Cold 400-60 | 10 | 0,733 / 0,911 | 8 / 10 |
| Xyzall 5mg | 10 | 0,397 / 0,816 | 5 / 10 |
| blue | 11 | 0,807 / 0,945 | 8 / 11 |
| pink | 15 | 0,933 / 0,931 | 15 / 15 |
| red | 15 | 0,618 / 0,903 | 10 / 15 |
| white | 10 | 0,739 / 0,843 | 8 / 9 |

A IoU 0,50, el recall máximo de las detecciones exportadas de YOLO es aproximadamente 77 % en `Cipro 500` y 70 % en `Xyzall 5mg`, frente a 96 % y 100 % en Cascade. Las curvas PR usan la cuadrícula de recall de COCO. En `Ibuphil 600 mg`, el AP de YOLO depende de una única caja correcta con score ≈0,015. Como análisis de sensibilidad, el AP medio de las otras siete clases es 0,688 en YOLO y 0,891 en Cascade. [PR de YOLO](results/yolo/evaluation/pr_curves.json) · [PR de Cascade](results/cascade/evaluation/pr_curves.json)

### Clasificación, confianza y localización

El diagnóstico cruza las predicciones exportadas con las anotaciones COCO de `valid`. Para los recuentos a umbral fijo, el emparejamiento es por clase y en orden decreciente de score. Examinar también otras clases y scores permite identificar errores que pueden coexistir sobre la misma instancia. [Predicciones YOLO](results/yolo/evaluation/predictions.json) · [Predicciones Cascade](results/cascade/evaluation/predictions.json)

- **Clasificación.** En seis FN de `Cipro 500`, YOLO predice `Xyzall 5mg` con score ≥ 0,25 e IoU ≥ 0,50. En la imagen 57 asigna la clase equivocada con IoU 0,96 y Cascade acierta con IoU 0,97. Ambos predicen `Ibuphil Cold 400-60` con score ≥ 0,25 sobre el único `Ibuphil 600 mg`. [Imagen 57, YOLO](results/yolo/evaluation/examples/failure-57.jpg) · [Imagen 61, Cascade](results/cascade/evaluation/examples/failure-61.jpg)
- **Confianza y fondo.** En 20 de los 30 FN de YOLO hay una caja exportada de la clase correcta con IoU ≥ 0,50 y score < 0,25. Reducir el corte a 0,10 recupera seis TP y añade ocho FP. En la imagen 86, ambos aciertan `Xyzall 5mg` y predicen `white` sobre lunares del fondo, con tres FP en YOLO y cinco en Cascade. [Imagen 86, YOLO](results/yolo/evaluation/examples/failure-86.jpg) · [Imagen 86, Cascade](results/cascade/evaluation/examples/failure-86.jpg)
- **Localización y escala.** En los 67 objetos acertados por ambos, las medianas de IoU son 0,934 y 0,957 para YOLO y Cascade. En las ocho cajas con área inferior al 1 % de la imagen, YOLO omite seis y Cascade dos. El rango pequeño de COCO, con otro corte, contiene solo dos cajas. En la escena de tres pastillas de la imagen 19, YOLO omite las tres y Cascade detecta dos. [Acierto común, imagen 1](results/yolo/evaluation/examples/correct-1.jpg) · [Imagen 19, YOLO](results/yolo/evaluation/examples/failure-19.jpg) · [Imagen 19, Cascade](results/cascade/evaluation/examples/failure-19.jpg)

### Evolución del entrenamiento

| Detector | Trabajo (30 épocas) | Mejor mAP en el historial | Época (iteración) | Tiempo SLURM | Checkpoints `best` y `last` |
| --- | --- | ---: | --- | --- | --- |
| YOLOv10-S | `11930429` | 0,7030092778429936 | 23 | 5 min 30 s | `weights/best.pt`, `weights/last.pt` |
| Cascade R-CNN | `11931348` | 0,8020639283459134 | 25 (1975) | 12 min 7 s | `best/model_best.pth`, `model_final.pth` |

- **YOLO.** El AP de validación es cero en las épocas 2 y 3. La pérdida de clasificación de validación alcanza 50 317 300 en la época 2 y después se recupera. Entre la mejor época (23) y la 30, el mAP baja de 0,703 a 0,653, mientras `train/cls_loss` pasa de 0,324 a 0,198 y `val/cls_loss` de 1,857 a 2,020. El mejor mAP del historial y el de la evaluación independiente del `best` corresponden a mediciones distintas. [Historial YOLO](results/yolo/training/history.csv) · [Curvas YOLO](results/yolo/training/training.png)
- **Cascade.** El mAP termina en 0,798. La curva se aproxima a una meseta. La pérdida disponible es la de entrenamiento. [Historial Cascade](results/cascade/training/history.csv) · [Curvas Cascade](results/cascade/training/training.png)

### Verificaciones registradas

La inicialización se completó en `11930809` (Cascade) y `11930232` (YOLO). Cascade verificó pausa y reanudación en `11931119` y `11931120`. YOLO completó dos épocas en una sesión (`11930428`). Queda pendiente su prueba de pausa y reanudación. [Protocolo de verificación](README.md#protocolo-de-verificación)

### Limitaciones del dataset

- Solo 23 imágenes tienen varias cajas (19 en `train`, 3 en `valid` y 1 en `test`). El bajo soporte de `Ibuphil 600 mg` limita su interpretación en validación y su AP en `test` será indefinido. El EDA atribuye siete clases al guion original, pendiente de consultar, mientras que ocho categorías del COCO tienen anotaciones. [Recuentos y mapeo](README.md#h-dataset-mapeo-e-integridad) · [EDA](../../notebooks/01_eda.ipynb)
- La mediana del área relativa de las cajas se sitúa entre 3 % y 3,5 % por partición. Hay 41 cajas con área inferior al 1 % de la imagen, dos de ellas en `test`. [EDA](../../notebooks/01_eda.ipynb)
- En la imagen 80, una anotación `Cipro 500` mide 1 px². Otras dos cajas, `Cipro 500` e `Ibuphil Cold 400-60`, se solapan con IoU 0,957. Son incidencias para revisar la etiqueta, la caja y su efecto sobre los FN. Se conservan en las métricas oficiales del baseline. [Imagen 80](results/cascade/evaluation/examples/failure-80.jpg)
- Según los nombres de archivo, las clases de color y las de medicamento quedan separadas por fecha en `train` y `valid`. El EDA detectó cero duplicados exactos por bytes o píxeles y seis pares candidatos a similitud perceptual por dHash, tres entre particiones. Los dos pares `train`–`valid` muestran fondos distintos en la inspección visual. [EDA](../../notebooks/01_eda.ipynb)
- Las medianas de luminancia media fueron 136,32, 128,27 y 128,80 en `train`, `valid` y `test`, respectivamente. Entre las 22 imágenes más oscuras de `valid`, YOLO omite 9 de 28 instancias. Entre las 22 más claras, omite 11 de 24. [EDA](../../notebooks/01_eda.ipynb)

## Interpretaciones e hipótesis priorizadas

La divergencia tardía entre entrenamiento y validación de YOLO es compatible con sobreajuste. La inestabilidad inicial puede estar relacionada con la dinámica de optimización. Una ejecución por modelo limita la atribución al LR y la estimación de variabilidad. La separación por fechas permite plantear un efecto de sesión o fondo, que queda por aislar del tipo de clase.

1. **LR de YOLO.** La inestabilidad temprana motiva la comparación controlada definida en [EXP-001](../exp-001-yolo-lr/README.md).
2. **Geometría moderada.** Rotación pequeña, traslación y escala isotrópica podrían mejorar la cobertura. Vigilar el recorte de instancias, la caja tras rotación y el detalle de grabados y ranuras tras interpolar. Comparar AP75 y recall por tamaño.
3. **Fotometría.** Probar por separado una variación leve de luminancia. Mantener tono y saturación y revisar por clase las confusiones de color. La diferencia global de luminancia justifica una prueba, con evidencia débil de concentración de errores en escenas oscuras.

### Factores para futuras comparaciones

Esta priorización orienta comparaciones controladas en `valid`. La ficha de EXP-001 concreta la primera prueba. Las prioridades de los demás factores dependen de las señales indicadas.

| Factor y prioridad inicial | Qué controla | Relevancia para pastillas y diferencia entre detectores | Señal que justificaría estudiarlo |
| --- | --- | --- | --- |
| Alta · LR, warm-up y scheduler | Magnitud de las actualizaciones y su evolución | Con 316 imágenes, un LR alto puede degradar las características transferidas y uno bajo puede limitar la adaptación. Cascade ajusta conjuntamente RPN y ROI con SGD. En YOLO, AdamW y el mínimo de warm-up condicionan las primeras actualizaciones. | Pérdidas oscilantes, AP inicial inestable o curvas estancadas con LR bajo. |
| Alta · Épocas | Cuántas veces se recorre el conjunto y presupuesto de actualizaciones | El dataset pequeño permite repetir pronto las mismas escenas. En Cascade, las iteraciones lógicas dependen del muestreador infinito. En YOLO, EMA puede suavizar las oscilaciones de validación. | Pérdidas de train que bajan con AP de validación estancado o descendente, o AP aún ascendente al final. |
| Alta · Aumentos | Variabilidad geométrica y fotométrica vista durante el ajuste | Flips o brillo podrían mejorar cobertura y conservar el significado de clase. Los cambios de tono pueden confundir clases cromáticas. La cascada reajusta propuestas y regiones y YOLO asigna positivos a dos ramas, lo que puede generar respuestas diferentes a cambios de densidad y escala. | Errores concentrados por iluminación, pose, fondo o escenas con varias pastillas en validación. |
| Media · Congelación y descongelación | Qué capas reciben actualizaciones y cuándo | Ajustar menos capas puede preservar rasgos preentrenados, a costa de limitar la adaptación del dominio. En Cascade conviene separar backbone y FPN de RPN/ROI. En YOLO conviene distinguir backbone, cuello y ramas, dejando DFL con su política interna. | Buena adaptación inicial de cabezas seguida de deterioro en validación, o evidencia de que el escenario falla solo en algunas clases de poco soporte. |
| Media · Resolución | Píxeles que representan un objeto en cada nivel de características | Una entrada mayor puede beneficiar a cajas pequeñas, con más coste y un original ya de 640. En Cascade puede cambiar propuestas útiles y niveles FPN. En YOLO cambia el número de posiciones disponibles para cada stride. | AP/recall bajo para cajas pequeñas y localización sistemáticamente pobre tras revisar aciertos y fallos. |
| Media · Lote | Ruido del gradiente, memoria y frecuencia de actualizaciones | Pocos ejemplos por paso amplifican variabilidad entre imágenes. FrozenBN reduce la dependencia directa de Cascade del lote en sus estadísticas. La BN de YOLO sigue aprendiendo estadísticas y el lote altera el último minibatch. | Picos de pérdida persistentes, AP poco estable entre ejecuciones, o saturación de memoria que limita la resolución. |
| Condicional · Weight decay | Fuerza de la regularización de los pesos | Puede ayudar si las cabezas memorizan ejemplos raros. Con SGD opera en el paso del gradiente y con AdamW se desacopla de la actualización adaptativa. | Brecha persistente de generalización con ajuste estable y LR razonable. |
| Condicional · Optimizador | Regla de actualización, momentum y preacondicionamiento | SGD y AdamW pueden recorrer trayectorias distintas con pocos pasos, especialmente en cabezas recién inicializadas. Comparar cambios exige controlar LR y presupuesto razonables para cada caso. | Convergencia insuficiente pese a un LR y scheduler defendibles o variabilidad repetida con la receta actual. |

## Decisiones

- Mantener ambos baselines como referencia en `valid`, con las ocho clases activas y el evaluador COCO común. Acompañar el mAP oficial con la sensibilidad a la clase de soporte unitario, y el punto operativo ilustrativo de 0,25 con las curvas PR.
- Conservar las anotaciones originales en el baseline. Revisar las incidencias señaladas antes de interpretar los resultados de las cajas pequeñas.
- Priorizar EXP-001 y reservar `test` hasta fijar las recetas. [Orden de trabajo](../README.md#próximos-pasos)
