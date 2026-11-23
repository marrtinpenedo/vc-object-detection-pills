# Baselines de detección de pastillas

## A. Objetivo y uso de los datos

La comparación utiliza el mismo dataset, resolución nominal y evaluador para ambos detectores. Cada implementación conserva su optimización y posprocesado, por lo que los resultados caracterizan las recetas completas.

`train` se usa para entrenar, `valid` para seleccionar el checkpoint y analizar errores, y `test` para la evaluación final una vez cerradas las configuraciones. Se conservan las particiones originales. La clave `val` de Ultralytics apunta a `valid`.

## B. Modelos e inicialización

| Detector | Inicialización y adaptación |
| --- | --- |
| Cascade R-CNN, ResNet-50 + FPN | Detectron2 usa `Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml` y reemplaza sus pesos de ImageNet por la copia local del checkpoint COCO `model_final_480dd8.pkl`, identificado por `model_zoo.get_checkpoint_url`. Con `MASK_ON=False` se conservan el RPN, la regresión agnóstica a clase y las tres etapas de cajas. Se descartan las claves de máscaras y se reinicializan los pesos y sesgos de los tres clasificadores `cls_score`, que pasan de 81 a 9 salidas (ocho clases y fondo). |
| YOLOv10-S | Ultralytics parte de `yolov10s.pt`, con cabeza `v10Detect` end-to-end de 80 clases. El YAML del checkpoint reconstruye ambas ramas para ocho clases. Se comprueba la transferencia de tensores compatibles y se registran las reinicializaciones en `cv3` y `one2one_cv3`. |

La carga rechaza incompatibilidades ajenas a las indicadas y registra el SHA-256 de los pesos utilizados. [Configuración de Cascade](https://github.com/facebookresearch/detectron2/blob/main/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml) · [Cabezas de Ultralytics 8.3.161](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/nn/modules/head.py)

## C. Fine-tuning completo

Se ajustan todos los parámetros entrenables para adaptar las representaciones de COCO a la escala, textura y color de las pastillas. Se inspecciona la evolución del AP en validación por el riesgo de sobreajuste con 316 imágenes de entrenamiento.

En Cascade se fija `BACKBONE.FREEZE_AT=0`. Los buffers de `FrozenBatchNorm2d` permanecen fijos por la arquitectura heredada. En YOLO se fija `freeze=0`. Ultralytics conserva fijas las proyecciones DFL y la cabeza one-to-one recibe características desacopladas del gradiente. La rama one-to-many retropropaga al backbone. `parameters.json` informa de parámetros entrenables y congelados y de buffers FrozenBN. [Entrenador de Ultralytics](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/trainer.py) · [Cabeza de YOLOv10](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/nn/modules/head.py)

## D–E. Preprocesamiento, resolución y aumentos

La resolución nominal de 640 × 640 conserva la escala original de los JPEG y las cajas. La mediana del área relativa por partición se sitúa entre 3 % y 3,5 %. Hay 41 cajas con área relativa inferior al 1 %, dos de ellas en `test`.

- **Cascade.** BGR, `ResizeShortestEdge` con un único tamaño inferior de 640 y lado máximo de 640 para train y validación. A tamaño nativo es identidad. Se conservan la media BGR y la desviación del YAML resuelto, la transformación de cajas correspondiente y el padding interno de la FPN al múltiplo requerido. El tensor se rellena dentro del modelo después de normalizar. `RANDOM_FLIP=none`, crop desactivado, muestreo de resize con un valor fijo y TTA desactivada. El muestreo de imágenes y propuestas sigue siendo parte del entrenamiento.
- **YOLO.** Carga OpenCV con resize proporcional determinista, `LetterBox` cuadrado y centrado con relleno 114 en caso necesario, `Format` con cajas normalizadas, RGB CHW y división de píxeles por 255 en entrenamiento y validación. En evaluación el adaptador reproduce el preprocesamiento y revierte escala y padding. Con los JPEG de este dataset la escala es uno y el padding geométrico es cero. El dataset se construye con `augment=False` para excluir `v8_transforms` y Albumentations. Mosaic, mixup, cutmix, copy-paste, HSV, rotación, traslación, escala, cizalla, perspectiva, flips y permuta BGR quedan en cero. También quedan desactivados multi-scale, close-mosaic y TTA. Auto-augment y random erasing pertenecen al pipeline de clasificación.
- Los aumentos aleatorios ampliarían la variabilidad de entrenamiento y podrían mejorar generalización. Desactivarlos en la primera referencia permite medir por separado su contribución en una futura comparación. Los cambios cromáticos merecen especial cuidado porque hay clases definidas por color.

[Mapeador de Detectron2](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/dataset_mapper.py) · [Transformaciones de Detectron2](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/detection_utils.py) · [Datasets de Ultralytics 8.3.161](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/data/dataset.py)

## F. Recetas e hiperparámetros

| Parámetro | Cascade R-CNN | YOLOv10-S |
| --- | --- | --- |
| Optimizador | SGD, momento 0,9 | AdamW, beta1 0,9 y beta2 0,999 del entrenador |
| LR inicial de referencia | 0,0025 | 0,001 |
| Scheduler y warm-up | Coseno en iteraciones, 79 iteraciones lineales desde 0,1× LR. En una ejecución de 1 época se trunca a `MAX_ITER - 1`. | Coseno por época, factor final 0,05. Warm-up configurado de 1 época con mínimo interno de 100 iteraciones, inclusive en el índice 100. Sesgos desde 0,001 y otros grupos desde cero. El warm-up de momentum opera solo sobre grupos con campo `momentum`, sin interpolar los betas de AdamW. |
| Weight decay | 0,0001, y 0 en parámetros de normalización | 0,0005 en el grupo de decay. Sesgos y normalizaciones quedan excluidos por el optimizador nativo. |
| Lote por GPU y efectivo | 4, sin acumulación | 8, `nbs=batch` y acumulación 1. Último lote de 4 con 316 imágenes. |
| Épocas y pasos máximos | 30 × 79 = 2.370 iteraciones | 30 × 40 = 1.200 minibatches |
| Precisión y semilla | FP32, TF32 desactivado, 2026 | FP32, AMP y TF32 desactivados, 2026 |
| Evaluación y guardado | Cada época lógica, usando `ceil(N/batch)` pasos del muestreador infinito | Cada época, con los pesos EMA de Ultralytics |
| Selección y parada | Mayor AP@[.50:.95] común en `valid`. En empate se conserva el primero. Se completa el presupuesto. | Mismo AP en `valid`. El guardado nativo conserva el último empate. Se completa el presupuesto con `patience=0`. |

Cascade desactiva el modo determinista de PyTorch para usar ROIAlign nativo de torchvision 0.20.1 en CUDA con las extensiones disponibles. Los resultados pueden diferir bit a bit entre ejecuciones con la misma semilla. [Selección de ROIAlign en Torchvision 0.20.1](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/ops/roi_align.py#L227-L236)

Cascade usa SGD con una tasa conservadora para un lote pequeño y un backbone preentrenado. YOLO usa AdamW con una tasa explícita. El recorte de la norma del gradiente se aplica por parámetro en Cascade y globalmente en YOLO. Las 30 épocas fijan una exposición nominal comparable pese a las diferencias de lote y número de pasos.

`configs/baseline/*.json` contiene la receta declarada. `--epochs`, `--batch`, `--imgsz`, `--workers`, `--seed` y `--lr` permiten cambios explícitos. Se guardan `detectron2.yaml` y `cascade_runtime.json` o `args.yaml` y `ultralytics_runtime.json` con valores efectivos. Los CSV dejan trazas de LR y pérdidas. En Cascade la pérdida de validación requeriría activar rutas de RPN/ROI, muestreo y comportamiento de entrenamiento. Se usa la curva COCO de validación para leer generalización. En YOLO se conserva la pérdida de validación que calcula el validador nativo con EMA.

[Scheduler de Detectron2](https://github.com/facebookresearch/detectron2/blob/main/detectron2/solver/build.py) · [Entrenador de Ultralytics 8.3.161](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/trainer.py)

### Sensibilidad potencial y prioridad de estudio

Los efectos propuestos son hipótesis para experimentos controlados en `valid`.

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

## G. Métricas y protocolo de inferencia

Los adaptadores convierten las cajas a COCO `[x, y, ancho, alto]` en píxeles de la imagen original, usando la inversa de clases. Se usa `pycocotools.COCOeval` para ambos detectores y para seleccionar los checkpoints. IoU .50:.05:.95, recall 0:.01:1, `maxDets=[1,10,100]`, tope global de 100 detecciones del detector por imagen y score mínimo nominal 0,001. Cascade aplica su NMS por clase con IoU 0,5. YOLOv10 exporta la rama one-to-one, con top-k y filtro de score end-to-end. La comparación mantiene el posprocesado inherente a cada detector.

- Se informa de **mAP@0.5:0.95**, **AP50**, **AP75** y **AP por clase** en escala 0–1, con sus recuentos de anotaciones. Una clase sin anotaciones tiene AP `null`.
- Las **curvas PR** conservan la precisión interpolada por clase y umbral de IoU. En el punto operativo ilustrativo fijado a score 0,25 e IoU 0,50, se reportan precision y recall micro, TP, FP y FN con el emparejamiento COCO por clase. Las visualizaciones acotadas usan ese mismo punto. No se eligen umbrales con test.
- **AP por tamaño** usa los rangos absolutos en px² de COCO con cortes `32²` y `96²`. El EDA usa cocientes caja/imagen con cortes de 1 % y 10 %. Conviene leer ambas medidas con sus respectivas definiciones.
- Las **curvas de entrenamiento** muestran pérdidas y AP de validación. Las pérdidas son interpretables dentro de cada detector y provienen de objetivos, asignaciones y escalas diferentes.
- **Latencia.** Ambos detectores usan el mismo conjunto ordenado de hasta 16 imágenes ya decodificadas, en la misma GPU y asignación, con lote 1, resolución nominal 640, FP32 y un hilo en el host. Se usan 20 calentamientos y 100 iteraciones por detector. El intervalo se sincroniza con CUDA e incluye preprocesado desde BGR en RAM, transferencia, inferencia, posprocesado nativo y retorno del array Nx6 a RAM. La medición comienza tras la decodificación y termina antes de exportar las predicciones y calcular métricas. Se registran tiempos individuales, media, mediana, p95, FPS (1000/media en ms), hardware, versión CUDA, driver y memoria asignada y reservada por PyTorch.

[Implementación de COCOeval](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) · [Posprocesado end-to-end en Ultralytics](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/utils/ops.py)

## H. Dataset, mapeo e integridad

El EDA y los JSON COCO de la revisión fijada confirman 451 imágenes y 496 instancias. Los JSON declaran 0 `pills` junto a los IDs activos 1–8. Dado que 0 carece de instancias, el baseline tiene ocho clases activas, frente a las siete del guion. La inversa de la tabla recupera el ID del evaluador.

| ID original COCO | ID interno | Clase | train | valid | test |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | Cipro 500 | 79 | 26 | 8 |
| 2 | 1 | Ibuphil 600 mg | 1 | 1 | 0 |
| 3 | 2 | Ibuphil Cold 400-60 | 55 | 10 | 7 |
| 4 | 3 | Xyzall 5mg | 57 | 10 | 5 |
| 5 | 4 | blue | 42 | 11 | 6 |
| 6 | 5 | pink | 35 | 15 | 8 |
| 7 | 6 | red | 40 | 15 | 5 |
| 8 | 7 | white | 42 | 10 | 8 |
| Total instancias | | | 351 | 98 | 47 |
| Total imágenes | | | 316 | 90 | 45 |

La preparación reutiliza la auditoría del descargador. Comprueba IDs, referencias, dimensiones, JPEG, cajas, duplicados a precisión YOLO y recuentos, y falla detalladamente ante una discrepancia. El derivado contiene copias íntegras de JPEG, un COCO con las ocho categorías y TXT YOLO normalizados. Las anotaciones del COCO derivado se numeran desde 1 porque COCOeval emplea el 0 como centinela de emparejamiento. Se conservan los IDs de imagen y categoría, el área y las coordenadas originales. `manifest.json` contiene recuentos y huellas SHA-256. Repetir la preparación comprueba la equivalencia del derivado existente.

Ibuphil 600 mg tiene una anotación en `train` y una en `valid`. Su AP en `test` será indefinido. Solo 23 imágenes tienen varias cajas (19/3/1). Las medianas de luminancia media fueron 136,32, 128,27 y 128,80 en `train`, `valid` y `test`, respectivamente. dHash señaló seis pares candidatos a similitud perceptual, tres entre particiones, sin duplicados exactos por bytes o píxeles. La interpretación de los resultados debe considerar los recuentos por subconjunto y la revisión visual de los pares entre particiones. [EDA del repositorio](../../notebooks/01_eda.ipynb)

## I. Protocolo experimental en FinisTerrae III

El protocolo utiliza los entornos, rutas y opciones `SLURM_LOGS` de la [guía de CESGA](../../docs/cesga.md).

### Resultados observados en CESGA

#### Entrenamiento

| Detector | Trabajo (30 épocas) | Mejor mAP@0.5:0.95 en validación durante el entrenamiento | Época (iteración) | Tiempo SLURM | Checkpoints best y last |
| --- | --- | ---: | --- | --- | --- |
| Cascade R-CNN | `11931348` | 0.8020639283459134 | 25 (1975) | 12 min 7 s | `best/model_best.pth`, `model_final.pth` |
| YOLOv10-S | `11930429` | 0.7030092778429936 | 23 | 5 min 30 s | `weights/best.pt`, `weights/last.pt` |

La inicialización se completó en `11930809` (Cascade) y `11930232` (YOLO). El smoke test de Cascade verificó pausa y reanudación en `11931119` y `11931120`. El de YOLO completó dos épocas en una sesión (`11930428`).

#### Evaluación independiente en validación

| Métrica | Cascade R-CNN | YOLOv10-S |
| --- | ---: | ---: |
| mAP@0.5:0.95 | 0.80206393 | 0.70209365 |
| AP50 | 0.88029961 | 0.83319268 |
| AP75 | 0.88029961 | 0.82607634 |
| Latencia media | 29.39 ms | 10.83 ms |
| FPS | 34.02 | 92.32 |

La tabla corresponde a la evaluación del lanzador versionado `slurm/evaluate.sh` en `11945016` (`COMPLETED`, código `0:0`). Las tres métricas de AP coincidieron con la evaluación `11944826` a ocho decimales. También coincidieron los SHA-256 del checkpoint de cada detector, del manifiesto del dataset y del COCO de validación.

Quedan pendientes el smoke test de pausa y reanudación de YOLO y la evaluación final en `test`.

### Protocolo de verificación

**Fase A. Entorno y datos.** En un nodo CPU, con los módulos cargados, complete la [preparación del dataset](../../docs/cesga.md#datos-y-pesos) y repítala sobre la misma salida para verificar la identidad del derivado. Compruebe los recuentos del manifiesto con la tabla de particiones. Ejecute las verificaciones de sintaxis, importación, configuración y pruebas sintéticas desde la raíz del proyecto.

```bash
"$CASCADE_ENV/bin/python" -m compileall -q scripts tests
bash -n slurm/cascade.sh && bash -n slurm/yolo.sh && bash -n slurm/evaluate.sh
"$CASCADE_ENV/bin/python" -c 'import torch, torchvision, detectron2._C, pycocotools.coco; import scripts.baseline_cascade'
"$YOLO_ENV/bin/python" -c 'import torch, torchvision, ultralytics, pycocotools.coco; import scripts.baseline_yolo'
"$CASCADE_ENV/bin/python" -m scripts.train_baseline cascade --check-config
"$YOLO_ENV/bin/python" -m scripts.train_baseline yolo --check-config
"$CASCADE_ENV/bin/python" -m unittest discover -s tests -v
```

**Fase B. Inicialización.** Los lanzadores ejecutan inferencia y retropropagación en un lote real, sin paso de optimizador.

```bash
export VERIFY_ROOT="$(mktemp -d "$BASELINE_RESULTS/verify-XXXXXX")"
printf '%s\n' "$VERIFY_ROOT"
sbatch "${SLURM_LOGS[@]}" --time=01:00:00 slurm/cascade.sh --data "$BASELINE_DATA" --weights "$CASCADE_WEIGHTS" \
  --output "$VERIFY_ROOT/init-cascade" --initialize
sbatch "${SLURM_LOGS[@]}" --time=01:00:00 slurm/yolo.sh --data "$BASELINE_DATA" --weights "$YOLO_WEIGHTS" \
  --output "$VERIFY_ROOT/init-yolo" --initialize
```

Confirme `COMPLETED` y `0:0` en SLURM. En `initialization.json`, compruebe ocho clases, inferencia completada y pérdidas finitas. Revise la transferencia de pesos en `weight_transfer.json` y los gradientes en el backbone y las cabezas de ambos detectores, así como en el RPN de Cascade. `parameters.json` registra las proyecciones DFL y los buffers FrozenBN.

**Fase C. Smoke test y reanudación.** Ejecute dos épocas con una pausa tras el primer checkpoint.

```bash
sbatch "${SLURM_LOGS[@]}" --time=02:00:00 slurm/cascade.sh --data "$BASELINE_DATA" --weights "$CASCADE_WEIGHTS" \
  --output "$VERIFY_ROOT/smoke-cascade" --epochs 2 --pause-after 1
sbatch "${SLURM_LOGS[@]}" --time=02:00:00 slurm/yolo.sh --data "$BASELINE_DATA" --weights "$YOLO_WEIGHTS" \
  --output "$VERIFY_ROOT/smoke-yolo" --epochs 2 --pause-after 1
```

Una vez finalizados ambos trabajos, compruebe que `status.json` indique `paused` y conserve `VERIFY_ROOT` para reanudar.

```bash
sbatch "${SLURM_LOGS[@]}" --time=02:00:00 slurm/cascade.sh --data "$BASELINE_DATA" --weights "$CASCADE_WEIGHTS" \
  --output "$VERIFY_ROOT/smoke-cascade" --epochs 2 --resume
sbatch "${SLURM_LOGS[@]}" --time=02:00:00 slurm/yolo.sh --data "$BASELINE_DATA" --weights "$YOLO_WEIGHTS" \
  --output "$VERIFY_ROOT/smoke-yolo" --epochs 2 --resume
```

Compruebe `complete`, dos validaciones con pasos crecientes, pérdidas finitas y cambios de los parámetros de control en `parameter_updates.json`. El warm-up de YOLO abarca las dos épocas del smoke test. La reanudación parte del último checkpoint completo y no garantiza equivalencia bit a bit. [Consulta de trabajos y recuperación de checkpoints](../../docs/cesga.md#ejecución-de-trabajos-slurm)

**Fase D. Métricas y artefactos.** Para ejecuciones de dos épocas, el verificador exige el registro de pausa y reanudación de la fase C. Con ambos entrenamientos en estado `complete`, defina las rutas y use el [envío de evaluación conjunta](../../docs/cesga.md#evaluación-conjunta).

```bash
CASCADE_RUN="$VERIFY_ROOT/smoke-cascade"
YOLO_RUN="$VERIFY_ROOT/smoke-yolo"
EVAL_ROOT="$VERIFY_ROOT/eval-valid"
```

El verificador comprueba el dataset, los checkpoints, el recálculo del AP y las curvas PR, y la equivalencia de las condiciones del benchmark. Inspeccione `training.png`, `pr_curves.png` y los ejemplos para comprobar la escala de las cajas y la visualización de TP, FP y FN.

### Entrenamiento de 30 épocas y evaluación

Una vez completadas las verificaciones, envíe la receta base en una tanda nueva. Conserve el valor de `RUN_ROOT` al cambiar de sesión.

```bash
export RUN_ROOT="$(mktemp -d "$BASELINE_RESULTS/baseline-XXXXXX")"
printf '%s\n' "$RUN_ROOT"
sbatch "${SLURM_LOGS[@]}" slurm/cascade.sh --data "$BASELINE_DATA" --weights "$CASCADE_WEIGHTS" \
  --output "$RUN_ROOT/cascade"
sbatch "${SLURM_LOGS[@]}" slurm/yolo.sh --data "$BASELINE_DATA" --weights "$YOLO_WEIGHTS" \
  --output "$RUN_ROOT/yolo"
```

Para reanudar, Cascade usa `last_checkpoint` y recupera el optimizador, el scheduler, la iteración y el estado del mejor AP. YOLO usa `weights/last.pt` con EMA, época y estado del optimizador. Los estados RNG y del muestreador pueden diferir tras reanudar. `status.json` acumula la duración de las sesiones registradas. Para una terminación abrupta, consulte el tiempo en `sacct`.

Con ambos entrenamientos completos, defina las rutas y use el mismo [envío de evaluación conjunta](../../docs/cesga.md#evaluación-conjunta).

```bash
CASCADE_RUN="$RUN_ROOT/cascade"
YOLO_RUN="$RUN_ROOT/yolo"
EVAL_ROOT="$RUN_ROOT/eval-valid"
```

Una vez cerrados los experimentos, la evaluación sobre `test` usa los checkpoints ya seleccionados en validación. Ejecute estos comandos en la misma [sesión GPU interactiva](../../docs/cesga.md#sesión-gpu-interactiva), con los módulos cargados y desde la raíz del proyecto.

```bash
"$CASCADE_ENV/bin/python" -m scripts.evaluate_baseline --data "$BASELINE_DATA" \
  --run "$RUN_ROOT/cascade" --split test --checkpoint best \
  --output "$RUN_ROOT/eval-cascade-test" --benchmark --warmup 20 --iterations 100
"$YOLO_ENV/bin/python" -m scripts.evaluate_baseline --data "$BASELINE_DATA" \
  --run "$RUN_ROOT/yolo" --split test --checkpoint best \
  --output "$RUN_ROOT/eval-yolo-test" --benchmark --warmup 20 --iterations 100
```

### Artefactos

- **Entrenamiento.** `run.json` registra la procedencia y la configuración. `status.json` indica el mejor paso, la duración, la memoria y las rutas de los checkpoints. Cascade guarda `metrics.json` en JSON Lines y YOLO, `results.csv`. Ambos generan `history.csv` y `training.png`.
- **Evaluación.** Cada detector genera `metrics.json`, `predictions.json`, `pr_curves.json`, `pr_curves.png` y hasta cinco ejemplos por tipo. El AP de YOLO calculado al recargar los pesos EMA puede diferir del registrado al seleccionar el checkpoint.

[Recuperación de artefactos](../../docs/cesga.md#recuperar-resultados) · [Figura de pérdidas de YOLOv10-S](results/README.md#historial-de-yolov10-s)
