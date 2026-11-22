# Baselines de detección de pastillas

## A. Objetivo y uso de los datos

El baseline es la primera referencia con una receta explícita por detector. Comparte los datos, el tamaño nominal y el evaluador, y admite diferencias de optimización y posprocesado propias de cada implementación. Las diferencias de rendimiento describirán estas dos recetas completas. Los resultados del smoke test sirven exclusivamente para verificar el software.

`train` se usa para aprender, `valid` para elegir el checkpoint y analizar errores, y `test` se reserva hasta cerrar las configuraciones experimentales. La carpeta original de validación se llama `valid` y la clave del YAML de Ultralytics se llama `val`. La preparación mantiene la pertenencia original de cada imagen.

## B. Modelos e inicialización

| Detector | Inicialización y adaptación |
| --- | --- |
| Cascade R-CNN, ResNet-50 + FPN | Detectron2 usa `Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml` para definir la arquitectura. Ese YAML declara pesos de ImageNet. Se comprueba la URL de `model_zoo.get_checkpoint_url` y se carga explícitamente su copia local COCO `model_final_480dd8.pkl`. `MASK_ON=False` omite la rama de máscaras. Se conservan las tres etapas de cajas, el RPN y la regresión agnóstica a clase. Las tres capas `cls_score` pasan de 81 a 9 salidas (ocho clases y fondo). Sus pesos y sesgos son las seis incompatibilidades esperadas. Las claves sobrantes de la rama de máscaras son esperadas. Cualquier otra incompatibilidad de carga es un error. |
| YOLOv10-S | Ultralytics usa el archivo oficial `yolov10s.pt`. Se exige una cabeza `v10Detect` end-to-end de 80 clases en la inicialización. El YAML incluido en el checkpoint reconstruye ambas ramas de predicción con ocho clases. Se comprueba la transferencia de todos los tensores compatibles y se registran los que se reinicializan en las ramas de clasificación `cv3` y `one2one_cv3`. La anchura de sus subredes puede depender del número de clases. Una incompatibilidad fuera de esas ramas es un error. |

Los binarios de pesos se cargan mediante los deserializadores nativos de cada proyecto. Usad archivos de las fuentes oficiales y checkpoints propios. El SHA-256 de cada peso usado queda registrado. [Zoo de Detectron2](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) · [YAML asignado](https://github.com/facebookresearch/detectron2/blob/main/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml) · [YOLOv10 y pesos oficiales](https://docs.ultralytics.com/models/yolov10/) · [Cabezas de Ultralytics 8.3.161](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/nn/modules/head.py)

## C. Fine-tuning completo

Ambos modelos parten de características aprendidas en COCO. Entrenar todas las capas entrenables permite adaptar las representaciones a objetos pequeños, texturas finas y colores que pueden discriminar clases de pastillas. Con una sola receta inicial se puede observar la adaptación del detector completo. Pocas muestras elevan el riesgo de sobreajuste, y su efecto depende de la transferencia previa, la tasa de aprendizaje y el presupuesto. La evolución de AP en validación permite valorar si aparece ese problema.

En Cascade se fija `BACKBONE.FREEZE_AT=0`. Los buffers de `FrozenBatchNorm2d` permanecen fijos por la arquitectura heredada. En YOLO se fija `freeze=0`. Ultralytics conserva fijas las proyecciones DFL y la cabeza one-to-one recibe características desacopladas del gradiente. La rama one-to-many retropropaga al backbone. `parameters.json` informa de parámetros entrenables y congelados y de buffers FrozenBN. [Entrenador de Ultralytics](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/trainer.py) · [Cabeza de YOLOv10](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/nn/modules/head.py)

## D–E. Preprocesamiento, resolución y aumentos

Fijamos 640 × 640 nominal porque los 451 JPEG de esta revisión ya se exportaron a ese tamaño. Así, la escala original del raster y de las cajas se mantiene en el baseline. Las medianas de área relativa por split rondan 3–3,5 %, con 41 cajas por debajo de 1 %. Conservar 640 evita reducirlas de partida. La estimación de rendimiento en cajas pequeñas sigue condicionada por el soporte, en especial por las dos cajas de ese rango relativo en test.

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

SGD se mantiene para la cascada y la tasa es conservadora para un lote pequeño y un backbone preentrenado. AdamW da un paso adaptativo a las distintas capas de YOLO, con LR explícito y sin el modo `auto`. Los dos optimizadores limitan la norma de gradiente con el mecanismo nativo. En Cascade se limita por parámetro y en YOLO globalmente. Los decays se mantienen moderados y el coseno reduce la magnitud de los cambios al final. El mismo número de épocas fija exposiciones nominales comparables con la receta inicial. El número de actualizaciones y sus efectos difieren por tamaño de lote, asignación y objetivos de pérdida.

`configs/baseline/*.json` contiene la receta declarada. `--epochs`, `--batch`, `--imgsz`, `--workers`, `--seed` y `--lr` permiten cambios explícitos. Se guardan `detectron2.yaml` y `cascade_runtime.json` o `args.yaml` y `ultralytics_runtime.json` con valores efectivos. Los CSV dejan trazas de LR y pérdidas. En Cascade la pérdida de validación requeriría activar rutas de RPN/ROI, muestreo y comportamiento de entrenamiento. Se usa la curva COCO de validación para leer generalización. En YOLO se conserva la pérdida de validación que calcula el validador nativo con EMA.

[Scheduler de Detectron2](https://github.com/facebookresearch/detectron2/blob/main/detectron2/solver/build.py) · [Entrenador de Ultralytics 8.3.161](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/trainer.py)

### Sensibilidad potencial y prioridad de estudio

Los mecanismos de la segunda columna son conocidos. Las consecuencias y señales de las dos columnas siguientes son hipótesis y criterios para decidir si merece la pena experimentar. El rendimiento observado del baseline y los efectos de futuros cambios están pendientes de medición.

| Factor y prioridad inicial | Qué controla | Relevancia para pastillas y diferencia entre detectores | Señal que justificaría estudiarlo |
| --- | --- | --- | --- |
| Alta · LR, warm-up y scheduler | Magnitud de las actualizaciones y su evolución | En un train de 316 imágenes, pasos demasiado grandes podrían degradar características transferidas y pasos pequeños podrían dejar las cabezas poco adaptadas. En Cascade se observa SGD por iteración y la adaptación conjunta del RPN y las ROI. AdamW de YOLO adapta las actualizaciones y tiene un mínimo de warm-up importante a esta escala. | Pérdidas oscilantes o no finitas, AP inestable al principio, o curvas estancadas con LR ya bajo. |
| Alta · Épocas | Cuántas veces se recorre el conjunto y presupuesto de actualizaciones | El dataset pequeño permite repetir pronto las mismas escenas. En Cascade, las iteraciones lógicas dependen del muestreador infinito. En YOLO, EMA puede suavizar las oscilaciones de validación. | Pérdidas de train que bajan con AP de validación estancado o descendente, o AP aún ascendente al final. |
| Alta · Aumentos | Variabilidad geométrica y fotométrica vista durante el ajuste | Flips o brillo podrían mejorar cobertura y conservar el significado de clase. Los cambios de tono pueden confundir clases cromáticas. La cascada reajusta propuestas y regiones y YOLO asigna positivos a dos ramas, lo que puede generar respuestas diferentes a cambios de densidad y escala. | Errores concentrados por iluminación, pose, fondo o escenas con varias pastillas en validación. |
| Media · Congelación y descongelación | Qué capas reciben actualizaciones y cuándo | Ajustar menos capas puede preservar rasgos preentrenados, a costa de limitar la adaptación del dominio. En Cascade conviene separar backbone y FPN de RPN/ROI. En YOLO conviene distinguir backbone, cuello y ramas, dejando DFL con su política interna. | Buena adaptación inicial de cabezas seguida de deterioro en validación, o evidencia de que el escenario falla solo en algunas clases de poco soporte. |
| Media · Resolución | Píxeles que representan un objeto en cada nivel de características | Una entrada mayor puede beneficiar a cajas pequeñas, con más coste y un original ya de 640. En Cascade puede cambiar propuestas útiles y niveles FPN. En YOLO cambia el número de posiciones disponibles para cada stride. | AP/recall bajo para cajas pequeñas y localización sistemáticamente pobre tras revisar aciertos y fallos. |
| Media · Lote | Ruido del gradiente, memoria y frecuencia de actualizaciones | Pocos ejemplos por paso amplifican variabilidad entre imágenes. FrozenBN reduce la dependencia directa de Cascade del lote en sus estadísticas. La BN de YOLO sigue aprendiendo estadísticas y el lote altera el último minibatch. | Picos de pérdida persistentes, AP poco estable entre ejecuciones, o saturación de memoria que limita la resolución. |
| Condicional · Weight decay | Fuerza de la regularización de los pesos | Puede ayudar si las cabezas memorizan ejemplos raros. Con SGD opera en el paso del gradiente y con AdamW se desacopla de la actualización adaptativa. | Brecha persistente de generalización con ajuste estable y LR razonable. |
| Condicional · Optimizador | Regla de actualización, momentum y preacondicionamiento | SGD y AdamW pueden recorrer trayectorias distintas con pocos pasos, especialmente en cabezas recién inicializadas. Comparar cambios exige controlar LR y presupuesto razonables para cada caso. | Convergencia insuficiente pese a un LR y scheduler defendibles o variabilidad repetida con la receta actual. |

## G. Métricas y protocolo de inferencia

Los adaptadores convierten las cajas a COCO `[x, y, ancho, alto]` en píxeles de la imagen original, usando la inversa de clases. Se usa `pycocotools.COCOeval` para ambos detectores y para seleccionar los checkpoints. IoU .50:.05:.95, recall 0:.01:1, `maxDets=[1,10,100]`, tope global de 100 detecciones del detector por imagen y score mínimo nominal 0,001. Cascade aplica su NMS por clase con IoU 0,5. YOLOv10 exporta la rama one-to-one, con top-k y filtro de score end-to-end. La comparación mantiene el posprocesado inherente a cada detector.

- **mAP@0.5** mide detección con un criterio de solapamiento moderado. **AP@0.75** exige mejor localización y **mAP@0.5:0.95** resume varios niveles de exigencia. **AP por clase** y sus soportes permiten leer desbalances. Una clase sin ground truths tiene AP `null`. Las estadísticas están en escala 0–1.
- Las **curvas PR** conservan la precisión interpolada por clase y umbral de IoU. En el punto operativo ilustrativo fijado a score 0,25 e IoU 0,50, se reportan precision y recall micro, TP, FP y FN con el emparejamiento COCO por clase. Las visualizaciones acotadas usan ese mismo punto. No se eligen umbrales con test.
- **AP por tamaño** usa los rangos absolutos en px² de COCO con cortes `32²` y `96²`. El EDA usa cocientes caja/imagen con cortes de 1 % y 10 %. Conviene leer ambas medidas con sus respectivas definiciones.
- Las **curvas de entrenamiento** muestran pérdidas y AP de validación. Las pérdidas son interpretables dentro de cada detector y provienen de objetivos, asignaciones y escalas diferentes.
- Para **latencia** se mide el mismo conjunto ordenado de hasta 16 imágenes ya decodificadas, en una sola A100, batch 1, 640 nominal, FP32, un hilo host, 20 warm-ups y 100 iteraciones por defecto. El intervalo, sincronizado con CUDA antes y después, incluye preprocesado desde BGR en RAM, transferencia, forward, posprocesado nativo y retorno del array Nx6 a RAM. La decodificación, exportación COCO, scoring y figuras quedan fuera del intervalo. Se guardan los tiempos, media, mediana, p95, FPS como 1000/media_ms, versión CUDA, driver, GPU y memoria de PyTorch asignada y reservada. Para comparar, ejecutad ambos en el mismo nodo y con la misma asignación y carga. La memoria medida representa el allocator de PyTorch.

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

Con una sola anotación en train y otra en validación, Ibuphil 600 mg tiene soporte muy escaso; en test su AP será indefinido. Solo 23 imágenes tienen varias cajas (19/3/1). La mediana de luminancia media fue 136,32 en train frente a 128,27 y 128,80 en validación y test. Es una comparación descriptiva. dHash señaló seis pares candidatos a similitud perceptual, tres entre splits, con cero grupos idénticos por bytes o píxeles. Revisad visualmente esos tres pares antes de interpretar como definitiva la evaluación. Las métricas por clase, tamaño y subconjunto deben leerse junto a sus recuentos. [EDA del repositorio](../../notebooks/01_eda.ipynb)

## I. Reproducibilidad y ejecución en FinisTerrae III

Los siguientes comandos son para la persona que vaya a verificar el baseline. Usad un filesystem persistente de la cuenta para pesos, entorno y checkpoints. Ejecutad las tareas de cómputo en una asignación SLURM. Antes de preparar entornos, consultad `module spider Python`, `module spider CUDA` y `module spider GCC`. Cargad los módulos disponibles que proporcionen Python 3.11 y `nvcc` 11.8 con un compilador compatible. Si necesitáis esos módulos en batch, cargadlos en el entorno de envío o añadid las líneas `module load` verificadas por vuestra cuenta. Dentro de la asignación comprobad `nvidia-smi`, `nvcc --version` y `module list`. La documentación del centro indica 32 CPU por A100 en el nodo estándar. Los scripts solicitan una A100, 32 CPU, 64 GiB y seis horas. Revisad QOS, límites y acceso al almacenamiento antes de enviar. [Guía FT3](https://cesga-docs.gitlab.io/ft3-user-guide/index.html) · [Nodos GPU](https://cesga-docs.gitlab.io/ft3-user-guide/gpu_nodes.html) · [Comandos de SLURM](https://cesga-docs.gitlab.io/ft3-user-guide/batch_basic_commands.html)

### Entornos y pesos

La combinación propuesta es CPython 3.11, PyTorch 2.5.1 + torchvision 0.20.1 del índice cu118, Detectron2 compilado con CUDA 11.8 al commit indicado y Ultralytics 8.3.161 en un entorno aparte. Se fija `sm_80` para A100. Compilar Detectron2 exige que toolkit y PyTorch sean compatibles. El driver real de vuestra GPU y la disponibilidad de módulos quedan para la fase B. `uv.lock` sigue perteneciendo al EDA. `constraints.txt` fija componentes críticos de los entornos GPU, y cada `run.json` guarda todas las versiones instaladas y los hashes de código para reconstrucción posterior. [PyTorch cu118](https://pytorch.org/get-started/previous-versions/#v251) · [Instalación de Detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) · [Commit fijado](https://github.com/facebookresearch/detectron2/commit/fd27788985af0f4ca800bca563acdb700bb890e2) · [Compatibilidad CUDA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

Desde la raíz del repositorio. Cambiad `BASELINE_WORK` si vuestra cuenta recomienda otra ubicación persistente.

```bash
export BASELINE_WORK="$HOME/pills-baseline"
export CASCADE_ENV="$BASELINE_WORK/envs/cascade"
export YOLO_ENV="$BASELINE_WORK/envs/yolo"
export BASELINE_DATA="$BASELINE_WORK/data/pills-prepared"
export BASELINE_RESULTS="$BASELINE_WORK/results"
export CASCADE_WEIGHTS="$BASELINE_WORK/checkpoints/model_final_480dd8.pkl"
export YOLO_WEIGHTS="$BASELINE_WORK/checkpoints/yolov10s.pt"
export YOLO_CONFIG_DIR="$BASELINE_WORK/ultralytics"
mkdir -p "$BASELINE_WORK/checkpoints" "$BASELINE_RESULTS" "$YOLO_CONFIG_DIR" results/slurm
```

Preparad los entornos en un nodo de cómputo y con los módulos ya elegidos. Para compilar, podéis iniciar una sesión CPU con `srun -c 4 --mem=16G -t 02:00:00 --pty bash`, ajustando los recursos a vuestro proyecto. Si la red de los nodos está restringida, preparad el caché o un wheelhouse en un entorno autorizado y trasladadlo.

```bash
uv venv --python 3.11 "$CASCADE_ENV"
uv venv --python 3.11 "$YOLO_ENV"
for env_path in "$CASCADE_ENV" "$YOLO_ENV"; do
  uv pip install --python "$env_path/bin/python" \
    --index-url https://download.pytorch.org/whl/cu118 torch==2.5.1+cu118 torchvision==0.20.1+cu118
  uv pip install --python "$env_path/bin/python" -c configs/baseline/constraints.txt \
    numpy opencv-python pycocotools Pillow matplotlib setuptools wheel ninja
done
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 \
  uv pip install --python "$CASCADE_ENV/bin/python" --no-build-isolation \
    -c configs/baseline/constraints.txt fvcore \
    'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@fd27788985af0f4ca800bca563acdb700bb890e2'
uv pip install --python "$YOLO_ENV/bin/python" -c configs/baseline/constraints.txt 'ultralytics==8.3.161'
uv pip check --python "$CASCADE_ENV/bin/python"
uv pip check --python "$YOLO_ENV/bin/python"
uv pip freeze --python "$CASCADE_ENV/bin/python" > "$BASELINE_WORK/cascade.freeze.txt"
uv pip freeze --python "$YOLO_ENV/bin/python" > "$BASELINE_WORK/yolo.freeze.txt"
```

Aprovisionad los pesos oficiales antes de enviar trabajos. Se pueden copiar desde una máquina con acceso a Internet. El archivo de Cascade corresponde a la URL que resuelve el zoo. La documentación de YOLOv10 enlaza la familia de pesos distribuida por Ultralytics. La fuente Arial es una necesidad del verificador de datasets de Ultralytics 8.3.161, incluso con gráficos nativos desactivados, y se puede preparar en su caché para trabajar sin acceso al exterior.

```bash
curl -fL https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl \
  -o "$CASCADE_WEIGHTS"
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10s.pt -o "$YOLO_WEIGHTS"
curl -fL https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf -o "$YOLO_CONFIG_DIR/Arial.ttf"
sha256sum "$CASCADE_WEIGHTS" "$YOLO_WEIGHTS"
```

### Protocolo previo al commit

**Fase A. Sintaxis, entornos y datos.** Si faltan los datos originales, usad el flujo existente con `uv sync --locked` y `uv run --locked python scripts/download_dataset.py`, o copiad la revisión fijada al repositorio. En una asignación de CPU ejecutad lo siguiente.

```bash
"$CASCADE_ENV/bin/python" -m compileall -q scripts tests
bash -n slurm/cascade.sh && bash -n slurm/yolo.sh
"$CASCADE_ENV/bin/python" -c 'import torch, torchvision, detectron2._C, pycocotools.coco; import scripts.baseline_cascade'
"$YOLO_ENV/bin/python" -c 'import torch, torchvision, ultralytics, pycocotools.coco; import scripts.baseline_yolo'
"$CASCADE_ENV/bin/python" -m scripts.train_baseline cascade --check-config
"$YOLO_ENV/bin/python" -m scripts.train_baseline yolo --check-config
"$CASCADE_ENV/bin/python" -m unittest discover -s tests -v
"$CASCADE_ENV/bin/python" -m scripts.baseline_data --data data/pills-sxdht --output "$BASELINE_DATA"
"$CASCADE_ENV/bin/python" -m scripts.baseline_data --data data/pills-sxdht --output "$BASELINE_DATA"
```

- [ ] Importan las extensiones en sus respectivos entornos y los tests pasan. Los sintéticos comprueban cajas límite, IDs inválidos, mapeo e inversa, AP perfecto, AP cero, GT ausente y detecciones vacías.
- [ ] Las dos preparaciones informan de 316/90/45 imágenes y 351/98/47 instancias. Inspectad `manifest.json` y el YAML portable. Las imágenes se copian byte a byte y los splits y los originales se conservan. Usad rutas exportadas de vuestra cuenta.

**Fase B. Inicialización en GPU.** Las dos sondas leen un lote real, hacen inferencia y retropropagación, y registran los gradientes sin ejecutar un paso de optimizador.

```bash
sbatch --time=01:00:00 slurm/cascade.sh --data "$BASELINE_DATA" --weights "$CASCADE_WEIGHTS" \
  --output "$BASELINE_RESULTS/init-cascade" --initialize
sbatch --time=01:00:00 slurm/yolo.sh --data "$BASELINE_DATA" --weights "$YOLO_WEIGHTS" \
  --output "$BASELINE_RESULTS/init-yolo" --initialize
squeue -u "$USER"
```

- [ ] SLURM informa `COMPLETED` y `0:0`. `run.json` identifica la GPU, CUDA, configuración, código y hash de los pesos.
- [ ] Ambos `initialization.json` informan ocho clases, pérdidas finitas e inferencia completada. Revisad `weight_transfer.json` y el log. En Cascade solo clasificadores y rama de máscaras. En YOLO solo incompatibilidades de clasificación de las dos ramas.
- [ ] Los gradientes alcanzan backbone, RPN y cabezas de Cascade, y backbone y ambos clasificadores de YOLO. `parameters.json` refleja que DFL y buffers FrozenBN siguen la política de cada arquitectura.

**Fase C. Smoke test y reanudación.** Pedid dos épocas y una pausa controlada después del primer checkpoint. Cada pareja de comandos usa un directorio exclusivo y el mismo presupuesto. Lanzad las reanudaciones al terminar las pausas.

```bash
sbatch --time=02:00:00 slurm/cascade.sh --data "$BASELINE_DATA" --weights "$CASCADE_WEIGHTS" \
  --output "$BASELINE_RESULTS/smoke-cascade" --epochs 2 --pause-after 1
sbatch --time=02:00:00 slurm/yolo.sh --data "$BASELINE_DATA" --weights "$YOLO_WEIGHTS" \
  --output "$BASELINE_RESULTS/smoke-yolo" --epochs 2 --pause-after 1
squeue -u "$USER"

sbatch --time=02:00:00 slurm/cascade.sh --data "$BASELINE_DATA" --weights "$CASCADE_WEIGHTS" \
  --output "$BASELINE_RESULTS/smoke-cascade" --epochs 2 --resume
sbatch --time=02:00:00 slurm/yolo.sh --data "$BASELINE_DATA" --weights "$YOLO_WEIGHTS" \
  --output "$BASELINE_RESULTS/smoke-yolo" --epochs 2 --resume
```

Para cada ID devuelto por `sbatch`, comprobad el estado, salida, tiempo y memoria.

```bash
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES
```

- [ ] Después de la pausa, `status.json` dice `paused` y existen best y last con una validación. Tras reanudar dice `complete` y el historial contiene las dos validaciones con pasos crecientes. Revisad logs, cargas de dataloader, LR, pérdidas finitas y presencia de estados del optimizador. Un fallo real del proceso debe dar código distinto de cero.
- [ ] El modelo se actualiza. `parameter_updates.json` comprueba que cambian un tensor del backbone y uno de clasificación después de la primera época de cada sesión. Revisad los pasos y pérdidas de ambas épocas. El warm-up de YOLO abarca todo este smoke test.

**Fase D. Métricas, procedencia y artefactos.** Tras la fase C, ejecutad ambos comandos de evaluación en **la misma** sesión GPU interactiva. Usad los dos entornos mediante sus ejecutables absolutos.

```bash
srun --gres=gpu:a100:1 -c 32 --mem=64G -t 01:30:00 --pty bash
```

Dentro de la sesión, desde el repositorio:

```bash
"$CASCADE_ENV/bin/python" -m scripts.evaluate_baseline --data "$BASELINE_DATA" \
  --run "$BASELINE_RESULTS/smoke-cascade" --split valid --checkpoint best \
  --output "$BASELINE_RESULTS/smoke-eval-cascade" --benchmark --warmup 20 --iterations 100
"$YOLO_ENV/bin/python" -m scripts.evaluate_baseline --data "$BASELINE_DATA" \
  --run "$BASELINE_RESULTS/smoke-yolo" --split valid --checkpoint best \
  --output "$BASELINE_RESULTS/smoke-eval-yolo" --benchmark --warmup 20 --iterations 100
"$CASCADE_ENV/bin/python" -m scripts.verify_baseline --data "$BASELINE_DATA" --split valid \
  --runs "$BASELINE_RESULTS/smoke-cascade" "$BASELINE_RESULTS/smoke-yolo" \
  --evaluations "$BASELINE_RESULTS/smoke-eval-cascade" "$BASELINE_RESULTS/smoke-eval-yolo"
```

- [ ] El verificador reproduce las métricas desde predicciones, comprueba el mismo COCO, IDs, rango de coordenadas, soportes, checkpoint, AP definido o nulo y condiciones de latencia.
- [ ] Abrid `training.png`, `pr_curves.png` y los JPEG de ejemplos. Comprobad su legibilidad y que cajas TP, FP y FN se dibujan en la escala original. Las cantidades TP/FP/FN y las curvas corresponden a sus umbrales declarados.
- [ ] La inicialización, pausa, reanudación y evaluación han sido ejecutadas en CESGA y revisadas antes de aceptar los cambios. El smoke test solo valida el circuito.

### Entrenamiento definitivo, evaluación y recuperación

Tras aceptar A–D, lanzad trabajos nuevos con la receta base y salidas nuevas.

```bash
sbatch slurm/cascade.sh --data "$BASELINE_DATA" --weights "$CASCADE_WEIGHTS" \
  --output "$BASELINE_RESULTS/baseline-cascade"
sbatch slurm/yolo.sh --data "$BASELINE_DATA" --weights "$YOLO_WEIGHTS" \
  --output "$BASELINE_RESULTS/baseline-yolo"
squeue -u "$USER"
```

Ante una interrupción, repetid el comando correspondiente con `--resume`, los mismos argumentos de configuración, datos, pesos y salida. Verificad que el último checkpoint exista. Cascade usa `last_checkpoint` y recupera optimizador, scheduler, iteración y el estado del hook del mejor AP. YOLO recupera su formato nativo `weights/last.pt`, que guarda EMA, época y optimizador y reconstruye la agenda. El final automático de Ultralytics que despoja al checkpoint del optimizador se sustituye por la evaluación común separada. La reanudación desde un checkpoint no restaura todos los estados RNG, del sampler o las copias FP32 de YOLO que el formato nativo no guarda. No se espera equivalencia bit a bit. `status.json` acumula tiempos de sesiones que alcanzaron el cierre; tras una terminación abrupta, consultad también `sacct` para contabilizar esa sesión. La recuperación empieza en el último límite completo de checkpoint.

En una asignación A100 como la de la fase D, ejecutad lo siguiente para evaluar validación. Repetidlo con `EVALUATION_SPLIT=test` exclusivamente cuando el protocolo y los modelos estén cerrados. Los checkpoints se seleccionan en validación en ambos casos.

```bash
EVALUATION_SPLIT=valid
"$CASCADE_ENV/bin/python" -m scripts.evaluate_baseline --data "$BASELINE_DATA" \
  --run "$BASELINE_RESULTS/baseline-cascade" --split "$EVALUATION_SPLIT" \
  --output "$BASELINE_RESULTS/eval-cascade-$EVALUATION_SPLIT" --benchmark --warmup 20 --iterations 100
"$YOLO_ENV/bin/python" -m scripts.evaluate_baseline --data "$BASELINE_DATA" \
  --run "$BASELINE_RESULTS/baseline-yolo" --split "$EVALUATION_SPLIT" \
  --output "$BASELINE_RESULTS/eval-yolo-$EVALUATION_SPLIT" --benchmark --warmup 20 --iterations 100
```

En cada run, `run.json` registra procedencia y configuración declarada, los archivos del framework conservan valores efectivos y `status.json` indica el mejor paso, duración, memoria y rutas relativas de best/last. Cascade escribe su `metrics.json` nativo en JSON Lines y YOLO su `results.csv`. Para análisis conjunto se guardan `history.csv` y `training.png`. La evaluación produce `metrics.json`, `predictions.json`, `pr_curves.json`, `pr_curves.png` y hasta cinco ejemplos por tipo. Para copiar resultados ligeros, definid el acceso y la ruta remota de vuestro proyecto. Los pesos se pueden recuperar quitando los filtros de `rsync`; los logs están en `results/slurm/` del checkout de CESGA. En YOLO los checkpoints serializan EMA y estado en el formato nativo, lo que puede dar pequeñas diferencias entre AP observado en la selección y AP del archivo vuelto a cargar. Revisad en A–D si representan un problema. Los modelos, pesos y módulos de CESGA quedan pendientes de prueba real.

```bash
CESGA_LOGIN='usuario@host-cesga'
CESGA_RESULTS='/ruta/persistente/pills-baseline/results'
mkdir -p results/cesga
rsync -av --exclude='*.pt' --exclude='*.pth' --exclude='*.tfevents.*' \
  "$CESGA_LOGIN:$CESGA_RESULTS/" results/cesga/
```

## J. Hipótesis para futuros experimentos

Si validación muestra sensibilidad a iluminación, fondo o pose, la primera comparación sería con aumentos fotométricos o geométricos acotados que mantengan las señales de clase. Si el ajuste se degrada tras una mejora inicial, contrastar tasas de aprendizaje más bajas, ajuste parcial y descongelación progresiva separaría adaptación útil de sobreajuste. Si el problema persiste con curvas estables, optimizador, weight decay u otra regularización merecerían una comparación controlada. Si el error se concentra en cajas pequeñas y se confirma al revisar ejemplos, se puede estudiar mayor resolución con el mismo protocolo de latencia. Estas comparaciones se decidirán a partir de validación y con soporte suficiente por categoría.
