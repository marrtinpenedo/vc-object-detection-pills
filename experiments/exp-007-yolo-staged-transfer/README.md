# EXP-007 · Fine-tuning por etapas de YOLOv10-S

**Estado** Fase A verificada. B y C pendientes.

[Hallazgos](findings.md) · [Inventario de evidencias](results/README.md)

## Motivación e hipótesis

La baseline ajusta toda la red desde la primera época con 316 imágenes. El mAP desciende entre su mejor época y la última, y la revisión de errores identifica confusiones entre clases. EXP-001 y EXP-006 observaron mejoras de estabilidad inicial acompañadas de descensos del mAP oficial. Estos resultados motivan probar si adaptar primero el neck y las cabezas, conservando temporalmente las representaciones del backbone, mejora la adaptación a las pastillas. No se ha establecido que el sobreajuste o la degradación del backbone expliquen las caídas observadas. [Baseline](../baseline/findings.md) · [EXP-001](../exp-001-yolo-lr/findings.md) · [EXP-006](../exp-006-yolo-warmup/findings.md)

Vasconcelos et al. observaron que un backbone preentrenado en clasificación y congelado durante toda la detección puede ser ventajoso con componentes de detección de capacidad suficiente. Este experimento parte de pesos de detección COCO y estudia duraciones de congelación hasta cubrir el entrenamiento completo. [Vasconcelos et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Vasconcelos_Proper_Reuse_of_Image_Classification_Features_Improves_Object_Detection_CVPR_2022_paper.pdf)

## Arquitectura y condiciones

Se comprobó el `yolov10s.pt` COCO del proyecto por SHA-256 y se construyó en CPU, con Ultralytics 8.3.161, el modelo adaptado a ocho clases mediante el adaptador del proyecto. Su YAML embebido y el grafo construido coinciden en la separación siguiente. El entrenamiento comprueba tipos, índices y conexiones tanto en los pesos de origen como en el modelo adaptado. [Pesos y huellas](../../docs/cesga.md#datos-y-pesos) · [Comprobación de arquitectura](../../scripts/yolo_staged.py)

| Parte | Índices y tipos |
| --- | --- |
| Backbone, 0–10 | 0–1 Conv, 2 C2f, 3 Conv, 4 C2f, 5 SCDown, 6 C2f, 7 SCDown, 8 C2fCIB, 9 SPPF, 10 PSA |
| Neck, 11–22 | 11 Upsample, 12 Concat, 13 C2f, 14 Upsample, 15 Concat, 16 C2f, 17 Conv, 18 Concat, 19 C2f, 20 SCDown, 21 Concat, 22 C2fCIB |
| Detector, 23 | v10Detect con entradas 16, 19 y 22 |

De los 8 072 544 escalares de parámetros del modelo adaptado, 3 888 800 pertenecen al backbone. La proyección DFL conserva sus 16 escalares congelados en ambas fases. Las ramas one-to-many y one-to-one siguen el flujo nativo. La primera pasa gradientes hacia el neck y el backbone entrenable. La segunda recibe las características desacopladas y ajusta sus propios parámetros. [Cabeza de Ultralytics 8.3.161](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/nn/modules/head.py)

| Brazo | Backbone congelado | Fine-tuning completo | Total |
| --- | ---: | ---: | ---: |
| [control](../../configs/baseline/yolo.json) | 0 | 30 | 30 |
| [freeze-5](../../configs/experiments/yolo-freeze-5.json) | 5 | 25 | 30 |
| [freeze-10](../../configs/experiments/yolo-freeze-10.json) | 10 | 20 | 30 |
| [freeze-15](../../configs/experiments/yolo-freeze-15.json) | 15 | 15 | 30 |

La opción `freeze_backbone_epochs` se admite solo en YOLO y debe ser un entero entre cero y el presupuesto, inclusive. Si se omite, es efectiva como cero y se mantiene la ruta del entrenador actual. Todas las recetas conservan YOLOv10-S, pesos COCO, datos y ocho clases, batch 8, 640, FP32, AdamW, LR 0,001, momentum 0,9, weight decay 0,0005, coseno y factor final 0,05, sin aumentos. El warm-up original conserva los parámetros de sesgos y momentum y termina en el índice 100 inclusive con 40 lotes por época. El evaluador COCO y la selección de `best` siguen siendo los de la baseline. [Receta de referencia](../baseline/README.md#f-recetas-e-hiperparámetros) · [Protocolo común](../baseline/README.md#g-métricas-y-protocolo-de-inferencia)

### Transición y auditoría

Una única llamada al entrenador ejecuta las 30 épocas. Tras la transferencia, Ultralytics construye AdamW, scheduler y EMA con el modelo completo. Al final de la preparación se desactivan los gradientes del backbone y se fijan sus Batch Normalization en modo evaluación. AdamW conserva la pertenencia original a los grupos de sesgos, pesos con decay y normalización sin decay. Con `grad=None`, omite tanto su actualización como el weight decay y la creación de estado. La subclase aplica el modo de las Batch Normalization después de la llamada nativa a `model.train()`. En la época N+1 restaura el estado entrenable original del backbone y sus Batch Normalization antes del primer minibatch. No se reinician grupos, estados acumulados, scheduler, warm-up, EMA, historial ni selección de checkpoints. Si N coincide con el presupuesto, nunca se alcanza N+1. El backbone queda congelado, `thaw_epoch` es nulo y AdamW mantiene a cero sus pasos. [Trainer de Ultralytics 8.3.161](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/trainer.py)

`parameters.json` inventaría el estado inicial y `staged_transfer.json` registra los grupos AdamW, módulos, totales, porcentaje de congelación y la época de transición. Por época se comprueban los SHA-256 de parámetros y buffers del modelo de entrenamiento, los contadores Batch Normalization y la ausencia de gradientes en parámetros congelados. Se muestrean gradientes útiles en el primer paso de cada época y se cotejan los cambios al final en neck, ambas cabezas y, tras la transición, backbone. Se conservan los objetos AdamW, scheduler y EMA y se verifican sus contadores. Las huellas concatenan los bytes de los tensores ordenados por nombre. Se calculan al preparar, en los límites de época y antes de descongelar. El guardado normal de EMA puede usar otra precisión. La comparación exacta se aplica a los tensores de entrenamiento.

`--initialize` verifica inferencia y backward con el backbone congelado, sin realizar pasos del optimizador. Las variantes rechazan explícitamente `--pause-after` y `--resume`. Ante una interrupción, iniciar una ejecución nueva de la variante. El control conserva sus opciones originales.

## Fases y criterio de decisión

**A, exploración.** Reentrenar los cuatro brazos con semilla 2026 y la misma revisión. El control permite cotejar la receta, la ruta de ejecución y el orden de magnitud de los resultados con la baseline histórica. Evaluar los cuatro `best` en `valid` dentro de una asignación. Una variante es elegible si pierde como máximo 0,02 de recall operativo y cumple una de estas condiciones:

1. Δ mAP oficial ≥ 0,01 y Δ del mAP de siete clases ≥ 0.
2. Δ del mAP de siete clases ≥ 0,02 y Δ mAP oficial ≥ 0.

Los deltas son variante menos control y los márgenes son absolutos, con finalidad exploratoria. Seleccionar como máximo una variante ordenando por mAP oficial, mAP sin Ibuphil 600 mg, recall operativo y, por último, menor duración. Si ninguna es elegible, cerrar en A.

**B, confirmación.** Entrenar únicamente el control y la variante seleccionada con semillas 2027 y 2028. Evaluar cada pareja en una asignación común. Describir A, B y las tres parejas por separado. Adoptar solo si se cumplen todos estos criterios:

- Δ medio positivo de mAP oficial en las tres parejas y mejora estricta en al menos dos semillas.
- Δ medio positivo de mAP oficial en las dos parejas de confirmación.
- Δ medio no negativo del mAP sin Ibuphil 600 mg y pérdida media de recall ≤ 0,02 en las tres parejas.

Examinar la consistencia por clase y el benchmark al aplicar la decisión. Esta decisión se toma con el protocolo original y sus ocho entrenamientos como máximo. Se analiza aparte de C. Se reserva `test` para la evaluación final del proyecto.

**C, ampliación posterior a A.** El mAP creció con la duración en la semilla 2026 y el mejor checkpoint de `freeze-15` se obtuvo en la época 15, justo antes de descongelar. Se amplía el intervalo para estudiar si esa pauta se mantiene, incluida la congelación permanente. El resultado de una semilla deja abierta la respuesta con otras duraciones y semillas.

| Brazo | Épocas congelado | Épocas descongelado | Semilla 2026 | Semilla 2027 | Semilla 2028 |
| --- | ---: | ---: | --- | --- | --- |
| control | 0 | 30 | A, hecho | B | B |
| freeze-5 | 5 | 25 | A, hecho | C | C |
| freeze-10 | 10 | 20 | A, hecho | C | C |
| freeze-15 | 15 | 15 | A, hecho | B | B |
| [freeze-20](../../configs/experiments/yolo-freeze-20.json) | 20 | 10 | C | C | C |
| [freeze-25](../../configs/experiments/yolo-freeze-25.json) | 25 | 5 | C | C | C |
| [freeze-30](../../configs/experiments/yolo-freeze-30.json) | 30 | 0 | C | C | C |

Son 21 entrenamientos únicos, cuatro recuperados de A, cuatro pendientes en B y 13 pendientes en C. En C se evalúan siete checkpoints por semilla en una misma asignación, con el protocolo común y solo en `valid`. Los AP, curvas PR y punto operativo de los checkpoints reutilizados deben reproducir exactamente los de A o B. Si difieren, se detiene el análisis para investigar. Las latencias de una asignación previa se conservan aparte.

C considera candidatas las variantes con Δ medio de mAP oficial ≥ 0,01, Δ medio de mAP de siete clases ≥ 0, mejora oficial en al menos dos semillas y pérdida media de recall ≤ 0,02 frente al control. Se ordenan por mAP oficial medio, mAP de siete clases medio, recall medio y menor duración. El resultado es una selección descriptiva con las mismas semillas usadas para buscar la duración. La confirmación independiente de la decisión exploratoria sobre `freeze-15` sigue siendo B.

### Métricas

El análisis verifica mAP COCO, AP50, AP75, AP por clase y tamaño, soporte y curvas PR desde las predicciones, incluye el promedio sin Ibuphil 600 mg como sensibilidad y registra precisión, recall y TP/FP/FN a score ≥ 0,25 e IoU ≥ 0,50. El benchmark común aporta latencia y FPS. La evaluación genera las escenas 57, 61 y 80 para cada brazo. Se archivan pares seleccionados. [Ejemplos y criterio de exportación](results/README.md)

La gráfica con escalas compartidas muestra mAP, pérdidas de train y validación y LR al final de cada época, con líneas verticales en la primera época descongelada. Las pérdidas usan escala logarítmica simétrica para conservar los picos y los ceros. El resumen registra el pico de clasificación y los mAP nulos iniciales, la mejor época, su brecha con la última, tiempo y memoria GPU. Incluye el mAP y las pérdidas de clasificación en las dos épocas que rodean cada transición, con las mismas épocas del control. La medición del entrenamiento incluye la auditoría de las variantes. La brecha entre máximo y última época describe la evolución en validación. Interpretarla junto a las pérdidas de train y valid y al AP por clase. La semilla exploratoria aporta una comparación descriptiva.

## Ejecución en CESGA

### Revisiones y reutilización

A y B usan el commit `ed7f9a6ef75afa9c66673bf8f03d34be2fa0c228` para entrenar y evaluar. El análisis de las descargas brutas se puede repetir con el analizador de ese commit. Las evidencias archivadas se reproducen con el mismo criterio y el checkout original como referencia de procedencia. Su copia es `~/vc-object-detection-pills-ed7f9a6ef75a` y su campaña es `$STORE/pills-baseline/results/exp-007-yolo-staged-HhuHld`. El plan reutiliza los cuatro checkpoints y la evaluación ya verificada de A.

Los 13 entrenamientos nuevos de C usan el checkout de la ampliación. Las evaluaciones de C se ejecutan con el código histórico para que sean comparables con A y B. Antes de enviar, el plan verifica todas las huellas de scripts, lanzadores y configuraciones. Autoriza solamente el cambio de dos líneas que acepta N igual al presupuesto y los tres JSON que añaden duraciones. El entrenador, la regla de transición, el predictor y el evaluador conservan sus bytes. La condición N < presupuesto sigue siendo válida para los cuatro brazos originales y el control conserva la ruta sin congelación. El análisis comprueba además esas huellas contra los blobs de Git. [Política de revisiones](provenance.py)

### Desplegar y encolar la campaña

Ejecutar en Bash desde el clon local después de versionar la revisión aprobada. Definir `CESGA_SSH` como en la [guía de transferencia](../../docs/cesga.md#transferencia-y-actualización-del-código). Se transfiere un checkout nuevo y se conserva intacta la copia histórica.

```bash
(
  set -euo pipefail
  : "${CESGA_SSH:?}"
  revision=$(git rev-parse --verify HEAD)
  test "$revision" != ed7f9a6ef75afa9c66673bf8f03d34be2fa0c228
  git cat-file -e "$revision:experiments/exp-007-yolo-staged-transfer/campaign.py"
  project="vc-object-detection-pills-${revision:0:12}"
  stage=$(mktemp -d)
  trap 'rm -rf -- "$stage"' EXIT
  git archive "$revision" | tar -x -C "$stage"
  printf '%s\n' "$revision" > "$stage/.exp007-revision"
  ssh "$CESGA_SSH" "mkdir -- '$project'"
  rsync -a -e ssh "$stage/" "$CESGA_SSH:$project/"
  printf 'export EXP_REVISION=%s\n' "$revision"
)
```

En una sesión Bash en el login de CESGA, ejecutar el `export` impreso y lanzar el plan una vez. Usa el entorno y los pesos compartidos del enunciado y el dataset preparado de la campaña histórica.

```bash
(
  set -euo pipefail
  : "${EXP_REVISION:?}" "${STORE:?}"
  cd "$HOME/vc-object-detection-pills-${EXP_REVISION:0:12}"
  "$STORE/pills-baseline/envs/yolo/bin/python" \
    -m experiments.exp-007-yolo-staged-transfer.campaign submit --revision "$EXP_REVISION"
)
```

El plan encola primero `exp007-yolo-smoke-permanent-s2026` en A100. Ejecuta el entrenador con tres épocas y `freeze_backbone_epochs=3`. El propio trabajo vuelve a comprobar las tres filas, las huellas invariantes, Batch Normalization, ausencia de gradientes, AdamW, scheduler, EMA y DFL. Solo escribe `smoke/verified.ok` si pasan todos los controles. Un fallo devuelve un estado distinto de cero y bloquea la cadena `afterok`.

Después se encolan B y C. Se serializan los trabajos de la campaña con datos compartidos. Cada entrenamiento usa un directorio Ultralytics exclusivo. Las evaluaciones llevan nombres `exp007-eval-b-s2027`, `exp007-eval-b-s2028` y `exp007-eval-c-sSEMILLA`. El plan comprueba las ejecuciones registradas y los nombres activos en SLURM antes de enviar. Conserva las salidas verificadas y escribe los IDs y argumentos de los nuevos envíos en `queue/`. La extensión queda en `$STORE/pills-baseline/results/exp-007-yolo-staged-HhuHld-extension`. [Planificador](campaign.py)

Para consultar el avance, usar `sacct -j ID --format=JobID,JobName,State,ExitCode,Elapsed` y los `.out` y `.err` bajo `slurm/` de cada campaña. Si se interrumpe el proceso de envío, repetir el mismo comando desde el mismo checkout. Si falla un trabajo, esperar que SLURM cancele los descendientes con dependencias inválidas, revisar sus logs y añadir `--retry-failed` al comando. Las salidas fallidas se conservan en `attempts/`. Si queda un envío sin ID confirmado, el plan se detiene. Contrastar `squeue --me --name=NOMBRE` y `sacct --name=NOMBRE --starttime=FECHA` con el comando y el directorio de trabajo del registro. Recuperar el ID verificado en el JSON antes de repetir. Si se comprueba que SLURM nunca aceptó el envío, conservar bajo otro nombre el registro incompleto y su salida preparada antes de repetir. [Consulta SLURM](../../docs/cesga.md#ejecución-de-trabajos-slurm)

## Recuperación y reproducción local

Recuperar B y C por separado con el inventario. Definir `PHASE_B` y `PHASE_C` con las rutas locales a las descargas verificadas o a los archivos publicados. Sin esas variables se reproduce A. Con solo `PHASE_B` se reproducen A y la decisión de B. Ejecutar el bloque en Bash desde el clon local con `valid/_annotations.coco.json` del dataset original.

```bash
(
  set -euo pipefail
  root=$PWD
  base="$root/experiments/exp-007-yolo-staged-transfer/results"
  a="$base/phase-a"
  data=$(realpath "${EXP_DATASET:-data/pills-sxdht}")
  confirmation=()
  if [[ -n ${PHASE_B:-} ]]; then
    b=$(realpath "$PHASE_B")
    confirmation=(--confirmation "$b")
  fi
  if [[ -n ${PHASE_C:-} ]]; then
    : "${PHASE_B:?Para reproducir C se requieren los artefactos de B}"
    c=$(realpath "$PHASE_C")
  fi
  old=$(mktemp -d)
  new=$(mktemp -d)
  trap 'rm -rf -- "$old" "$new"' EXIT
  git archive ed7f9a6ef75afa9c66673bf8f03d34be2fa0c228 | tar -x -C "$old"
  mkdir -p results
  out=$(mktemp -d "$PWD/results/exp-007-analysis-XXXXXX")
  py=(uv run --offline --no-project --python 3.11 --with pycocotools==2.0.8 --with numpy==1.26.4
      --with pillow==10.4.0 --with matplotlib==3.9.4 python -m)
  export MPLBACKEND=Agg
  "${py[@]}" experiments.exp-007-yolo-staged-transfer.analyze \
    --source "$a" "${confirmation[@]}" --original-checkout "$old" --data "$data" --output "$out/original"
  if [[ -n ${c:-} ]]; then
    revision=$(git -C "$root" rev-parse --verify "$(cat "$c/source-commit")^{commit}")
    git -C "$root" archive "$revision" | tar -x -C "$new"
    cd "$new"
    "${py[@]}" experiments.exp-007-yolo-staged-transfer.analyze_c --source "$a" \
      --confirmation "$b" --extension "$c" --original-checkout "$old" --git-repo "$root" \
      --data "$data" --output "$out/c"
  fi
  printf '%s\n' "$out"
)
```

Si faltan dependencias en caché, quitar `--offline`. El [analizador de C](analyze_c.py) identifica revisión y checkpoint por brazo y verifica COCO desde las predicciones exportadas. Genera `duracion.png` con trazas por semilla, medias y desviaciones estándar muestrales, `prefijo-congelado.png` para las primeras cinco épocas comunes, historiales por semilla y un resumen JSON. Este incluye los máximos antes y después de cada transición y los deltas máximos durante el prefijo congelado compartido por cada pareja de duraciones. En `freeze-30` el máximo posterior es nulo. Las diferencias del prefijo se informan como diagnóstico, sin asumir igualdad bit a bit del entrenamiento.
