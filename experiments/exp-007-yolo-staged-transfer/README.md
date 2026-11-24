# EXP-007 · Fine-tuning por etapas de YOLOv10-S

**Estado** `propuesto`.

[Hallazgos](findings.md) · [Inventario de evidencias](results/README.md)

## Motivación e hipótesis

La baseline ajusta toda la red desde la primera época con 316 imágenes. El mAP desciende entre su mejor época y la última, y la revisión de errores identifica confusiones entre clases. EXP-001 y EXP-006 observaron mejoras de estabilidad inicial acompañadas de descensos del mAP oficial. Estos resultados motivan probar si adaptar primero el neck y las cabezas, conservando temporalmente las representaciones del backbone, mejora la adaptación a las pastillas. No se ha establecido que el sobreajuste o la degradación del backbone expliquen las caídas observadas. [Baseline](../baseline/findings.md) · [EXP-001](../exp-001-yolo-lr/findings.md) · [EXP-006](../exp-006-yolo-warmup/findings.md)

Vasconcelos et al. observaron que un backbone preentrenado en clasificación y congelado durante toda la detección puede ser ventajoso con componentes de detección de capacidad suficiente. Este experimento parte de pesos de detección COCO y descongela el backbone dentro del mismo entrenamiento. El artículo motiva la hipótesis de preservar características al inicio, con un punto de partida y una duración distintos. [Vasconcelos et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Vasconcelos_Proper_Reuse_of_Image_Classification_Features_Improves_Object_Detection_CVPR_2022_paper.pdf)

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

La opción `freeze_backbone_epochs` se admite solo en YOLO y debe ser un entero entre cero y el presupuesto menos uno. Si se omite, es efectiva como cero y se mantiene la ruta del entrenador actual. Todas las recetas conservan YOLOv10-S, pesos COCO, datos y ocho clases, batch 8, 640, FP32, AdamW, LR 0,001, momentum 0,9, weight decay 0,0005, coseno y factor final 0,05, sin aumentos. El warm-up original conserva los parámetros de sesgos y momentum y termina en el índice 100 inclusive con 40 lotes por época. El evaluador COCO y la selección de `best` siguen siendo los de la baseline. [Receta de referencia](../baseline/README.md#f-recetas-e-hiperparámetros) · [Protocolo común](../baseline/README.md#g-métricas-y-protocolo-de-inferencia)

### Transición y auditoría

Una única llamada al entrenador ejecuta las 30 épocas. Tras la transferencia, Ultralytics construye AdamW, scheduler y EMA con el modelo completo. Al final de la preparación se desactivan los gradientes del backbone y se fijan sus Batch Normalization en modo evaluación. AdamW conserva la pertenencia original a los grupos de sesgos, pesos con decay y normalización sin decay. Con `grad=None`, omite tanto su actualización como el weight decay y la creación de estado. La subclase aplica el modo de las Batch Normalization después de la llamada nativa a `model.train()`. En la época N+1 restaura el estado entrenable original del backbone y sus Batch Normalization antes del primer minibatch. No se reinician grupos, estados acumulados, scheduler, warm-up, EMA, historial ni selección de checkpoints. [Trainer de Ultralytics 8.3.161](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/trainer.py)

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

Examinar la consistencia por clase y el benchmark al aplicar la decisión. La búsqueda queda limitada a los cuatro brazos fijados y a ocho entrenamientos completos como máximo. Se reserva `test` para la evaluación final del proyecto.

### Métricas

El análisis verifica mAP COCO, AP50, AP75, AP por clase y tamaño, soporte y curvas PR desde las predicciones, incluye el promedio sin Ibuphil 600 mg como sensibilidad y registra precisión, recall y TP/FP/FN a score ≥ 0,25 e IoU ≥ 0,50. El benchmark común aporta latencia y FPS. Los JPEG 57, 61 y 80 permanecen emparejados.

La gráfica con escalas compartidas muestra mAP, pérdidas de train y validación y LR al final de cada época, con líneas verticales en la primera época descongelada. Las pérdidas usan escala logarítmica simétrica para conservar los picos y los ceros. El resumen registra el pico de clasificación y los mAP nulos iniciales, la mejor época, su brecha con la última, tiempo y memoria GPU. Incluye el mAP y las pérdidas de clasificación en las dos épocas que rodean cada transición, con las mismas épocas del control. La medición del entrenamiento incluye la auditoría de las variantes. La brecha entre máximo y última época describe la evolución en validación. Interpretarla junto a las pérdidas de train y valid y al AP por clase. La semilla exploratoria aporta una comparación descriptiva.

## Ejecución en CESGA

### Desplegar y preparar una tanda

Ejecutar los bloques en Bash. Desde el clon local, tras versionar la revisión aprobada, obtener su SHA con `git rev-parse HEAD` y transferir ese commit a una carpeta nueva mediante `git archive` y la guía de CESGA. En CESGA se usa esa misma copia para entrenamiento y evaluación. Definir `EXP_COMMIT` con el SHA transferido y las rutas `BASELINE_RESULTS`, `BASELINE_DATA`, `YOLO_ENV` y `YOLO_WEIGHTS` de la guía. [Transferencia](../../docs/cesga.md#transferencia-y-actualización-del-código) · [Rutas y entorno](../../docs/cesga.md#acceso-y-almacenamiento)

Desde la raíz desplegada, comprobar las huellas de los pesos conforme a la guía de CESGA, crear una tanda exclusiva y conservar `EXP_ROOT`. Se copian únicamente los archivos preparados del manifiesto para que las cachés de etiquetas queden aisladas de otras tandas. [Pesos e integridad](../../docs/cesga.md#datos-y-pesos)

```bash
export EXP_ROOT
EXP_ROOT=$(mktemp -d "${BASELINE_RESULTS:?}/exp-007-yolo-staged-XXXXXX")
(
  set -euo pipefail
  : "${EXP_ROOT:?}"
  [[ "${EXP_COMMIT:?}" =~ ^[0-9a-f]{40}$ ]]
  printf '%s\n' "$EXP_COMMIT" > "$EXP_ROOT/source-commit"
  mkdir "$EXP_ROOT/data" "$EXP_ROOT/slurm"
  "$YOLO_ENV/bin/python" -c 'import json, sys; print("manifest.json", *sorted(json.load(open(sys.argv[1]))["files"]), sep="\n")' \
    "$BASELINE_DATA/manifest.json" > "$EXP_ROOT/data.files"
  rsync -a --files-from="$EXP_ROOT/data.files" "$BASELINE_DATA/" "$EXP_ROOT/data/"
  sha256sum scripts/*.py slurm/{yolo.sh,evaluate.sh,evaluate.py} configs/baseline/yolo.json \
    configs/experiments/yolo-freeze-{5,10,15}.json > "$EXP_ROOT/code-config.sha256"
  sha256sum "$EXP_ROOT/data/manifest.json" "$EXP_ROOT/data/coco/valid.json" "$YOLO_WEIGHTS" \
    > "$EXP_ROOT/inputs.sha256"
) && printf '%s\n' "$EXP_ROOT"
```

### Smoke test antes de la fase A

Enviar tres épocas con una congelada. La configuración y la salida del smoke test quedan fuera del inventario científico. Esperar a que termine antes de enviar otros entrenamientos de la misma tanda.

```bash
(
  set -euo pipefail
  sha256sum --check --quiet "$EXP_ROOT/code-config.sha256" "$EXP_ROOT/inputs.sha256"
  mkdir "$EXP_ROOT/smoke"
  "$YOLO_ENV/bin/python" -c 'import json; c=json.load(open("configs/baseline/yolo.json")); c.update(epochs=3,freeze_backbone_epochs=1); print(json.dumps(c))' \
    > "$EXP_ROOT/smoke/config.json"
  PILLS_YOLO_CONFIG_DIR="$EXP_ROOT/smoke/ultralytics" sbatch --parsable --export=ALL \
    --job-name=exp007-yolo-smoke-s2026 --output="$EXP_ROOT/slurm/%x-%j.out" \
    --error="$EXP_ROOT/slurm/%x-%j.err" slurm/yolo.sh --data "$EXP_ROOT/data" --weights "$YOLO_WEIGHTS" \
    --config "$EXP_ROOT/smoke/config.json" --output "$EXP_ROOT/smoke/train"
)
```

Comprobar `COMPLETED` y `0:0` en SLURM. La comprobación siguiente usa únicamente los artefactos del smoke test.

```bash
"$YOLO_ENV/bin/python" - "$EXP_ROOT/smoke/train" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
status = json.loads((root / "status.json").read_text())
audit = json.loads((root / "staged_transfer.json").read_text())
assert status["status"] == "complete"
assert audit["thaw_epoch"] == 2
rows = audit["epochs"]
assert [row["epoch"] for row in rows] == [1, 2, 3]
assert [row["backbone_frozen"] for row in rows] == [True, False, False]
assert [row["ema_updates"] for row in rows] == [40, 80, 120]
assert rows[0]["backbone_sha256"] == audit["initial_backbone_sha256"]
assert rows[1]["nonzero_gradients"]["backbone"] and rows[1]["updated"]["backbone"]
PY
```

### Entrenar y evaluar

Definir las funciones en la misma sesión Bash. Los entrenamientos de cada llamada se encadenan para que solo uno escriba la caché de etiquetas de la tanda. Cada brazo usa su propia configuración de Ultralytics. Lanzar la evaluación cuando todos figuren como `COMPLETED|0:0`. El lanzador lo comprueba y registra antes de medir los checkpoints.

```bash
train_exp007() (
  set -euo pipefail
  sha256sum --check --quiet "$EXP_ROOT/code-config.sha256" "$EXP_ROOT/inputs.sha256"
  seed=$1
  shift
  dep=()
  for arm; do
    case "$arm" in
      control) config=configs/baseline/yolo.json ;;
      freeze-5|freeze-10|freeze-15) config="configs/experiments/yolo-$arm.json" ;;
      *) exit 2 ;;
    esac
    slot="$EXP_ROOT/s$seed/$arm"
    mkdir -p "$EXP_ROOT/s$seed"
    mkdir "$slot"
    "$YOLO_ENV/bin/python" -m scripts.train_baseline yolo --check-config --config "$config" \
      --seed "$seed" > "$slot/config.json"
    printf '%s\n' "$slot/train" > "$slot/train.path"
    job=$(PILLS_YOLO_CONFIG_DIR="$slot/ultralytics" sbatch --parsable --export=ALL "${dep[@]}" \
      --job-name="exp007-yolo-$arm-s$seed" --output="$EXP_ROOT/slurm/%x-%j.out" \
      --error="$EXP_ROOT/slurm/%x-%j.err" slurm/yolo.sh --data "$EXP_ROOT/data" \
      --weights "$YOLO_WEIGHTS" --config "$config" --seed "$seed" --output "$slot/train")
    printf '%s\n' "${job%%;*}" | tee "$slot/train.job"
    dep=(--dependency="afterok:${job%%;*}")
  done
)

eval_exp007() (
  set -euo pipefail
  sha256sum --check --quiet "$EXP_ROOT/code-config.sha256" "$EXP_ROOT/inputs.sha256"
  phase=$1 seed=$2
  shift 2
  runs=()
  for arm; do runs+=(--run "$arm" "$EXP_ROOT/s$seed/$arm/train"); done
  sbatch --parsable --job-name="exp007-eval-$phase-s$seed" --output="$EXP_ROOT/slurm/%x-%j.out" \
    --error="$EXP_ROOT/slurm/%x-%j.err" slurm/evaluate.sh --data "$EXP_ROOT/data" --split valid \
    --checkpoint best --check-slurm --output "$EXP_ROOT/eval-$phase-s$seed" \
    --example-image-id 57 --example-image-id 61 --example-image-id 80 "${runs[@]}"
)
```

```bash
train_exp007 2026 control freeze-5 freeze-10 freeze-15
```

Tras completar los cuatro trabajos, ejecutar:

```bash
eval_exp007 a 2026 control freeze-5 freeze-10 freeze-15
```

Si el análisis de A selecciona un brazo, anotar esa decisión y asignarlo a `SELECTED`. Para cada semilla 2027 y 2028, usar `train_exp007 SEMILLA control "$SELECTED"`, esperar a que termine la pareja y usar `eval_exp007 b SEMILLA control "$SELECTED"`. Completar los entrenamientos de 2027 antes de enviar los de 2028 dentro de la misma tanda. Confirmar `verified.ok` después de cada evaluación. [Lanzador y seguimiento](../../docs/cesga.md#evaluación-conjunta)

## Recuperación y reproducción local

Descargar los archivos de la lista selectiva. Desde la raíz del clon en Bash, definir `LOCAL_EXP` con la descarga o evidencia de A y, para B, `LOCAL_EXP_B` con su evidencia. `EXP_DATASET` puede indicar otra ubicación del COCO original. La revisión anotada debe estar disponible en Git. [Inventario y recuperación](results/README.md)

```bash
(
  set -euo pipefail
  source=$(realpath "${LOCAL_EXP:?}")
  dataset=$(realpath "${EXP_DATASET:-data/pills-sxdht}")
  verify_evidence() (
    cd "$1"
    sums=downloaded.sha256
    if [[ -f evidence.sha256 ]]; then sums=evidence.sha256; fi
    sha256sum --check --quiet "$sums"
  )
  verify_evidence "$source"
  confirmation=()
  if [[ -n ${LOCAL_EXP_B:-} ]]; then
    second=$(realpath "$LOCAL_EXP_B")
    verify_evidence "$second"
    confirmation=(--confirmation "$second")
  fi
  revision=$(git rev-parse --verify "$(cat "$source/source-commit")^{commit}")
  checkout=$(mktemp -d)
  trap 'rm -rf -- "$checkout"' EXIT
  git archive "$revision" | tar -x -C "$checkout"
  mkdir -p results
  out=$(mktemp -d "$PWD/results/exp-007-analysis-XXXXXX")
  cd "$checkout"
  sha256sum --check --quiet "$source/code-config.sha256"
  if [[ -n ${LOCAL_EXP_B:-} ]]; then sha256sum --check --quiet "$second/code-config.sha256"; fi
  MPLBACKEND=Agg uv run --offline --no-project --python 3.11 \
    --with pycocotools==2.0.8 --with numpy==1.26.4 --with pillow==10.4.0 --with matplotlib==3.9.4 \
    python -m experiments.exp-007-yolo-staged-transfer.analyze --source "$source" --data "$dataset" \
    "${confirmation[@]}" --output "$out/report"
  printf '%s\n' "$out/report"
)
```

Si faltan dependencias en la caché, quitar `--offline`. El analizador usa las anotaciones originales de `valid`, reconstruye el COCO y exige el mismo SHA-256 de la evaluación. Verifica métricas, procedencia y evidencias de la transición antes de aplicar el criterio. Sus salidas son `summary.json` y una figura por semilla. El campo `exploration.selected` es nulo si ninguna variante pasa el filtro.
