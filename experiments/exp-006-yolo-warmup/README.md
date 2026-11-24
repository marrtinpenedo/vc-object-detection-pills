# EXP-006 · Duración del warm-up de YOLOv10-S

**Estado** `propuesto`

[Hallazgos y decisión](findings.md) · [Artefactos](results/README.md)

## Motivación e hipótesis

El baseline registró picos de `val/cls_loss` y épocas con mAP nulo al inicio. En EXP-001, reducir el LR de 0,001 a 0,0005 atenuó el pico inicial en las tres semillas. El mAP COCO medio pasó de 0,7450 a 0,6947 en ocho clases y de 0,7181 a 0,7511 al promediar las otras siete. Ibuphil 600 mg aporta una de las ocho clases al promedio oficial y tiene una instancia en `train` y una en `valid`. El experimento previo muestra que estabilidad y rendimiento pueden evolucionar de manera distinta. La causa de los picos sigue sin aislarse. [Baseline](../baseline/findings.md#evolución-del-entrenamiento) · [EXP-001](../exp-001-yolo-lr/findings.md)

**Hipótesis.** Prolongar el warm-up con LR inicial de 0,001 reducirá la inestabilidad de las primeras cinco épocas sin perjudicar la adaptación posterior. La mejora de rendimiento requiere además mayor mAP de los checkpoints `best` reevaluados en `valid`.

## Diseño y criterios de comparación

| Brazo | `warmup_epochs` | Último índice | Minibatches con interpolación | Configuración |
| --- | ---: | ---: | ---: | --- |
| Control | 1,0 | 100 | 101 | [Baseline](../../configs/baseline/yolo.json) |
| A · `warmup-5` | 5,0 | 200 | 201 | [Variante](../../configs/experiments/yolo-warmup-5.json) |
| B · `warmup-3` | 3,0 | 120 | 121 | [Variante](../../configs/experiments/yolo-warmup-3.json) |

Solo cambia `warmup_epochs`. Se mantienen YOLOv10-S, los pesos COCO `yolov10s.pt`, las particiones y las ocho clases, 30 épocas, lote 8, tamaño 640, FP32, AdamW, LR inicial 0,001 y scheduler coseno con factor final 0,05. Se conservan weight decay 0,0005, momentum 0,9, `warmup_momentum=0.8`, `warmup_bias_lr=0.001` y la ausencia de augmentación. Cada época valida y actualiza los checkpoints conforme a la receta original, con `best` seleccionado por mAP COCO. [Receta de referencia](../baseline/README.md#f-recetas-e-hiperparámetros)

Con 316 imágenes y lote 8 se esperan 40 minibatches por época. Ultralytics 8.3.161 interpola hasta el índice global `max(round(warmup_epochs * nb), 100)`, inclusive desde cero en un entrenamiento nuevo. La tabla cuenta minibatches dentro del presupuesto normal de 30 épocas. `record_runtime()` reproduce ese cálculo con la longitud real del cargador y guarda el índice en `ultralytics_runtime.json`. Durante el warm-up, el LR de cada grupo interpola hacia su LR inicial multiplicado por el factor coseno de la época. El grupo de sesgos empieza en 0,001 y los otros grupos en cero. El warm-up de momentum solo actúa sobre grupos con un campo `momentum`. AdamW conserva sus betas. [Ultralytics 8.3.161](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/trainer.py) · [Registro del repositorio](../../scripts/baseline_yolo.py)

### Fases y decisión

1. **Fase A, exploración.** Entrenar el control y `warmup-5` con semilla 2026 en una misma tanda y revisión.
2. **Fase B, refinamiento.** Probar `warmup-3` con 2026 si `warmup-5` mejora el mAP oficial o muestra un pico inicial menor sin aumentar las épocas a cero ni reducir el mAP de las primeras cinco épocas. En el segundo caso se admite una caída de hasta 0,01 absoluto de mAP oficial, fijada aquí como margen operativo para explorar. Reutilizar el control de A si coinciden revisión, datos, pesos, entorno y receta. Si la señal resulta insuficiente, registrar el resultado negativo y valorar el cierre sin ampliar la búsqueda.
3. **Fase C, robustez.** Seleccionar como máximo una variante que cumpla el filtro exploratorio frente al control. Si hay dos elegibles, elegir la de mayor mAP oficial en 2026. En caso de empate, aplicar en orden menor pico de pérdida, menos épocas a cero en 1–5, mayor mAP medio en 1–5 y menor duración del warm-up. Registrar la elección antes de entrenar controles emparejados con 2027 y 2028. Informar esas dos semillas por separado de la selección exploratoria.

Adoptar una variante exige un Δ medio de mAP oficial positivo y una mejora estricta en al menos dos de las tres parejas. Revisar por separado los deltas de confirmación y el AP por clase, en particular si el resultado global depende de Ibuphil 600 mg o si se repiten caídas por clase. Un arranque más estable solo respalda la hipótesis de estabilidad. Reservar `test` para el procedimiento final del proyecto. Un cambio del scheduler o del factor final queda como hipótesis futura, condicionada a la trayectoria observada en estas fases.

### Métricas

- **Principal.** mAP@0,50:0,95 COCO en `valid` de cada `best` reevaluado. Registrar AP50, AP75, AP y soporte por clase. Calcular también el promedio de AP sin Ibuphil 600 mg como análisis de sensibilidad.
- **Complementarias.** Precisión, recall, TP, FP y FN a score ≥ 0,25 e IoU ≥ 0,50. Conservar las curvas PR, latencia y FPS del benchmark común y ejemplos de las mismas imágenes.
- **Estabilidad.** Medir el pico de `val/cls_loss` en épocas 1–5, el número de épocas con mAP exactamente cero en 1–5 y 1–30, el mAP medio en 1–5, la mejor época y la brecha entre su mAP y el de la época 30. Representar las pérdidas de `train` y `valid` y el mAP de las 30 épocas.
- **LR observado.** Usar las columnas `lr/pg0` (sesgos), `lr/pg1` (pesos con decay) y `lr/pg2` (grupo sin decay) cuando estén disponibles. Registran el LR al final de cada época. Si se reconstruye la interpolación por minibatch, etiquetarla como teórica y conservarla separada de las muestras registradas.

Informar deltas variante menos control por semilla, media y desviación estándar muestral de los brazos y de los deltas. Con una semilla, la desviación estándar queda indefinida. [Protocolo COCO y benchmark](../baseline/README.md#g-métricas-y-protocolo-de-inferencia)

## Ejecución en CESGA

Ejecutar los bloques de esta sección en Bash. Si la sesión se abre en Zsh, iniciar `bash` antes de definir las variables o funciones.

### Preparar una tanda

Tras revisar y versionar EXP-006, desplegar una copia nueva del commit exacto siguiendo la [guía de transferencia](../../docs/cesga.md#transferencia-y-actualización-del-código). Conservarla durante toda la tanda. En CESGA, usar Bash desde esa raíz y definir las [rutas de la guía](../../docs/cesga.md#acceso-y-almacenamiento). Asignar a `EXP_COMMIT` el SHA completo devuelto por `git rev-parse HEAD` en el clon desde el que se transfirió la copia. Si no contiene `.git`, conservar `source-commit` con las huellas del código.

El bloque prepara una copia privada de los archivos declarados en el manifiesto para aislar la caché de etiquetas de Ultralytics. Los pesos y el entorno siguen siendo los instalados. Conservar el valor de `EXP_ROOT` para las fases posteriores.

```bash
export EXP_ROOT
EXP_ROOT=$(mktemp -d "${BASELINE_RESULTS:?}/exp-006-yolo-warmup-XXXXXX")
(
  set -euo pipefail
  : "${EXP_ROOT:?}"
  [[ "${EXP_COMMIT:?}" =~ ^[0-9a-f]{40}$ ]]
  printf '%s\n' "$EXP_COMMIT" > "$EXP_ROOT/source-commit"
  mkdir "$EXP_ROOT/data" "$EXP_ROOT/slurm"
  "$YOLO_ENV/bin/python" -c 'import json, sys; print("manifest.json", *sorted(json.load(open(sys.argv[1]))["files"]), sep="\n")' \
    "$BASELINE_DATA/manifest.json" > "$EXP_ROOT/data.files"
  rsync -a --files-from="$EXP_ROOT/data.files" "$BASELINE_DATA/" "$EXP_ROOT/data/"
  sha256sum scripts/*.py slurm/yolo.sh configs/baseline/yolo.json \
    configs/experiments/yolo-warmup-{3,5}.json > "$EXP_ROOT/code-config.sha256"
  sha256sum "$EXP_ROOT/data/manifest.json" "$EXP_ROOT/data/coco/valid.json" "$YOLO_WEIGHTS" \
    > "$EXP_ROOT/inputs.sha256"
) && printf '%s\n' "$EXP_ROOT"
```

### Enviar entrenamientos

Definir esta función en la sesión Bash de envío. Cada brazo tiene una salida, configuración de Ultralytics, configuración efectiva e ID propios. Dentro de una pareja, el segundo entrenamiento espera a que termine el primero para evitar escrituras simultáneas en la caché de etiquetas de la tanda.

```bash
train_exp() (
  set -euo pipefail
  : "${EXP_ROOT:?}"
  seed=$1
  shift
  sha256sum --check --quiet "$EXP_ROOT/code-config.sha256" "$EXP_ROOT/inputs.sha256"
  dep=()
  for arm; do
    case "$arm" in
      control) config=configs/baseline/yolo.json ;;
      warmup-3|warmup-5) config="configs/experiments/yolo-$arm.json" ;;
      *) exit 2 ;;
    esac
    mkdir -p "$EXP_ROOT/s$seed"
    slot="$EXP_ROOT/s$seed/$arm"
    mkdir "$slot"
    "$YOLO_ENV/bin/python" -m scripts.train_baseline yolo --check-config \
      --config "$config" --seed "$seed" > "$slot/config.json"
    printf '%s\n' "$slot/train" > "$slot/train.path"
    job=$(PILLS_YOLO_CONFIG_DIR="$slot/ultralytics" sbatch --parsable --export=ALL "${dep[@]}" \
      --output="$EXP_ROOT/slurm/%x-%j.out" --error="$EXP_ROOT/slurm/%x-%j.err" slurm/yolo.sh \
      --data "$EXP_ROOT/data" --weights "$YOLO_WEIGHTS" --config "$config" \
      --seed "$seed" --output "$slot/train")
    printf '%s\n' "${job%%;*}" | tee "$slot/train.job"
    dep=(--dependency="afterok:${job%%;*}")
  done
)

train_exp 2026 control warmup-5
```

La última línea inicia A. Si se aprueba B, usar `train_exp 2026 warmup-3` tras terminar A. Para C, asignar a `SELECTED` el brazo elegido, ejecutar `train_exp 2027 control "$SELECTED"` y, cuando ambos trabajos terminen, `train_exp 2028 control "$SELECTED"`. Consultar el estado con `sacct -j ID --allocations --format=JobID,State,ExitCode` y los archivos de `$EXP_ROOT/slurm/`. Conservar las salidas fallidas antes de repetir en una tanda nueva. [Seguimiento SLURM](../../docs/cesga.md#ejecución-de-trabajos-slurm)

### Evaluar sobre la misma GPU

Solicitar una [sesión interactiva en una A100](../../docs/cesga.md#sesión-gpu-interactiva) con 32 CPU. Cargar los [módulos de la guía](../../docs/cesga.md#entornos-y-dependencias) dentro de la asignación. Restablecer `EXP_ROOT` con la ruta de la tanda y situarse en la misma copia del código. Comprobar todos los brazos antes de iniciar la evaluación.

```bash
eval_exp() (
  set -euo pipefail
  : "${EXP_ROOT:?}" "${SLURM_JOB_ID:?Requiere una asignación GPU}"
  phase=$1 seed=$2
  shift 2
  sha256sum --check --quiet "$EXP_ROOT/code-config.sha256" "$EXP_ROOT/inputs.sha256"
  unset PYTHONHOME PYTHONPATH
  export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONNOUSERSITE=1
  out="$EXP_ROOT/eval-$phase-s$seed"
  mkdir "$out"
  export YOLO_CONFIG_DIR="$out/ultralytics"
  py="$YOLO_ENV/bin/python"
  data="$EXP_ROOT/data"
  for arm; do
    slot="$EXP_ROOT/s$seed/$arm"
    state=$(sacct -j "$(cat "$slot/train.job")" --allocations --noheader --parsable2 --format=State,ExitCode)
    [[ "$state" == 'COMPLETED|0:0' ]]
    printf '%s\n' "$state" > "$out/$arm.train.sacct"
    "$py" -c 'import json, sys; assert json.load(open(sys.argv[1]))["status"] == "complete"' "$slot/train/status.json"
  done
  printf '%s\n' "$SLURM_JOB_ID" > "$out/eval.job"
  for arm; do
    run="$EXP_ROOT/s$seed/$arm/train"
    "$py" -m scripts.evaluate_baseline --data "$data" --run "$run" \
      --split valid --checkpoint best --benchmark --warmup 20 --iterations 100 \
      --examples 0 --output "$out/$arm"
    "$py" - "$data" "$run" "$out/$arm" <<'PY'
from pathlib import Path
import sys
from scripts.baseline_data import read_json, sha256
from scripts.baseline_metrics import coco_eval, operating_point, save_examples
from scripts.verify_baseline import verify_evaluation, verify_run

data, run, output = map(Path, sys.argv[1:])
coco = data / "coco/valid.json"
truth = read_json(coco)
digest = sha256(data / "manifest.json")
_, best = verify_run(run, digest)
metrics = verify_evaluation(output, truth, sha256(coco), digest, "valid")
if best != metrics["checkpoint_sha256"]:
    raise ValueError("La evaluación no corresponde a best")
ev = coco_eval(truth, read_json(output / "predictions.json"))
_, examples = operating_point(ev)
ids = (57, 61, 80)
folder = output / "examples"
save_examples(ev, {i: examples[i] for i in ids}, data / "images/valid", folder, len(ids))
for image in folder.glob("*.jpg"):
    image.rename(folder / f"image-{image.name.split('-', 1)[1]}")
if {image.name for image in folder.glob("*.jpg")} != {f"image-{i}.jpg" for i in ids}:
    raise ValueError("Faltan ejemplos emparejados")
PY
  done
  touch "$out/verified.ok"
)

eval_exp a 2026 control warmup-5
```

En B, usar `eval_exp b 2026 control warmup-5 warmup-3` para medir todos los brazos comparables en la misma asignación. En C, usar `eval_exp c 2027 control "$SELECTED"` y `eval_exp c 2028 control "$SELECTED"` tras comprobar cada pareja. Los ejemplos fijan los IDs 57, 61 y 80 de `valid` antes de ver resultados. Si una evaluación falla, conservar su directorio bajo otro nombre antes de reintentar. Los checkpoints permanecen en CESGA.

## Recuperación y reproducción del análisis

Recuperar la selección del [inventario de resultados](results/README.md#recuperación-selectiva) o usar una tanda del [archivo de evidencia](results/README.md#evidencia-para-versionar). Desde la raíz del clon local, iniciar `bash` si el shell habitual es Zsh. Definir `LOCAL_EXP` con la ruta de la tanda y, si procede, `EXP_DATASET` con la ruta del dataset COCO original. Para verificar las métricas basta con su anotación `valid/_annotations.coco.json`.

El bloque comprueba las huellas de los artefactos y extrae en un directorio temporal el commit de `source-commit`, que debe estar disponible en el clon. Los comandos funcionan desde un HEAD posterior al commit de entrenamiento. El análisis reconstruye el COCO con el mapeo del manifiesto, coteja las huellas del original y del COCO preparado, y llama a `verify_evaluation()` para recalcular mAP, AP por clase y curvas PR sobre las predicciones exportadas. El informe indica `coco_verified: true` si todas las evaluaciones de la comparación superan la comprobación. [Verificador común](../../scripts/verify_baseline.py)

```bash
(
  set -euo pipefail
  source=$(realpath "${LOCAL_EXP:?}")
  dataset=$(realpath "${EXP_DATASET:-data/pills-sxdht}")
  sums=downloaded.sha256
  if [[ -f "$source/evidence.sha256" ]]; then sums=evidence.sha256; fi
  (cd "$source" && sha256sum --check --quiet "$sums")
  revision=$(git rev-parse --verify "$(cat "$source/source-commit")^{commit}")
  checkout=$(mktemp -d)
  trap 'rm -rf -- "$checkout"' EXIT
  git archive "$revision" | tar -x -C "$checkout"
  mkdir -p results
  out=$(mktemp -d "$PWD/results/exp-006-analysis-XXXXXX")
  cd "$checkout"
  sha256sum --check --quiet "$source/code-config.sha256"
  MPLBACKEND=Agg uv run --offline --no-project --python 3.11 \
    --with pycocotools==2.0.8 --with numpy==1.26.4 \
    --with pillow==10.4.0 --with matplotlib==3.9.4 \
    python -m experiments.exp-006-yolo-warmup.analyze --source "$source" --data "$dataset" \
    --variant warmup-5 --pair 2026 a --plots --output "$out/a"
  printf '%s\n' "$out/a"
)
```

Si faltan dependencias en la caché de `uv`, quitar `--offline` para preparar el entorno fijado. Para B, comparar cada variante con `--pair 2026 b` y una salida distinta. Para C, usar la variante elegida, añadir `--pair 2027 c --pair 2028 c` a su pareja exploratoria y usar otra salida. El informe distingue todas las parejas de las dos confirmaciones y mantiene por separado las métricas de reevaluación y el historial. El resumen sin `--data` sirve para consulta y deja `coco_verified: false`. Registrar la decisión a partir del informe verificado. [Hallazgos](findings.md)
