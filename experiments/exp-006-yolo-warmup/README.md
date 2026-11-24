# EXP-006 · Duración del warm-up de YOLOv10-S

**Estado** `cerrado` en fase A. Se conserva el warm-up de la baseline. Las fases B y C quedaron sin ejecutar.

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

La tanda `exp-006-yolo-warmup-wqn08F` comparó el control (`11994691`) y `warmup-5` (`11994693`) con semilla 2026. El entrenamiento usó `slurm/yolo.sh` de la [revisión registrada](results/phase-a/source-commit). Se conservan la configuración efectiva, las huellas de datos y pesos y el estado de cada ejecución. [Inventario de fase A](results/README.md)

### Evaluar sobre la misma GPU

El trabajo `12001925` evaluó ambos checkpoints `best` en `valid` con el lanzador de la revisión `11b223b9a494e97886e715c0b043702ebdd24ba5` desplegado aparte. El evaluador y el verificador se ejecutaron desde la copia del código de entrenamiento. `launch.json` registra las huellas de ambos archivos del lanzador, los checkpoints y los parámetros del benchmark. El envío histórico se puede consultar con `git show 11b223b:experiments/exp-006-yolo-warmup/README.md`. [Lanzamiento registrado](results/phase-a/eval-a-s2026/launch.json)

## Recuperación y reproducción del análisis

Desde la raíz del clon local, ejecutar el bloque en Bash. Por defecto usa la evidencia archivada de la fase A. Se puede indicar otra copia con `LOCAL_EXP` y ubicar el dataset COCO original mediante `EXP_DATASET`. Para el recálculo basta con su anotación `valid/_annotations.coco.json`. [Inventario de resultados](results/README.md#recuperación-selectiva)

El bloque comprueba las huellas de los artefactos y extrae en un directorio temporal el commit de `source-commit`, que debe estar disponible en el clon. Los comandos funcionan desde un HEAD posterior al commit de entrenamiento. El análisis reconstruye el COCO con el mapeo del manifiesto, coteja las huellas del original y del COCO preparado, y llama a `verify_evaluation()` para recalcular mAP, AP por clase y curvas PR sobre las predicciones exportadas. El informe indica `coco_verified: true` si todas las evaluaciones de la comparación superan la comprobación. [Verificador común](../../scripts/verify_baseline.py)

```bash
(
  set -euo pipefail
  source=$(realpath "${LOCAL_EXP:-experiments/exp-006-yolo-warmup/results/phase-a}")
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

Si faltan dependencias en la caché de `uv`, quitar `--offline` para preparar el entorno fijado. Comparar el `summary.json` regenerado con el [resumen archivado](results/analysis/summary.json). El informe sin `--data` queda marcado con `coco_verified: false`.
