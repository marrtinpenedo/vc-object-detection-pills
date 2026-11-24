# EXP-002 · Augmentación fotométrica controlada

**Estado** `propuesto`, implementación preparada.

[Verificaciones y hallazgos](findings.md) · [Artefactos](results/README.md)

## Motivación e hipótesis

Una variación moderada de iluminación durante el entrenamiento puede mejorar la generalización de YOLOv10-S y Cascade R-CNN sin perjudicar las señales cromáticas de las clases. La hipótesis se contrasta por arquitectura con su propio control. [Limitaciones observadas en el baseline](../baseline/findings.md#limitaciones-del-dataset)

## Diseño

| Detector | Control | Variante | LR en ambos brazos | Semilla | Épocas |
| --- | --- | --- | ---: | ---: | ---: |
| YOLOv10-S | [Baseline](../../configs/baseline/yolo.json) | [Fotometría](../../configs/experiments/yolo-photometric.json) | 0,001 | 2026 | 30 |
| Cascade R-CNN | [Baseline](../../configs/baseline/cascade.json) | [Fotometría](../../configs/experiments/cascade-photometric.json) | 0,0025 | 2026 | 30 |

Ejecutar cuatro trabajos con la misma revisión, datos preparados y tipo de GPU. Dentro de cada arquitectura se mantienen los pesos iniciales, optimizador, planificador, warm-up, lote, resolución, precisión, posprocesado y selección del checkpoint `best` por COCO en `valid`. Usar controles nuevos en la misma revisión para estimar el efecto de la augmentación. [Recetas de referencia](../baseline/README.md#f-recetas-e-hiperparámetros)

Solo las variantes incluyen el bloque `photometric`. En cada lectura de una imagen de `train` se aplica con probabilidad 0,5. Las ganancias independientes de brillo y contraste se muestrean uniformemente en [0,90, 1,10]. A partir de Y, la luminancia codificada del píxel, y μ, la media de la imagen, se calcula Y′ = brillo × (μ + contraste × (Y − μ)). Se aplica el cociente Y′/Y a los tres canales por igual, con un límite común para mantener el canal de mayor valor entre 1 y 255. Los píxeles negros se mantienen. Las proporciones entre canales conservan el tono y la saturación HSV salvo el redondeo a 8 bits. Las ganancias acotan los factores por separado y el efecto conjunto en un píxel depende de su luminancia. El RNG fotométrico es independiente del muestreador del entrenamiento y usa la semilla del worker o la de la ejecución cuando no hay workers.

- **YOLO.** `YOLODataset(..., augment=False)` conserva `LetterBox → Format`. La variante inserta un único paso fotométrico antes de `LetterBox` y solo para el modo `train`. Mosaic, MixUp, CutMix, Copy-Paste, albumentations, HSV, flips y demás augmentaciones nativas quedan fuera de la cadena. Multi-scale sigue desactivado.
- **Cascade.** Solo `build_train_loader` de la variante antepone la transformación al `ResizeShortestEdge` del mapper base. La transformación deja intactas las coordenadas y las máscaras. El resize sigue siendo fijo y los flips, recorte y TTA permanecen desactivados.
- Ambos brazos consumen el mismo dataset en disco. Los mappers de validación conservan el preprocesado base. `test` queda reservado.

### Comparación en validación

- **Principal.** Informar Δ de mAP COCO@0,50:0,95 de las ocho clases en `valid` por arquitectura. Comparar los checkpoints `best` reevaluados con el mismo protocolo.
- **Clases de color.** Informar el AP de `blue`, `pink`, `red` y `white` y su media sin ponderar. Para una arquitectura, la hipótesis inicial recibe apoyo si sube el mAP oficial y la media de estas cuatro clases se mantiene o sube. Registrar todas las pérdidas individuales y revisar las confusiones antes de proponer adopción.
- **Diagnósticos.** Acompañar de AP50, AP75, AP por clase con soporte, curvas PR, latencias y ejemplos comunes. Incluir la sensibilidad sin `Ibuphil 600 mg` como comprobación complementaria. Una pareja por arquitectura es exploratoria y no estima variabilidad entre semillas. [Protocolo COCO común](../baseline/README.md#g-métricas-y-protocolo-de-inferencia)

## Reproducción local de la vista previa

Desde la raíz del repositorio, con NumPy, Pillow, Matplotlib y el dataset local, ejecutar esta orden. La figura usa una imagen de `train` por clase y dos ganancias fijas en los extremos para hacer visibles los cambios. Incluye las cajas COCO originales en los tres paneles.

```bash
MPLBACKEND=Agg .venv/bin/python -m experiments.exp-002-photometric.preview
```

## Ejecución en CESGA

Ejecutar desde la raíz de una copia de la misma revisión para los cuatro trabajos. Definir las variables de almacenamiento y pesos y configurar los logs persistentes según la guía. [Almacenamiento](../../docs/cesga.md#acceso-y-almacenamiento) · [SLURM](../../docs/cesga.md#ejecución-de-trabajos-slurm)

### Verificar antes de enviar

Ejecutar desde la raíz de la copia que contiene EXP-002, con `EXP_SOURCE_COMMIT` apuntando al commit desplegado. Usar los entornos existentes, los pesos COCO locales y el dataset preparado de la guía. Pedir una sola [GPU A100 interactiva](../../docs/cesga.md#sesión-gpu-interactiva) y cargar los módulos de CESGA dentro de esa asignación. Las pruebas de cargadores usan CPU. El último paso usa la GPU para inicializar las dos variantes con sus recetas completas.

| Prueba omitida en local | Entorno y dependencia que activa la prueba |
| --- | --- |
| `test_photometric.YoloPhotometricLoaderTest.test_train_pixels_change_but_labels_and_validation_do_not` | YOLO, Ultralytics 8.3.161 |
| `test_photometric.CascadePhotometricLoaderTest.test_train_pixels_change_but_labels_and_validation_do_not` | Cascade, Detectron2 0.6 |
| `test_baseline.CascadeInitializationTest.test_initialization_has_event_storage_and_disables_determinism` | Cascade, Detectron2 0.6 y su `EventStorage` |

Para cada detector se comprueba que el control entrega los píxeles originales con su preprocesado habitual. Ocho lecturas consecutivas de la variante a probabilidad 0,5 deben incluir píxeles aumentados y sin aumentar. En validación se comprueba que control y variante coinciden. También se comprueban las coordenadas y etiquetas de cada lote en entrenamiento. YOLO las conserva en validación, y el registro de anotaciones de Cascade sigue intacto tras pasar por sus mappers. La prueba unitaria de Cascade comprueba `EventStorage`, gradientes y que la inicialización se detiene antes de `trainer.train()`.

`--initialize` usa los pesos y datos reales. Hace una inferencia y un backward sobre un lote de entrenamiento sin ejecutar épocas ni pasos de optimizador. Usar directorios separados para esas comprobaciones y para los entrenamientos de 30 épocas.

```bash
export EXP_ROOT="$(mktemp -d "${BASELINE_RESULTS:?Define BASELINE_RESULTS}/exp-002-photometric-XXXXXX")"
(
  set -euo pipefail
  : "${EXP_ROOT:?Falta la carpeta nueva de la tanda}"
  unset PYTHONHOME PYTHONPATH
  export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONNOUSERSITE=1
  export YOLO_CONFIG_DIR="$BASELINE_WORK/cache/ultralytics"
  mkdir "$EXP_ROOT/preflight"
  printf '%s\n' "$EXP_ROOT"
  printf '%s\n' "${EXP_SOURCE_COMMIT:?Indica el commit desplegado}" > "$EXP_ROOT/source-commit"
  sha256sum scripts/*.py tests/test_{baseline,photometric}.py \
    configs/baseline/{yolo,cascade}.json configs/experiments/{yolo,cascade}-photometric.json \
    slurm/{yolo,cascade}.sh > "$EXP_ROOT/code-config.sha256"
  sha256sum "$BASELINE_DATA/manifest.json" "$BASELINE_DATA/coco/valid.json" \
    "$YOLO_WEIGHTS" "$CASCADE_WEIGHTS" > "$EXP_ROOT/inputs.sha256"

  run_required_tests() {
    local py="$1"
    shift
    PYTHONPATH="$PWD/tests" "$py" - "$@" <<'PY_TESTS'
import sys
import unittest

suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.testsRun and result.wasSuccessful() and not result.skipped else 1)
PY_TESTS
  }

  "$YOLO_ENV/bin/python" -c 'import cv2, torch, ultralytics; assert ultralytics.__version__ == "8.3.161"; print(torch.__version__, cv2.__version__, ultralytics.__version__)'
  run_required_tests "$YOLO_ENV/bin/python" \
    test_photometric.PhotometricTest test_photometric.YoloPhotometricLoaderTest \
    2>&1 | tee "$EXP_ROOT/preflight/yolo-tests.log"

  "$CASCADE_ENV/bin/python" -c 'import cv2, torch, detectron2, detectron2._C; assert detectron2.__version__ == "0.6"; print(torch.__version__, cv2.__version__, detectron2.__version__)'
  run_required_tests "$CASCADE_ENV/bin/python" \
    test_photometric.PhotometricTest test_photometric.CascadePhotometricLoaderTest \
    test_baseline.CascadeInitializationTest \
    2>&1 | tee "$EXP_ROOT/preflight/cascade-tests.log"

  for detector in yolo cascade; do
    case "$detector" in
      yolo) py="$YOLO_ENV/bin/python"; weights="$YOLO_WEIGHTS" ;;
      cascade) py="$CASCADE_ENV/bin/python"; weights="$CASCADE_WEIGHTS" ;;
    esac
    "$py" -m scripts.train_baseline "$detector" --initialize \
      --data "$BASELINE_DATA" --weights "$weights" \
      --config "configs/experiments/$detector-photometric.json" \
      --output "$EXP_ROOT/preflight/$detector" \
      2>&1 | tee "$EXP_ROOT/preflight/$detector-initialize.log"
  done

  "$CASCADE_ENV/bin/python" - "$EXP_ROOT/preflight" <<'PY_REPORTS'
import json
import math
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
for detector in ("yolo", "cascade"):
    output = root / detector
    run = json.loads((output / "run.json").read_text())
    status = json.loads((output / "status.json").read_text())
    report = json.loads((output / "initialization.json").read_text())
    spec = json.loads(Path(f"configs/experiments/{detector}-photometric.json").read_text())
    assert run["mode"] == "initialize" and run["run"]["config"] == spec, detector
    assert status["status"] == "initialized" and report["classes"] == 8, detector
    assert report["losses"] and all(map(math.isfinite, report["losses"].values())), detector
    assert "nonzero" in report["gradients"].values(), detector

    if detector == "yolo":
        runtime = json.loads((output / "ultralytics_runtime.json").read_text())
        assert runtime["photometric"] == spec["photometric"]
        assert runtime["train_transforms"] == ["YoloPhotometric", "LetterBox", "Format"]
        assert not runtime["args"]["augment"]
    else:
        cfg = yaml.safe_load((output / "detectron2.yaml").read_text())
        assert cfg["INPUT"]["PHOTOMETRIC"] == spec["photometric"]
        assert cfg["INPUT"]["RANDOM_FLIP"] == "none" and not cfg["INPUT"]["CROP"]["ENABLED"]
    print(f"{detector}: inicialización, ocho clases y fotometría verificadas")
PY_REPORTS
  printf 'OK\n' > "$EXP_ROOT/preflight.ok"
)
```

Los envíos requieren `preflight.ok` en esa misma tanda. Se genera solo cuando las pruebas obligatorias terminan sin omisiones, ambos sondeos devuelven `initialized` con pérdidas finitas, gradientes útiles y ocho clases, y los registros confirman la configuración fotométrica. Si falla un paso, revisar el log de `preflight/` correspondiente antes de usar el bloque de entrenamiento.

### Entrenar

Desde la raíz de la misma copia, conservar o volver a exportar `EXP_ROOT` con la ruta impresa en la verificación. Definir las variables de almacenamiento y logs si se abrió una sesión nueva. El envío comprueba las huellas de la revisión, las recetas, los datos y los pesos registrados en la comprobación.

```bash
(
  set -euo pipefail
  grep -qx OK "${EXP_ROOT:?Indica la tanda verificada}/preflight.ok"
  sha256sum --check --quiet "$EXP_ROOT/code-config.sha256"
  sha256sum "$BASELINE_DATA/manifest.json" "$BASELINE_DATA/coco/valid.json" \
    "$YOLO_WEIGHTS" "$CASCADE_WEIGHTS" | cmp - "$EXP_ROOT/inputs.sha256"

  for detector in yolo cascade; do
    case "$detector" in
      yolo) weights="$YOLO_WEIGHTS" ;;
      cascade) weights="$CASCADE_WEIGHTS" ;;
    esac
    for arm in control variante; do
      case "$arm" in
        control) config="configs/baseline/$detector.json" ;;
        variante) config="configs/experiments/$detector-photometric.json" ;;
      esac
      run="$detector-$arm-s2026"
      cp -- "$config" "$EXP_ROOT/$run.config.json"
      sbatch --parsable "${SLURM_LOGS[@]}" "slurm/$detector.sh" \
        --data "$BASELINE_DATA" --weights "$weights" \
        --config "$EXP_ROOT/$run.config.json" --output "$EXP_ROOT/$run" \
        > "$EXP_ROOT/$run.job"
      cat "$EXP_ROOT/$run.job"
    done
  done
)
```

### Evaluar en validación

Confirmar `COMPLETED` y `0:0` en SLURM y `complete` en los cuatro `status.json`. Usar una misma [sesión GPU interactiva](../../docs/cesga.md#sesión-gpu-interactiva) para medir ambos brazos de cada modelo. Definir `EXP_ROOT` con el valor de la tanda y cargar los módulos de la guía. El bloque crea una carpeta distinta por intento y comprueba cada par de evaluaciones del mismo brazo con el verificador existente.

```bash
(
  set -euo pipefail
  unset PYTHONHOME PYTHONPATH
  export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONNOUSERSITE=1
  export YOLO_CONFIG_DIR="$BASELINE_WORK/cache/ultralytics"
  eval_root="$(mktemp -d "$EXP_ROOT/eval-valid-XXXXXX")"
  printf '%s\n' "$eval_root"

  for detector in yolo cascade; do
    case "$detector" in
      yolo) py="$YOLO_ENV/bin/python" ;;
      cascade) py="$CASCADE_ENV/bin/python" ;;
    esac
    for arm in control variante; do
      run="$detector-$arm-s2026"
      "$py" -m scripts.evaluate_baseline --data "$BASELINE_DATA" \
        --run "$EXP_ROOT/$run" --split valid --checkpoint best \
        --benchmark --warmup 20 --iterations 100 --output "$eval_root/$run"
    done
  done

  for arm in control variante; do
    "$CASCADE_ENV/bin/python" -m scripts.verify_baseline --data "$BASELINE_DATA" --split valid \
      --runs "$EXP_ROOT/cascade-$arm-s2026" "$EXP_ROOT/yolo-$arm-s2026" \
      --evaluations "$eval_root/cascade-$arm-s2026" "$eval_root/yolo-$arm-s2026"
  done
)
```

Conservar los registros de la tanda y los [artefactos de entrenamiento y validación](results/README.md).
