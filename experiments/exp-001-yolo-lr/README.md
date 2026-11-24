# EXP-001 · Tasa de aprendizaje inicial de YOLOv10-S

**Estado** `cerrado` tras evaluación en `valid`

[Hallazgos y decisión en validación](findings.md) · [Inventario y procedencia](results/README.md)

## Motivación e hipótesis

El AP nulo y el pico de pérdida al inicio del baseline de YOLO motivan una prueba controlada del LR inicial. [Evolución del entrenamiento del baseline](../baseline/findings.md#evolución-del-entrenamiento)

**Hipótesis.** Reducir el LR inicial de 0,001 a 0,0005 disminuirá la inestabilidad de las primeras cinco épocas y aumentará el mAP@0,50:0,95 del checkpoint `best` en `valid` frente al control de la misma semilla. Un arranque más estable con mAP similar respaldaría solo la primera parte.

## Diseño y configuración

| Brazo | LR inicial | Configuración |
| --- | ---: | --- |
| Control | 0,001 | [Configuración del baseline](../../configs/baseline/yolo.json) |
| Variante | 0,0005 | [Configuración EXP-001](../../configs/experiments/yolo-lr-0005.json) |

Ambos brazos comparten los pesos COCO `yolov10s.pt`, los datos preparados, la revisión del código, el entorno, el tipo de GPU y la semilla de cada pareja. Conservan la receta de referencia, incluido el presupuesto de 30 épocas, los aumentos desactivados y la selección de `best` por mAP COCO en `valid`. Se mantienen la política coseno, el factor final y el warm-up, incluido el LR inicial de sesgos de 0,001. Las tasas del scheduler que dependen de `lr` cambian con la variante. [Receta de referencia](../baseline/README.md#f-recetas-e-hiperparámetros)

La primera pareja usa la semilla 2026 y las repeticiones usan 2027 y 2028. Cada control comparte tanda y presupuesto con su variante. La decisión se basa en esas tres parejas.

### Métricas y criterios de comparación

- **Estabilidad temprana.** Comparar el pico de `val/cls_loss`, el número de épocas con AP cero y el mAP de las cinco primeras épocas. Un pico menor acompañado de un AP inicial peor puede indicar adaptación más lenta. Examinar también la diferencia entre el mejor mAP y el de la época 30.
- **Rendimiento en validación.** La métrica primaria es la diferencia pareada de mAP@0,50:0,95 COCO sobre ocho clases entre los checkpoints `best` reevaluados. Registrar las métricas, curvas PR y latencias del protocolo común, con ejemplos de las mismas imágenes. Revisar especialmente `Cipro 500` y `Xyzall 5mg`. Con las predicciones guardadas, barrer los umbrales de score y comparar el recall micro máximo alcanzable con precisión micro ≥ 0,82 a IoU 0,50. [Protocolo COCO](../baseline/README.md#g-métricas-y-protocolo-de-inferencia)
- **Decisión.** Favorecer la variante si el mAP mejora en al menos dos de las tres parejas y en la media, con las diferencias por semilla y por clase a la vista. Una mejora restringida al arranque indica estabilidad. Un efecto pequeño, variable o acompañado por caídas repetidas en las clases de interés queda como evidencia inconclusa para adoptar la variante.

## Ejecución de referencia

La revisión ejecutada fue `e7ad7adb512b5f5274428087c5bb328200f8b7f0`. El procedimiento de referencia usa el [entorno YOLO](../../docs/cesga.md#entornos-y-dependencias), los [datos y pesos preparados](../../docs/cesga.md#datos-y-pesos) y los [logs persistentes de SLURM](../../docs/cesga.md#ejecución-de-trabajos-slurm). Desde la raíz de esa revisión, crear una salida nueva por tanda y guardar `EXP_ROOT` y los ID de trabajo. Para cada repetición, cambiar `EXP_SEED` y enviar los dos brazos dentro de la misma tanda.

```bash
export EXP_ROOT="$(mktemp -d "$BASELINE_RESULTS/exp-001-yolo-lr-XXXXXX")"
printf '%s\n' "$EXP_ROOT"
export EXP_SEED=2026

sbatch --parsable "${SLURM_LOGS[@]}" slurm/yolo.sh \
  --data "$BASELINE_DATA" --weights "$YOLO_WEIGHTS" \
  --config configs/baseline/yolo.json --seed "$EXP_SEED" \
  --output "$EXP_ROOT/control-s$EXP_SEED"
sbatch --parsable "${SLURM_LOGS[@]}" slurm/yolo.sh \
  --data "$BASELINE_DATA" --weights "$YOLO_WEIGHTS" \
  --config configs/experiments/yolo-lr-0005.json --seed "$EXP_SEED" \
  --output "$EXP_ROOT/variante-s$EXP_SEED"
```

Cuando ambos trabajos terminen con `COMPLETED`, código `0:0` y estado `complete` en `status.json`, evaluar sus checkpoints `best` en una misma [sesión GPU interactiva](../../docs/cesga.md#sesión-gpu-interactiva), con los módulos de la guía cargados. Para comparar las dos ejecuciones de YOLO, use directamente el [evaluador común](../../scripts/evaluate_baseline.py).

```bash
(
  set -e
  unset PYTHONHOME PYTHONPATH
  export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONNOUSERSITE=1
  for brazo in control variante; do
    "$YOLO_ENV/bin/python" -m scripts.evaluate_baseline \
      --data "$BASELINE_DATA" --run "$EXP_ROOT/$brazo-s$EXP_SEED" \
      --split valid --checkpoint best --benchmark --warmup 20 --iterations 100 \
      --output "$EXP_ROOT/eval-$brazo-s$EXP_SEED"
  done
)
```

## Reproducción local del análisis

Ejecutar esta orden desde la raíz del repositorio con el dataset COCO original en `data/pills-sxdht` y las dependencias ya disponibles en la caché de `uv`.

```bash
MPLBACKEND=Agg uv run --offline --no-project \
  --with pycocotools==2.0.8 --with numpy==1.26.4 \
  --with pillow==10.4.0 --with matplotlib==3.9.4 \
  python -m experiments.exp-001-yolo-lr.analyze
```

El análisis verifica los artefactos descargados, contrasta la revisión local y recalcula COCO a partir de las predicciones exportadas y las anotaciones de `valid`. La opción `--data` permite indicar otra copia local del dataset. Las salidas se regeneran en [`results/analysis/`](results/README.md#análisis-derivado). [Código del análisis](analyze.py)
