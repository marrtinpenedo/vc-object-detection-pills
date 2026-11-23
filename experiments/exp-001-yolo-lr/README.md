# EXP-001 · Tasa de aprendizaje inicial de YOLOv10-S

**Estado** `propuesto`

## Motivación e hipótesis

El AP nulo y el pico de pérdida al inicio del baseline de YOLO motivan una prueba controlada del LR inicial. [Evolución del entrenamiento del baseline](../baseline/findings.md#evolución-del-entrenamiento)

**Hipótesis.** Reducir el LR inicial de 0,001 a 0,0005 disminuirá la inestabilidad de las primeras cinco épocas y aumentará el mAP@0,50:0,95 del checkpoint `best` en `valid` frente al control de la misma semilla. Un arranque más estable con mAP similar respaldaría solo la primera parte.

## Diseño y configuración

| Brazo | LR inicial | Configuración |
| --- | ---: | --- |
| Control | 0,001 | [Configuración del baseline](../../configs/baseline/yolo.json) |
| Variante | 0,0005 | [Configuración EXP-001](../../configs/experiments/yolo-lr-0005.json) |

Ambos brazos comparten los pesos COCO `yolov10s.pt`, los datos preparados, la revisión del código, el entorno, el tipo de GPU y la semilla de cada pareja. Conservan la receta de referencia, incluido el presupuesto de 30 épocas, los aumentos desactivados y la selección de `best` por mAP COCO en `valid`. Se mantienen la política coseno, el factor final y el warm-up, incluido el LR inicial de sesgos de 0,001. Las tasas del scheduler que dependen de `lr` cambian con la variante. [Receta de referencia](../baseline/README.md#f-recetas-e-hiperparámetros)

La primera pareja usará la semilla 2026 y las repeticiones usarán 2027 y 2028. Cada control se ejecutará en la misma tanda que su variante y tendrá el mismo presupuesto. La decisión se basa en esos controles pareados. El baseline publicado da contexto.

### Métricas y criterios de comparación

- **Estabilidad temprana.** Comparar el pico de `val/cls_loss`, el número de épocas con AP cero y el mAP de las cinco primeras épocas. Un pico menor acompañado de un AP inicial peor puede indicar adaptación más lenta. Examinar también la diferencia entre el mejor mAP y el de la época 30.
- **Rendimiento en validación.** La métrica primaria es la diferencia pareada de mAP@0,50:0,95 entre los checkpoints `best` reevaluados. Registrar las métricas, curvas PR y latencias del protocolo común, con ejemplos de las mismas imágenes. Revisar especialmente `Cipro 500` y `Xyzall 5mg`. Con las predicciones guardadas, barrer los umbrales de score y comparar el recall micro máximo alcanzable con precisión micro ≥ 0,82 a IoU 0,50. [Protocolo COCO](../baseline/README.md#g-métricas-y-protocolo-de-inferencia)
- **Decisión.** Favorecer la variante si el mAP mejora en al menos dos de las tres parejas y en la media, con las diferencias por semilla y por clase a la vista. Una mejora restringida al arranque indica estabilidad. Un efecto pequeño, variable o acompañado por caídas repetidas en las clases de interés queda como evidencia inconclusa para adoptar la variante.

### Limitaciones previstas

Un LR más bajo puede necesitar más épocas y quedar por detrás con el mismo presupuesto. Los cambios en el arranque corresponden al cambio de LR bajo la política de warm-up fijada. El soporte de `valid` limita las conclusiones por clase, en particular para `Ibuphil 600 mg`. Las repeticiones estiman sensibilidad a la semilla en la misma partición. Las decisiones se toman en `valid` y la evaluación final se reserva para `test`.

## Ejecución y artefactos

Usar el [entorno YOLO](../../docs/cesga.md#entornos-y-dependencias), los [datos y pesos preparados](../../docs/cesga.md#datos-y-pesos) y los [logs persistentes de SLURM](../../docs/cesga.md#ejecución-de-trabajos-slurm). Desde la raíz de la revisión que se vaya a ejecutar, crear una salida nueva para la tanda. Conservar `EXP_ROOT` y los ID devueltos por `sbatch`. Para cada repetición, cambiar `EXP_SEED` y repetir ambos envíos dentro de la misma tanda.

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

Registrar revisión, trabajos, rutas, huellas de datos y pesos, versiones, GPU y parámetros efectivos. Conservar `run.json`, `status.json`, `ultralytics_runtime.json` e `history.csv` del entrenamiento y `metrics.json`, `predictions.json`, curvas PR y ejemplos de la evaluación. Añadir los enlaces a los artefactos al evaluar la ficha. Repetir una evaluación en una ruta nueva.

## Resultados

**PENDIENTES.** La ficha se encuentra en fase de propuesta.

## Conclusiones y decisión

**PENDIENTES.** Tras comparar las parejas, registrar si se conserva el control, se adopta la variante o se requiere más evidencia. Actualizar el [índice](../README.md).
