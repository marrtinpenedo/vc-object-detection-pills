# Experimentos de detección de pastillas

| ID | Detector y comparación | Estado |
| --- | --- | --- |
| `BASE-YOLO` | YOLOv10-S, ajuste completo sin aumentos | `evaluado` en `valid` |
| `BASE-CASCADE` | Cascade R-CNN con ResNet-50 + FPN, ajuste completo sin aumentos | `evaluado` en `valid` |
| [EXP-001](exp-001-yolo-lr/README.md) | YOLOv10-S, tasa de aprendizaje inicial | `cerrado` tras evaluación en `valid` · [hallazgos](exp-001-yolo-lr/findings.md) |
| [EXP-002](exp-002-photometric/README.md) | YOLOv10-S y Cascade R-CNN, augmentación fotométrica controlada | `cerrado` tras evaluación en `valid` · [hallazgos](exp-002-photometric/findings.md) |
| [EXP-003](exp-003-ensemble/README.md) | Ensemble de YOLOv10-S y Cascade R-CNN con predicciones archivadas | Evaluado en `valid` · [hallazgos](exp-003-ensemble/findings.md) |
| [EXP-004](exp-004-yolo-factorial/README.md) | Factorial épocas × afín y Mosaic en YOLOv10-S con ablación a 100 épocas | Cerrado en `valid` · [hallazgos](exp-004-yolo-factorial/findings.md) |
| [EXP-005](exp-005-cascade-geometric/README.md) | Factorial épocas × afín en Cascade R-CNN | Cerrado en `valid` · [hallazgos](exp-005-cascade-geometric/findings.md) |
| [EXP-006](exp-006-yolo-warmup/README.md) | YOLOv10-S, duración del warm-up con LR inicial fijo | `cerrado` en fase A. Se conserva el warm-up de referencia · [hallazgos](exp-006-yolo-warmup/findings.md) |
| [EXP-007](exp-007-yolo-staged-transfer/README.md) | YOLOv10-S, duración de la congelación del backbone | `cerrado` en `valid`. B confirma 15 épocas y C selecciona 20 · [hallazgos](exp-007-yolo-staged-transfer/findings.md) |
| [EXP-008](exp-008-cascade-freezing/README.md) | Cascade R-CNN, duración de la congelación de ResNet-50 | Cerrado en `valid`, sin confirmación. El freezing no mejora los controles · [hallazgos](exp-008-cascade-freezing/findings.md) |

[Metodología de los baselines](baseline/README.md) · [Resultados, diagnósticos y decisiones](baseline/findings.md)

## Próximos pasos

La numeración de la campaña asigna EXP-003 al ensemble y EXP-004/005 a los factoriales. Las propuestas al cierre de EXP-002 usaban una numeración provisional.

La [campaña EXP-003 a EXP-005](../docs/campaigns/exp-003-005/README.md) completó los dos factoriales. El [cierre de validación](../docs/campaigns/exp-003-005/findings.md) recoge los nueve resultados reproducidos, el contraste con EXP-007 y los candidatos para combinar geometría y congelación en YOLO.

La evaluación final en `test` se reserva hasta fijar las recetas, los checkpoints y el punto operativo.

## Registro de un experimento

Crear `experiments/exp-NNN-descripcion/` a partir de la [plantilla](TEMPLATE.md). Usar `README.md` para el protocolo reproducible, `findings.md` para resultados, interpretación y decisión, y `results/` para los artefactos. Identificar el brazo y la semilla de cada repetición. Usar `propuesto` para un diseño documentado, `en ejecución` desde su primer trabajo, `evaluado` cuando consten resultados e interpretación en `valid`, y `cerrado` cuando se hayan completado las evaluaciones previstas y registrado la decisión.
