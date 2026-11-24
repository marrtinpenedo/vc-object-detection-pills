# Experimentos de detección de pastillas

| ID | Detector y comparación | Estado |
| --- | --- | --- |
| `BASE-YOLO` | YOLOv10-S, ajuste completo sin aumentos | `evaluado` en `valid` |
| `BASE-CASCADE` | Cascade R-CNN con ResNet-50 + FPN, ajuste completo sin aumentos | `evaluado` en `valid` |
| [EXP-001](exp-001-yolo-lr/README.md) | YOLOv10-S, tasa de aprendizaje inicial | `cerrado` tras evaluación en `valid` · [hallazgos](exp-001-yolo-lr/findings.md) |
| [EXP-002](exp-002-photometric/README.md) | YOLOv10-S y Cascade R-CNN, augmentación fotométrica controlada | `propuesto` |
| EXP-003 | Augmentación geométrica moderada | Planificado, sujeto a viabilidad |

[Metodología de los baselines](baseline/README.md) · [Resultados, diagnósticos y decisiones](baseline/findings.md)

## Próximos pasos

1. Ejecutar [EXP-002, augmentación fotométrica controlada](exp-002-photometric/README.md), como siguiente comparación.
2. Preparar EXP-003, augmentación geométrica moderada, si resulta viable. Revisar las [anotaciones señaladas](baseline/findings.md#limitaciones-del-dataset) antes de interpretar su efecto por clase.
3. Considerar como líneas posteriores el [warm-up extendido](exp-001-yolo-lr/findings.md#decisión-y-siguientes-líneas-de-investigación), Copy-Paste y ensemble.

La evaluación final en `test` se reserva hasta fijar las recetas, los checkpoints y el punto operativo.

## Registro de un experimento

Crear `experiments/exp-NNN-descripcion/` a partir de la [plantilla](TEMPLATE.md). Usar `README.md` para el protocolo reproducible, `findings.md` para resultados, interpretación y decisión, y `results/` para los artefactos. Identificar el brazo y la semilla de cada repetición. Usar `propuesto` para un diseño documentado, `en ejecución` desde su primer trabajo, `evaluado` cuando consten resultados e interpretación en `valid`, y `cerrado` cuando se hayan completado las evaluaciones previstas y registrado la decisión.
