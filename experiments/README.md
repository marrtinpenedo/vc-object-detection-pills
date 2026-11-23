# Experimentos de detección de pastillas

| ID | Detector y comparación | Estado |
| --- | --- | --- |
| `BASE-YOLO` | YOLOv10-S, ajuste completo sin aumentos | `evaluado` en `valid` |
| `BASE-CASCADE` | Cascade R-CNN con ResNet-50 + FPN, ajuste completo sin aumentos | `evaluado` en `valid` |
| [EXP-001](exp-001-yolo-lr/README.md) | YOLOv10-S, tasa de aprendizaje inicial | `propuesto`, resultados pendientes |

[Metodología de los baselines](baseline/README.md) · [Resultados, diagnósticos y decisiones](baseline/findings.md)

## Próximos pasos

1. Ejecutar EXP-001 con las comparaciones pareadas previstas.
2. Revisar las [anotaciones señaladas](baseline/findings.md#limitaciones-del-dataset) y evaluar por separado las [hipótesis de aumentos](baseline/findings.md#interpretaciones-e-hipótesis-priorizadas).
3. Evaluar en `test` una vez fijados las recetas, los checkpoints y el punto operativo.

## Registro de un experimento

Crear `experiments/exp-NNN-descripcion/README.md` a partir de la [plantilla](TEMPLATE.md). Registrar las repeticiones por variante y semilla en la misma ficha y actualizar su estado en el índice. Usar `propuesto` para un diseño documentado, `en ejecución` desde su primer trabajo, `evaluado` cuando consten resultados e interpretación en `valid`, y `cerrado` cuando se hayan completado las evaluaciones previstas y registrado la decisión.
