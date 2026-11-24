# Experimentos de detección de pastillas

| ID | Detector y comparación | Estado |
| --- | --- | --- |
| `BASE-YOLO` | YOLOv10-S, ajuste completo sin aumentos | `evaluado` en `valid` |
| `BASE-CASCADE` | Cascade R-CNN con ResNet-50 + FPN, ajuste completo sin aumentos | `evaluado` en `valid` |
| [EXP-001](exp-001-yolo-lr/README.md) | YOLOv10-S, tasa de aprendizaje inicial | `cerrado` tras evaluación en `valid` · [hallazgos](exp-001-yolo-lr/findings.md) |
| [EXP-002](exp-002-photometric/README.md) | YOLOv10-S y Cascade R-CNN, augmentación fotométrica controlada | `cerrado` tras evaluación en `valid` · [hallazgos](exp-002-photometric/findings.md) |
| EXP-003 | Augmentación geométrica moderada | Planificado, sujeto a viabilidad |
| EXP-004 | Ensemble de YOLOv10-S y Cascade R-CNN | Propuesto con predicciones guardadas |

[Metodología de los baselines](baseline/README.md) · [Resultados, diagnósticos y decisiones](baseline/findings.md)

## Próximos pasos

1. **EXP-004.** Probar una regla de fusión por clase y unos parámetros fijados de antemano sobre las predicciones `valid` de los dos controles de EXP-002. Comparar con el mAP de Cascade control, incluidas las siete clases con más soporte y el punto operativo. La prueba reutiliza las cajas exportadas y evita entrenar. El margen de aciertos exclusivos de YOLO es estrecho al corte operativo, así que conviene limitar la búsqueda de parámetros. [Diagnóstico de EXP-002](exp-002-photometric/findings.md#conclusiones)
2. **EXP-003.** Si la fusión queda empatada o pierde y hay GPU antes de la entrega, priorizar Cascade. Probar un cambio geométrico moderado (p. ej., traslación y escala acotadas con actualización de cajas) solo en `train`. Mantener pesos, semilla, presupuesto y reglas de selección. Verificar las cajas transformadas y el cargador de `valid`, y usar un control emparejado dentro de la misma revisión. [Anotaciones señaladas](baseline/findings.md#limitaciones-del-dataset)
3. Reservar tiempo para fijar recetas, checkpoints y punto operativo y evaluar en `test`. Si la fusión se selecciona en `valid`, generar las predicciones `test` de sus dos componentes y aplicar la regla congelada. Dejar warm-up extendido, Copy-Paste y más pruebas para después de esta evaluación.

La evaluación final en `test` se reserva hasta fijar las recetas, los checkpoints y el punto operativo.

## Registro de un experimento

Crear `experiments/exp-NNN-descripcion/` a partir de la [plantilla](TEMPLATE.md). Usar `README.md` para el protocolo reproducible, `findings.md` para resultados, interpretación y decisión, y `results/` para los artefactos. Identificar el brazo y la semilla de cada repetición. Usar `propuesto` para un diseño documentado, `en ejecución` desde su primer trabajo, `evaluado` cuando consten resultados e interpretación en `valid`, y `cerrado` cuando se hayan completado las evaluaciones previstas y registrado la decisión.
