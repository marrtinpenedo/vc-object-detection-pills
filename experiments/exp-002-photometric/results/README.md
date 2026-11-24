# Artefactos de EXP-002

[Vista previa sobre las ocho clases de entrenamiento](preview.jpg) · [Cómo regenerarla](../README.md#reproducción-local-de-la-vista-previa)

Al recuperar resultados, conservar `source-commit`, `inputs.sha256`, los cuatro IDs de SLURM, la raíz de la tanda y las rutas originales de evaluación. Organizar la copia en `s2026/<detector>/<brazo>/training/` y `s2026/<detector>/<brazo>/evaluation/`.

- De la verificación previa, conservar `code-config.sha256`, `preflight.ok`, los logs y los informes `initialization.json` de ambos detectores.
- En entrenamiento, guardar la configuración efectiva, `run.json`, `status.json`, el historial y el registro del entorno nativo. En YOLO, `ultralytics_runtime.json` enumera las transformaciones de la variante. En Cascade, `detectron2.yaml` registra el bloque fotométrico.
- En evaluación, guardar `metrics.json`, incluida la muestra de latencias, `predictions.json`, curvas PR y ejemplos pertinentes de `valid`. Mantener las huellas y las rutas de los checkpoints en la procedencia.
