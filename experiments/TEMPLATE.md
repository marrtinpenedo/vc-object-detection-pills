# EXP-NNN · Título

**Estado** `propuesto`

## Motivación e hipótesis

Enlazar la evidencia que motiva el cambio. Formular el efecto esperado y qué resultado cuestionaría la hipótesis.

## Diseño y criterios de comparación

Fijar control, variante, factor modificado, constantes, semillas, repeticiones y criterio de decisión antes de ejecutar. Usar `train` para entrenar y `valid` para seleccionar checkpoints, comparar variantes y analizar errores. Reservar `test` para la evaluación final con la receta y el punto operativo fijados.

Indicar la métrica principal y las comprobaciones específicas. Aplicar el [protocolo COCO común](baseline/README.md#g-métricas-y-protocolo-de-inferencia) para AP global y por clase con soporte, curvas PR, latencia comparable y ejemplos de las mismas imágenes. Ajustar el enlace al copiar la plantilla a la carpeta del experimento.

## Ejecución y artefactos

Enlazar la configuración y el comando o procedimiento. Registrar la configuración efectiva, la revisión del código, las huellas de datos y pesos, el entorno y hardware, los ID de SLURM y las rutas de salida. Usar una salida nueva por ejecución.

## Resultados

**Pendientes.** Enlazar los artefactos y presentar las diferencias frente al control por semilla, con métricas, curvas y ejemplos.

## Conclusiones y decisión

**Pendientes.** Interpretar los resultados y sus limitaciones. Registrar si se adopta, repite o descarta la variante y actualizar el índice de experimentos.
