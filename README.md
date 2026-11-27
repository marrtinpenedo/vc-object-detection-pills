# Detección de pastillas

Estudio comparativo de dos arquitecturas de detección de objetos, Cascade R-CNN (two-stage) y YOLOv10-S (one-stage), preentrenadas en MS COCO y ajustadas mediante fine-tuning sobre el conjunto de datos `pills-sxdht`. El proyecto establece baselines y compara distintas estrategias de entrenamiento, aumento de datos y congelación de capas mediante experimentos reproducibles ejecutados en el supercomputador FinisTerrae III del CESGA, con gestión de trabajos mediante SLURM. La evaluación incluye precisión, recall, AP, latencia y análisis de errores.

**Autores:** Martín Rodríguez Penedo y Santiago Pereira Olmeda.  
**Equipo:** I.

## Documentación

- [Datos y análisis exploratorio](experiments/baseline/README.md#descarga-y-exploración-local) y [notebook del EDA](notebooks/01_eda.ipynb).
- [Metodología de los baselines](experiments/baseline/README.md) y [hallazgos en validación](experiments/baseline/findings.md).
- [Índice de experimentos](experiments/README.md) con estado y próximos pasos.
- [Resultados finales de EXP-009](experiments/exp-009-final-optimization/final-results.md) y [reproducción a partir de predicciones](experiments/exp-009-final-optimization/results/evidence/README.md).
- [Cierre de EXP-008](experiments/exp-008-cascade-freezing/findings.md). Ninguna variante de freezing mejoró a su control de Cascade en validación.
- [Guía de CESGA](docs/cesga.md) para preparar el entorno, enviar trabajos y recuperar artefactos. El [lanzador de evaluación](docs/cesga.md#evaluación-conjunta) admite uno o varios checkpoints en una misma GPU.
