# Detección de pastillas

Práctica de Visión por Computador del Grado en Inteligencia Artificial de la USC, curso 2026–2027. El equipo I estudia dos detectores preentrenados en MS COCO, Cascade R-CNN y YOLOv10-S, sobre `pills-sxdht`. El proyecto establece baselines y evalúa mejoras mediante precisión, recall, AP, latencia y análisis de errores.

**Autores:** Martín Rodríguez Penedo y Santiago Pereira Olmeda.  
**Equipo:** I.

## Documentación

- [Datos y análisis exploratorio](experiments/baseline/README.md#descarga-y-exploración-local) y [notebook del EDA](notebooks/01_eda.ipynb).
- [Metodología de los baselines](experiments/baseline/README.md) y [hallazgos en validación](experiments/baseline/findings.md).
- [Índice de experimentos](experiments/README.md) con estado y próximos pasos.
- [Cierre de EXP-008](experiments/exp-008-cascade-freezing/findings.md). Ninguna variante de freezing mejoró a su control de Cascade en validación.
- [Guía de CESGA](docs/cesga.md) para preparar el entorno, enviar trabajos y recuperar artefactos. El [lanzador de evaluación](docs/cesga.md#evaluación-conjunta) admite uno o varios checkpoints en una misma GPU.
