# Artefactos de EXP-001

Cada brazo tiene `training/` y `evaluation/`. [El inventario](provenance/downloaded.tsv) conserva la ruta de descarga, la ruta local actual y el SHA-256 de los 69 archivos recuperados, que se movieron byte a byte.

| Semilla | Control, ID de entrenamiento | Variante, ID de entrenamiento | ID de evaluación | Carpeta de evaluación en la descarga |
| --- | --- | --- | --- | --- |
| 2026 | [`11953121`](s2026/control/training/slurm.job) | [`11953122`](s2026/variante/training/slurm.job) | [`11974093`](s2026/control/evaluation/metrics.json) | `s2026/eval-valid-jrDSk6/{control,variante}` |
| 2027 | [`11974352`](s2027/control/training/slurm.job) | [`11974353`](s2027/variante/training/slurm.job) | [`11976857`](s2027/control/evaluation/metrics.json) | `repeticiones/eval-valid-7fEylD/{control,variante}-s2027` |
| 2028 | [`11974354`](s2028/control/training/slurm.job) | [`11974355`](s2028/variante/training/slurm.job) | [`11976857`](s2028/control/evaluation/metrics.json) | `repeticiones/eval-valid-7fEylD/{control,variante}-s2028` |

- `training/` contiene la configuración recuperada, el ID SLURM, la invocación y el entorno en `run.json`, el estado y la selección de checkpoint en `status.json`, los argumentos efectivos de Ultralytics y la evolución en `history.csv` y `training.png`. En 2026 los argumentos de configuración apuntan a las rutas del repositorio. En las repeticiones apuntan a los JSON materializados en la carpeta de la tanda.
- `evaluation/` contiene las métricas y muestras de latencia, las predicciones COCO con cajas en píxeles y las curvas PR en JSON y PNG. `metrics.json` guarda la ruta y huella del checkpoint, el entorno evaluador, las imágenes medidas y el ID SLURM.
- `provenance/` conserva los registros de revisión de las tandas [2026](provenance/s2026/source-commit) y [2027–2028](provenance/repeticiones/source-commit) y las [huellas de entrada registradas en 2026](provenance/s2026/inputs.sha256). Los metadatos de las seis parejas de entrenamiento/evaluación coinciden con esas huellas. `git_head` figura como `null` en los JSON. Los archivos Python se identifican por sus hashes, cotejados con el commit local que declaran los dos `source-commit`.

Las raíces de entrenamiento registradas en CESGA están bajo `/mnt/netapp2/Store_uni/home/usc/cursos/curso1014/pills-baseline/results/`. La tanda 2026 usó `exp-001-s2026-8t2uBp` y la de 2027–2028 usó `exp-001-reps-2ELqgV`. Cada una contiene sus ejecuciones `{control,variante}-s<semilla>`, y las rutas originales completas de `weights/best.pt` quedan en `metrics.json`. Se conservan los nombres de descarga de las carpetas de evaluación en la tabla. Sus rutas absolutas de salida no figuran en los metadatos recuperados.

## Análisis derivado

- [`analysis/resumen.csv`](analysis/resumen.csv) contiene métricas globales, AP por clase y diagnósticos de detección y entrenamiento. Cada fila muestra control, variante y diferencia (variante menos control) por semilla, más las medias y desviaciones estándar muestrales de los tres pares. La columna `n_valid` indica el soporte para las métricas por clase.
- [`analysis/ibuphil.csv`](analysis/ibuphil.csv) incluye todas las predicciones de la imagen de `Ibuphil 600 mg`, todas las exportadas como esa clase y las que tienen IoU ≥ 0,50 con alguna anotación de `Ibuphil Cold 400-60`. `prediction_index` es la posición en `predictions.json` empezando por cero. Se incluyen la caja `[x, y, ancho, alto]`, el score, la anotación de mayor IoU y el mayor IoU con una anotación Cold de esa imagen.
- [`analysis/entrenamiento.png`](analysis/entrenamiento.png) compara los historiales de las tres parejas.

[Reproducción local](../README.md#reproducción-local-del-análisis) · [Interpretación](../findings.md)
