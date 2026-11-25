# EXP-009 A · Operación y recuperación

Esta campaña envía **29 trabajos**: 1 preflight, 14 entrenamientos y 14 evaluaciones. El preflight ejecuta la prueba nativa de cajas y composición, un sondeo de la baseline y dos smokes de tres épocas con datos reales y pesos COCO. Los smokes cubren la transición y la congelación permanente. Cada entrenamiento espera al preflight por `afterok`. Cada evaluación espera a su propio entrenamiento. Las evaluaciones trabajan con `best` en `valid`. [Protocolo y criterios de selección](../../experiments/exp-009-final-optimization/README.md)

Se reutiliza la asignación de campañas previa, una A100, 32 CPU y 64 GB por job. EXP-004 midió 16,50–16,72 min y 4,76 GiB de pico asignado en los controles relevantes de 100 épocas. Usar 16,7 minutos para los 14 entrenamientos da unas **3 h 54 min GPU acumuladas**, antes de preflight y evaluaciones. La congelación prolongada puede reducir ese tiempo. EXP-007 empleó 3,40–3,93 min para freeze-20 a freeze-30 con presupuesto de 30, frente a aproximadamente cinco minutos para controles de 30 de campañas previas. Como previsión de planificación, reservar **3,5–4,5 horas GPU acumuladas** para A. Las asignaciones pueden solaparse. La previsión no incluye esperas de cola. [EXP-004](../../experiments/exp-004-yolo-factorial/findings.md) · [EXP-007](../../experiments/exp-007-yolo-staged-transfer/findings.md)

## Transferencia y lanzamiento

En el portátil, desde la raíz del checkout de EXP-009:

```bash
python3 -m scripts.campaign --exp009-a plan
(cd results && sha256sum -c exp-009-a.tar.gz.sha256)
scp results/exp-009-a.tar.gz{,.sha256} USUARIO_CESGA@ft3.cesga.es:
```

En CESGA:

```bash
(
  set -euo pipefail
  cd "$HOME"
  sha256sum -c exp-009-a.tar.gz.sha256
  mkdir "$HOME/pills-exp009-a"
  tar -xzf exp-009-a.tar.gz -C "$HOME/pills-exp009-a"
  bash "$HOME/pills-exp009-a/slurm/campaign.sh" exp009-a launch
)
```

El trabajo usa `$STORE/pills-baseline` o el valor explícito de `PILLS_CAMPAIGN_WORK`, con su entorno `envs/yolo`, dataset preparado y pesos COCO. La raíz de campaña predeterminada es `results/exp-009-a` dentro de ese árbol. Usar el mismo `PILLS_CAMPAIGN_WORK` en las operaciones posteriores. Se puede elegir otra raíz con `--root` al lanzar, consultar o exportar.

El bundle archiva el manifiesto y los fuentes con SHA-256 aunque la rama de trabajo tenga cambios locales. Su identificador es la huella del `snapshot.json` extraído. Para volver a empaquetar tras editar el código, usar un nombre nuevo con `python3 -m scripts.campaign --exp009-a bundle RUTA_NUEVA.tar.gz`, generar su archivo `.sha256` y usar un directorio de despliegue y una raíz de resultados nuevos. El control de artefactos compara receta, pesos y datos con lo declarado al inicializar la campaña.

## Seguimiento y recuperación

Desde el checkout desplegado en CESGA:

```bash
bash slurm/campaign.sh exp009-a status
bash slurm/campaign.sh exp009-a launch --retry-failed
bash slurm/campaign.sh exp009-a collect "$HOME/exp009-a-results.tar.gz"
```

Antes de reintentar, consultar `jobs.json`, `logs/` y `sacct -j ID --format=JobID,JobName,State,ExitCode,Elapsed`. Los trabajos activos o completados conservan su intento. Los fallos se reintentan en un directorio nuevo. Una evaluación fallida se puede repetir desde el entrenamiento completado. Los entrenamientos interrumpidos se reintentan desde COCO, conservando los artefactos previos. Para un envío de estado incierto usar `reconcile NODO --job-id ID` tras verificar su nombre en SLURM. Si se confirma que nunca se envió, usar `reconcile NODO --confirmed-not-submitted`.

`collect` usa un destino nuevo y exporta predicciones, curvas PR, historiales, inventario de parámetros, auditoría de entrenamiento, semillas, estados y huellas. Los checkpoints y las imágenes permanecen en su ubicación original. Puede recopilarse un estado parcial, pero el análisis final exige las 14 cadenas completas.

## Análisis local

```bash
scp USUARIO_CESGA@ft3.cesga.es:exp009-a-results.tar.gz results/
python3 -m scripts.campaign_analysis.import_results results/exp009-a-results.tar.gz results/exp009-a-import
snapshot=$(tar -xOf results/exp-009-a.tar.gz snapshot.json | sha256sum | cut -d' ' -f1)
MPLBACKEND=Agg uv run --no-project --python 3.11 \
  --with pycocotools==2.0.8 --with numpy==1.26.4 --with pillow==10.4.0 \
  --with matplotlib==3.9.4 --with pyyaml==6.0.4 \
  python -m experiments.exp-009-final-optimization.analyze \
  results/exp009-a-import results/exp009-a-analysis --snapshot-sha256 "$snapshot" \
  --data /ruta/a/pills-sxdht
```

Usar destinos nuevos. El analizador exige las anotaciones originales, pero puede generar tablas sin imágenes disponibles. Produce `resumen.csv`, `factorial.csv`, AP por clase, confusiones, historiales, ventanas de transición, curvas y `fase-b-propuesta.json`. La propuesta queda para revisión y no actúa como lanzador de B.

Para reproducir las figuras complementarias de interacción, AP por clase, descongelación y coste, usar el mismo entorno Python del analizador:

```bash
python -m experiments.exp-009-final-optimization.diagnostics results/exp009-a-import results/exp009-a-analysis
```
