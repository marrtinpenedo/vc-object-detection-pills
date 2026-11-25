# EXP-009 C · Confirmación final de YOLO

C usa el snapshot inmutable de EXP-009 A y B, `8254e3e0bfa68bebfd6e8a8090b8c8a55fb759989bdf345b6954762357114ef3`. El paquete comprueba los 20 archivos del snapshot, la receta de cada referencia de 2026 y sus hashes de inventario, predicciones y `best`. Cuatro referencias corresponden a A y dos a B. El manifiesto incluye sus IDs de preflight, entrenamiento y evaluación. [Manifiesto C](../../configs/campaign/exp-009-c.json) · [Resultados que motivan la confirmación](../../experiments/exp-009-final-optimization/findings.md#fase-c--diseño-definitivo-propuesto)

| ID | Freezing | Afín + Mosaic | Fotometría | Referencia de 2026 |
| --- | ---: | :---: | :---: | --- |
| A | 0 | — | — | `f000-control` |
| B | 0 | Sí | — | `f000-geom` |
| C | 80 | — | — | `f080-control` |
| D | 80 | Sí | — | `f080-geom` |
| E | 80 | — | Sí | `f080-control-photo` |
| F | 80 | Sí | Sí | `f080-geom-photo` |

Se ejecutan A–F con las semillas 2027 y 2028. La receta cambia exclusivamente la semilla de su referencia. Todas usan 100 épocas, COCO, batch 8, resolución 640, FP32, AdamW con LR 0,001, warm-up original y scheduler coseno. La política conjunta usa Mosaic 0,30 durante todo el entrenamiento y la afín histórica. La fotometría conserva ±10 % y probabilidad 0,5. Los brazos congelados reactivan el backbone en la época 81 con el mecanismo histórico. El `best` se elige por el mAP interno en validación.

El DAG consta de **25 trabajos**: un preflight, 12 entrenamientos y 12 evaluaciones. El preflight ejecuta la prueba nativa del dataloader, inicializa el control con pesos COCO y hace un smoke con afín + Mosaic + fotometría. El smoke dura tres épocas y exige la descongelación en la época 2 con 0, 40 y 80 pasos acumulados del backbone. Los entrenamientos dependen del preflight y cada evaluación de su propio entrenamiento por `afterok` mientras su padre sigue activo. La asignación histórica pide una A100, 32 CPU y 64 GB por job. Los tiempos internos de las seis referencias suman unos 82 min. Para dos semillas, preflight y evaluación, presupuestar alrededor de **3 horas GPU acumuladas**, con ejecución paralela según disponibilidad y sin incluir cola.

## Despliegue y lanzamiento

En el portátil, desde la raíz del checkout EXP-009:

```bash
(cd results && sha256sum -c exp-009-c.tar.gz.sha256)
scp results/exp-009-c.tar.gz{,.sha256} USUARIO_CESGA@ft3.cesga.es:
```

En CESGA:

```bash
(
  set -euo pipefail
  cd "$HOME"
  sha256sum -c exp-009-c.tar.gz.sha256
  mkdir "$HOME/pills-exp009-c"
  tar -xzf exp-009-c.tar.gz -C "$HOME/pills-exp009-c"
  bash "$HOME/pills-exp009-c/campaign.sh" launch
)
```

El lanzador usa `$STORE/pills-baseline` o el árbol especificado con `PILLS_CAMPAIGN_WORK`. La raíz de resultados predeterminada es `results/exp-009-c` dentro de ese árbol. Las operaciones de seguimiento y exportación deben usar la misma raíz y el mismo controlador.

## Seguimiento y recuperación

En CESGA:

```bash
controlador="$HOME/pills-exp009-c/campaign.sh"
bash "$controlador" status
bash "$controlador" launch --retry-failed
bash "$controlador" collect "$HOME/exp009-c-results.tar.gz"
(cd "$HOME" && sha256sum exp009-c-results.tar.gz > exp009-c-results.tar.gz.sha256)
```

Usar el reintento tras comprobar los fallos en `logs/`, `jobs.json` y `sacct -j ID`. Se conservan los intentos anteriores y se reutilizan los entrenamientos completados si falla su evaluación. Para una solicitud de envío sin ID, reconciliar con `bash "$controlador" reconcile NODO --job-id ID` tras comprobar el nombre en SLURM. Si se confirma que nunca se envió, usar `--confirmed-not-submitted`. El export contiene los recibos, el inventario SHA-256, las huellas de `best`, las predicciones, las PR, los historiales, las auditorías de descongelación, el tiempo interno y la memoria. Los checkpoints permanecen en CESGA.

## Comprobación del código de salida

El tar incluye estados SLURM y recibos de los trabajos. Para certificar `COMPLETED 0:0` hace falta la salida de contabilidad. En CESGA, para los 25 IDs del primer intento de la campaña ejecutada:

```bash
sacct -X -n -P -j "$(seq -s, 12031892 12031916)" -o JobIDRaw,State,ExitCode > "$HOME/exp009-c-sacct.txt"
```

Copiar ese archivo al checkout local como `results/exp009-c-sacct.txt`. El analizador exige una fila `COMPLETED|0:0` por cada ID aceptado del ledger. `--diagnostic` permite reproducir métricas sin el extracto y marca la selección como provisional. Usar un directorio nuevo al ejecutar el análisis certificado.

## Importación y análisis local

Usar el [entorno CPU de A](exp-009-a.md#análisis-local). La importación y el análisis requieren destinos nuevos. Las importaciones de A y B se reutilizan.

```bash
scp USUARIO_CESGA@ft3.cesga.es:exp009-c-results.tar.gz \
    USUARIO_CESGA@ft3.cesga.es:exp009-c-results.tar.gz.sha256 \
    USUARIO_CESGA@ft3.cesga.es:exp009-c-sacct.txt results/
(cd results && sha256sum -c exp009-c-results.tar.gz.sha256)
python -m scripts.campaign_analysis.import_results results/exp009-c-results.tar.gz results/exp009-c-import
tar -xOf results/exp-009-c.tar.gz controller.json > results/exp009-c-controller.json
MPLBACKEND=Agg python -m experiments.exp-009-final-optimization.confirmation \
  results/exp009-c-import results/exp009-a-import-20261125 results/exp009-b-import \
  data/pills-sxdht results/exp009-c-analysis-verified results/exp009-c-controller.json \
  --slurm-accounting results/exp009-c-sacct.txt
```

El análisis exige los 25 trabajos completos y valida sus recibos, reproduce las **18 evaluaciones** COCO y sus PR desde las predicciones y verifica las recetas efectivas al cambiar la semilla. Genera métricas por semilla, descriptivos de tres semillas, AP por clase, curvas y diagnósticos de la época de thaw, `best` y última época. Los contrastes muestran 2026 como exploración y 2027/2028 como confirmación, con diferencias individuales y la media de las dos semillas nuevas. El archivo `candidato-yolo-valid.json` registra la receta y el checkpoint propuestos, con su ruta, semilla, SHA-256 y evidencia para la revisión final. [Protocolo de selección y comparación final](../../experiments/exp-009-final-optimization/final-evaluation.md)

Para combinar el análisis certificado con D y generar el registro de los dos modelos:

```bash
MPLBACKEND=Agg python -m experiments.exp-009-final-optimization.closeout \
  results/exp009-c-analysis-verified results/exp009-c-import \
  results/exp009-d-analysis results/exp009-d-import results/exp009-final-selection-verified
```

Para regenerar exactamente el paquete en un destino nuevo:

```bash
python -m scripts.exp009_followup.bundle C results/exp009-a-import-20261125 /tmp/exp-009-c.tar.gz \
  --reference-b results/exp009-b-import
```
