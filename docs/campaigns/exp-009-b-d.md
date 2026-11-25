# EXP-009 B y D · Despliegues independientes

B tiene **9 trabajos**. Un preflight YOLO valida el cargador con ejemplos deterministas, hace un forward/backward en el control con fotometría y entrena un smoke de tres épocas con pesos COCO y la composición completa. El smoke abre el backbone en la época 2 y exige 40 pasos de AdamW del backbone en esa época. Cuatro trabajos entrenan las únicas celdas nuevas. Cada evaluación depende de su entrenamiento. D tiene **5 trabajos**. Un preflight Detectron2 verifica el cargador, los pesos y el forward/backward; lo siguen dos pares entrenamiento/evaluación. Todos usan dependencias `afterok` mientras el padre esté activo y comprueban su recibo de éxito antes de empezar. [B](../../configs/campaign/exp-009-b.json) · [D](../../configs/campaign/exp-009-d.json)

Los paquetes llevan el snapshot histórico completo de A o EXP-008, byte por byte. Un inventario separado identifica el controlador y el plan. El ledger conserva los intentos, las versiones y los IDs SLURM. Cada despliegue valida el entorno, la preparación de datos y los pesos, y tiene su propia raíz. La documentación científica fija cómo comparar los brazos. [Diseño](../../experiments/exp-009-final-optimization/README.md) · [Auditoría de A](../../experiments/exp-009-final-optimization/thaw-audit.md)

## Transferencia y lanzamiento

Desde la raíz del checkout local de EXP-009. Sustituir `USUARIO_CESGA` por la cuenta propia. Para lanzar solo una fase, transferir únicamente sus dos archivos.

```bash
(cd results && sha256sum -c exp-009-b.tar.gz.sha256 exp-009-d.tar.gz.sha256)
scp results/exp-009-{b,d}.tar.gz{,.sha256} USUARIO_CESGA@ft3.cesga.es:
```

Desde CESGA, seleccionar `fase=b` o `fase=d`. Repetir únicamente el bloque de despliegue para la otra fase si se quiere ejecutarla en paralelo.

```bash
(
  set -euo pipefail
  fase=b
  cd "$HOME"
  sha256sum -c "exp-009-${fase}.tar.gz.sha256"
  mkdir "$HOME/pills-exp009-${fase}"
  tar -xzf "exp-009-${fase}.tar.gz" -C "$HOME/pills-exp009-${fase}"
  bash "$HOME/pills-exp009-${fase}/campaign.sh" launch
)
```

Se usa `$STORE/pills-baseline` con el entorno, los datos y los pesos existentes; se puede fijar `PILLS_CAMPAIGN_WORK=/ruta/a/pills-baseline` si están en otro árbol. Las raíces predeterminadas son `results/exp-009-b` y `results/exp-009-d` dentro de ese árbol. Cada operación acepta `--root` para usar una raíz explícita. La GPU se usa exclusivamente en los jobs SLURM. A partir de las cuatro referencias de A, presupuestar alrededor de **60–70 min GPU acumulados** para B. EXP-008 midió 39,36 min para el control de 100 épocas, por lo que D requiere aproximadamente **80–90 min GPU acumulados** con preflight y evaluaciones. Se solicita una A100, 32 CPU y 64 GB por trabajo según campañas previas. Las asignaciones pueden solaparse y la previsión excluye cola.

## Seguimiento, reintento y exportación

En CESGA, mantener el mismo valor de `PILLS_CAMPAIGN_WORK` y la misma raíz que en el lanzamiento. Elegir `fase=b` o `fase=d`.

```bash
fase=b
controlador="$HOME/pills-exp009-${fase}/campaign.sh"
bash "$controlador" status
# Si hay fallos confirmados y se desea reintentar
bash "$controlador" launch --retry-failed
# Cuando la fase haya terminado
bash "$controlador" collect "$HOME/exp009-${fase}-results.tar.gz"
(cd "$HOME" && sha256sum "exp009-${fase}-results.tar.gz" > "exp009-${fase}-results.tar.gz.sha256")
```

Consultar `logs/`, `jobs.json` y `sacct -j ID --format=JobID,JobName,State,ExitCode,Elapsed` antes de reintentar. Reintentar conserva trabajos activos y completos. Las evaluaciones fallidas pueden reutilizar su entrenamiento. Un entrenamiento fallido comienza otro intento desde los pesos COCO y deja el directorio anterior disponible. Si el envío carece de ID confirmado, usar `bash "$controlador" reconcile NODO --job-id ID` tras confirmar el nombre con SLURM. Si se confirma que el envío nunca ocurrió, usar `--confirmed-not-submitted`. Una raíz rechaza cambios de controlador o de snapshot. Para código revisado usar un nuevo paquete y una raíz diferente.

## Importación y análisis local

Usar un entorno CPU con las dependencias del [analizador de A](exp-009-a.md#análisis-local) y el checkout de EXP-009. Para D puede extraerse la referencia archivada de EXP-008 al destino local propuesto, o utilizar su importación ya verificada. Destinos de importación y análisis deben ser nuevos.

```bash
fase=b
scp "USUARIO_CESGA@ft3.cesga.es:exp009-${fase}-results.tar.gz" \
    "USUARIO_CESGA@ft3.cesga.es:exp009-${fase}-results.tar.gz.sha256" results/
(cd results && sha256sum -c "exp009-${fase}-results.tar.gz.sha256")
python -m scripts.campaign_analysis.import_results "results/exp009-${fase}-results.tar.gz" "results/exp009-${fase}-import"
tar -xOf "results/exp-009-${fase}.tar.gz" controller.json > "results/exp009-${fase}-controller.json"
```

Para B, usar las referencias de A ya importadas.

```bash
MPLBACKEND=Agg python -m experiments.exp-009-final-optimization.followup B \
  results/exp009-b-import results/exp009-a-import-20261125 data/pills-sxdht \
  results/exp009-b-analysis results/exp009-b-controller.json
```

Para D, ejecutar tras descargar su export siguiendo el bloque anterior con `fase=d`.

```bash
python -m scripts.campaign_analysis.import_results \
  experiments/exp-008-cascade-freezing/results/evidence/exp008-validation.tar.xz results/exp008-reference
MPLBACKEND=Agg python -m experiments.exp-009-final-optimization.followup D \
  results/exp009-d-import results/exp008-reference data/pills-sxdht \
  results/exp009-d-analysis results/exp009-d-controller.json
```

El análisis comprueba el inventario SHA-256, los `best`, las cadenas de trabajos y la equivalencia de las recetas efectivas. Reproduce COCO y PR desde las predicciones. Guarda resumen, AP por clase, métricas operativas, latencias, historiales, figuras y auditoría. B desglosa efectos e interacciones y D calcula media y dispersión entre semillas. Las latencias describen asignaciones independientes. Los checkpoints permanecen en CESGA.

## Regeneración local

El empaquetador usa el tar histórico de A en `results/exp-009-a.tar.gz` o el tar versionado de EXP-008. Verifica el inventario de la referencia y sus cadenas completas. Genera el tar y el fichero `.sha256` en un destino nuevo. B requiere NumPy para validar la política geométrica.

```bash
python -m scripts.exp009_followup.bundle B results/exp009-a-import-20261125 /tmp/exp-009-b.tar.gz
python -m scripts.exp009_followup.bundle D results/exp008-reference /tmp/exp-009-d.tar.gz
```
