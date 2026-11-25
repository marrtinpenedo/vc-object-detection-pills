# Campaña EXP-003 a EXP-005

[Ensemble local](../../../experiments/exp-003-ensemble/README.md) · [Factorial YOLO](../../../experiments/exp-004-yolo-factorial/README.md) · [Factorial Cascade](../../../experiments/exp-005-cascade-geometric/README.md) · [Cierre de validación](findings.md) · [Manifiesto](../../../configs/campaign/exp-003-005.json)

La receta queda congelada tras contrastar los lanzadores históricos de RF100 y auditar las cajas de entrenamiento. [Informe de política geométrica](geometric-policy.md)

## Matriz y dependencias

| Arquitectura | Condiciones GPU | Dependencia |
| --- | --- | --- |
| YOLOv10-S | A/B sin aumentos, C/D afín + Mosaic, E solo afín a 100 épocas | `preflight-yolo → train-exp004-{a,b,c,d,e} → eval-exp004-{a,b,c,d,e}` |
| Cascade R-CNN | A/B sin aumentos, C/D solo afín | `preflight-cascade → train-exp005-{a,b,c,d} → eval-exp005-{a,b,c,d}` |

A y C duran 30 épocas. B, D y E duran 100. Todos los entrenamientos usan semilla 2026. Las evaluaciones recalculan COCO de `valid` a partir de `best` con 20 iteraciones de calentamiento y 100 muestras de latencia. Cada evaluación depende de su propio entrenamiento. En el primer lanzamiento hay **20 trabajos** (2 preflights, 9 entrenamientos, 9 evaluaciones). No hay dependencias entre familias ni condiciones por resultados de mAP. EXP-003 reutiliza predicciones y se completa en CPU local.

`python -m scripts.campaign plan` imprime el grafo sin enviar trabajos. El lanzador guarda un manifiesto resuelto con recetas, rutas, hashes y dependencias, y materializa los JSON para el lanzador base. Verifica el SHA-256 del manifiesto de datos, el COCO de `valid`, los pesos y el snapshot. Los jobs también verifican entradas, receta y procedencia. Los dos preflights ejecutan el contrato de cargador real para su framework y un sondeo con pesos COCO, inferencia y forward/backward de A y C. Se usa C para las transformaciones que también usan D; los presupuestos de 100 épocas se verifican por configuración y auditoría. Las pruebas con mocks quedan en la suite local. El lanzador de login nunca importa el runtime GPU.

## Transferencia y lanzamiento históricos

El [archivo original](../../../experiments/exp-004-yolo-factorial/results/evidence/campaign-003-005.tar.gz) conserva los bytes usados en CESGA y su descriptor `snapshot.json`. El hash del descriptor identifica el snapshot. El commit base indica el punto de partida de un árbol que todavía tenía cambios locales. Para reproducir esa ejecución hay que extraer el archivo original. Un `bundle` desde el checkout integrado genera un snapshot distinto. La [evidencia archivada](../../../experiments/exp-004-yolo-factorial/results/evidence/README.md) conserva las dos versiones y el parche.

```bash
scp experiments/exp-004-yolo-factorial/results/evidence/campaign-003-005.tar.gz USUARIO_CESGA@ft3.cesga.es:
```

Después de la transferencia, ejecutar **una vez** en CESGA. Para un snapshot nuevo usar un directorio de código distinto y añadir `--root RUTA_DE_RESULTADOS_NUEVA` al lanzamiento.

```bash
mkdir "$HOME/pills-campaign-003-005" && tar -xzf "$HOME/campaign-003-005.tar.gz" -C "$HOME/pills-campaign-003-005" && bash "$HOME/pills-campaign-003-005/slurm/campaign.sh" launch
```

El script carga los módulos CESGA del proyecto. Por defecto usa `$STORE/pills-baseline`, con `envs/{yolo,cascade}`, `data/pills-prepared` y `checkpoints`. Si la raíz de ese árbol es otra, prefijar la última orden con `PILLS_CAMPAIGN_WORK=/ruta/absoluta/pills-baseline` y usar el mismo prefijo para los comandos de operación. La raíz por defecto de resultados es `$PILLS_CAMPAIGN_WORK/results/campaign-003-005` o su equivalente en `$STORE`. Los jobs reciben rutas absolutas y cargan módulos por separado.

## Seguimiento, reintentos y recuperación

Para el preflight YOLO fallido `11994101`, seguir la [recuperación YOLO entre snapshots](yolo-recovery.md). Cascade conserva los resultados originales. El flujo ordinario de abajo aplica cuando se mantiene un único snapshot.

Desde el directorio desplegado:

```bash
bash slurm/campaign.sh status
```

El registro `jobs.json` conserva los IDs, el intento y la dependencia. Cada intento escribe en `jobs/<nodo>/attempt-NN/` y recibe su propio log `logs/<nombre>-<ID>.{out,err}`. `done.json` indica que se publicaron los artefactos del intento. Ante un fallo, revisar `status`, `sacct -j ID`, el stderr y el `status.json` del entrenador. La política `--kill-on-invalid-dep=yes` cancela dependencias inviables. Cada evaluación tiene su propia dependencia. [Referencia de dependencias de SLURM](https://slurm.schedmd.com/sbatch.html#OPT_dependency)

```bash
bash slurm/campaign.sh launch --retry-failed
```

Con un bloqueo de lanzamiento se evita el envío simultáneo. Un reintento repite solo los trabajos fallidos, en directorios nuevos, y conserva los activos y completados. Si el padre terminó correctamente se verifica su recibo; un ID que ya salió del controlador puede dejar de admitir un `afterok` nuevo. Un entrenamiento fallido se reinicia desde los pesos COCO. Los parciales y los checkpoints anteriores permanecen disponibles para diagnóstico. Si un envío queda `UNRECORDED`, buscar el nombre exacto con `squeue` y `sacct`. Registrar el ID confirmado con `bash slurm/campaign.sh reconcile NODO --job-id ID`. Si se comprobó que el controlador nunca lo aceptó, registrar `--confirmed-not-submitted` antes de reintentar. Un estado `UNKNOWN` conserva el intento hasta que contabilidad informe su estado.

Para recoger la evidencia, ejecutar en CESGA y descargar únicamente el tar indicado. Se puede exportar antes del final para analizar las condiciones completas.

```bash
bash slurm/campaign.sh collect "$HOME/campaign-003-005-results.tar.gz"
```

Desde el portátil, con la ruta de acceso habitual:

```bash
scp USUARIO_CESGA@ft3.cesga.es:campaign-003-005-results.tar.gz results/
```

El export selectivo guarda configuraciones efectivas, recibos, estados, historiales, métricas, predicciones, PR, hashes de los checkpoints y extractos de logs para fallos. Los checkpoints, los datos de imágenes, los entornos y TensorBoard permanecen en CESGA. Para repetir la inferencia se necesita el `best` cuyo SHA-256 quedó registrado.

## Análisis local

Para la campaña final con los dos snapshots, descargar la exportación indicada en el [cierre de validación](findings.md). El comando de cierre fija las dos huellas y exige nueve cadenas exitosas.

Desde el portátil, importar en un destino nuevo. El paquete compartido se conserva dentro de los [resultados de EXP-004](../../../experiments/exp-004-yolo-factorial/results/evidence/README.md) y también contiene EXP-005. La implementación de análisis, importación y auditoría se trasladó de `experiments/campaign-003-005/` a `scripts/campaign_analysis/`. Los manifiestos de ejecución conservan `configs/campaign/exp-003-005.json`, `scripts/campaign.py`, `slurm/campaign.sh` y las rutas de salida originales.

```bash
.venv/bin/python -m scripts.campaign_analysis.import_results \
  experiments/exp-004-yolo-factorial/results/evidence/campaign-003-005-results-recovered.tar.gz results/campaign-003-005-import
MPLBACKEND=Agg uv run --no-project --python 3.11 \
  --with pycocotools==2.0.8 --with numpy==1.26.4 --with pillow==10.4.0 \
  --with matplotlib==3.9.4 --with pyyaml==6.0.4 \
  python -m scripts.campaign_analysis.analyze \
  results/campaign-003-005-import results/campaign-003-005-analysis
```

Añadir `--snapshot-sha256` con la huella del descriptor original. Para verificar qué intentos ejecutaron el parche, añadir `--recovery-sha256` con la huella del descriptor corregido. Las dependencias son de CPU. Se puede usar `--offline` si están en la caché de `uv`, o el intérprete local que ya las tenga.

El analizador verifica el manifiesto, la cadena de intentos terminados, el COCO reconstruido y las firmas de los checkpoints. Recalcula curvas PR, métricas y percentiles de latencia. Escribe las tablas de rendimiento (`resumen.csv`, `por_clase.csv`, `contrastes.csv` y `aporte_por_clase.csv`), aprendizaje (`historial.csv`, `diagnostico_historial.csv`) y errores (`errores_por_imagen.csv`, `confusiones.csv`, `errores_instancias.csv`, `ibuphil.csv`). Las figuras se generan a partir de esas mediciones locales. Si las imágenes de validación están disponibles, guarda ejemplos anotados por condición. Genera también `auditoria.json`, `geometria_comun_100.csv`, `resumen-ejecutivo.md` y un borrador `findings.md` por detector. Un contraste necesita todas sus condiciones. Las latencias son descriptivas por asignación.

## Recursos y límites

Cada trabajo usa una A100, 32 CPU y 64 GB de RAM, con los límites SLURM existentes de 6 h por entrenamiento y 1 h por preflight o evaluación. Los entrenamientos de 30 épocas en EXP-002 tardaron unos 5,2 min en YOLO y 13,1–13,3 min en Cascade, con aproximadamente 4,76 y 2,95 GiB asignados en GPU. Extrapolando 360 épocas YOLO y 260 Cascade, la base es **~3 h GPU acumuladas**. Planificar **3,2–4 h GPU acumuladas** para incluir Mosaic, sondeos, E/S y evaluaciones. Si las asignaciones llegan en paralelo, el camino crítico será un preflight, un Cascade de 100 épocas (referencia de ~44 min) y su evaluación. Estas cifras excluyen la cola y los reintentos. El coste y la memoria de Mosaic, y la disponibilidad de nodos, se confirman durante la campaña.

`valid` selecciona checkpoints y estrategias. Se conserva `test` para la evaluación final con recetas congeladas. Las dos matrices usan una semilla. Más épocas cambian el horizonte de los schedulers y aumentan el número de checkpoints candidatos. Mosaic es una diferencia de política entre detectores; el contraste afín a 100 épocas aprovecha E en YOLO para reducir esa diferencia.
