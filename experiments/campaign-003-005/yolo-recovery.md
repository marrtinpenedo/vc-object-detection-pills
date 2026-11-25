# Recuperación del preflight YOLO 11994101

El mock anterior devolvía 64 o 32 para toda llamada a `random.uniform`. En Ultralytics 8.3.161, `BaseMixTransform` llama primero a `uniform(0, 1)` y sale si el número supera `p`. Con `p=1`, el test continuaba con la imagen original y la afín, sin componer Mosaic. Una reconstrucción local de los JPEG de prueba reprodujo los 6 717 elementos distintos de 12 288. Los sorteos separados del test fijan la activación, el centro Y/X y el paso de BGR a RGB de `Format`. La cadena es `Mosaic → afín → LetterBox → Format` en entrenamiento y `LetterBox → Format` en validación. Se conserva la igualdad exacta de píxeles para el caso determinista y se añaden comprobaciones explícitas de clases, cajas recortadas, colación e integridad de las anotaciones. La receta, el código de entrenamiento y los sondeos con inicialización COCO, forward y backward conservan sus huellas. [Test nativo](../../tests/test_geometry.py) · [Implementación fijada de Ultralytics](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/data/augment.py)

## Transferir y reintentar

Desde la raíz del repositorio en el portátil, transferir **solo** el tar incremental. Contiene `scripts/campaign.py`, `tests/test_geometry.py` y el descriptor `snapshot.json`.

```bash
scp results/campaign-003-005-yolo-recovery.delta.tar.gz USUARIO_CESGA@ft3.cesga.es:
```

En CESGA, conservar la copia desplegada inicialmente y extraer el parche sobre una copia nueva. El ejemplo usa las rutas del despliegue documentado. Si se usó otra ruta de código, ajustar `original`. Si el lanzamiento inicial usó `PILLS_CAMPAIGN_WORK` o `--root`, proporcionar exactamente los mismos valores al ejecutar los comandos de campaña.

```bash
delta="$HOME/campaign-003-005-yolo-recovery.delta.tar.gz"
original="$HOME/pills-campaign-003-005"
corrected="$HOME/pills-campaign-003-005-yolo-recovery"
printf '%s  %s\n' '33786bed75f0f903fe78c64d38611e1b72059e809c715ac56796926780f19d65' "$delta" |
  sha256sum -c - &&
  mkdir "$corrected" &&
  cp -a "$original/." "$corrected/" &&
  tar -xzf "$delta" -C "$corrected" &&
  bash "$corrected/slurm/campaign.sh" recover-yolo --failed-preflight 11994101
```

`recover-yolo` mantiene la raíz de resultados original. Comprueba las huellas de ambos snapshots, que las únicas diferencias de código sean el lanzador y el test, y que manifiesto, rutas, recetas, datos, pesos y grafo coincidan. Confirma con SLURM el fallo original de 11994101 y que los intentos YOLO originales estén fallidos o cancelados y sin recibos de éxito. Registra el descriptor corregido y el SHA-256 completo por intento en `jobs.json`. Solo considera los once nodos YOLO y conserva los recibos, intentos y resultados de Cascade. Una vez completado el preflight corregido, los cinco entrenamientos se liberan por `afterok` y cada evaluación espera su propio entrenamiento. Un estado original desconocido impide el reenvío.

Para consultar o repetir el envío, usar la copia corregida. Repetir el comando de recuperación si contiene intentos fallidos recuperables. Conserva los intentos activos o completos. Un envío ambiguo se trata con `reconcile` como en la guía general. `launch --retry-failed` pertenece al flujo de una única versión y el lanzador corregido rechaza usarlo con el manifiesto original.

```bash
cd "$HOME/pills-campaign-003-005-yolo-recovery"
bash slurm/campaign.sh status
bash slurm/campaign.sh recover-yolo --failed-preflight 11994101
```

Al terminar, exportar desde la copia corregida para que el tar de resultados incluya los dos descriptores y el registro completo:

```bash
bash slurm/campaign.sh collect "$HOME/campaign-003-005-results-recovered.tar.gz"
```

El [análisis local](README.md#análisis-local) audita cada cadena contra su propio snapshot. Para el parámetro `--snapshot-sha256`, usar la huella del descriptor **original**. El informe `auditoria.json` registra además el descriptor corregido y la huella de cada cadena completa.

## Huellas de despliegue

| Elemento | SHA-256 |
| --- | --- |
| Descriptor original | `2b98ece5c736735e4602a992c1847958be74cbee6a52b3daa03143cf91f1b3d9` |
| Descriptor corregido | `23e048675002dd0bbd678b184ef674288813a10517179fedb32914caad762978` |
| Tar incremental a transferir | `33786bed75f0f903fe78c64d38611e1b72059e809c715ac56796926780f19d65` |
| Tar completo corregido de respaldo | `692a9222996bc2b6f31df4428906f089bc6e3ce8cc780976c4c460bb4470f95a` |
