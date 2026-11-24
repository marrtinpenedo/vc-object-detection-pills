# Artefactos de EXP-002

[Vista previa sobre las ocho clases de entrenamiento](preview.jpg) · [Cómo regenerarla](../README.md#reproducción-local-de-la-vista-previa)

## Evidencia para versionar

La evidencia reproducible está en `s2026/` y los resultados derivados en `analysis/`. Se conservan las recetas efectivas, estados, historiales, predicciones, métricas, datos PR, procedencia y los cuatro JPEG comentados en los hallazgos. Los historiales conservan sus bytes originales para verificar sus SHA-256. El árbol local `descarga/`, los pesos, los logs y las copias de consulta adicionales quedan fuera de Git. El inventario conserva los SHA-256 de toda la importación y el analizador permite omitir los archivos opcionales ausentes, como los PNG por brazo y los logs del preflight.

Los ejemplos y la vista previa usan imágenes de `pills-sxdht` de Mohamed Attia, distribuidas en Roboflow 100. Se añadieron cajas y etiquetas a los ejemplos, y variantes de brillo y contraste a la vista previa. [Atribución y licencia](../../baseline/results/README.md#atribución-de-las-imágenes)

## Qué descargar

Colocar los archivos de una sola tanda en `results/descarga/` dentro de este experimento. Conservar los nombres de las cuatro ejecuciones y de la carpeta de evaluación conjunta.

```text
descarga/
├── source-commit
├── inputs.sha256
├── manifest.json
├── yolo-control-s2026.config.json
├── yolo-control-s2026.job
├── yolo-variante-s2026.config.json
├── yolo-variante-s2026.job
├── cascade-control-s2026.config.json
├── cascade-control-s2026.job
├── cascade-variante-s2026.config.json
├── cascade-variante-s2026.job
├── yolo-control-s2026/
├── yolo-variante-s2026/
├── cascade-control-s2026/
├── cascade-variante-s2026/
└── eval-valid-<id>/
    ├── yolo-control-s2026/
    ├── yolo-variante-s2026/
    ├── cascade-control-s2026/
    └── cascade-variante-s2026/
```

- En la raíz, usar `source-commit`, `inputs.sha256` y los archivos `.config.json` y `.job` de la tanda. Copiar también el `manifest.json` exacto de los datos preparados (`$BASELINE_DATA/manifest.json`).
- En cada carpeta de entrenamiento, incluir `run.json`, `status.json` y `history.csv`. Añadir `ultralytics_runtime.json` en YOLO y tanto `cascade_runtime.json` como `detectron2.yaml` en Cascade. Si hubo reanudación bajo otro ID SLURM, incluir el `resume_runtime.json` de la sesión que completó el trabajo.
- En cada carpeta de evaluación, incluir `metrics.json`, `predictions.json` y `pr_curves.json`. Usar una evaluación de los cuatro `best` en `valid` con las muestras del benchmark. El informe conserva la ruta original y el SHA-256 del checkpoint, además del ID SLURM de evaluación.
- Para consulta, se pueden incluir `training.png`, `pr_curves.png`, los JPEG de `examples/`, `code-config.sha256`, `preflight.ok`, los logs `preflight/{yolo,cascade}-{tests,initialize}.log` y los informes `preflight/{yolo,cascade}/initialization.json`.

El recálculo usa las anotaciones `train/_annotations.coco.json` y `valid/_annotations.coco.json` del dataset original local en `data/pills-sxdht/`. Para revisar imágenes en la auditoría de errores, los CSV identifican los JPEG de `valid` por nombre.

## Importación local

Ejecutar este comando desde la raíz del repositorio cuando la carpeta de descarga esté completa.

```bash
.venv/bin/python -m experiments.exp-002-photometric.import_results \
  experiments/exp-002-photometric/results/descarga
```

Para elegir entre varias evaluaciones, añadir `--evaluation eval-valid-<id>` con el nombre exacto. El importador conserva los originales y copia la selección en `s2026/<detector>/<brazo>/training/` y `s2026/<detector>/<brazo>/evaluation/`. El inventario `s2026/provenance/downloaded.tsv` registra la ruta de descarga, la ruta importada y el SHA-256. Si falta un archivo requerido, el importador lo indica antes de publicar la carpeta. Para una importación alternativa, usar `--results RUTA` con un destino nuevo y pasar la misma opción al analizador. [Comando de recálculo](../README.md#análisis-local)

### Manifiesto reconstruido para la tanda `eval-valid-UgqskU`

La descarga de esta tanda carecía de `manifest.json`. Con el dataset original completo en disco se pudo reconstruir sin modificar la descarga. El importador verifica la huella contra `inputs.sha256` y registra `local-reconstruction/manifest.json` como origen en el inventario. Para reproducir esta importación desde la raíz del repositorio con un destino nuevo, añadir `--results RUTA` en el comando de importación.

```bash
workdir=$(mktemp -d)
.venv/bin/python -m scripts.baseline_data --output "$workdir/prepared" &&
.venv/bin/python -m experiments.exp-002-photometric.import_results \
  experiments/exp-002-photometric/results/descarga \
  --evaluation eval-valid-UgqskU --manifest "$workdir/prepared/manifest.json"
```

### Comprobación binaria opcional

Para repetir la inferencia se necesitan los cuatro `best` de la raíz de entrenamiento `exp-002-photometric-llaPwF`. Sus SHA-256 declarados constan en `metrics.json` y en `verificacion.json`. La comprobación de pesos COCO iniciales requiere además `checkpoints/yolov10s.pt` y `checkpoints/model_final_480dd8.pkl`, identificados por las huellas de `inputs.sha256`. Los seis binarios quedan fuera de la descarga local usada para el recálculo. [Rutas y huellas de la tanda](analysis/verificacion.json)

| Checkpoint | Ruta dentro de la tanda |
| --- | --- |
| YOLO control | `yolo-control-s2026/weights/best.pt` |
| YOLO variante | `yolo-variante-s2026/weights/best.pt` |
| Cascade control | `cascade-control-s2026/best/model_best.pth` |
| Cascade variante | `cascade-variante-s2026/best/model_best.pth` |

## Salidas del análisis

Los CSV conservan el AP en escala 0–1. Los deltas pareados son variante menos control. Las columnas entre detectores usan Cascade menos YOLO con el mismo brazo. La última columna estima la diferencia descriptiva de mejoras.

| En `analysis/` | Contenido |
| --- | --- |
| `verificacion.json` | Revisión, hashes de entrada, checkpoints, IDs SLURM, particiones, hardware, comparabilidad de las latencias y solapamiento de TP entre arquitecturas (por brazo al punto operativo). |
| `resumen.csv` | mAP COCO oficial de ocho clases, AP50, AP75, AP por tamaño, mAP sin la clase de soporte unitario, AP medio de clases de color, métricas operativas, latencias y selección de época. Un delta de latencia vacío indica que el benchmark carece de condiciones comparables verificadas. |
| `por_clase.csv` | Soporte en train y valid, AP, AP50, AP75, TP exportados a IoU 0,50 y 0,75 y TP, FP y FN al punto operativo. |
| `confusiones.csv` | Recuento por par de clases. Para cada FN a score ≥ 0,25 e IoU ≥ 0,50 se cuenta como candidata la caja de clase distinta de mayor score que alcance ambos umbrales. |
| `errores_instancias.csv` | Cajas COCO, scores, IoU y estado del emparejamiento por anotación de las clases de interés. La mejor caja de la clase correcta se elige por IoU y la candidata de otra clase por score con IoU ≥ 0,50. |
| `ibuphil.csv` | Todas las predicciones de las imágenes de 600 mg, las predicciones exportadas como 600 mg y las que solapan una anotación Cold con IoU ≥ 0,50. Incluye las cajas, la posición en `predictions.json` desde cero y el ranking por score de la clase 600 mg cuando corresponde. |
| `entrenamiento.png` | AP de validación por época, época seleccionada y pérdida de entrenamiento propia de cada detector. Cascade se alinea por épocas lógicas a partir de sus iteraciones. |
| `rendimiento.png` y `ap_por_clase.png` | Paneles para presentación con las métricas principales y los cambios de AP por clase en puntos porcentuales. |
