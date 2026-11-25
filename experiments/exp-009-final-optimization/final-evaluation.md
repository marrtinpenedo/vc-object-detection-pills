# Selección de candidatos y evaluación final

## Elegir los candidatos usando validación

**YOLOv10-S.** El informe de C presenta primero **F−D** (fotometría con freeze-80 y geometría), **E−C** (fotometría con freeze-80 sin geometría) y **D−B** (freeze-80 con geometría), seguidos de las demás parejas y las dos interacciones identificables. Cada contraste muestra 2027 y 2028 por separado y 2026 como exploración auditada. Se informa media y desviación estándar muestral. Dos semillas nuevas sobre las mismas imágenes acotan la repetibilidad, y diferencias pequeñas siguen siendo descriptivas.

Elegir una receta con la regla prefijada. B y C se comparan con A. D, E y F deben superar tanto A como sus controles directos B, C y D, respectivamente. Cada puerta requiere **≥1 pp de ganancia oficial media**, cambio medio a siete clases **≥0** y ganancia oficial estrictamente positiva en **al menos dos de tres semillas**. Entre las recetas elegibles, escoger la de mayor mAP oficial medio, con desempate por mAP medio a siete clases y luego por orden A–F. A queda disponible si ninguna variante pasa. Examinar los cambios por clase, precisión, recall y latencia al justificar la decisión. El punto operativo conserva el umbral de validación. Dentro de la receta elegida, proponer el `best` de la semilla mediana por mAP oficial y desempatar la ordenación por mAP a siete clases y semilla. Así se utiliza un checkpoint representativo de la receta, con una regla fijada antes de conocer C.

**Cascade R-CNN.** Usar el control de 100 épocas, con pesos COCO, sin freezing ni augmentación de campaña. La misma regla de semilla mediana selecciona **2027**, con mAP oficial **0,8379**, a siete clases **0,8933** y `best` de la época **34**. Las tres semillas de validación promedian 0,8569 oficial y 0,8936 a siete clases. El SHA-256 registrado de `best/model_best.pth` es `744e7237849b83607a770f8cd230d37e462bed2edd47ce81e4ba240b09a8c4c8`, en la cadena 12029017 → 12029019 de D. [Resumen de D](results/analysis/d/resumen.csv) · [Auditoría de D](results/analysis/d/auditoria.json)

**YOLO tras certificar C.** La regla selecciona **D**, con 100 épocas, freeze-80, afín + Mosaic y sin fotometría. La semilla mediana es **2027**, `best` en época **97**, mAP oficial **0,8839**, **0,8816** a siete clases. La huella registrada de `weights/best.pt` es `0607143d11c8ed014a7e77ac19dbe9fa297e72fe5c4ad519a9f22ccb9d8e9473` y corresponde a la cadena 12031899 → 12031900. Se han verificado los 25 IDs y códigos `COMPLETED|0:0`. [Registro certificado de ambos modelos](results/analysis/seleccion/seleccion-valid.json) · [Resultados de C](findings.md#fase-c--resultados-de-la-confirmación)

## Registrar la selección antes de abrir test

Con C completa y revisada, guardar un registro fechado para ambos detectores que incluya:

- Receta efectiva, semilla, pesos COCO, hashes del dataset preparado y de las anotaciones de validación.
- Época seleccionada, ruta exacta y SHA-256 de cada `best`, IDs SLURM, snapshot del código nativo y huella de `scripts/evaluate_baseline.py`. Verificar físicamente el SHA-256 de cada peso en CESGA y contrastarlo con el recibo de entrenamiento y la evaluación de validación.
- Métricas que motivan la selección y el criterio fijo de operación de **score ≥0,25 e IoU ≥0,50**.
- Hash del protocolo y del código de evaluación, mapeo de categorías y huella de las anotaciones del conjunto reservado. Identificar las salidas de la evaluación final con raíces nuevas.

Una vez aprobado ese registro, evaluar una vez cada checkpoint en la misma partición reservada con el evaluador común de su snapshot verificado. El criterio principal será COCO bbox mAP@0.5:0.95 sobre las ocho clases, con score de exportación 0,001 y hasta 100 detecciones por imagen. Informar AP50, AP75, AP por clase, métricas operativas y la sensibilidad sin Ibuphil como complemento. El benchmark usa FP32, batch 1, tamaño nominal 640, 20 warm-ups y 100 iteraciones. Describir el alcance medido y realizar las mediciones bajo condiciones de hardware comparables. Archivar las predicciones y sus huellas, las PR, las métricas y los recibos de los dos checkpoints fijados. C se limita a `valid`.


## Preparación y comprobación de pesos

El paquete final lleva el registro certificado, el lanzador, el exportador y las dos huellas físicas esperadas. La evaluación usa el snapshot de C desplegado en `$HOME/pills-exp009-c/runtime`. Su evaluador, su predictor Cascade y su código COCO coinciden con D.

Desde el checkout local de EXP-009, sustituir `USUARIO_CESGA` por la cuenta propia:

```bash
(cd results && sha256sum -c exp-009-final-test.tar.gz.sha256)
scp results/exp-009-final-test.tar.gz{,.sha256} USUARIO_CESGA@ft3.cesga.es:
```

En CESGA, estos pasos **solo comprueban archivos**:

```bash
(
  set -euo pipefail
  cd "$HOME"
  sha256sum -c exp-009-final-test.tar.gz.sha256
  mkdir "$HOME/pills-exp009-final"
  tar -xzf exp-009-final-test.tar.gz -C "$HOME/pills-exp009-final"
  cd "$HOME/pills-exp009-final"
  sha256sum -c archivos.sha256
  sha256sum -c pesos.sha256 | tee "$HOME/exp009-final-pesos-verificados.txt"
)
```

`pesos.sha256` contiene las rutas **absolutas** y las huellas extraídas del registro certificado. Este es el mismo chequeo escrito de forma explícita:

```bash
sha256sum -c <<'SHA256'
0607143d11c8ed014a7e77ac19dbe9fa297e72fe5c4ad519a9f22ccb9d8e9473  /mnt/netapp2/Store_uni/home/usc/cursos/curso1014/pills-baseline/results/exp-009-c/jobs/train-d-s2027/attempt-01/training/weights/best.pt
744e7237849b83607a770f8cd230d37e462bed2edd47ce81e4ba240b09a8c4c8  /mnt/netapp2/Store_uni/home/usc/cursos/curso1014/pills-baseline/results/exp-009-d/jobs/train-e100-f000-s2027/attempt-01/training/best/model_best.pth
SHA256
```

Antes de aprobar, conservar la salida con dos `OK`, las recetas y la [selección certificada](results/analysis/seleccion/seleccion-valid.json). El programa exige las huellas otra vez justo antes de inferir y comprueba la receta, el trabajo de entrenamiento y la época de cada `best`.

## Lanzamiento reservado para después de aprobar

La evaluación pide **una A100 para los dos modelos**, secuenciales, en una misma asignación. El controlador valida el snapshot y la huella de las anotaciones COCO preparadas. Exige **45 imágenes y ocho clases**, compone las rutas de inferencia solo con `images/test` y registra los 45 IDs en ambas evaluaciones. El dataset preparado contiene 47 anotaciones de test según el manifiesto. La comprobación de integridad de los archivos preparados cubre todas las particiones. La inferencia, COCO y el benchmark de esta asignación se limitan a test. Se reutiliza la misma selección de validación y el punto operativo 0,25/0,50.

```bash
(
  set -euo pipefail
  work=/mnt/netapp2/Store_uni/home/usc/cursos/curso1014/pills-baseline
  test "$(realpath -e "$STORE/pills-baseline")" = "$work"
  launcher="$HOME/pills-exp009-final"
  (cd "$launcher" && sha256sum -c archivos.sha256 && sha256sum -c pesos.sha256)
  cd "$HOME/pills-exp009-c/runtime"
  sbatch --export="ALL,PILLS_EVAL_LAUNCHER_DIR=$launcher" "$launcher/evaluate.sh" \
    --selection "$launcher/seleccion-valid.json" \
    --data "$work/data/pills-prepared" --split test --checkpoint best --check-slurm \
    --run yolo "$work/results/exp-009-c/jobs/train-d-s2027/attempt-01/training" \
    --run cascade "$work/results/exp-009-d/jobs/train-e100-f000-s2027/attempt-01/training" \
    --output "$work/results/exp-009-final-test" --benchmark --warmup 20 --iterations 100
)
```

El código verifica y conserva `launch.json`, `eval.job`, los recibos de los dos entrenamientos y `verified.ok`. El benchmark repite inferencias en las mismas 16 imágenes de test en el mismo orden. El mAP se calcula con una única colección de predicciones por detector sobre las 45 imágenes.

## Recuperación, importación y análisis

Una vez terminada la asignación, desde CESGA:

```bash
work=/mnt/netapp2/Store_uni/home/usc/cursos/curso1014/pills-baseline
python3 "$HOME/pills-exp009-final/final_export.py" "$work/results/exp-009-final-test" \
    "$HOME/exp009-final-test-results.tar.gz"
(cd "$HOME" && sha256sum exp009-final-test-results.tar.gz > exp009-final-test-results.tar.gz.sha256)
```

El exportador exige un único recibo `COMPLETED|0:0` para el job final y exporta solo métricas, predicciones, PR y recibos con inventario SHA-256. En el portátil, usar el entorno CPU de las campañas anteriores y destinos nuevos:

```bash
scp USUARIO_CESGA@ft3.cesga.es:exp009-final-test-results.tar.gz \
    USUARIO_CESGA@ft3.cesga.es:exp009-final-test-results.tar.gz.sha256 results/
(cd results && sha256sum -c exp009-final-test-results.tar.gz.sha256)
python -m scripts.campaign_analysis.import_results \
    results/exp009-final-test-results.tar.gz results/exp009-final-test-import
MPLBACKEND=Agg python -m experiments.exp-009-final-optimization.final_analysis \
    results/exp009-final-test-import results/exp009-final-selection-verified/seleccion-valid.json \
    results/exp-009-final-test.tar.gz results/exp009-c-import/snapshot.json \
    results/exp009-c-import/prepared-manifest.json data/pills-sxdht results/exp009-final-test-analysis
```

El análisis reconstituye COCO de test desde las anotaciones originales y exige su huella preparada, los 45 IDs y las mismas clases. Reproduce métricas y PR desde predicciones, coteja los pesos, la asignación GPU y el benchmark. Genera `resumen.csv`, `por_clase.csv`, `comparacion.csv`, `ap_por_clase.png` y `auditoria.json`. Si una clase carece de anotaciones, su AP queda indefinido y se informa el número de clases con soporte.

## Importar la copia completa recibida

La descarga del directorio raíz `exp-009-final-test/` contiene los artefactos de evaluación, ejemplos y ajustes temporales de Ultralytics. Para el tar de este formato, proyectar los archivos requeridos al layout del importador oficial. La proyección registra el SHA-256 del tar original y genera un inventario SHA-256 local. El tar original queda intacto.

```bash
python -m scripts.exp009_followup.final_import \
    results/exp009-final-test-results.tar.gz results/exp009-final-test-import
MPLBACKEND=Agg python -m experiments.exp-009-final-optimization.final_analysis \
    results/exp009-final-test-import results/exp009-final-selection-verified/seleccion-valid.json \
    results/exp-009-final-test.tar.gz results/exp009-c-import/snapshot.json \
    results/exp009-c-import/prepared-manifest.json data/pills-sxdht results/exp009-final-test-analysis \
    --slurm-accounting results/exp009-final-test-sacct.txt
```

El recibo se obtiene con `sacct -X -n -P -j 12035833 -o JobIDRaw,State,ExitCode`. Para reproducir las métricas cuando falta ese archivo, usar `--allow-missing-accounting` en lugar del argumento de contabilidad. En ambos casos se mantienen las comprobaciones de procedencia, huellas y métricas. El informe indica si la contabilidad se verificó de manera independiente. [Resultados de la evaluación ejecutada](final-results.md)
