# EXP-004 y EXP-005 · Cierre de validación

Se reprodujeron las nueve condiciones desde sus predicciones oficiales. El importador verificó el inventario de la descarga `81d797854c9d5c1fcc32fb44e5eadeb05551dfe979fd962d5df027ec2f4673df`. El análisis comprobó el COCO de validación, los hashes de `best`, las curvas PR y las cadenas completas, con Cascade en `2b98ece5c736` y YOLO en `23e048675002`. Los resultados reproducidos coinciden con los nueve valores comunicados de mAP oficial y los nueve de sensibilidad al redondear a cuatro decimales. [Auditoría](../../results/campaign-003-005-analysis/auditoria.json) · [Tabla completa](../../results/campaign-003-005-analysis/resumen.csv) · [Resumen ejecutivo](../../results/campaign-003-005-analysis/resumen-ejecutivo.md) · [Figura para la presentación](../../results/campaign-003-005-analysis/map_condiciones.png)

## Resumen para la presentación

**Método.** Factoriales 30/100 épocas × control/geometría con semilla 2026, más YOLO E para la ablación de Mosaic. Validación de `best` con 90 imágenes y 98 instancias de ocho clases. El promedio de siete clases es un análisis de sensibilidad.

| Detector | Condición | mAP oficial | mAP sin Ibuphil 600 mg |
| --- | --- | ---: | ---: |
| YOLOv10-S | A · 30, control | 0,7021 | 0,6881 |
| YOLOv10-S | B · 100, control | 0,7078 | 0,6803 |
| YOLOv10-S | C · 30, afín + Mosaic | 0,8035 | 0,8040 |
| YOLOv10-S | D · 100, afín + Mosaic | **0,8200** | **0,8086** |
| YOLOv10-S | E · 100, solo afín | 0,7805 | 0,7635 |
| Cascade R-CNN | A · 30, control | 0,8012 | **0,8942** |
| Cascade R-CNN | B · 100, control | 0,8026 | 0,8851 |
| Cascade R-CNN | C · 30, solo afín | **0,8830** | 0,8805 |
| Cascade R-CNN | D · 100, solo afín | 0,8069 | 0,8579 |

- **YOLO.** D−B mejora +11,22 pp oficialmente y +12,82 pp sin Ibuphil. D−E mejora +3,95 y +4,51 pp respectivamente. Cipro, Ibuphil Cold, Xyzall y red impulsan D−B, y `white` retrocede. D alcanza 93,88 % de recall con 18 FP. C conserva mejor precisión (86,41 % frente a 83,64 %) y mayor AP50/AP75 con un presupuesto de 30 épocas. [EXP-004](../exp-004-yolo-factorial/findings.md)
- **Cascade.** C−A mejora +8,18 pp oficialmente y pierde 1,37 pp a siete clases. Ibuphil aporta +9,38 pp al mAP oficial y las otras siete restan 1,20 pp. D−C cae 7,61 pp oficiales y 2,26 pp sin Ibuphil. C y D conservan 94,90 % de recall operativo. En D, un FP de baja confianza pasa por delante de la única detección correcta de Ibuphil. [EXP-005](../exp-005-cascade-geometric/findings.md)
- **Épocas y tiempo de inferencia.** B−A cambia +0,57 pp en YOLO y +0,14 pp en Cascade, con descensos a siete clases. El scheduler coseno recorre horizontes distintos y con 100 épocas hay más oportunidades de seleccionar el máximo. YOLO D promedió 10,87 ms por imagen y Cascade C 29,50 ms, en asignaciones A100 independientes con batch 1. [Curvas YOLO](../../results/campaign-003-005-analysis/exp-004/entrenamiento.png) · [Curvas Cascade](../../results/campaign-003-005-analysis/exp-005/entrenamiento.png)

## Comparación con las baselines y EXP-007

Se recalcularon dos baselines y doce evaluaciones de referencia de EXP-007 fase C, cuatro por semilla. Todas coinciden con sus JSON y curvas PR publicados y comparten los hashes de datos de esta campaña. Las baselines históricas son referencias descriptivas. Los contrastes factoriales usan los controles contemporáneos. [Baseline histórica](../baseline/findings.md) · [EXP-007, fase C](https://github.com/marrtinpenedo/vc-object-detection-pills/blob/6acc2efea86324de1cd039fbcb34df2f0252d95a/experiments/exp-007-yolo-staged-transfer/findings.md#fase-c--duración-de-la-congelación) · [Tabla de referencias](../../results/campaign-003-005-analysis/referencias_validacion.csv)

| Referencia | mAP oficial | mAP sin Ibuphil |
| --- | ---: | ---: |
| Baseline YOLO, 2026 | 0,7021 | 0,6881 |
| Baseline Cascade, 2026 | 0,8021 | 0,8909 |
| EXP-007 `freeze-15`, media de tres semillas | 0,8037 | 0,7947 |
| EXP-007 `freeze-20`, semilla 2026 | 0,8125 | 0,7999 |
| EXP-007 `freeze-20`, media de tres semillas | 0,8373 | 0,8236 |
| EXP-007 `freeze-30`, media de tres semillas | 0,8335 | 0,8192 |

YOLO D gana 11,79 pp oficiales frente a la baseline. Frente a `freeze-20` en la semilla 2026, D gana 0,76 pp oficiales y 0,86 pp a siete clases, con un presupuesto de 100 frente a 30 épocas. `freeze-20` gana a su control en las tres semillas de EXP-007, con +9,23 pp oficiales de media y DE de 2,21 pp para el mAP de la variante. La fase B había confirmado `freeze-15` y la fase C seleccionó `freeze-20` al ampliar las duraciones sobre las mismas semillas. Toda la comparación comparte el conjunto de validación. La geometría y el freezing se han probado por separado.

## Candidatos para el experimento combinado

1. **Prioridad · YOLO durante 30 épocas, receta C + `freeze-20`.** Combina la política afín + Mosaic de EXP-004 con las primeras 20 épocas de backbone congelado y diez de ajuste completo de EXP-007. Probar en validación un factorial de cuatro brazos bajo el mismo código y semillas: control, C, `freeze-20` y C + `freeze-20`. El contraste de interacción es `combinado − C − freeze-20 + control`. Mantener fijos los demás hiperparámetros y reportar la métrica oficial, el promedio de siete clases y el punto operativo.
2. **Extensión · D + `freeze-20` a 100 épocas.** D es el mejor YOLO de esta campaña. El contraste D−C solo gana 0,45 pp sin Ibuphil y las 20 épocas congeladas ocupan una fracción distinta del entrenamiento largo. Si se prueba, emparejarlo con D y con el brazo de 100 épocas con freezing y sin aumentos bajo el mismo scheduler. La eficacia de esa combinación queda abierta.
3. **Referencia Cascade.** Conservar C por el criterio oficial y A como candidato de 30 épocas con el mayor mAP a siete clases y el mismo recall. El freezing de EXP-007 es propio de YOLO. Si solo se elige un Cascade por el criterio primario, C lidera el mAP oficial en validación, con sensibilidad explícita a una sola instancia.

La geometría de YOLO ofrece una ganancia consistente entre las dos duraciones en la semilla observada. La afín de Cascade merece replicación antes de adoptarse por su aparente mejora oficial. La selección de `best` y las comparaciones se hicieron sobre las mismas 90 imágenes. La campaña factorial aporta una sola semilla. El test queda reservado.

## Exportación selectiva

En CESGA, desde la copia de la recuperación. El lanzador usa la raíz `results/campaign-003-005` dentro de `PILLS_CAMPAIGN_WORK`.

```bash
PILLS_CAMPAIGN_WORK=/mnt/netapp2/Store_uni/home/usc/cursos/curso1014/pills-baseline \
  bash "$HOME/pills-campaign-003-005-yolo-recovery/slurm/campaign.sh" collect \
    "$HOME/campaign-003-005-results-recovered.tar.gz"
```

La descarga se ejecuta desde la raíz del repositorio en el portátil.

```bash
scp USUARIO_CESGA@ft3.cesga.es:campaign-003-005-results-recovered.tar.gz results/
```

El tar contiene los descriptores, el registro de intentos, los estados de SLURM, las configuraciones, las predicciones, las curvas PR, los historiales y las huellas de los checkpoints. Los pesos y las imágenes quedan en sus ubicaciones originales. [Recuperación de YOLO y huellas](yolo-recovery.md)

## Reproducción local

```bash
.venv/bin/python -m experiments.campaign-003-005.import_results \
  results/campaign-003-005-results-recovered.tar.gz results/campaign-003-005-import

MPLBACKEND=Agg uv run --no-project --python 3.11 \
  --with pycocotools==2.0.8 --with numpy==1.26.4 --with pillow==10.4.0 \
  --with matplotlib==3.9.4 --with pyyaml==6.0.4 \
  python -m experiments.campaign-003-005.analyze \
  results/campaign-003-005-import results/campaign-003-005-analysis \
  --snapshot-sha256 2b98ece5c736735e4602a992c1847958be74cbee6a52b3daa03143cf91f1b3d9 \
  --recovery-sha256 23e048675002dd0bbd678b184ef674288813a10517179fedb32914caad762978 \
  --require-complete
```

El cierre exige nueve cadenas `preflight → train → eval` completas. Cascade debe corresponder al descriptor original y YOLO al de recuperación. El análisis verifica el inventario y coteja el COCO de `valid`, el `best` evaluado y la procedencia de cada etapa. `resumen-ejecutivo.md` y `map_condiciones.png` reúnen los dos detectores. Cada subdirectorio genera su informe, efectos, AP por clase, errores y evolución de aprendizaje.

## Criterio de lectura

El resultado principal es mAP COCO@0,50:0,95 de las ocho clases. Los contrastes muestran cuánto aporta Ibuphil 600 mg al cambio oficial y cuánto aportan las otras siete. La sensibilidad usa el promedio de siete. La precisión y el recall describen el punto operativo con confianza 0,25 e IoU 0,50. El análisis utiliza semilla 2026 y las 90 imágenes de validación. La evaluación de test queda reservada para candidatos congelados.

[EXP-004](../exp-004-yolo-factorial/findings.md) · [EXP-005](../exp-005-cascade-geometric/findings.md)
