# EXP-004 · Hallazgos en validación

Se reprodujeron las cinco evaluaciones COCO a partir de las predicciones exportadas de `best`, bajo el snapshot de recuperación `23e048675002`. El control A reproduce el mAP de la baseline histórica, 0,7021. La semilla es 2026 y las 90 imágenes contienen 98 instancias. [Informe completo, incluidas las ocho clases](../../results/campaign-003-005-analysis/exp-004/findings.md) · [Baseline](../baseline/findings.md) · [Auditoría](../../results/campaign-003-005-analysis/auditoria.json)

## Efectos y ablación

| Contraste | Δ mAP oficial (pp) | Δ sin Ibuphil (pp) |
| --- | ---: | ---: |
| B−A · Épocas en el control | +0,57 | −0,78 |
| D−C · Épocas con geometría | +1,65 | +0,45 |
| C−A · Afín + Mosaic a 30 épocas | +10,14 | +11,59 |
| D−B · Afín + Mosaic a 100 épocas | +11,22 | +12,82 |
| (D−B)−(C−A) · Interacción | +1,08 | +1,23 |
| E−B · Solo afín a 100 épocas | +7,27 | +8,31 |
| D−E · Habilitar Mosaic a 100 épocas | +3,95 | +4,51 |

D lidera las cinco condiciones con mAP oficial **0,8200** y **0,8086** a siete clases. Frente a B suben seis de las otras siete clases. Los mayores cambios son Cipro 500 (+24,99 pp), Ibuphil Cold 400-60 (+21,91), Xyzall 5mg (+19,63) y red (+19,01). `white` baja 8,54 pp e Ibuphil 600 mg queda en 0,90 en ambos. Frente a E, D gana principalmente en Xyzall (+19,14) y red (+8,77), con una pérdida de 12,60 pp en `white`. D−E incorpora la composición y el reescalado de Mosaic, junto con un cambio estimado de 49,5 % a 42,9 % de afín efectiva. [AP y contribuciones por clase](../../results/campaign-003-005-analysis/exp-004/aporte_por_clase.csv) · [Auditoría geométrica](../campaign-003-005/geometric-policy.md#auditoría-de-la-receta-sobre-train)

C alcanza **0,8035** y **0,8040** en 30 épocas. Su AP50, AP75 y precisión operativa son mayores que los de D. D aporta +1,65 pp oficiales frente a C, de los que +1,25 pp corresponden a Ibuphil. Solo +0,45 pp separan sus medias de siete clases. Esta diferencia y la interacción pequeña requieren replicación antes de priorizar el presupuesto largo. [Figura factorial](../../results/campaign-003-005-analysis/exp-004/efectos_factoriales.png)

## Curvas y errores

Con confianza ≥ 0,25 e IoU ≥ 0,50, D recupera 30 TP frente a B (92 frente a 62), con 18 FP frente a 10. En la escena 19, D detecta las tres pastillas y B las omite. En la escena 86, D detecta Xyzall y acumula ocho FP, seis de clase `white`. En la escena 61, la mejor caja de Ibuphil de D tiene score 0,1356 e IoU 0,9043. Su AP de clase es 0,90 y la instancia sigue como FN al punto operativo. [Errores por imagen](../../results/campaign-003-005-analysis/exp-004/errores_por_imagen.csv) · [Escena 19, B](../../results/campaign-003-005-analysis/exp-004/casos/B/failure-19.jpg) · [Escena 19, D](../../results/campaign-003-005-analysis/exp-004/casos/D/correct-19.jpg) · [Escena 86, D](../../results/campaign-003-005-analysis/exp-004/ejemplos/D/failure-86.jpg) · [PR de D](../../results/campaign-003-005-analysis/exp-004/curvas_pr/D.png)

B selecciona la época 38 y termina en 0,5633 de mAP de historial, 14,44 pp por debajo del máximo. En ese tramo, la pérdida de clasificación baja en train de 0,323 a 0,069 y sube en validación de 2,02 a 2,85. Es compatible con sobreajuste en B. D alcanza su máximo en la época 86, termina en 0,7912 y su pérdida de clasificación de validación es similar en esas dos épocas (0,837 y 0,829). C se selecciona en la época 27 y acaba a 1,05 pp de su máximo. Al final de la época 30, el LR de los brazos de 100 épocas es 0,000816 frente a 0,0000526 en los de 30. Los brazos largos ya llevan otra trayectoria en esa época. El historial valora el último estado durante el entrenamiento. El evaluador común reproduce por separado el checkpoint `best`. [Curvas de entrenamiento](../../results/campaign-003-005-analysis/exp-004/entrenamiento.png) · [Diagnóstico de historial](../../results/campaign-003-005-analysis/exp-004/diagnostico_historial.csv)

La geometría mejora el mAP oficial y la sensibilidad a siete clases en ambos presupuestos de esta semilla. Para combinarla con el freezing de EXP-007, C permite probar primero ambas intervenciones a 30 épocas. D queda como receta geométrica de mayor mAP observado y candidato para un contraste posterior emparejado a 100 épocas. [Comparación con EXP-007 y diseño combinado](../campaign-003-005/findings.md#comparación-con-las-baselines-y-exp-007)
