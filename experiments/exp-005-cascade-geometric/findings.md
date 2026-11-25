# EXP-005 · Hallazgos en validación

Se reprodujeron las cuatro evaluaciones COCO de `best` con los artefactos del snapshot original `2b98ece5c736`. C, con afín y 30 épocas, lidera el mAP oficial con **0,8830**. A, el control de 30 épocas, lidera el análisis sin Ibuphil 600 mg con **0,8942**. Su mAP oficial, 0,8012, queda a 0,09 pp por debajo de la baseline histórica de Cascade (0,8021). [Informe completo](../../results/campaign-003-005-analysis/exp-005/findings.md) · [Auditoría](../../results/campaign-003-005-analysis/auditoria.json)

## Efectos factoriales

| Contraste | Δ mAP oficial (pp) | Δ sin Ibuphil (pp) | Aporte Ibuphil al Δ oficial (pp) |
| --- | ---: | ---: | ---: |
| B−A · Épocas en el control | +0,14 | −0,91 | +0,94 |
| D−C · Épocas con geometría | −7,61 | −2,26 | −5,625 |
| C−A · Afín a 30 épocas | +8,18 | −1,37 | +9,38 |
| D−B · Afín a 100 épocas | +0,43 | −2,72 | +2,81 |
| (D−B)−(C−A) · Interacción | −7,75 | −1,36 | −6,56 |

En C−A, el AP de Ibuphil sube de 0,15 a 0,90. Su peso de 1/8 aporta +9,38 pp al mAP oficial, mientras el resto aporta −1,20 pp. Seis de esas siete clases empeoran. En D−C, Ibuphil baja de 0,90 a 0,45. Las otras siete pierden de media 2,26 pp y aportan −1,98 pp al cambio oficial. Empeoran seis, especialmente Xyzall 5mg (−6,80 pp), `white` (−5,33) y `blue` (−3,12). El resultado amplio de la afín a 100 épocas también es desfavorable frente al control B. [Contribuciones por clase](../../results/campaign-003-005-analysis/exp-005/aporte_por_clase.csv) · [Figura factorial](../../results/campaign-003-005-analysis/exp-005/efectos_factoriales.png)

## Qué cambia en la única instancia de Ibuphil

En la imagen 61, las cajas correctas de C y D tienen IoU similares, 0,9325 y 0,9247. En C, el acierto encabeza el ranking con score 0,00629. En D, un falso positivo de la imagen 13 (score 0,03002) precede al acierto (0,01737). En un recálculo diagnóstico en memoria, retirar solo ese falso positivo eleva el AP de Ibuphil en D de 0,45 a 0,90 y el mAP de 0,8069 a 0,8631, con el AP de las otras clases y el punto operativo iguales. La métrica oficial conserva todas las predicciones. [Ranking y cajas](../../results/campaign-003-005-analysis/exp-005/ibuphil.csv) · [Contrafactual](../../results/campaign-003-005-analysis/exp-005/diagnostico_ranking_ibuphil.json)

Las cuatro condiciones tienen 93 TP y 5 FN a confianza ≥ 0,25 e IoU ≥ 0,50. C y D tienen respectivamente 10 y 9 FP. La escena 61 conserva un FN de Ibuphil y un FP de Ibuphil Cold en ambos brazos. Este contraste entre AP y el punto operativo explica por qué C gana mucho mAP oficial sin un salto de recall. [Escena 61, C](../../results/campaign-003-005-analysis/exp-005/ejemplos/C-ibuphil/failure-61.jpg) · [Escena 61, D](../../results/campaign-003-005-analysis/exp-005/ejemplos/D-ibuphil/failure-61.jpg)

## Dinámica de las 100 épocas

C llega a 0,8830 en la época 24 y termina en 0,8137. D alcanza 0,8069 en la época 18, ya está en 0,7552 a la 24 y termina en 0,7597. Su pérdida de entrenamiento baja de 0,268 a 0,165 desde `best` hasta la última época y su mAP de validación se estabiliza por debajo del máximo. Es compatible con degradación de generalización en esa ejecución. Faltan pérdidas de validación y AP por clase de los checkpoints intermedios para distinguir sobreajuste de inestabilidad de ranking en cada época. En la época 24, C tiene LR 0,000240 y D 0,00216, aproximadamente nueve veces más alto. El presupuesto y el horizonte del coseno cambian juntos. B apenas mejora el máximo de A (+0,14 pp oficiales) y termina en una meseta de 0,7906 frente a 0,7994 de A. [Curvas y LR](../../results/campaign-003-005-analysis/exp-005/entrenamiento.png) · [Diagnóstico de historial](../../results/campaign-003-005-analysis/exp-005/diagnostico_historial.csv)

C es el candidato de Cascade por la métrica oficial. A es la referencia más sólida frente a la exclusión de Ibuphil y conserva el mismo recall operativo. Mantendría ambos para una comparación final si el protocolo admite dos Cascade. La geometría de 100 épocas queda fuera de la selección actual. Una réplica emparejada de A y C permitiría comprobar si la ventaja oficial de C persiste en otra semilla. [Decisión global de campaña](../campaign-003-005/findings.md)
