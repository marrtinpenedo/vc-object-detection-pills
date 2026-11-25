# EXP-003 · Hallazgos en validación

La fusión fija subió el mAP oficial de **0,7887 a 0,8194 (+3,08 pp)** frente a Cascade control. Al promediar las siete clases con más soporte, bajó de **0,8914 a 0,8793 (−1,21 pp)**. El resultado da apoyo parcial a la hipótesis y la decisión para esta entrega es conservar Cascade como modelo de referencia.

| Métrica en validación | Cascade | YOLO | Ensemble |
| --- | ---: | ---: | ---: |
| mAP COCO oficial | 0,7887 | 0,7021 | 0,8194 |
| AP50 | 0,8653 | 0,8332 | 0,9045 |
| AP75 | 0,8653 | 0,8261 | 0,9045 |
| mAP sin Ibuphil 600 mg | 0,8914 | 0,6881 | 0,8793 |
| Precisión micro | 0,9029 | 0,8193 | 0,8774 |
| Recall micro | 0,9490 | 0,6939 | 0,9490 |
| TP / FP / FN | 93 / 10 / 5 | 68 / 15 / 30 | 93 / 13 / 5 |

El AP de `Ibuphil 600 mg` pasó de 0,0692 a 0,4000, lo que aporta +4,13 pp al mAP oficial. Las otras siete clases aportaron −1,06 pp. `blue` y `pink` ganaron 2,40 y 2,49 pp. Xyzall perdió 3,35 pp y `white` perdió 3,31 pp. A score 0,25 e IoU 0,50, el total de aciertos permaneció igual y se añadieron tres FP. Las diferencias de AP también reflejan la ordenación de scores y la localización a varios IoU. [Resumen y AP por clase](results/resumen.csv) · [Curvas PR de la fusión](results/pr_curves.png)

Se reconstruyó el COCO de las 90 imágenes y 98 instancias de `valid`. Los dos controles, del trabajo de evaluación `11987778`, reprodujeron sus métricas y curvas archivadas. Se ejecutó una sola regla fijada de antemano sobre esas predicciones. Los resultados corresponden a esta partición y pareja de modelos. El coste de inferir con ambos modelos queda por medir. [Regla y reproducción](README.md) · [Huellas y procedencia](results/provenance.json)
