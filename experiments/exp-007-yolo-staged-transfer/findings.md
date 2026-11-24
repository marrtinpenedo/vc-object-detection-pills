# EXP-007 · Hallazgos en validación

## Fase A

Se comprobó la descarga por SHA-256 y se ejecutó el analizador de `ed7f9a6ef75afa9c66673bf8f03d34be2fa0c228`. El recálculo COCO reproduce el AP y las curvas PR de los cuatro checkpoints `best` evaluados en la misma asignación `12005631`. El protocolo anterior incluyó el smoke en A100 con transición, completado según el registro de la campaña. La evidencia local de A contiene las auditorías de los tres brazos congelados. [Resumen verificado](results/analysis/phase-a.json) · [Inventario](results/README.md)

| Brazo | mAP oficial | mAP de siete clases | Recall operativo | Mejor época | Último mAP del historial |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0,70209365 | 0,688107 | 0,693878 | 23 | 0,653203 |
| freeze-5 | 0,73572803 | 0,712261 | 0,673469 | 29 | 0,730831 |
| freeze-10 | 0,76169139 | 0,741933 | 0,714286 | 27 | 0,678327 |
| freeze-15 | 0,80732573 | 0,794087 | 0,816327 | 15 | 0,688505 |

La regla original seleccionó `freeze-15`. `freeze-10` también supera los márgenes de exploración. `freeze-5` pierde 0,020408 de recall operativo y rebasa el margen de 0,02. El mejor checkpoint de `freeze-15` se obtiene en la última época congelada. El orden observado y la evolución del historial motivan C con una sola semilla disponible. No establecen un perjuicio general de la descongelación. [Criterios](README.md#fases-y-criterio-de-decisión) · [Historial comparativo](results/analysis/phase-a-history-s2026.png)

En la escena 10 ambos modelos detectan el objeto. El control añade un falso positivo y `freeze-15` conserva únicamente el acierto. Ambos presentan fallos en las escenas 57, 61 y 80. [Escena 10, control](results/phase-a/eval-a-s2026/control/examples/failure-10.jpg) · [Escena 10, freeze-15](results/phase-a/eval-a-s2026/freeze-15/examples/correct-10.jpg)

## Fases B y C

Pendientes. B mantiene la confirmación preestablecida de `freeze-15`. C se definió después de observar A y compara las siete duraciones con tres semillas. Sus resultados servirán para seleccionar una receta candidata de forma descriptiva.
