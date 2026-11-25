# EXP-009 · Informe final de test

Los dos checkpoints fueron seleccionados en validación con la regla de la semilla mediana. La evaluación reservada contiene **45 imágenes y 47 anotaciones**. Cascade obtuvo **0,9189 mAP COCO**, **2,17 puntos porcentuales** más que YOLO D. YOLO obtuvo **97,92 % de precisión operativa**, frente a **94,00 %**, y procesó **2,72 veces** más imágenes por segundo en la misma GPU. Ambos recuperaron las 47 anotaciones. [Resumen CSV](results/analysis/test/resumen.csv) · [Comparación CSV](results/analysis/test/comparacion.csv)

## Resultados principales

| Modelo y checkpoint | mAP@0.50:0.95 | AP50 / AP75 | Precisión | Recall | TP / FP / FN | Latencia media / p95 (ms/img) | FPS |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| YOLOv10-S D · freeze-80 + afín + Mosaic · 2027, época 97 | 0,8971 | 1,0000 / 1,0000 | 97,92 % | 100 % | 47 / 1 / 0 | 10,74 / 10,78 | 93,09 |
| Cascade R-CNN · control, 100 épocas · 2027, época 34 | 0,9189 | 1,0000 / 1,0000 | 94,00 % | 100 % | 47 / 3 / 0 | 29,24 / 29,62 | 34,20 |

El punto operativo conserva score ≥0,25 e IoU ≥0,50. COCO usa las predicciones exportadas con el protocolo congelado. Los AP50 y AP75 perfectos pueden coexistir con falsos positivos a score ≥0,25 porque COCO ordena las detecciones por score e interpola la precisión. En IoU 0,85, YOLO obtiene 1,0000 y Cascade 0,9673. En IoU 0,90 y 0,95, Cascade alcanza **0,8739 / 0,3475** frente a **0,7692 / 0,2021** de YOLO. Esos dos umbrales explican la ventaja neta de Cascade en el promedio de diez IoU. [AP por IoU](results/analysis/test/ap_por_iou.csv) · [Figura AP por IoU](results/analysis/test/ap_por_iou.png)

El benchmark se ejecutó de forma secuencial en la misma **A100 PCIe de 40 GB**, nodo `c210-5`, batch 1, FP32, tamaño nominal 640, 20 warm-ups y 100 mediciones por modelo en las mismas 16 imágenes de test. La medición empieza con la imagen BGR decodificada y acaba con las detecciones devueltas a CPU. FPS es el inverso de la latencia media. [Figura de velocidad](results/analysis/test/velocidad_test.png) · [Auditoría de la asignación](results/analysis/test/auditoria.json)

## AP por clase

| Clase | GT en test | AP YOLO D | AP Cascade | Δ Cascade−YOLO (pp) |
| --- | ---: | ---: | ---: | ---: |
| Cipro 500 | 8 | 0,8488 | 0,9570 | +10,82 |
| Ibuphil 600 mg | 0 | — | — | — |
| Ibuphil Cold 400-60 | 7 | 0,9339 | 0,8891 | −4,48 |
| Xyzall 5mg | 5 | 0,8906 | 0,9042 | +1,36 |
| blue | 6 | 0,9442 | 0,9484 | +0,42 |
| pink | 8 | 0,8877 | 0,9337 | +4,60 |
| red | 5 | 0,8854 | 0,9307 | +4,53 |
| white | 8 | 0,8893 | 0,8690 | −2,03 |

El AP de **Ibuphil 600 mg es indefinido**. COCO promedia las siete clases con anotaciones y el mAP oficial de test coincide con la sensibilidad sin Ibuphil. Ambos detectores emitieron algunas predicciones de Ibuphil por debajo del umbral operativo, donde la clase carece de TP y FN observables. El test no permite estimar su capacidad para detectar Ibuphil 600 mg. [Tabla con AP50/AP75 por clase](results/analysis/test/por_clase.csv) · [Figura AP por clase](results/analysis/test/ap_por_clase.png)

## Errores al punto operativo

| Modelo | ID COCO | Clase predicha | Score | Interpretación |
| --- | ---: | --- | ---: | --- |
| YOLO D | 3 | pink | 0,398 | Detección duplicada de la pastilla rosa, IoU 0,909 |
| Cascade | 18 | Xyzall 5mg | 0,353 | Predicción sobre un Cipro detectado correctamente, IoU 0,888 |
| Cascade | 19 | Ibuphil Cold 400-60 | 0,361 | Caja sin solape con la anotación |
| Cascade | 28 | red | 0,439 | Predicción sobre un Ibuphil Cold detectado correctamente, IoU 0,899 |

Cada imagen con falso positivo conservó su detección correcta. Los diagnósticos se obtuvieron comparando las cajas exportadas con las anotaciones mediante el emparejamiento COCO. La figura usa las imágenes originales tras cotejar sus SHA-256 con el manifiesto. [Tabla de errores y cajas](results/analysis/test/errores_operativos.csv) · [Panel de errores](results/analysis/test/errores_operativos.png)

## Validación frente a test

La comparación usa los **mismos checkpoints de 2027**, con 90 imágenes en validación y 45 en test. En validación el mAP oficial incluye ocho clases. La comparación de siete clases mantiene el soporte de categorías común.

| Modelo | mAP oficial valid → test | mAP siete clases valid → test | Δ común (pp) | Precisión valid → test | Recall valid → test |
| --- | --- | --- | ---: | --- | --- |
| YOLO D | 0,8839 → 0,8971 | 0,8816 → 0,8971 | +1,55 | 95,83 % → 97,92 % | 93,88 % → 100 % |
| Cascade | 0,8379 → 0,9189 | 0,8933 → 0,9189 | +2,56 | 90,29 % → 94,00 % | 94,90 % → 100 % |

La subida oficial de Cascade entre particiones es de **8,10 pp**. Al controlar el conjunto de clases, la diferencia es de **2,56 pp**. El soporte de Ibuphil explica buena parte de esa discrepancia. La dificultad y la composición de las imágenes también varían entre particiones. La diferencia de test describe estos checkpoints en 45 imágenes. El experimento no replica la evaluación de test con otras semillas. [Tabla validación-test](results/analysis/test/validacion_test.csv) · [Figura validación-test](results/analysis/test/validacion_test.png)

## Procedencia y reproducción

- La asignación registrada es `12035833`. `launch.json`, ambos `metrics.json` y `verified.ok` concuerdan. Coinciden el SHA-256 de la selección, las dos huellas del lanzador aprobado y el inventario del código nativo del snapshot de C. El launcher comprobó físicamente ambos pesos antes de inferir. Las huellas anotadas en el lanzamiento y al producir métricas coinciden con el registro de validación.
- Se reconstruyeron las anotaciones de test con la huella congelada `eda7d1faf0e441fde5a81825b27efc073dba00595d7b4fb5f5fff793f3f5f2d4`. Se comprobaron los 45 IDs y se reprodujeron exactamente COCO, AP por clase, punto operativo y curvas PR desde las 87 predicciones YOLO y las 368 Cascade. Se recalcularon latencia media, mediana, p95 y FPS desde las muestras guardadas.
- El tar recibido tiene SHA-256 `4b48b4268d13f72d7e1048a63a0019979d597456579b1c1acbf5e1ed0435a7f9`. Es una copia del directorio. Se conservó el original y se proyectaron los artefactos requeridos al formato del importador, byte a byte. El inventario generado es local. El recibo SLURM independiente acredita `12035833|COMPLETED|0:0` y coincide con el lanzamiento y las métricas. Su SHA-256 es `aa5782a0eb04890a5eaaefde2f808ec7d0e68b07b7f2fb7dfd0b31ac8eb93118`. [Recibo SLURM](results/evidence/exp009-final-test-sacct.txt) [Auditoría local](results/analysis/test/auditoria.json) · [Recibo de importación](results/evidence/directory-import.json) · [Registro congelado](results/analysis/seleccion/seleccion-valid.json)

**Para la presentación.** Cascade lidera el mAP de este test por el ajuste de cajas a IoU alto. YOLO D recupera las mismas anotaciones, genera dos falsos positivos menos y reduce la latencia. La validación en tres semillas respalda freeze-80 con afín + Mosaic para YOLO y muestra que las ganancias fotométricas de 2026 no se repitieron. El test mide los dos modelos ya elegidos.
