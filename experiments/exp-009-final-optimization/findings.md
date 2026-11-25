# EXP-009 · Validación multifactorial y evaluación final

**Selección certificada en validación.** En C, YOLO D (freeze-80 + afín + Mosaic) cumple la regla prefijada y elige `best` de la semilla mediana 2027. Las ganancias fotométricas observadas en 2026 se revierten en 2027 y 2028. Para Cascade se mantiene el control de 100 épocas, también con la semilla mediana 2027. [Resultados de C](#fase-c--resultados-de-la-confirmación) · [Protocolo final](final-evaluation.md)

**Resultado de A.** Freeze-80 encabeza las dos recetas, tanto con las ocho clases como al excluir Ibuphil 600 mg. Con afín + Mosaic alcanza **0,8578** de mAP COCO y **0,8518** a siete clases. El criterio fijado para B nomina **80 épocas** congeladas.

## Fase A · Freezing y afín + Mosaic en YOLOv10-S

### Verificación antes de aceptar las métricas

El SHA-256 del tar recibido es `6bb76e5a15bdb5a66e2aa4e36e1063ffdc887e297055b7bd52096ce6710e1b6b`. El importador verificó los 222 archivos inventariados. El descriptor `8254e3e0bfa68bebfd6e8a8090b8c8a55fb759989bdf345b6954762357114ef3` coincide con el del bundle transferido. Se cotejaron las 20 huellas de fuente con el paquete y el checkout.

Los 29 trabajos constan como `COMPLETED` en el primer intento. El preflight `12026504` incluye la inicialización del control y los smokes temporal y permanente. Se cotejaron los recibos `done.json`, las relaciones `afterok`, las recetas efectivas, los pesos iniciales y el dataset en las 14 cadenas de entrenamiento y evaluación. Para cada brazo se verificó el SHA-256 registrado del `best` contra el checkpoint indicado en su evaluación. Los pesos permanecen en CESGA. El analizador reproduce desde las predicciones las métricas, el AP por clase y las PR exportadas para las 90 imágenes y 98 anotaciones de `valid`. Todos los brazos usan semilla 2026. [Auditoría](results/analysis/a/auditoria.json) · [Procedimiento](../../docs/campaigns/exp-009-a.md#análisis-local)

### Resultados de los 14 brazos

Precisión y recall usan score ≥0,25 e IoU ≥0,50. `geom` aplica la afín auditada más Mosaic 0,30. Los mAP de la tabla son los de la evaluación independiente de `best`. El CSV incluye también mAP de la última época, tiempos y memoria.

| Freeze | Receta | mAP COCO | mAP sin Ibuphil | AP50 | AP75 | Precisión (%) | Recall (%) | TP/FP/FN | Época best | Entrenamiento (min) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | Control | 0,7078 | 0,6803 | 0,8398 | 0,7941 | 86,11 | 63,27 | 62/10/36 | 38 | 16,36 |
| 0 | Afín + Mosaic | 0,8200 | 0,8086 | 0,9555 | 0,9462 | 83,64 | 93,88 | 92/18/6 | 86 | 16,72 |
| 10 | Control | 0,7233 | 0,7124 | 0,9031 | 0,8659 | 78,22 | 80,61 | 79/22/19 | 10 | 16,22 |
| 10 | Afín + Mosaic | 0,8136 | 0,8013 | 0,9556 | 0,9279 | 87,38 | 91,84 | 90/13/8 | 86 | 16,15 |
| 20 | Control | 0,7233 | 0,7124 | 0,9031 | 0,8659 | 78,22 | 80,61 | 79/22/19 | 10 | 15,39 |
| 20 | Afín + Mosaic | 0,8403 | 0,8318 | 0,9766 | 0,9728 | 87,88 | 88,78 | 87/12/11 | 70 | 15,50 |
| 40 | Control | 0,8318 | 0,8221 | 0,9617 | 0,9540 | 92,78 | 91,84 | 90/7/8 | 37 | 14,39 |
| 40 | Afín + Mosaic | 0,8213 | 0,8101 | 0,9664 | 0,9552 | 94,85 | 93,88 | 92/5/6 | 88 | 14,38 |
| 60 | Control | 0,8318 | 0,8221 | 0,9617 | 0,9540 | 92,78 | 91,84 | 90/7/8 | 37 | 13,16 |
| 60 | Afín + Mosaic | 0,8416 | 0,8332 | 0,9656 | 0,9481 | 89,00 | 90,82 | 89/11/9 | 78 | 13,21 |
| 80 | Control | 0,8384 | 0,8296 | 0,9619 | 0,9536 | 93,48 | 87,76 | 86/6/12 | 98 | 12,14 |
| 80 | Afín + Mosaic | 0,8578 | 0,8518 | 0,9790 | 0,9790 | 93,88 | 93,88 | 92/6/6 | 89 | 12,18 |
| 100 | Control | 0,8318 | 0,8221 | 0,9617 | 0,9540 | 92,78 | 91,84 | 90/7/8 | 37 | 11,11 |
| 100 | Afín + Mosaic | 0,7654 | 0,7605 | 0,9600 | 0,9190 | 92,71 | 90,82 | 89/7/9 | 24 | 11,16 |

[Tabla completa](results/analysis/a/resumen.csv) · [mAP frente a duración](results/analysis/a/duracion.png) · [AP por clase y punto operativo](results/analysis/a/por_clase.csv)

### Efectos e interacción

Los contrastes se expresan en puntos porcentuales. El efecto de añadir geometría compara ambas celdas de la misma duración. El efecto del freezing se calcula frente a freeze-0 de su propia receta. La interacción resta al efecto geométrico el de freeze-0. La columna de siete clases es una comprobación de sensibilidad.

| Freeze | Δ geometría, oficial | Δ geometría, siete | Δ freezing sin geom., oficial | Δ freezing con geom., oficial | Interacción oficial | Interacción siete |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +11,22 | +12,82 | +0,00 | +0,00 | +0,00 | +0,00 |
| 10 | +9,03 | +8,89 | +1,55 | −0,64 | −2,19 | −3,93 |
| 20 | +11,70 | +11,94 | +1,55 | +2,03 | +0,48 | −0,88 |
| 40 | −1,05 | −1,20 | +12,41 | +0,13 | −12,27 | −14,03 |
| 60 | +0,97 | +1,11 | +12,41 | +2,15 | −10,25 | −11,71 |
| 80 | +1,94 | +2,22 | +13,06 | +3,78 | −9,28 | −10,61 |
| 100 | −6,64 | −6,16 | +12,41 | −5,46 | −17,86 | −18,99 |

Al promediar los siete niveles por igual, añadir afín + Mosaic aporta **+3,88 pp oficiales y +4,23 pp a siete clases**. Ese promedio reúne respuestas distintas. La augmentación aporta **+11,70 pp** con freeze-20 y **+1,94 pp** con freeze-80, pasa a **−1,05 pp** con freeze-40 y a **−6,64 pp** con congelación permanente. La caída de la permanente sigue en las siete clases (−6,16 pp). El efecto de freeze-80 frente a freeze-0, promediado por igual entre ambas recetas, es **+8,42 pp oficiales y +9,62 pp a siete clases**.

El óptimo observado en la cuadrícula coincide en **80 épocas**, y la geometría reduce la diferencia entre ese nivel y freeze-0. Freeze-80 suma **+13,06 pp** oficiales sin augmentación y **+3,78 pp** con afín + Mosaic, con una interacción de **−9,28 pp** oficiales y **−10,61 pp** a siete clases. La combinación registra el mejor mAP aunque su ganancia es menor que la suma de los efectos calculados en freeze-0 y control limpio. Son diferencias descriptivas de una semilla. El factorial estima afín + Mosaic como un solo factor. [Contrastes completos por métrica y clase](results/analysis/a/factorial.csv) · [Figura de interacción](results/analysis/a/diagnosticos/interaccion.png)

### Qué clases explican freeze-80

Con afín + Mosaic, pasar de freeze-0 a freeze-80 sube el AP en seis de las otras siete clases. Destacan **white (+11,75 pp), Xyzall (+5,65), blue (+5,52) y Cipro (+4,07)**. Ibuphil Cold cae 1,88 pp. El AP de Ibuphil 600 mg sigue en 0,90 y aporta cero al salto de **+3,78 pp** oficiales. Sin augmentación, freeze-80 mejora las siete clases frente a freeze-0. Xyzall (+23,89 pp), Ibuphil Cold (+21,01), Cipro (+20,24) y red (+19,52) aportan las mayores mejoras y el AP de Ibuphil 600 mg también queda en 0,90.

Al **añadir geometría al propio freeze-80**, Cipro aporta el mayor cambio de AP (+8,83 pp). Afín + Mosaic eleva el punto operativo de 86 a 92 TP con 6 FP en ambos casos, pasando de 87,76 a 93,88 % de recall y de 93,48 a 93,88 % de precisión. Frente a freeze-0 con geometría, el mejor brazo mantiene 92 TP y baja de 18 a 6 FP. [Δ AP por clase](results/analysis/a/diagnosticos/ap_por_clase_delta.png) · [Confusiones](results/analysis/a/confusiones.csv) · [PR de freeze-80 con geometría](results/analysis/a/pr-f080-geom.png)

### Prefijos, descongelación y selección de best

Los 30 pares de duraciones positivas dentro de una misma receta tienen filas idénticas en toda su parte congelada común al excluir la columna de tiempo. En el control, freeze-10 y freeze-20 eligen la época **10** y exportan predicciones y PR idénticas. Freeze-40, freeze-60 y freeze-100 eligen la época **37** y forman otro grupo idéntico en predicciones y PR. Las huellas de los archivos de checkpoint difieren. El empate de detecciones corresponde al prefijo compartido y no equivale a replicaciones independientes. Los dos brazos contemporáneos de freeze-0 también reproducen byte por byte las predicciones de EXP-004 B y D. [Auditoría de prefijos](results/analysis/a/diagnosticos/prefijos.csv) · [SHA-256 de predicciones y checkpoints](results/analysis/a/diagnosticos/predicciones_best.csv)

En los diez brazos temporales baja el mAP de historial en la primera época descongelada. A freeze-10, freeze-20 y freeze-40, las seis trayectorias temporales caen a mAP cero o prácticamente cero antes de recuperarse. A freeze-80 la caída es más moderada: de **0,7120 a 0,6156** sin augmentación y de **0,6764 a 0,6501** con ella. Los brazos permanentes de las mismas recetas registran 0,7146 y 0,6875 en la época 81. El LR registrado en esa época para los temporales es 0,0001407, frente a 0,0009768 al descongelar tras diez épocas. El momento del cambio, el LR y la trayectoria de las partes entrenables evolucionan juntos. La pauta observada no identifica cuál explica la recuperación.

Con freeze-80 ambos mejores checkpoints llegan después de descongelar. El control selecciona la época **98** y la geometría la **89**. El mAP de historial de freeze-80 + geometría termina en **0,7567**, 10,11 pp por debajo de su máximo. Esa brecha refuerza la necesidad de fijar la selección por `best` y replicar antes de generalizar. La auditoría certifica la transición y la continuidad de AdamW, Batch Normalization y EMA por época. [Curvas de aprendizaje](results/analysis/a/evolucion.png) · [Ventana de descongelación](results/analysis/a/diagnosticos/ventana_descongelacion.png) · [Valores y pérdidas](results/analysis/a/diagnosticos/descongelacion.csv)

### Coste y nominación para B

Los 14 entrenamientos suman **198,06 min de tiempo interno** (unas 3,30 horas GPU de entrenamiento). Freeze-80 con geometría requiere **12,18 min**, frente a **16,72 min** para su freeze-0. El pico asignado ronda **4,76 GiB** en los temporales y baja a **2,65 GiB** en los dos permanentes. La latencia media por imagen varía entre **10,66 y 11,12 ms** en asignaciones independientes. El export permite comparar las muestras de latencia, aunque no registra tiempos GPU totales de los trabajos de evaluación. [Coste y rendimiento](results/analysis/a/diagnosticos/coste_gpu.png) · [Tiempo y memoria por brazo](results/analysis/a/resumen.csv)

**Llevar freeze-80 a B**, con `f000-control`, `f000-geom`, `f080-control` y `f080-geom` como las cuatro celdas sin fotometría, sujetas a verificar su equivalencia con el despliegue de B. En la celda geométrica, freeze-80 supera el mínimo prefijado de +1 pp con **+3,78 pp oficiales** y conserva una ganancia de **+4,32 pp** entre las siete clases. Es una nominación exploratoria para el factorial fotométrico. La confirmación con 2027 y 2028 queda para la fase correspondiente. [Propuesta y control de reutilización](results/analysis/a/fase-b-propuesta.json) · [Diseño para B y C](README.md#fase-b--fotometría-e-interacciones)

### Auditoría de seguimiento

La [auditoría de descongelación](thaw-audit.md) documenta seis colapsos transitorios con continuidad de AdamW, BN, scheduler y EMA, los checkpoints con predicciones coincidentes y el carácter puntual del máximo de freeze-80 con geometría. B conserva exactamente ese código.

## Fases B y D · Integridad y procedencia

| Fase | Archivos del inventario SHA-256 | Trabajos completados | Preflight | Evaluaciones nuevas / referencias reproducidas |
| --- | ---: | ---: | ---: | ---: |
| B | 75 | 9/9 | 12028987 | 4 / 4 de A |
| D | 40 | 5/5 | 12029015 | 2 / 1 de EXP-008 |

Los 14 trabajos nuevos terminaron al primer intento. El registro conserva las 12 dependencias `afterok` y el analizador coteja los recibos y el padre efectivo de cada trabajo. Los hashes de los controladores y los snapshots coinciden con los paquetes de despliegue, incluidos `8254e3e0bfa68beb…` de A y `8d479bfd14b07905…` de EXP-008. Los cuatro pares A/B reproducen la misma receta efectiva al descontar la fotometría; en D la receta efectiva solo difiere en `SEED`. Los listados completos de versiones de los entrenamientos y las evaluaciones coinciden en cada par. El smoke nativo de B registra la cadena completa y la descongelación en la época 2. El `best` de cada brazo coincide por SHA-256 entre el recibo del entrenamiento y la evaluación. Los exports ligeros incluyen las huellas, con los pesos en CESGA.

Se reprodujeron las once evaluaciones COCO, AP por clase y curvas PR desde predicciones de las 90 imágenes de `valid`, con 98 anotaciones y ocho clases. Precisión, recall y recuentos usan score ≥0,25 e IoU ≥0,50. SHA-256 locales de los tar recibidos: B `6167bfec3097040a7f64f191410178db24ae67d855fa9a8fb0fa74b67fb72fb2` y D `bd168930abe6387b49b08729a2f2b728bf06b577d2625d7c3f36e6ca5dbad8b8`. [Auditoría B](results/analysis/b/auditoria.json) · [Auditoría D](results/analysis/d/auditoria.json) · [Procedimiento reproducible](../../docs/campaigns/exp-009-b-d.md#análisis-local)

## Fase B · Fotometría en YOLOv10-S

Los ocho brazos usan 100 épocas y la semilla 2026. La intervención fotométrica es brillo y contraste ±10 % con probabilidad 0,5. `Geom.` significa afín + Mosaic con probabilidad de Mosaic 0,30 y `close_mosaic=0`. Las cuatro filas sin fotometría provienen de A.

| Freeze | Geom. | Foto | mAP COCO | mAP sin Ibuphil | AP50 | AP75 | Precisión (%) | Recall (%) | TP/FP/FN | Época best |
| ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | — | — | 0,7078 | 0,6803 | 0,8398 | 0,7941 | 86,11 | 63,27 | 62/10/36 | 38 |
| 0 | — | Sí | 0,6740 | 0,6275 | 0,7894 | 0,7601 | 83,82 | 58,16 | 57/11/41 | 30 |
| 0 | Sí | — | 0,8200 | 0,8086 | 0,9555 | 0,9462 | 83,64 | 93,88 | 92/18/6 | 86 |
| 0 | Sí | Sí | 0,8228 | 0,8117 | 0,9785 | 0,9410 | 84,55 | 94,90 | 93/17/5 | 70 |
| 80 | — | — | 0,8384 | 0,8296 | 0,9619 | 0,9536 | 93,48 | 87,76 | 86/6/12 | 98 |
| 80 | — | Sí | 0,8545 | 0,8337 | 0,9552 | 0,9447 | 87,76 | 87,76 | 86/12/12 | 89 |
| 80 | Sí | — | 0,8578 | 0,8518 | 0,9790 | 0,9790 | 93,88 | 93,88 | 92/6/6 | 89 |
| 80 | Sí | Sí | **0,8710** | **0,8669** | **0,9804** | **0,9804** | 91,26 | 95,92 | 94/9/4 | 95 |

[Tabla con latencia, tiempo y memoria](results/analysis/b/resumen.csv) · [Historial por época](results/analysis/b/historial.csv) · [Comparación y curvas de aprendizaje](results/analysis/b/rendimiento.png) · [Curva PR del candidato](results/analysis/b/pr-f080-geom-photo.png)

### Contrastes factoriales

Los efectos principales son diferencias de medias entre niveles. Las interacciones dobles son diferencias de diferencias promediadas sobre el tercer factor. La triple compara la interacción geometría × fotometría en freeze-80 frente a freeze-0. Todas las cifras están en pp.

| Contraste | Δ oficial | Δ sin Ibuphil |
| --- | ---: | ---: |
| Freezing | +9,93 | +11,35 |
| Geometría + Mosaic | +7,42 | +9,20 |
| Fotometría | −0,04 | −0,76 |
| Freezing × geometría | −11,25 | −12,86 |
| Freezing × fotometría | +3,01 | +3,44 |
| Geometría × fotometría | +1,68 | +3,35 |
| Freezing × geometría × fotometría | −3,94 | −4,51 |

| Añadir fotometría a | Δ oficial | Δ sin Ibuphil | Aporte de Ibuphil al Δ oficial | Δ precisión | Δ recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Freeze-0 sin geometría | −3,38 | −5,29 | +1,25 | −2,29 | −5,10 |
| Freeze-0 con geometría | +0,28 | +0,32 | 0,00 | +0,91 | +1,02 |
| Freeze-80 sin geometría | +1,61 | +0,41 | +1,25 | −5,72 | 0,00 |
| Freeze-80 con geometría | +1,32 | +1,51 | 0,00 | −2,62 | +2,04 |

El efecto medio fotométrico queda cerca de cero en la métrica oficial y cae en la sensibilidad a siete clases. Con freeze-0, la geometría modifica su efecto en **+3,65 pp oficiales**. Con freeze-80, pasar de sin a con geometría modifica el efecto fotométrico en **−0,29 pp** oficiales y **+1,10 pp** a siete clases. La interacción triple recoge ese cambio. Su estimación es exploratoria, con una ejecución por celda y el mismo conjunto de validación. [Factorial completo por clase](results/analysis/b/factorial.csv) · [Contrastes de AP50/AP75, precisión y recall](results/analysis/b/efectos_metricas.csv) · [Figura de fotometría por receta](results/analysis/b/fotometria_por_receta.png)

### Qué sostiene la mejora

- **Freeze-80 con geometría y fotometría.** El AP de Ibuphil 600 mg sigue en 0,90. Mejoran cinco de las otras siete clases, sobre todo `white` (**+6,03 pp**), Xyzall (**+5,32**) e Ibuphil Cold (**+3,53**). Bajan `blue` (**−3,68**) y Cipro (**−1,51**). Se recuperan dos TP y se suman tres FP. El mAP interno llega a 0,8708 en la época 95 y a 0,8658 en la 91. El último es 0,7761. La ganancia en `best` coexiste con variación importante entre épocas.
- **Freeze-80 con fotometría y sin geometría.** Ibuphil aporta **1,25 de los 1,61 pp** del aumento oficial. Su mejor IoU sube de 0,934 a 0,977 y la detección sigue bajo el umbral operativo. Mejoran Cipro (+4,49 pp), `blue` (+4,35) y `white` (+2,30). Las otras cuatro clases comunes bajan. El recall conserva 86 TP y los FP se duplican de 6 a 12. Su curva final es más estable, con nueve épocas a menos de 1 pp del máximo, pero su ganancia operativa queda por confirmar.
- **Freeze-0.** Sin geometría la fotometría degrada ambas métricas agregadas y pierde cinco TP. Con geometría apenas mejora el mAP. `white` (+15,75 pp) y Xyzall (+7,75) compensan pérdidas como Ibuphil Cold (−10,25) y Cipro (−7,45). Ninguna de las dos recetas freeze-0 con fotometría alcanza el criterio de C.

La latencia media de los ocho brazos queda entre 10,71 y 11,00 ms por imagen en asignaciones A100 independientes. Los cuatro entrenamientos con fotometría sumaron 57,72 min internos, con un pico asignado de unos 4,76 GiB por brazo. [AP, TP, FP y FN por clase](results/analysis/b/por_clase.csv) · [Diagnóstico de Ibuphil](results/analysis/b/ibuphil.csv)

## Fase D · Repetibilidad de Cascade a 100 épocas

| Semilla | mAP COCO | mAP sin Ibuphil | AP Ibuphil | AP50 / AP75 | Precisión (%) | Recall (%) | TP/FP/FN | Época best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2026 · EXP-008 | 0,8996 | 0,8996 | 0,9000 | 0,9793 | 91,18 | 94,90 | 93/9/5 | 46 |
| 2027 | 0,8379 | 0,8933 | 0,4500 | 0,9187 | 90,29 | 94,90 | 93/10/5 | 34 |
| 2028 | 0,8331 | 0,8878 | 0,4500 | 0,9166 | 87,74 | 94,90 | 93/13/5 | 26 |
| **Media ± DE muestral** | **0,8569 ± 0,0371** | **0,8936 ± 0,0059** | **0,6000 ± 0,2598** | **0,9382 ± 0,0356** | **89,73 ± 1,79** | **94,90 ± 0,00** | — | — |

**El 0,8996 oficial fue excepcional entre estas tres semillas.** Ibuphil explica 5,625 pp de las caídas oficiales de 6,18 pp en 2027 y 6,66 pp en 2028, es decir, el 91,1 % y el 84,5 %, respectivamente. La brecha sin Ibuphil baja a 0,63 y 1,18 pp. Las tres semillas conservan 93 TP. En las siete clases comunes, destacan las caídas de Xyzall (−1,93 pp) en 2027 y de Ibuphil Cold (−3,18 pp) y `blue` (−2,03 pp) en 2028. `white` sube en las dos réplicas. [Diferencias entre semillas](results/analysis/d/diferencias_semillas.csv) · [AP y dispersión por clase](results/analysis/d/diferencias_por_clase.csv) · [Figura por clase](results/analysis/d/ap_por_semilla.png)

La detección correcta de Ibuphil tiene IoU **0,918 / 0,932 / 0,943**. Cada una alcanza nueve de los diez umbrales COCO, hasta 0,90. En 2026 ocupa el primer puesto por score. En 2027 la precede un falso positivo sobre un Ibuphil Cold y en 2028 uno sobre una pastilla `pink`. En las dos réplicas la precisión en el punto de recuperación de esa instancia es 1/2 en lugar de 1. El AP pasa de 0,90 a 0,45 aunque la caja correcta sea algo más ajustada. Las tres detecciones correctas quedan con score <0,25 y son FN en el punto operativo.

El mAP de última época queda en **0,8159 / 0,8062 / 0,8052**, mientras el máximo se selecciona entre las épocas 26 y 46. En 2026 seis épocas quedan a menos de 1 pp de su propio máximo. Esa semilla fue la mejor entre las tres y su ventaja oficial se concentra en la única anotación de Ibuphil. La estabilidad de 0,8878–0,8996 en las otras siete clases respalda conservar la receta Cascade de 100 épocas como candidata. Las tres semillas comparten validación y solo describen variabilidad de entrenamiento en esas imágenes. [Ranking e IoU de Ibuphil](results/analysis/d/ibuphil.csv) · [Curvas por semilla](results/analysis/d/rendimiento.png) · [PR de las tres semillas](results/analysis/d/)

## Fase C · Diseño definitivo propuesto

**Seis recetas × dos semillas nuevas = 12 entrenamientos.** Para cada una de 2027 y 2028, repetir el factorial completo freeze-0/80 × geometría sí/no sin fotometría. Añadir las dos recetas freeze-80 con fotometría junto con sus controles exactos. Ambas cumplen el filtro de B de ≥1 pp oficial y sensibilidad sin Ibuphil no negativa. El brazo con los tres factores tiene prioridad por su mejora en las siete clases. El brazo sin geometría permite comprobar si la ganancia se repite cuando la señal de 2026 depende más de Ibuphil y cae la precisión operativa.

Los seis brazos de 2026 ya están reproducidos. Presentar primero los contrastes de las dos semillas nuevas, y después la media y DE muestrales de las tres con 2026 identificado como exploratorio. Calcular efectos emparejados por semilla de freezing y de geometría en el factorial sin fotometría. Para los dos brazos freeze-80 con fotometría, comparar con su pareja sin ella y estimar geometría × fotometría a freeze-80. La interacción triple completa y el efecto fotométrico a freeze-0 quedan como señales exploratorias de B. Mantener los criterios de adopción fijados y el punto operativo común. Cada receta conserva 100 épocas, pesos y datos con las huellas auditadas, selección de `best` por mAP de validación y el mismo protocolo COCO. [Recetas, contrastes y referencias de C](results/analysis/b/fase-c-propuesta.json) · [Criterios de adopción](README.md#fase-c--confirmación-emparejada)

## Fase C · Resultados de la confirmación

**Estado de la evidencia.** El tar recibido tiene SHA-256 `e2c827a78bfd16602caefddc312eae9f09f8e528d61f9ac9614cae6291ab1550` y el importador verificó sus **187 archivos inventariados**. Los trabajos `12031892–12031916` constan como `COMPLETED` en el primer intento. Se verificaron sus recibos, sus 24 dependencias, el controlador, el snapshot histórico, el smoke nativo, los datos, las recetas efectivas y la concordancia de los hashes de `best`. Las 12 evaluaciones de C y las seis referencias de 2026 reprodujeron COCO, AP por clase y PR sin modificar las predicciones. La contabilidad adicional certifica **25 filas `COMPLETED|0:0`**, con exactamente los mismos IDs del ledger y sin duplicados. SHA-256 de la contabilidad: `ceb10221a20011f4de90d2cbaa6bc4dfc1add09d1727a1209a07bedbd6bb9990`. Se reprodujeron los 18 resultados en modo estricto y se confirmó la selección. [Auditoría certificada de C](results/analysis/c/auditoria.json) · [Comprobación de contabilidad](../../docs/campaigns/exp-009-c.md#comprobación-del-código-de-salida)

### Resultados de las seis recetas

Los tripletes siguen el orden **2026 exploratoria / 2027 / 2028**. «±» indica desviación estándar muestral de tres semillas, no un intervalo de confianza. Precisión y recall usan score ≥0,25 e IoU ≥0,50. El CSV contiene AP50, AP75, métricas operativas, memoria y latencia por semilla. Todos los modelos YOLO se entrenaron durante 100 épocas.

| Receta | mAP oficial por semilla | mAP oficial, media ± DE | mAP sin Ibuphil por semilla | mAP sin Ibuphil, media ± DE | Precisión / recall medios | Latencia media (ms) | Época best |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| A · Control | 0,7078 / 0,7020 / 0,7182 | 0,7093 ± 0,0082 | 0,6803 / 0,6594 / 0,6923 | 0,6773 ± 0,0166 | 79,74 / 70,41 % | 10,77 | 38 / 35 / 30 |
| B · Afín + Mosaic | 0,8200 / 0,8395 / 0,8315 | 0,8303 ± 0,0098 | 0,8086 / 0,8165 / 0,8182 | 0,8144 ± 0,0051 | 87,89 / 92,52 % | 10,99 | 86 / 73 / 73 |
| C · Freeze-80 | 0,8384 / 0,8352 / 0,8393 | 0,8377 ± 0,0021 | 0,8296 / 0,8117 / 0,8307 | 0,8240 ± 0,0107 | 90,73 / 88,10 % | 10,83 | 98 / 89 / 100 |
| **D · Freeze-80 + afín + Mosaic** | **0,8578 / 0,8839 / 0,8895** | **0,8771 ± 0,0169** | **0,8518 / 0,8816 / 0,8880** | **0,8738 ± 0,0193** | **94,89 / 94,56 %** | 11,08 | 89 / 97 / 100 |
| E · Freeze-80 + foto | 0,8545 / 0,7686 / 0,8215 | 0,8149 ± 0,0433 | 0,8337 / 0,7641 / 0,8103 | 0,8027 ± 0,0354 | 89,09 / 82,99 % | 10,90 | 89 / 16 / 33 |
| F · Freeze-80 + afín + Mosaic + foto | 0,8710 / 0,8155 / 0,8817 | 0,8561 ± 0,0355 | 0,8669 / 0,8035 / 0,8648 | 0,8451 ± 0,0360 | 88,09 / 93,54 % | 10,88 | 95 / 19 / 97 |

[Resultados por semilla](results/analysis/c/por_semilla.csv) · [Media y DE por métrica](results/analysis/c/resumen.csv) · [AP y errores por clase](results/analysis/c/por_clase.csv) · [Media y DE de AP por clase](results/analysis/c/media_ap_por_clase.csv) · [Comparación de recetas](results/analysis/c/map_por_receta.png) · [PR del candidato](results/analysis/c/pr-d-s2027.png)

### Contrastes emparejados e interpretación

Los efectos están en puntos porcentuales. La última columna compara solo las dos semillas nuevas.

| Contraste y métrica | 2026 | 2027 | 2028 | Media ± DE de tres | Media 2027/28 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F−D · oficial | +1,32 | −6,84 | −0,78 | −2,10 ± 4,23 | −3,81 |
| F−D · sin Ibuphil | +1,51 | −7,81 | −2,32 | −2,88 ± 4,68 | −5,07 |
| E−C · oficial | +1,61 | −6,67 | −1,78 | −2,28 ± 4,16 | −4,22 |
| E−C · sin Ibuphil | +0,41 | −4,76 | −2,04 | −2,13 ± 2,59 | −3,40 |
| D−B · oficial | +3,78 | +4,44 | +5,80 | +4,68 ± 1,03 | +5,12 |
| D−B · sin Ibuphil | +4,32 | +6,51 | +6,99 | +5,94 ± 1,42 | +6,75 |

**Fotometría.** Ninguno de los dos beneficios observados en 2026 se confirma en las semillas nuevas. En F−D destaca la caída media de Cipro (−6,44 pp), `blue` (−5,64) e Ibuphil Cold (−5,62). En promedio, F sacrifica 6,80 pp de precisión y 1,02 pp de recall frente a D. E pierde 5,10 pp de recall medio ante C. Las caídas persisten al excluir Ibuphil. El resultado se limita a la fotometría, duraciones y geometría estudiadas.

**Freezing con geometría.** D−B mejora en las tres semillas, con ganancia media de AP en las siete clases comunes. Xyzall (+9,29 pp), `blue` (+8,69) y Cipro (+6,65) son las que más ganan. El AP de Ibuphil cae 4,17 pp en promedio. Por eso la ganancia de D resulta mayor al excluirlo. Este patrón respalda la receta en las 90 imágenes de validación. La estimación de tres semillas comparte dichas imágenes. [Figura de contrastes prioritarios](results/analysis/seleccion/contrastes_prioritarios.png) · [Todos los contrastes](results/analysis/c/contrastes.csv) · [Contrastes por clase](results/analysis/c/contrastes_ap_por_clase.csv)

### Descongelación, estabilidad y alternativas

Los 12 brazos con freeze-80 registran caída de mAP interno al pasar de la época 80 a la 81. Las caídas oscilan entre −0,21 y −16,96 pp. En D son **−2,63 / −6,81 / −4,51 pp**. La auditoría reproduce el cambio de BN y los contadores de AdamW, scheduler y EMA. D elige `best` tras abrir el backbone en las tres semillas. Sus medias de las diez últimas épocas son **0,7967 / 0,8617 / 0,8727**. Las épocas a menos de 1 pp del máximo interno son **1 / 1 / 2**. En 2027 el último valor interno es 0,8691, a 1,41 pp del `best` interno. En E, los mejores checkpoints de 2027 y 2028 caen en las épocas 16 y 33, con el backbone aún congelado. También ocurre en F de 2027, época 19. Estas curvas muestran que abrir el backbone puede preceder tanto a una recuperación como a un máximo sin recuperar. El diseño no identifica una causa a partir de estas oscilaciones. [Curvas con los puntos best](results/analysis/c/curvas_por_receta.png) · [Ventanas de descongelación](results/analysis/c/descongelacion.csv)

**Aplicación de la regla prefijada.** D supera los criterios frente a A (**+16,77 pp** oficiales, **+19,65 pp** a siete) y frente a B. También es el mayor mAP oficial medio entre las recetas elegibles. A queda como referencia, B y C pasan sus comparaciones frente a A pero quedan por debajo de D. E y F quedan descartadas por sus contrastes directos E−C y F−D. Dentro de D, **la semilla mediana es 2027**, con `best` en la época **97**, mAP oficial **0,8839** y **0,8816** a siete clases. La semilla 2028 obtuvo el mayor mAP (0,8895). [Propuesta de checkpoint YOLO](results/analysis/c/candidato-yolo-valid.json)

### Los dos candidatos en validación

| Receta y cohorte de tres semillas | mAP oficial | mAP sin Ibuphil | Precisión | Recall | Latencia media |
| --- | --- | --- | ---: | ---: | ---: |
| YOLO D | 0,8771 ± 0,0169 | 0,8738 ± 0,0193 | 94,89 % | 94,56 % | 11,08 ms/img |
| Cascade control 100 épocas | 0,8569 ± 0,0371 | 0,8936 ± 0,0059 | 89,73 % | 94,90 % | 29,19 ms/img |

El orden del mAP cambia al excluir Ibuphil. YOLO D presenta mayor precisión media; Cascade conserva una ventaja pequeña de recall. Las latencias se midieron con el mismo protocolo y modelo de GPU en asignaciones independientes. En los dos checkpoints propuestos para la evaluación final, YOLO registra **0,8839 / 0,8816**, precisión **95,83 %**, recall **93,88 %** y **11,28 ms/img**, mientras Cascade registra **0,8379 / 0,8933**, **90,29 %**, **94,90 %** y **29,09 ms/img**, respectivamente. La comparación de validación motiva evaluar ambos en la partición reservada una vez aprobados los candidatos. [Comparación por semilla](results/analysis/seleccion/comparacion_por_semilla.csv) · [Figura de sensibilidad](results/analysis/seleccion/comparacion_validacion.png) · [Registro de ambos checkpoints](results/analysis/seleccion/seleccion-valid.json)


## Evaluación final en test

Los dos `best` de la semilla mediana 2027 se evaluaron en la misma A100, con 45 imágenes y 47 anotaciones. El mAP COCO reproducido fue **0,8971** para YOLO D y **0,9189** para Cascade. Ambos recuperaron las 47 instancias. YOLO obtuvo **97,92 %** de precisión y **93,09 FPS**, frente a **94,00 %** y **34,20 FPS** de Cascade. Ibuphil 600 mg tiene AP indefinido por falta de anotaciones. [Informe final, tablas y figuras](final-results.md)
