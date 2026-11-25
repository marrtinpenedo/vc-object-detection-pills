# EXP-008 · Congelación de ResNet en Cascade R-CNN

**Cerrado en validación, sin pasar a confirmación.** Las seis variantes de freezing quedan por debajo de su control emparejado en el mAP oficial y en la sensibilidad a siete clases. El control de 100 épocas obtuvo **0,8996** en ambas métricas al redondear a cuatro decimales.

## Procedencia y protocolo

Reproducción local con el importador y el analizador oficial. El tar de resultados tiene SHA-256 `0bf53563fccdbfc95b24de4038bcd00e71c5cc13e3013c93bc9e38d20d578cda`. Se verificaron 134 archivos, el descriptor del despliegue `8d479bfd14b079057f12d94fee5e3d412a3b0905ea114f39509695965f0ef170`, los pesos iniciales, el COCO de validación y las huellas de los ocho `best`. El registro tiene 17 trabajos `COMPLETED` en el primer intento, con preflight `12022434`. El recálculo desde las predicciones coincide con las métricas y curvas PR exportadas. La semilla es 2026. Se usaron 90 imágenes, 98 instancias y ocho clases de `valid`. Los trabajos de entrenamiento usaron A100 PCIe de 40 GB en varias asignaciones. [Auditoría](results/analysis/auditoria.json) · [Evidencia y reproducción](results/evidence/README.md) · [Protocolo](README.md)

El smoke nativo congeló la primera iteración y empezó a actualizar ResNet en la segunda. En los cuatro brazos temporales, el primer paso descongelado cambió la huella de ResNet, creó 53 buffers de momento y conservó los 46 buffers de las ramas activas. La auditoría verificó la continuidad del scheduler y las huellas de los buffers FrozenBN. En freeze-30 y freeze-100, los pesos de ResNet conservaron su huella inicial durante 2 370 y 7 900 pasos, respectivamente.

## Métricas reproducidas

Precisión, recall y recuentos usan score ≥0,25 e IoU ≥0,50. El mAP oficial incluye ocho clases. El de siete es una sensibilidad que excluye Ibuphil 600 mg.

| Épocas | Freeze | mAP oficial | mAP sin Ibuphil | AP50 | AP75 | Precisión | Recall | TP/FP/FN | Época best | mAP última época |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 0 | 0,7927 | 0,8953 | 0,8654 | 0,8654 | 92,08 % | 94,90 % | 93/8/5 | 21 | 0,7919 |
| 30 | 10 | 0,7814 | 0,8869 | 0,8632 | 0,8632 | 90,38 % | 95,92 % | 94/10/4 | 21 | 0,7753 |
| 30 | 20 | 0,7819 | 0,8861 | 0,8649 | 0,8649 | 86,24 % | 95,92 % | 94/15/4 | 9 | 0,7794 |
| 30 | 30 | 0,7839 | 0,8852 | 0,8695 | 0,8695 | 87,85 % | 95,92 % | 94/13/4 | 21 | 0,7803 |
| 100 | 0 | 0,8996 | 0,8996 | 0,9793 | 0,9793 | 91,18 % | 94,90 % | 93/9/5 | 46 | 0,8159 |
| 100 | 20 | 0,8064 | 0,8958 | 0,8816 | 0,8816 | 89,52 % | 95,92 % | 94/11/4 | 48 | 0,7882 |
| 100 | 50 | 0,7951 | 0,8903 | 0,8764 | 0,8764 | 89,52 % | 95,92 % | 94/11/4 | 42 | 0,7853 |
| 100 | 100 | 0,7906 | 0,8875 | 0,8740 | 0,8740 | 88,68 % | 95,92 % | 94/12/4 | 31 | 0,7820 |

Todas las variantes congeladas recuperan un TP de `white` frente a su control y aumentan el total de FP entre 2 y 7. El recall sube 1,02 pp en los seis brazos y la precisión baja entre 1,65 y 5,84 pp. El AP de `white` mejora con todas las variantes. Cipro 500, Xyzall 5mg, Ibuphil Cold y `blue` pierden AP en los seis contrastes. [Resumen CSV](results/analysis/resumen.csv) · [AP50, AP75 y errores por clase](results/analysis/por_clase.csv) · [mAP por presupuesto](results/analysis/diagnosticos/map_por_presupuesto.png)

### AP COCO por clase

| Clase | 30 control | 30 f10 | 30 f20 | 30 f30 | 100 control | 100 f20 | 100 f50 | 100 f100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cipro 500 | 0,8672 | 0,8476 | 0,8522 | 0,8508 | 0,8798 | 0,8557 | 0,8432 | 0,8425 |
| Ibuphil 600 mg | 0,0750 | 0,0429 | 0,0529 | 0,0750 | 0,9000 | 0,1800 | 0,1286 | 0,1125 |
| Ibuphil Cold 400-60 | 0,9458 | 0,9188 | 0,9425 | 0,9298 | 0,9549 | 0,9347 | 0,9401 | 0,9401 |
| Xyzall 5mg | 0,8200 | 0,7695 | 0,7550 | 0,7871 | 0,8270 | 0,8118 | 0,7748 | 0,7959 |
| blue | 0,9439 | 0,9374 | 0,9371 | 0,9270 | 0,9577 | 0,9423 | 0,9467 | 0,9402 |
| pink | 0,9324 | 0,9353 | 0,9525 | 0,9292 | 0,9510 | 0,9461 | 0,9439 | 0,9270 |
| red | 0,9180 | 0,9199 | 0,9145 | 0,9073 | 0,9172 | 0,9289 | 0,9096 | 0,8998 |
| white | 0,8395 | 0,8796 | 0,8486 | 0,8650 | 0,8095 | 0,8514 | 0,8740 | 0,8670 |

## Congelación temporal y permanente

| Presupuesto | Freeze | Δ mAP oficial (pp) | Δ sin Ibuphil (pp) | Aporte de Ibuphil al Δ oficial (pp) |
| --- | ---: | ---: | ---: | ---: |
| 30 | 10 | −1,13 | −0,84 | −0,40 |
| 30 | 20 | −1,08 | −0,92 | −0,28 |
| 30 | 30 | −0,88 | −1,01 | +0,00 |
| 100 | 20 | −9,33 | −0,38 | −9,00 |
| 100 | 50 | −10,45 | −0,93 | −9,64 |
| 100 | 100 | −10,90 | −1,21 | −9,84 |

- **30 épocas.** El control lidera en las dos métricas. La congelación permanente es la mejor entre los tres brazos congelados en mAP oficial (0,7839) y la última entre ellos a siete clases (0,8852). Freeze-10 lidera el grupo congelado a siete clases (0,8869). Ambas diferencias son pequeñas y las seis comparaciones contra los controles son negativas en las dos métricas.
- **100 épocas.** Freeze-20 lidera el grupo congelado en las dos métricas. Frente a la congelación permanente gana 1,57 pp oficiales y 0,83 pp a siete clases. Frente al control pierde 9,33 pp oficiales y 0,38 pp a siete clases. Freeze-50 y freeze-100 pierden 0,93 y 1,21 pp a siete clases frente al control. El criterio preregistrado de +1 pp oficial y sensibilidad no negativa queda incumplido en ambos presupuestos.
- **Ibuphil.** A 100 épocas el control sitúa la única detección correcta en el rango 1 del ranking, con IoU 0,9183 y score 0,00595. Los brazos congelados sitúan ese acierto en los rangos 5, 7 y 8, pese a IoU de 0,9366, 0,9256 y 0,9173. Los cuatro scores quedan por debajo de 0,25 y la instancia es FN al punto operativo en los ocho brazos. La pérdida de ranking explica entre 9,00 y 9,84 pp del déficit oficial de los tres congelados largos. [Contrastes](results/analysis/contrastes.csv) · [Δ AP por clase](results/analysis/diagnosticos/ap_delta_por_clase.png) · [Ranking de Ibuphil](results/analysis/diagnosticos/ranking_ibuphil.csv)

La congelación permanente redujo el tiempo interno de 12,25 a 10,42 min a 30 épocas y de 39,36 a 33,18 min a 100. La memoria pico asignada pasó de aproximadamente 2,95 a 1,85 GiB. Las asignaciones eran independientes. Este ahorro acompaña una pérdida de AP y el criterio de selección prioriza el rendimiento.

## Qué pasa al descongelar

| Presupuesto | Freeze | LR al descongelar | Δ mAP en primera época (pp) | Δ permanente en mismas épocas (pp) | Mejor antes | Mejor después |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 30 | 10 | 0,001875 | −1,04 | +0,13 | 0,7778 (e8) | 0,7814 (e21) |
| 30 | 20 | 0,000625 | −0,23 | +0,29 | 0,7819 (e9) | 0,7804 (e28) |
| 100 | 20 | 0,002261 | +0,27 | +0,75 | 0,7901 (e18) | 0,8064 (e48) |
| 100 | 50 | 0,001250 | −0,97 | +0,20 | 0,7951 (e42) | 0,7916 (e72) |

Tres de cuatro brazos pierden mAP al terminar su primera época descongelada. El cuarto, freeze-20 a 100 épocas, sube 0,27 pp en la primera y cae a 0,7588 en la segunda. En los cuatro, el cambio de mAP de la primera época descongelada es inferior a la variación del brazo permanente en esas mismas épocas. Más adelante los máximos de freeze-10 a 30 y freeze-20 a 100 superan sus respectivos picos congelados. Freeze-20 a 30 y freeze-50 a 100 eligen un checkpoint todavía congelado.

Los brazos permanentes y temporales ya tenían historiales de mAP distintos durante los prefijos congelados. La transición es real según pesos y SGD. La comparación de curvas en una sola semilla, con CUDA no determinista y distintas asignaciones, permite describir la dinámica y deja abierta su atribución causal. El scheduler conserva el horizonte original. El LR de freeze-20 al abrir ResNet en el brazo de 100 épocas es unas 3,62 veces el del brazo de 30. [Curvas de 30 épocas](results/analysis/entrenamiento-30.png) · [Curvas de 100 épocas](results/analysis/entrenamiento-100.png) · [Ventana comparada con congelación permanente](results/analysis/diagnosticos/ventana_descongelacion.png) · [Pérdidas y LR alrededor de la transición](results/analysis/transicion.csv)

## Referencias históricas y decisión

| Referencia de Cascade | mAP oficial | mAP sin Ibuphil | AP de Ibuphil |
| --- | ---: | ---: | ---: |
| Baseline histórica | 0,8021 | 0,8909 | 0,1800 |
| EXP-002 control | 0,7887 | 0,8914 | 0,0692 |
| EXP-005 A | 0,8012 | 0,8942 | 0,1500 |
| EXP-005 B | 0,8026 | 0,8851 | 0,2250 |
| EXP-005 C | 0,8830 | 0,8805 | 0,9000 |
| EXP-005 D | 0,8069 | 0,8579 | 0,4500 |

El control nuevo de 30 épocas está por debajo de la baseline histórica en mAP oficial (−0,93 pp) y por encima a siete clases (+0,43 pp). El control nuevo de 100 épocas supera a EXP-005 B en **9,71 pp oficiales** y **1,45 pp a siete clases**. Ibuphil explica 8,44 pp del salto oficial. Su `best` se seleccionó en la época 46. En el historial hay seis épocas por encima de 0,88 y el último mAP es 0,8159. Esa variación y la distancia entre tandas aconsejan conservar el control contemporáneo y replicar antes de generalizar la ventaja. El export contiene AP por clase de `best` y mAP agregado por época. No permite atribuir la variación de cada época a una clase concreta. [Referencias verificadas por huella de datos](results/analysis/diagnosticos/referencias_cascade.csv) · [Baseline](../baseline/findings.md) · [EXP-005](../exp-005-cascade-geometric/findings.md)

**Decisión.** Conservar el control Cascade de 100 épocas como referencia de EXP-008. El estudio queda cerrado sin confirmación con 2027 ni 2028. EXP-008 no designa una variante de freezing para la combinación final. La interacción con la afín no se midió. El conjunto de test permanece reservado.
