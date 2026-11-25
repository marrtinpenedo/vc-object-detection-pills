# EXP-009 · Optimización final de YOLOv10-S y Cascade R-CNN

**Estado.** Fase A reproducida en `valid` para 14 brazos YOLO en semilla 2026. Freeze-80 queda nominado para B. B y D están reproducidas en validación. C está cerrada en validación con 18 evaluaciones reproducidas y 25 trabajos certificados como `COMPLETED 0:0`. Los dos finalistas fijados se evaluaron en test y sus métricas COCO fueron reproducidas desde las predicciones. [Informe final](final-results.md) · [Evidencia y reproducción local](results/evidence/README.md) · [Manifiesto A](../../configs/campaign/exp-009-a.json) · [Operación y recuperación](../../docs/campaigns/exp-009-a.md) · [Hallazgos](findings.md) · [Auditoría de descongelación](thaw-audit.md) · [Operación de B y D](../../docs/campaigns/exp-009-b-d.md)

## Evidencia previa y decisión

| Experimento | Evidencia en validación | Decisión para EXP-009 |
| --- | --- | --- |
| [EXP-001](../exp-001-yolo-lr/findings.md) | Tres parejas. LR 0,0005 reduce el mAP oficial medio 5,03 pp frente a 0,001, aun con mejora a siete clases. | Mantener LR 0,001. |
| [EXP-002](../exp-002-photometric/findings.md) | Una semilla. Fotometría suave cambia el mAP oficial −10,78 pp en YOLO y +0,34 pp en Cascade. La ganancia de Cascade no se replica y conserva el mismo punto operativo. | Probar la interacción fotométrica después de A. |
| [EXP-003](../exp-003-ensemble/findings.md) | Una pareja de modelos. El ensemble gana +3,08 pp oficial y pierde −1,21 pp sin Ibuphil frente a Cascade. | Mantener la optimización por detector. |
| [EXP-004](../exp-004-yolo-factorial/findings.md) | Una semilla. Afín + Mosaic a 100 épocas gana +11,22 pp oficiales y +12,82 pp a siete. Habilitar Mosaic frente a solo afín suma +3,95 y +4,51 pp en ese presupuesto. | Conservar la política auditada como factor agrupado. |
| [EXP-005](../exp-005-cascade-geometric/findings.md) | Una semilla. La afín de 30 épocas gana +8,18 pp oficiales, con +9,38 pp de aporte de Ibuphil y −1,37 pp en el promedio de siete. Con 100 épocas también empeoran las siete clases. | Cascade sin afín como candidato por defecto. |
| [EXP-006](../exp-006-yolo-warmup/findings.md) | Una pareja. Warm-up de cinco épocas cambia el mAP oficial −0,87 pp y el de siete −1,00 pp. | Mantener el warm-up original. |
| [EXP-007](../exp-007-yolo-staged-transfer/findings.md) | Freeze-15 se confirma con dos semillas adicionales. Freeze-20 gana en tres semillas y promedia 0,8373, +9,23 pp frente a su control. Su elección de duración se hizo sobre esas semillas y con 30 épocas totales. | Barrer duraciones con presupuesto común de 100. La extrapolación de EXP-007 es una hipótesis. |
| [EXP-008](../exp-008-cascade-freezing/findings.md) | Una semilla. Las seis variantes de freezing quedan por debajo de sus controles, también a siete clases. El control de 100 obtiene 0,8996 en ambas métricas al redondear. | Mantener Cascade sin freezing y estimar estabilidad del control largo. |

Los controles históricos contextualizan las magnitudes. Los contrastes usan controles contemporáneos. Ibuphil 600 mg tiene una única instancia de validación y su AP pesa 1/8 en el mAP oficial. Tres semillas sobre las mismas imágenes informan de la variación entre entrenamientos.

## Fase A · Freezing × geometría a 100 épocas

El [manifiesto](../../configs/campaign/exp-009-a.json) cruza siete duraciones, 0, 10, 20, 40, 60, 80 y 100 épocas, con dos recetas, control y afín + Mosaic. Todas usan YOLOv10-S, semilla 2026, COCO preentrenado, imágenes de 640, batch 8, FP32, AdamW, LR 0,001, warm-up original y coseno. Cada entrenamiento usa 40 lotes por época y 4 000 actualizaciones nominales. El término del warm-up se fija por el mínimo nativo de 100 índices de lote. El coseno llega a un factor final nominal 0,05. Con presupuestos distintos, incluso los LR de épocas compartidas pueden cambiar. [Bucle nativo de Ultralytics 8.3.161](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/trainer.py#L319-L389)

| Épocas congeladas | Control | Afín + Mosaic | Primera época entrenable del backbone |
| ---: | --- | --- | --- |
| 0 | `f000-control` | `f000-geom` | 1 |
| 10 | `f010-control` | `f010-geom` | 11 |
| 20 | `f020-control` | `f020-geom` | 21 |
| 40 | `f040-control` | `f040-geom` | 41 |
| 60 | `f060-control` | `f060-geom` | 61 |
| 80 | `f080-control` | `f080-geom` | 81 |
| 100 | `f100-control` | `f100-geom` | — |

La política es la misma que en EXP-004. Mosaic nativo, probabilidad 0,30 y `close_mosaic=0`, precede una afín con intento 0,5, rotación ±8°, traslación ±4 %, escala 0,90–1,10 y flip horizontal condicional 0,5. La auditoría estima un 42,9 % de afín efectiva con Mosaic. Se conservan el rechazo de cajas inválidas, la asociación clase/caja y el recorte de teselas. La validación usa `LetterBox → Format` y la evaluación común lee solo `valid`. [Política y auditoría geométrica](../../docs/campaigns/exp-003-005/geometric-policy.md#auditoría-de-la-receta-sobre-train)

Se reutiliza el entrenador por etapas de EXP-007. AdamW se construye con los parámetros del backbone ya en sus grupos. El entrenador congela parámetros y estadísticas de BN de los módulos 0–10. Neck y cabezas siguen activos. Al comienzo de la época N+1 descongela el backbone con el mismo AdamW, scheduler y EMA. Freeze-100 conserva la huella inicial hasta el final y registra `thaw_epoch: null`. El preflight hace un sondeo del control y dos smokes de tres épocas con pesos reales y afín + Mosaic, uno con transición y otro permanente. [Auditoría del entrenador](../../scripts/yolo_staged.py)

### Lectura de A y nominación para B

Para cada duración `f`, el efecto geométrico es `geom(f) − control(f)`. Se calcula el efecto de congelar frente a duración 0 dentro de cada receta y la interacción `[geom(f) − control(f)] − [geom(0) − control(0)]`. El análisis reproduce COCO y PR desde predicciones, valida SHA-256 de `best` y audita Batch Normalization, EMA, AdamW y grupo de descongelación. Exporta AP por clase, dos mAP, AP50/AP75, precisión, recall, TP/FP/FN, confusiones, épocas `best` y final, ventanas de transición, tiempo y memoria. Las latencias se miden con el protocolo anterior en asignaciones independientes.

Una duración positiva es elegible si en alguna celda supera el brazo de duración 0 de esa misma receta en **al menos 1 pp oficial** y el promedio de siete clases es **igual o superior**. Se ordenan las celdas elegibles por mAP oficial, mAP de siete clases, menor duración y control ante un empate completo. `fase-b-propuesta.json` registra una propuesta y las cuatro referencias reutilizables, con sus recetas, trabajos y hashes. Si ninguna celda cumple, B queda en espera de revisión. Esta nominación es exploratoria.

## Fase B · Fotometría e interacciones

B cruza `F ∈ {0, 80}`, `G ∈ {sin augmentación, afín + Mosaic}` y `P ∈ {sin fotometría, fotometría EXP-002}`, con 100 épocas y semilla 2026. Las cuatro celdas sin fotometría provienen de A si coinciden los hashes del código de entrenamiento y evaluación, los datos y pesos, las recetas efectivas y el SHA-256 de cada `best`. El manifiesto fija las cuatro evaluaciones de A por inventario, trabajos, recetas y huellas. B emplea el código inmutable de A y entrena cuatro celdas nuevas. El analizador comprueba también los parámetros efectivos, las predicciones y las curvas PR.

La fotometría de brillo y contraste ±10 % tiene probabilidad 0,5. La composición es `Mosaic → afín → fotometría → LetterBox → Format` cuando están activos los tres pasos iniciales. El ajuste de luminancia se calcula sobre la imagen que llega tras los pasos geométricos y deja las anotaciones intactas. Las cuatro celdas con fotometría usan la misma regla de semilla separada por worker. La afín conserva su flujo aleatorio histórico. El flujo independiente de fotometría ya está en el snapshot de A. Las repeticiones exactas de EXP-002 usan su código archivado. La prueba del cargador verifica composición, píxeles, cajas, clases y validación sin augmentación. Un preflight obligatorio verifica la cadena conjunta en un smoke de tres épocas con freezing, descongelación y EMA. [Prueba de composición](../../tests/test_geometry.py)

Se estimará `P` dentro de cada celda `F×G` y las diferencias de diferencias. A y el factorial B identifican la política conjunta. La contribución de Mosaic separado de la afín se estudia con la ablación de solo afín sobre la duración elegida, si responde a una duda de A/B. También queda prevista la ablación Mosaic hasta la época 90, manteniendo la afín en 91–100. Esta segunda ablación requiere un preflight adicional. El cierre nativo vuelve a construir la cadena de transformaciones y el cargador del proyecto debe conservar su subcadena afín. Ninguna ablación está en el DAG de A. [Cierre nativo de Mosaic](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/engine/trainer.py#L737-L745)

## Fase C · Confirmación emparejada

[Operación y recuperación de C](../../docs/campaigns/exp-009-c.md) · [Manifiesto](../../configs/campaign/exp-009-c.json) · [Protocolo de selección final](final-evaluation.md)

Confirmar las cuatro recetas sin fotometría en semillas 2027 y 2028: control, freeze-80, afín + Mosaic y freeze-80 + afín + Mosaic. Conservar los resultados de 2026 identificados como exploración y mostrar por separado las dos nuevas semillas. Se añaden hasta dos celdas fotométricas de B si cada una mejora al menos 1 pp oficial y mantiene el promedio de siete clases frente a su pareja sin fotometría. Si cumplen más de dos, priorizar mayor mAP oficial, mayor mAP de siete y finalmente menor complejidad. En B cumplen los dos brazos freeze-80 con fotometría. C queda fijada en seis recetas y 12 entrenamientos. Informar por separado los contrastes de 2027/2028 y el resumen que incorpora 2026 como semilla exploratoria. [Diseño resultante](findings.md#fase-c--diseño-definitivo-propuesto)

Informar medias y desviaciones estándar muestrales, diferencias emparejadas y signos, AP por clase, precisión, recall y latencia. Para adoptar una mejora sobre el control pertinente, exigir media oficial superior en al menos 1 pp y media a siete clases igual o superior, con signo favorable oficial en al menos dos semillas. Entre las recetas elegibles, priorizar el mAP oficial medio y explicar las diferencias de sensibilidad, errores y punto operativo. Si ninguna cumple el criterio, retener el control. Reservar la selección del checkpoint concreto hasta cerrar los análisis de validación.

## Fase D · Estabilidad de Cascade

Replicar en 2027 y 2028 la receta de EXP-008 de 100 épocas sin freezing ni geometría. Usar el snapshot archivado de EXP-008 para añadir solo la semilla y mantener la comparabilidad con 2026. La fotometría de EXP-002 aportó una señal de una sola semilla, con AP50 y AP75 ligeramente inferiores. D contiene solo las dos réplicas sin augmentación. Una pareja adicional requeriría una hipótesis sobre estabilidad o contraste a 100 épocas después de la réplica. Aplicaría la misma política ±10 % y emparejaría por semilla con el control de 100 épocas. No hay trabajos Cascade ni confirmaciones en el DAG de A.

## Cierre de la selección

Antes de evaluar `test`, registrar el candidato de cada detector, receta efectiva, semilla, procedencia y hash del checkpoint. El punto operativo de diagnóstico permanece en score ≥0,25 e IoU ≥0,50. El protocolo COCO independiente usa las predicciones completas desde el umbral de exportación y queda fijado antes de abrir `test`. Tras esa evaluación se conservan las recetas registradas.
