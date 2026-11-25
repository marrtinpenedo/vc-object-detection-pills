# EXP-008 · Congelación temporal y permanente de Cascade R-CNN

**Estado** Cerrado en `valid`, sin confirmación. Ninguna variante de congelación mejora su control emparejado. El control de 100 épocas obtuvo 0,8996 de mAP oficial y 0,8996 sin Ibuphil 600 mg. [Hallazgos](findings.md) · [Evidencia](results/evidence/README.md)

## Pregunta y protocolo

¿Conservar el extractor ResNet-50 preentrenado en COCO durante parte o todo el entrenamiento mejora la detección de pastillas? Se modifica solo la duración de congelación de `backbone.bottom_up`. FPN, RPN y las tres etapas Cascade permanecen entrenables. Usamos semilla 2026, las mismas 316 imágenes de entrenamiento, 90 de validación, ocho clases y la receta sin augmentación de los controles de EXP-005. La evaluación final de cada brazo reproduce COCO sobre `valid` y el checkpoint `best`.

| Presupuesto | Congelación inicial | Primera época descongelada | Primer paso descongelado (base 1) |
| --- | ---: | ---: | ---: |
| 30 | 0 | 1 | 1 |
| 30 | 10 | 11 | 791 |
| 30 | 20 | 21 | 1 581 |
| 30 | 30 | — | — |
| 100 | 0 | 1 | 1 |
| 100 | 20 | 21 | 1 581 |
| 100 | 50 | 51 | 3 951 |
| 100 | 100 | — | — |

Cada época lógica tiene 79 pasos de SGD con batch 4. Los presupuestos son 2 370 y 7 900 pasos, con warm-up de 79 y scheduler coseno. Se conserva el diseño propuesto porque compara congelación parcial corta, más larga y permanente dentro de cada horizonte. Las dos condiciones de 20 épocas congeladas comparten el paso de transición y difieren en el LR que corresponde al horizonte del coseno. Los efectos se estiman contra el control de su propio presupuesto. Las ocho ejecuciones son nuevas. EXP-005 A/B documentan la misma intención de receta, pero pertenecen a otro snapshot y otra tanda. Los usamos como contexto y generamos controles contemporáneos bajo el mismo manifiesto y auditoría. [EXP-005](../exp-005-cascade-geometric/findings.md) · [EXP-007](../exp-007-yolo-staged-transfer/findings.md) · [Manifiesto](../../configs/campaign/exp-008.json)

## Implementación y verificaciones

Detectron2 construye el modelo y SGD antes de cargar el checkpoint COCO. El constructor del optimizador omite los parámetros con `requires_grad=False`. Se conserva `MODEL.BACKBONE.FREEZE_AT=0` y se congela ResNet solo después de construir SGD y cargar los pesos. El hook activa de nuevo sus gradientes inmediatamente antes del primer minibatch de la época N+1. Las referencias al optimizador y al scheduler se mantienen, junto con sus grupos, LR y estados. Los buffers de momento de ResNet empiezan con sus primeras actualizaciones. [Construcción de DefaultTrainer](https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/engine/defaults.py#L332-L362) · [Filtro de parámetros](https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/solver/build.py#L186-L201)

El adaptador exige FPN con `bottom_up` ResNet de etapas 3/4/6/3 y FrozenBatchNorm2d. Durante la fase congelada, ResNet se mantiene en modo evaluación. Se restaura ese modo antes de cada paso congelado porque la validación periódica vuelve a poner el modelo en modo entrenamiento. Al descongelar cambia el modo del extractor y se conservan los buffers de FrozenBN de COCO. La huella de todos los pesos ResNet debe coincidir al inicio y al fin de cada época congelada. Cada paso congelado exige ausencia de gradientes y estado SGD en ResNet. El primer paso descongelado acredita gradientes útiles, cambio de pesos, creación de momento y continuidad del contador del coseno. Se comprueban además gradientes y actualizaciones en ramas activas. [Código del hook y auditoría](../../scripts/cascade_staged.py) · [FPN v0.6](https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/modeling/backbone/fpn.py#L63-L89)

Se conserva el evaluador, la validación cada 79 iteraciones y la publicación del primer máximo de mAP en caso de empate. Los controles omiten `freeze_backbone_epochs` y ejecutan la ruta habitual. Las variantes congeladas requieren una sesión continua. Un fallo se reintenta como ejecución nueva desde COCO y en otro directorio de intento. El tiempo interno incluye la auditoría de huellas y se interpreta como descriptivo. Detectron2 mantiene el modo CUDA de la baseline.

## Ejecución y reproducción

La [guía histórica de CESGA](../../docs/campaigns/exp-008.md) documenta el despliegue, el preflight, los reintentos y la recuperación. Los resultados pueden reproducirse en CPU con la [evidencia del repositorio](results/evidence/README.md) y las anotaciones originales.

## Criterio de lectura y confirmación

El análisis exige las ocho cadenas exitosas, valida el smoke y los cambios de fase, recalcula COCO desde predicciones y produce tablas de mAP, AP50/AP75, AP por clase, sensibilidad a siete clases, errores, mejor época, tiempo, memoria, PR y curvas por presupuesto. La tabla de transición cubre las épocas N−1 a N+2. La métrica primaria es mAP de ocho clases. El criterio fijado para pasar a confirmación exigía una mejora oficial de al menos 1 pp y una diferencia de siete clases no negativa respecto al control. Ningún brazo cumplió esos requisitos. EXP-008 queda cerrado tras la exploración con semilla 2026. El conjunto de test se reserva para la evaluación final. [Hallazgos](findings.md)
