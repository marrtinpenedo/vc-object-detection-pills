# EXP-002 · Hallazgos

## Verificación local

- La variante de cada detector difiere de su control únicamente en el bloque fotométrico. EXP-001 conserva la configuración original.
- Las pruebas del operador verifican que brillo y contraste producen efectos separados, que se conservan las proporciones entre canales dentro del error de redondeo y que los sorteos son reproducibles por semilla de worker. El operador mantiene el tipo, tamaño y orden de los píxeles y deja intacta la entrada.
- La [vista previa](results/preview.jpg) contiene originales y dos variantes de las ocho clases con las mismas cajas COCO dibujadas. Los píxeles con un canal cerca de 255 tienen menos margen para aclararse. En tonos muy oscuros, el redondeo de cada canal representa una fracción mayor de su valor.
- La suite local completó 21 pruebas y omitió tres. Quedan por ejecutar los dos tests de los cargadores de EXP-002 y el test previo de inicialización de Cascade en sus entornos nativos. [Pruebas de EXP-002](../../tests/test_photometric.py)

## Resultados en validación

Pendientes de los cuatro entrenamientos y evaluaciones.
