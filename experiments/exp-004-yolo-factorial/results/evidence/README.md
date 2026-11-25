# Evidencia compartida de EXP-004 y EXP-005

`campaign-003-005-results-recovered.tar.gz` es la exportación selectiva original. Incluye el inventario SHA-256, los manifiestos, el registro SLURM, las configuraciones y las predicciones, curvas PR e historiales de validación. Los pesos, entornos y datos de imágenes no forman parte de la exportación. Se guarda una sola copia para los dos factoriales.

Los otros tres archivos son el código desplegado original, la copia completa corregida y el parche incremental. Los descriptores que identifican el código de ejecución son `2b98ece5c736735e4602a992c1847958be74cbee6a52b3daa03143cf91f1b3d9` para Cascade y `23e048675002dd0bbd678b184ef674288813a10517179fedb32914caad762978` para YOLO. Los archivos conservan las rutas que figuran en los manifiestos históricos.

La tabla conjunta, las referencias, la figura comparativa y la auditoría de las nueve evaluaciones están en este directorio. Los resultados por detector se guardan en los directorios `results/analysis/` de [EXP-004](../analysis/) y [EXP-005](../../../exp-005-cascade-geometric/results/analysis/). Las copias regeneradas y las extracciones locales se escriben en el directorio `results/` de la raíz, excluido de Git.

[Huellas de los archivos](SHA256SUMS) · [Protocolo y reproducción](../../../../docs/campaigns/exp-003-005/findings.md#reproducción-local)
