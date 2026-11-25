# Evidencia de validación de EXP-008

`exp008-validation.tar.xz` es una recompresión sin pérdida de la exportación selectiva de CESGA. Conserva el flujo TAR y los 134 archivos del inventario original, incluidas las predicciones, los historiales, las configuraciones y la auditoría de congelación. El gzip original `exp008-results-20261125.tar.gz` se conserva fuera de Git y tiene SHA-256 `0bf53563fccdbfc95b24de4038bcd00e71c5cc13e3013c93bc9e38d20d578cda`.

`exp-008-cascade-freezing.tar.gz` conserva el código desplegado. La huella SHA-256 de su descriptor interior `snapshot.json` es `8d479bfd14b079057f12d94fee5e3d412a3b0905ea114f39509695965f0ef170`. Coincide con el descriptor archivado en la exportación. Los pesos y las imágenes no forman parte de estos paquetes. [Huellas de los paquetes](SHA256SUMS) · [Procedimiento de CESGA](../../../../docs/campaigns/exp-008.md)

Las [tablas, figuras y auditoría derivadas](../analysis/) se incluyen para consulta. El CSV consolidado por iteración `historial.csv` se genera al reproducir el análisis. Los historiales originales están en el paquete comprimido. [Cierre del experimento](../../findings.md)

## Reproducción local

Desde la raíz del repositorio, con las anotaciones originales bajo `data/pills-sxdht` o en la ruta indicada por `PILLS_DATASET`. Para analizar el gzip original, sustituir la ruta de `exp008-validation.tar.xz` al importar.

```bash
(
  set -euo pipefail
  evidence=experiments/exp-008-cascade-freezing/results/evidence
  data="${PILLS_DATASET:-data/pills-sxdht}"
  (cd "$evidence" && sha256sum -c SHA256SUMS)
  snapshot=$(tar -xOf "$evidence/exp-008-cascade-freezing.tar.gz" snapshot.json | sha256sum | cut -d' ' -f1)
  python3 -m scripts.campaign_analysis.import_results \
    "$evidence/exp008-validation.tar.xz" results/exp-008-import
  cpu=(uv run --no-project --python 3.11 --with pycocotools==2.0.8 --with numpy==1.26.4
       --with pillow==10.4.0 --with matplotlib==3.9.4 --with pyyaml==6.0.4 python -m)
  export MPLBACKEND=Agg
  "${cpu[@]}" experiments.exp-008-cascade-freezing.analyze results/exp-008-import results/exp-008-analysis \
    --data "$data" --snapshot-sha256 "$snapshot"
  "${cpu[@]}" experiments.exp-008-cascade-freezing.diagnostics results/exp-008-import results/exp-008-analysis \
    --data "$data"
)
```

El importador y el analizador requieren destinos nuevos para conservar las reproducciones previas. La reproducción usa CPU y predicciones ya exportadas, con `valid` fijado en el manifiesto.
