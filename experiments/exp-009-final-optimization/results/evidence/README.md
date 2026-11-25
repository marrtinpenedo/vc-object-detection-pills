# Evidencia reproducible de EXP-009

Los cuatro archivos `*-validation.tar.xz` conservan el flujo TAR de las exportaciones selectivas de A, B, C y D. Los inventarios internos verifican las predicciones, configuraciones, historiales, hashes de checkpoints y recibos. Los paquetes `exp-009-*.tar.gz` son las versiones originales desplegadas. El archivo `exp009-final-test-projection.tar.xz` conserva byte a byte los artefactos de evaluación seleccionados de la descarga del directorio final. Su recibo contiene el SHA-256 del tar original y la lista de archivos auxiliares excluidos. El inventario de esa proyección es local. Los recibos SLURM de C y de test se conservan por separado. [SHA-256 de la evidencia](SHA256SUMS) · [Recibo de la proyección](directory-import.json) · [Informe final](../../final-results.md)

Las tablas y figuras publicadas están en [`analysis/`](../analysis/). Los archivos de entrada contienen el historial original de cada brazo. El análisis regenera las tablas extensas y las PR de todos los modelos. Los pesos, entornos y el dataset permanecen fuera de Git.

## Reproducción local

Ejecutar desde la raíz del repositorio con Python 3.11, NumPy 1.26.4, pycocotools 2.0.8, Pillow 10.4.0, matplotlib 3.9.4 y PyYAML 6.0.4. Definir `PILLS_PYTHON` si se usa un entorno CPU específico. `PILLS_DATASET` puede apuntar al directorio de las anotaciones originales. Las imágenes se usan para figuras de ejemplo cuando están disponibles. El procedimiento puntúa predicciones archivadas en CPU.

```bash
(
  set -euo pipefail
  evidence=experiments/exp-009-final-optimization/results/evidence
  baseline=experiments/exp-008-cascade-freezing/results/evidence/exp008-validation.tar.xz
  data="${PILLS_DATASET:-data/pills-sxdht}"
  py="${PILLS_PYTHON:-python3}"
  out=$(mktemp -d results/exp009-replay.XXXXXX)
  export MPLBACKEND=Agg
  (cd "$evidence" && sha256sum -c SHA256SUMS)

  for phase in a b c d; do
    "$py" -m scripts.campaign_analysis.import_results \
      "$evidence/exp009-$phase-validation.tar.xz" "$out/$phase-import"
  done
  "$py" -m scripts.campaign_analysis.import_results "$baseline" "$out/exp008"
  "$py" -m scripts.campaign_analysis.import_results \
    "$evidence/exp009-final-test-projection.tar.xz" "$out/test-import"
  for phase in b c d; do
    tar -xOf "$evidence/exp-009-$phase.tar.gz" controller.json > "$out/$phase-controller.json"
  done
  snapshot=$(sha256sum "$out/a-import/snapshot.json" | cut -d' ' -f1)

  "$py" -m experiments.exp-009-final-optimization.analyze \
    "$out/a-import" "$out/a" --snapshot-sha256 "$snapshot" --data "$data"
  "$py" -m experiments.exp-009-final-optimization.diagnostics "$out/a-import" "$out/a"
  "$py" -m experiments.exp-009-final-optimization.followup B \
    "$out/b-import" "$out/a-import" "$data" "$out/b" "$out/b-controller.json"
  "$py" -m experiments.exp-009-final-optimization.followup D \
    "$out/d-import" "$out/exp008" "$data" "$out/d" "$out/d-controller.json"
  "$py" -m experiments.exp-009-final-optimization.confirmation \
    "$out/c-import" "$out/a-import" "$out/b-import" "$data" "$out/c" "$out/c-controller.json" \
    --slurm-accounting "$evidence/exp009-c-sacct.txt"
  "$py" -m experiments.exp-009-final-optimization.closeout \
    "$out/c" "$out/c-import" "$out/d" "$out/d-import" "$out/seleccion"
  cmp "$out/seleccion/seleccion-valid.json" "$evidence/../analysis/seleccion/seleccion-valid.json"
  "$py" -m experiments.exp-009-final-optimization.final_analysis \
    "$out/test-import" "$out/seleccion/seleccion-valid.json" "$evidence/exp-009-final-test.tar.gz" \
    "$out/c-import/snapshot.json" "$out/c-import/prepared-manifest.json" "$data" "$out/test" \
    --slurm-accounting "$evidence/exp009-final-test-sacct.txt"
  printf 'Reproducción en %s\n' "$out"
)
```
