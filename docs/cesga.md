# Infraestructura y operación en CESGA FinisTerrae III

## Acceso y almacenamiento

En el equipo local, defina `CESGA_SSH` con el usuario autorizado y el servidor o con un alias de SSH.

```bash
export CESGA_SSH='USUARIO_CESGA@ft3.cesga.es'
ssh "$CESGA_SSH"
```

Utilice el nodo de login para transferencias, envío y consulta de trabajos. Reserve nodos de cómputo para compilar, entrenar y evaluar.

`$HOME` y `$STORE` son persistentes. Conserve el código en `$HOME` y los datos, entornos, pesos y resultados en `$STORE/pills-baseline`. Reserve `$LOCAL_SCRATCH` y `$TMPDIR` para archivos temporales. [Almacenamiento permanente](https://cesga-docs.gitlab.io/ft3-user-guide/storage.html) · [Almacenamiento temporal](https://cesga-docs.gitlab.io/ft3-user-guide/scratch_dirs.html)

```text
pills-baseline/
├── envs/
│   ├── cascade/      # Entorno Conda de Detectron2
│   └── yolo/         # Entorno Conda de Ultralytics
├── data/             # Datos originales y derivado preparado
├── checkpoints/      # Pesos de inicialización y sumas SHA-256
├── results/          # Ejecuciones, checkpoints ajustados y logs
└── cache/            # Cachés de Conda y pip, configuración y fuente de Ultralytics
```

Defina las rutas siguientes en las sesiones Bash de CESGA desde las que vaya a preparar o ejecutar el protocolo.

```bash
export BASELINE_WORK="$STORE/pills-baseline"
export CASCADE_ENV="$BASELINE_WORK/envs/cascade"
export YOLO_ENV="$BASELINE_WORK/envs/yolo"
export BASELINE_DATA="$BASELINE_WORK/data/pills-prepared"
export BASELINE_RESULTS="$BASELINE_WORK/results"
export CASCADE_WEIGHTS="$BASELINE_WORK/checkpoints/model_final_480dd8.pkl"
export YOLO_WEIGHTS="$BASELINE_WORK/checkpoints/yolov10s.pt"
export PIP_CACHE_DIR="$BASELINE_WORK/cache/pip"
export CONDA_PKGS_DIRS="$BASELINE_WORK/cache/conda-pkgs"
export YOLO_CONFIG_DIR="$BASELINE_WORK/cache/ultralytics"
mkdir -p "$BASELINE_WORK"/{envs,data,checkpoints,results} \
  "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$YOLO_CONFIG_DIR"
```

## Transferencia y actualización del código

Desde la raíz del repositorio local, seleccione la revisión versionada que contiene el código y la configuración del experimento. El ejemplo archiva `HEAD`. Si desea ejecutar otra revisión, indique su commit o referencia en `revision`. El envío crea una carpeta por commit en `$HOME` de CESGA.

```bash
(
  set -euo pipefail
  revision=HEAD
  commit=$(git rev-parse --verify "${revision}^{commit}")
  remote_project="vc-object-detection-pills-${commit:0:12}"
  stage=$(mktemp -d)
  trap 'rm -rf -- "$stage"' EXIT
  git archive "$commit" | tar -x -C "$stage"
  ssh "$CESGA_SSH" "mkdir -- '$remote_project'"
  rsync -a -e ssh "$stage/" "$CESGA_SSH:$remote_project/"
  printf 'En CESGA, usa ~/%s\n' "$remote_project"
)
```

En CESGA, sitúese en la carpeta indicada al finalizar la transferencia. Para actualizar el código, transfiera la nueva revisión a otra carpeta y mantenga las copias de los trabajos activos. Las ejecuciones de referencia utilizaron una copia con `rsync` sin directorio `.git`. En estas copias `git_head` puede quedar nulo. Registre el commit completo junto con las rutas y los ID de SLURM. El registro de procedencia incluye los SHA-256 de los scripts Python.

## Entornos y dependencias

La instalación de referencia utiliza dos entornos Conda con Python 3.11, PyTorch 2.5.1+cu121 y torchvision 0.20.1+cu121. Los comandos siguientes documentan su preparación. La instalación completa desde cero en otra cuenta queda por comprobar.

Desde la raíz del proyecto en CESGA, solicite un nodo CPU para crear los entornos y compilar Detectron2.

```bash
srun --partition=short \
  --cpus-per-task=4 \
  --mem=16G \
  --time=02:00:00 \
  --pty bash
```

En la sesión de cómputo, cargue los módulos y compruebe que las variables de [almacenamiento](#acceso-y-almacenamiento) estén definidas. Utilice el mismo bloque de módulos para las sesiones GPU interactivas.

```bash
module --force purge
module load cesga/2022
module load miniconda3/22.11.1-1
module load cuda/12.1.1
module list
nvcc --version
```

Si los entornos existen en su cuenta, compruebe las versiones al final de esta sección y use sus intérpretes. Para crearlos, ejecute desde la raíz del proyecto los siguientes comandos. [Restricciones del baseline](../configs/baseline/constraints.txt) · [Wheels de PyTorch 2.5.1](https://pytorch.org/get-started/previous-versions/#v251)

```bash
conda create --yes --prefix "$CASCADE_ENV" python=3.11 pip
conda create --yes --prefix "$YOLO_ENV" python=3.11 pip

conda install -y \
  --prefix "$CASCADE_ENV" \
  --override-channels -c conda-forge \
  "gcc_linux-64=11" \
  "gxx_linux-64=11" \
  "sysroot_linux-64=2.17"

for env_path in "$CASCADE_ENV" "$YOLO_ENV"; do
  "$env_path/bin/python" -m pip install -c configs/baseline/constraints.txt \
    --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121'
  "$env_path/bin/python" -m pip install -c configs/baseline/constraints.txt \
    numpy opencv-python pycocotools Pillow matplotlib
done

"$CASCADE_ENV/bin/python" -m pip install -c configs/baseline/constraints.txt \
  fvcore ninja 'setuptools==80.9.0' wheel \
  'termcolor>=1.1' 'yacs>=0.1.8' tabulate cloudpickle 'tqdm>4.29.0' tensorboard \
  'iopath>=0.1.7,<0.1.10' 'omegaconf>=2.1,<2.4' 'hydra-core>=1.1' black packaging
"$YOLO_ENV/bin/python" -m pip install -c configs/baseline/constraints.txt \
  'ultralytics==8.3.161'
```

### Compilación de Detectron2

Compile Detectron2 0.6 en el entorno Cascade con GCC/G++ 11, sysroot 2.17, CUDA 12.1.1 y arquitectura `sm_80`. El pin de setuptools conserva `pkg_resources`, requerido por `model_zoo`. [Dependencias del commit fijado](https://github.com/facebookresearch/detectron2/blob/fd27788985af0f4ca800bca563acdb700bb890e2/setup.py#L146-L180)

```bash
(
  set -e
  eval "$(conda shell.bash hook)"
  conda activate "$CASCADE_ENV"
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export CUDAHOSTCXX="$CXX"
  export FORCE_CUDA=1
  export TORCH_CUDA_ARCH_LIST="8.0"
  export MAX_JOBS=4
  "$CC" --version
  "$CXX" --version
  python -m pip install \
    --force-reinstall --no-cache-dir \
    --no-build-isolation --no-deps \
    "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@fd27788985af0f4ca800bca563acdb700bb890e2"
)
```

Compruebe las versiones y dependencias con los módulos cargados.

```bash
for env_path in "$CASCADE_ENV" "$YOLO_ENV"; do
  "$env_path/bin/python" --version
  "$env_path/bin/python" -m pip check
  "$env_path/bin/python" -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert (torch.__version__, torchvision.__version__, torch.version.cuda) == ("2.5.1+cu121", "0.20.1+cu121", "12.1")'
done
"$CASCADE_ENV/bin/python" -c 'import detectron2, detectron2._C; from detectron2 import model_zoo; print(detectron2.__version__); assert detectron2.__version__ == "0.6"'
"$YOLO_ENV/bin/python" -c 'import ultralytics; print(ultralytics.__version__); assert ultralytics.__version__ == "8.3.161"'
```

## Datos y pesos

Siga las [instrucciones de descarga](../experiments/baseline/README.md#descarga-y-exploración-local) y traslade los archivos de `data/pills-sxdht/` al almacenamiento persistente.

Si realizó la descarga en CESGA, ejecute desde la raíz del proyecto en CESGA el siguiente bloque.

```bash
mkdir -p "$BASELINE_WORK/data/pills-sxdht" &&
  rsync -a data/pills-sxdht/ "$BASELINE_WORK/data/pills-sxdht/"
```

Si realizó la descarga en el equipo local, ejecute desde la raíz del repositorio local el siguiente bloque.

```bash
remote_data=$(ssh "$CESGA_SSH" 'mkdir -p "$STORE/pills-baseline/data/pills-sxdht" &&
  printf "%s\n" "$STORE/pills-baseline/data/pills-sxdht"') &&
  rsync -a -e ssh data/pills-sxdht/ "$CESGA_SSH:$remote_data/"
```

Prepare el dataset en el nodo CPU con el entorno Cascade disponible. El manifiesto resultante identifica los datos utilizados para el entrenamiento y la evaluación. [Particiones y categorías](../experiments/baseline/README.md#h-dataset-mapeo-e-integridad)

```bash
"$CASCADE_ENV/bin/python" -m scripts.baseline_data --data "$BASELINE_WORK/data/pills-sxdht" --output "$BASELINE_DATA"
```

Descargue los pesos COCO en `$BASELINE_WORK/checkpoints` si aún no están disponibles. El archivo de YOLOv10-S se comprueba con el SHA-256 publicado. Para Cascade, se registra la suma del archivo descargado del zoo oficial. [Zoo de Detectron2](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) · [Pesos de Ultralytics v8.4.0](https://github.com/ultralytics/assets/releases/expanded_assets/v8.4.0)

```bash
(
  set -e
  cd "$BASELINE_WORK/checkpoints"
  curl -fL https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl \
    -o model_final_480dd8.pkl.part
  curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10s.pt \
    -o yolov10s.pt.part
  printf '%s  %s\n' '65fa8332c38a7189597a268897bce6c1026f9499711b9761fb0c2c639b91d4d6' yolov10s.pt.part \
    | sha256sum -c -
  mv -- model_final_480dd8.pkl.part model_final_480dd8.pkl
  mv -- yolov10s.pt.part yolov10s.pt
  sha256sum model_final_480dd8.pkl yolov10s.pt > SHA256SUMS
)
```

Verifique los pesos y prepare la fuente utilizada por el validador de Ultralytics.

```bash
(cd "$BASELINE_WORK/checkpoints" && sha256sum -c SHA256SUMS)
if [ ! -s "$YOLO_CONFIG_DIR/Arial.ttf" ]; then
  curl -fL https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    -o "$YOLO_CONFIG_DIR/Arial.ttf"
fi
```

## Ejecución de trabajos SLURM

Envíe los trabajos desde la raíz del proyecto. [slurm/cascade.sh](../slurm/cascade.sh), [slurm/yolo.sh](../slurm/yolo.sh) y [slurm/evaluate.sh](../slurm/evaluate.sh) cargan los módulos y usan los intérpretes de `$STORE/pills-baseline/envs` dentro de la asignación. Cada trabajo usa la partición `short`, una NVIDIA A100 PCIe de 40 GB, 32 CPU por GPU y 64 GB de RAM. El límite de los entrenamientos es de seis horas y el de la evaluación, una hora.

Configure los logs persistentes antes de enviar los trabajos. Conserve el ID que devuelve `sbatch` y la ruta de cada ejecución.

```bash
mkdir -p "$BASELINE_RESULTS/slurm"
SLURM_LOGS=(--output="$BASELINE_RESULTS/slurm/%x-%j.out" --error="$BASELINE_RESULTS/slurm/%x-%j.err")
```

Consulte los trabajos en cola y el estado de la ejecución. Sustituya `ID_DEL_TRABAJO` por el identificador devuelto al enviar el trabajo. [Comandos SLURM en FT3](https://cesga-docs.gitlab.io/ft3-user-guide/batch_basic_commands.html)

```bash
squeue --me
sacct -j ID_DEL_TRABAJO --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES
```

Para interrumpir un entrenamiento con `scancel ID_DEL_TRABAJO`, compruebe antes que haya un checkpoint disponible. Al reanudar, conserve los argumentos de datos, pesos, configuración y salida del trabajo original y añada `--resume`. Para una pausa planificada, use la secuencia del [protocolo de verificación](../experiments/baseline/README.md#protocolo-de-verificación).

### Sesión GPU interactiva

Para operaciones interactivas que requieran CUDA, solicite la misma clase de GPU y cargue los módulos dentro de la asignación.

```bash
srun --partition=short --gres=gpu:a100:1 \
  --cpus-per-task=32 --mem=64G --time=01:30:00 --pty bash
```

## Evaluación conjunta

Desde la raíz del código usado para evaluar, ejecute el lanzador con `--data`, `--output` y un `--run NOMBRE RUTA` por ejecución. Cada nombre debe ser único, empezar por letra o dígito y usar únicamente letras ASCII, dígitos, guiones o guiones bajos. Prepare el directorio padre de la salida y deje libre la ruta de salida. El lanzador detecta el backend en `run.json` y usa su entorno Conda. Todos los checkpoints se evalúan secuencialmente en la GPU asignada.

Los ejemplos de Bash usan las rutas y los logs definidos en esta guía. Asigne a cada variable de ejecución la ruta de un entrenamiento completo. En la comparación entre variantes, ambas deben usar los mismos datos.

```bash
sbatch --job-name=eval-yolo-valid "${SLURM_LOGS[@]}" slurm/evaluate.sh \
  --data "$BASELINE_DATA" --output "$BASELINE_RESULTS/eval-yolo-valid" --run yolo "$YOLO_RUN"

sbatch --job-name=eval-cascade-valid "${SLURM_LOGS[@]}" slurm/evaluate.sh \
  --data "$BASELINE_DATA" --output "$BASELINE_RESULTS/eval-cascade-valid" --run cascade "$CASCADE_RUN"

sbatch --job-name=eval-yolo-variantes-valid "${SLURM_LOGS[@]}" slurm/evaluate.sh \
  --data "$BASELINE_DATA" --output "$BASELINE_RESULTS/eval-yolo-variantes-valid" \
  --run control "$CONTROL_RUN" --run variante "$VARIANT_RUN"

sbatch --job-name=eval-baselines-valid "${SLURM_LOGS[@]}" slurm/evaluate.sh \
  --data "$BASELINE_DATA" --output "$BASELINE_RESULTS/eval-baselines-valid" \
  --run cascade "$CASCADE_RUN" --run yolo "$YOLO_RUN"
```

Use nombres descriptivos con `--job-name` para distinguir experimento, detector, brazo, fase o semilla según corresponda.

```bash
squeue --me --name=exp006-eval-a-s2026
sacct --name=exp006-eval-a-s2026 --starttime today --format=JobID,JobName%32,State,ExitCode
sacct -j ID_DEL_TRABAJO --format=JobID,JobName%32,State,ExitCode
```

- `--split valid|test` y `--checkpoint best|last` usan `valid` y `best` por defecto. Reserve `test` para la evaluación final.
- El benchmark usa 20 calentamientos y 100 iteraciones por defecto. Ajuste `--warmup` y `--iterations` en común para todos los modelos. Use la misma resolución para comparar latencias o indique `--no-benchmark`.
- `--examples N` elige hasta cinco ejemplos por tipo por defecto y admite cero. Para guardar las mismas imágenes por brazo, use uno o varios `--example-image-id ID` en su lugar.
- `--check-slurm` exige `COMPLETED` y código `0:0` para el último trabajo de entrenamiento de cada brazo y guarda su estado en `NOMBRE.train.sacct`. Consulte la ayuda con `bash slurm/evaluate.sh --help`.

La salida contiene `NOMBRE/` por ejecución, `eval.job` y `launch.json` con la procedencia y la selección de checkpoints. Antes de evaluar, se comprueban el estado del entrenamiento, los datos y el checkpoint. Después, se recalculan las métricas a partir de las predicciones y se cotejan las curvas PR, el hash del checkpoint y el protocolo de latencia. `verified.ok` se publica al terminar todas las comprobaciones. Confirme también `COMPLETED` y `0:0` en SLURM. Si hay un fallo, conserve la salida parcial y use otra ruta al repetir. La configuración de Ultralytics se guarda por trabajo. [Protocolo de medición](../experiments/baseline/README.md#g-métricas-y-protocolo-de-inferencia)

La interfaz posicional `sbatch slurm/evaluate.sh DATOS RUN_CASCADE RUN_YOLO SALIDA_NUEVA` mantiene `cascade/`, `yolo/`, los checkpoints `best`, el split `valid` y las comprobaciones adicionales de los entrenamientos del baseline.

Para evaluar desde una revisión histórica, puede desplegar juntos `evaluate.sh` y `evaluate.py` en una carpeta nueva. Envíe el trabajo desde la raíz del código histórico, con `PILLS_EVAL_LAUNCHER_DIR` exportada y con la ruta absoluta del lanzador en `sbatch`. El evaluador, el verificador y los hashes de `scripts/*.py` proceden de esa raíz. Conserve la carpeta del lanzador durante el trabajo. [Ejemplo con EXP-006](../experiments/exp-006-yolo-warmup/README.md#evaluar-sobre-la-misma-gpu)

## Recuperar resultados

Desde la raíz del proyecto en CESGA, indique las dos ejecuciones completas y su evaluación conjunta, todas bajo `$BASELINE_RESULTS`. El exportador comprueba el ZIP antes de publicarlo. Incluye los informes, historiales, gráficas, predicciones y hasta cinco ejemplos correctos y cinco fallidos por detector. Los checkpoints y logs se conservan en CESGA.

```bash
ZIP_NAME=baseline-results.zip
"${CASCADE_ENV:?Defina CASCADE_ENV}/bin/python" experiments/baseline/export_results.py \
  --results-root "$BASELINE_RESULTS" --cascade-run "$CASCADE_RUN" \
  --yolo-run "$YOLO_RUN" --evaluation "$EVAL_ROOT" \
  --output "${BASELINE_RESULTS:?Defina BASELINE_RESULTS}/$ZIP_NAME"
```

Desde el portátil, con `CESGA_SSH` definido, descargue únicamente el ZIP y compruebe su integridad.

```bash
ZIP_NAME=baseline-results.zip
remote_results=$(ssh "${CESGA_SSH:?Defina CESGA_SSH}" 'printf "%s\n" "${STORE:?}/pills-baseline/results"') &&
  local_results=$(mktemp -d ../baseline-results.XXXXXX) &&
  scp "$CESGA_SSH:$remote_results/$ZIP_NAME" "$local_results/$ZIP_NAME" &&
  python3 -c 'from sys import argv; from zipfile import ZipFile; raise SystemExit(ZipFile(argv[1]).testzip())' \
    "$local_results/$ZIP_NAME" &&
  printf '%s\n' "$local_results/$ZIP_NAME"
```
