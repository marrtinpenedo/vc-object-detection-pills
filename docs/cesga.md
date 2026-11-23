# Ejecución de los baselines en CESGA FinisTerrae III

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

Las ejecuciones de referencia utilizaron una copia transferida con `rsync` sin directorio `.git`. Desde la raíz del repositorio local, empaquete la revisión versionada de `feat/baseline-training` y transfiérala a `$HOME` en CESGA.

```bash
(
  set -euo pipefail
  commit=$(git rev-parse --verify feat/baseline-training)
  remote_project="vc-object-detection-pills-${commit:0:12}"
  stage=$(mktemp -d)
  trap 'rm -rf -- "$stage"' EXIT
  git archive "$commit" | tar -x -C "$stage"
  ssh "$CESGA_SSH" "mkdir -- '$remote_project'"
  rsync -a -e ssh "$stage/" "$CESGA_SSH:$remote_project/"
  printf 'En CESGA, usa ~/%s\n' "$remote_project"
)
```

En CESGA, sitúese en la carpeta indicada al finalizar la transferencia. Para actualizar el código, transfiera la nueva revisión a otra carpeta y mantenga las copias de los trabajos activos. En estas copias `git_head` puede quedar nulo. El registro de procedencia incluye los SHA-256 de los scripts Python y el nombre de la carpeta identifica la revisión del proyecto.

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
  --prefix "$BASELINE_WORK/envs/cascade" \
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
  conda activate "$BASELINE_WORK/envs/cascade"
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

El descargador de la revisión fijada de [Francesco/pills-sxdht](https://huggingface.co/datasets/Francesco/pills-sxdht) escribe en `data/pills-sxdht/`. Siga las [instrucciones de descarga](../README.md#datos-y-modelos) y traslade los archivos originales al almacenamiento persistente.

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

Configure los logs persistentes antes de enviar las fases de [inicialización, prueba corta y entrenamiento](../experiments/baseline/README.md#protocolo-de-verificación). Conserve el ID que devuelve `sbatch` y la ruta de cada ejecución.

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

Defina `CASCADE_RUN` y `YOLO_RUN` con dos ejecuciones completas que usaron el mismo `$BASELINE_DATA`. Indique en `EVAL_ROOT` un directorio nuevo fuera de los datos y de ambas ejecuciones, con el directorio padre ya creado. El [protocolo experimental](../experiments/baseline/README.md#protocolo-de-verificación) proporciona las rutas de cada fase.

```bash
eval_job=$(sbatch --parsable "${SLURM_LOGS[@]}" slurm/evaluate.sh \
  "$BASELINE_DATA" "$CASCADE_RUN" "$YOLO_RUN" "$EVAL_ROOT") &&
  printf '%s\n' "${eval_job%%;*}"
```

El lanzador evalúa secuencialmente los checkpoints `best` en `valid` sobre la GPU asignada y ejecuta `scripts.verify_baseline`. Las salidas quedan en `$EVAL_ROOT/cascade` y `$EVAL_ROOT/yolo`. Confirme el estado `COMPLETED`, el código `0:0` y el mensaje final del verificador en el log. Para repetir una evaluación, utilice otra salida. [Protocolo de medición](../experiments/baseline/README.md#g-métricas-y-protocolo-de-inferencia) · [Resultados observados](../experiments/baseline/README.md#resultados-observados-en-cesga)

## Recuperar resultados

Conserve los checkpoints entrenados en `$BASELINE_RESULTS`. Desde el equipo local, use `rsync` para recuperar en `results/cesga/` los informes, historiales, configuraciones, figuras, ejemplos y logs. Ajuste la ruta remota si almacenó los resultados en otro directorio.

```bash
remote_results=$(ssh "$CESGA_SSH" 'printf "%s\n" "$STORE/pills-baseline/results"') &&
  mkdir -p results/cesga &&
  rsync -av -e ssh --prune-empty-dirs \
    --include='*/' --include='*.json' --include='*.csv' --include='*.yaml' --include='*.yml' \
    --include='*.png' --include='*.jpg' --include='*.jpeg' --include='*.out' --include='*.err' \
    --exclude='*' \
    "$CESGA_SSH:$remote_results/" results/cesga/
```
