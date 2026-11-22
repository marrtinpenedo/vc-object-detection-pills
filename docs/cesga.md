# Baseline en CESGA FinisTerrae III

## Acceso y almacenamiento

Conecta con el acceso SSH autorizado de la cuenta.

```bash
ssh ft3.cesga.es
```

Usa el nodo de login para transferir archivos, actualizar el repositorio y enviar o consultar trabajos. Solicita un nodo CPU mediante `srun` para instalar o compilar y un nodo GPU para las comprobaciones CUDA, la inferencia y el entrenamiento.

- `$HOME` se conserva entre sesiones. Es adecuado para el checkout y la configuración personal.
- `$STORE` se conserva entre sesiones. Aquí están los entornos y el espacio de trabajo del baseline.
- `/scratch` es temporal. Para un trabajo, usa las rutas asignadas por SLURM, como `$LOCAL_SCRATCH` o `$TMPDIR`. Guarda los resultados que quieras conservar en `$STORE` antes de que termine el trabajo. [Almacenamiento de FT3](https://cesga-docs.gitlab.io/ft3-user-guide/storage.html) · [Scratch de FT3](https://cesga-docs.gitlab.io/ft3-user-guide/scratch_dirs.html)

Si ya hay un checkout en CESGA, usa su ubicación. Para transferir una primera copia a `$HOME`, ejecuta esto en Bash desde el repositorio local. La copia temporal contiene el historial de la rama y los archivos versionados.

```bash
(
  set -e
  stage=$(mktemp -d)
  trap 'rm -rf -- "$stage"' EXIT
  git clone --no-local --single-branch --branch feat/baseline-training \
    "$PWD" "$stage/vc-object-detection-pills"
  git -C "$stage/vc-object-detection-pills" remote set-url origin "$(git remote get-url origin)"
  ssh ft3.cesga.es 'test ! -e "$HOME/vc-object-detection-pills"'
  rsync -a "$stage/vc-object-detection-pills/" ft3.cesga.es:vc-object-detection-pills/
)
```

En CESGA, comprueba que el checkout esté limpio antes de actualizarlo.

```bash
cd "$HOME/vc-object-detection-pills"
git status --short
git switch feat/baseline-training
git pull --ff-only origin feat/baseline-training
```

El directorio de trabajo de la cuenta es `$STORE/pills-baseline`. Sus subdirectorios se crearon para lo siguiente.

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

Define estas variables en la terminal Bash que vayas a usar.

```bash
export BASELINE_WORK="$STORE/pills-baseline"
export CASCADE_ENV="$BASELINE_WORK/envs/cascade"
export YOLO_ENV="$BASELINE_WORK/envs/yolo"
export BASELINE_DATA="$BASELINE_WORK/data/pills-prepared"
export BASELINE_RESULTS="$BASELINE_WORK/results"
export CASCADE_WEIGHTS="$BASELINE_WORK/checkpoints/model_final_480dd8.pkl"
export YOLO_WEIGHTS="$BASELINE_WORK/checkpoints/yolov10s.pt"
export PIP_CACHE_DIR="$BASELINE_WORK/cache/pip"
export CONDA_PKGS_DIRS="$BASELINE_WORK/cache/conda"
export YOLO_CONFIG_DIR="$BASELINE_WORK/cache/ultralytics"
mkdir -p "$BASELINE_WORK"/{envs,data,checkpoints,results} \
  "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$YOLO_CONFIG_DIR"
```

## Sesión CPU y módulos

Se utilizó esta asignación para preparar los entornos. Ejecuta `srun` desde la raíz del repositorio.

```bash
srun --partition=short \
  --cpus-per-task=4 \
  --mem=16G \
  --time=02:00:00 \
  --pty bash
```

Una vez dentro del nodo asignado, carga los módulos. Repite este bloque en cada sesión interactiva, CPU o GPU. Comprueba también que las variables de rutas estén definidas en esa sesión.

```bash
module --force purge
module load cesga/2022
module load miniconda3/22.11.1-1
module load cuda/12.1.1
module list
nvcc --version
```

## Reutilización y reinstalación de los entornos

Los dos entornos ya están instalados con Python 3.11. Para reutilizarlos, carga los módulos y llama directamente a `"$CASCADE_ENV/bin/python"` o `"$YOLO_ENV/bin/python"`. Ve al bloque de comprobación de versiones de esta sección. Usa los comandos de creación cuando los prefijos estén libres.

Desde la raíz del repositorio, crea los entornos con Conda e instala paquetes Python con pip. El fichero de restricciones fija las versiones confirmadas. El sufijo `+cu121` selecciona los wheels CUDA 12.1 y la restricción se aplica también a las instalaciones posteriores. [Restricciones del baseline](../configs/baseline/constraints.txt) · [Wheels de PyTorch 2.5.1](https://pytorch.org/get-started/previous-versions/#v251)

```bash
conda create --yes --prefix "$CASCADE_ENV" python=3.11 pip
conda create --yes --prefix "$YOLO_ENV" python=3.11 pip

conda install --yes --prefix "$CASCADE_ENV" -c conda-forge \
  'gcc_linux-64=11' 'gxx_linux-64=11' 'sysroot_linux-64=2.17'

for env_path in "$CASCADE_ENV" "$YOLO_ENV"; do
  "$env_path/bin/python" -m pip install -c configs/baseline/constraints.txt \
    --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121'
  "$env_path/bin/python" -m pip install -c configs/baseline/constraints.txt \
    numpy opencv-python pycocotools Pillow matplotlib
done

"$CASCADE_ENV/bin/python" -m pip install -c configs/baseline/constraints.txt \
  fvcore ninja setuptools wheel
"$YOLO_ENV/bin/python" -m pip install -c configs/baseline/constraints.txt \
  'ultralytics==8.3.161'
```

### Compilación de Detectron2

Detectron2 0.6 quedó instalado desde el commit `fd27788985af0f4ca800bca563acdb700bb890e2`, con PyTorch 2.5.1+cu121, el toolkit CUDA 12.1.1 y código para `sm_80` (A100). GCC/G++ 11 y el sysroot 2.17 de Conda proporcionan la compatibilidad necesaria con el sistema. Activa el prefijo de Cascade en un subshell para aplicar también los ajustes de los paquetes de compilación. [Commit de Detectron2](https://github.com/facebookresearch/detectron2/commit/fd27788985af0f4ca800bca563acdb700bb890e2)

```bash
(
  set -e
  eval "$(conda shell.bash hook)"
  conda activate "$CASCADE_ENV"
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export CUDAHOSTCXX="$CXX"
  export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=4
  "$CC" --version
  "$CXX" --version
  "$CASCADE_ENV/bin/python" -m pip install --no-build-isolation \
    -c configs/baseline/constraints.txt \
    'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@fd27788985af0f4ca800bca563acdb700bb890e2'
)
```

Comprueba la coherencia de dependencias y las versiones de ambos entornos. Estas comprobaciones de importación pueden hacerse en el nodo CPU con los módulos cargados.

```bash
for env_path in "$CASCADE_ENV" "$YOLO_ENV"; do
  "$env_path/bin/python" --version
  "$env_path/bin/python" -m pip check
  "$env_path/bin/python" -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert (torch.__version__, torchvision.__version__, torch.version.cuda) == ("2.5.1+cu121", "0.20.1+cu121", "12.1")'
done
"$CASCADE_ENV/bin/python" -c 'import detectron2, detectron2._C; print(detectron2.__version__); assert detectron2.__version__ == "0.6"'
"$YOLO_ENV/bin/python" -c 'import ultralytics; print(ultralytics.__version__); assert ultralytics.__version__ == "8.3.161"'
```

## Datos y pesos pendientes

El dataset asignado es [Francesco/pills-sxdht](https://huggingface.co/datasets/Francesco/pills-sxdht). Comprueba primero qué hay en `$BASELINE_WORK/data`. La presencia del dataset preparado en CESGA está pendiente de confirmación. Si faltan los originales, sigue la [descarga de la revisión fijada](../README.md#datos-y-modelos) o transfiere esa misma revisión. Coloca la copia original en `$BASELINE_WORK/data/pills-sxdht` y usa la [preparación de la fase A](../experiments/baseline/README.md#protocolo-previo-al-commit) para generar `$BASELINE_DATA`. Se mantiene el [mapeo de ocho clases activas](../experiments/baseline/README.md#h-dataset-mapeo-e-integridad). El ID 0 `pills` carece de instancias.

Los frameworks están instalados. Quedan pendientes la descarga de los pesos preentrenados, su carga y la ejecución completa de los detectores.

- **Cascade R-CNN** usa ResNet-50 + FPN, `Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml`, `cfg.MODEL.MASK_ON=False` y los pesos COCO `model_final_480dd8.pkl` del zoo de Detectron2. [Zoo de Detectron2](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
- **YOLOv10-S** usa Ultralytics y los pesos COCO `yolov10s.pt` del tamaño S. GitHub lista el archivo de la release `v8.4.0` del guion y publica el SHA-256 que se comprueba abajo. La referencia previa del baseline era `v8.3.0`. [Asset publicado en v8.4.0](https://github.com/ultralytics/assets/releases/expanded_assets/v8.4.0)

Cuando aprovisiones los pesos por primera vez, descárgalos en el directorio persistente. El SHA-256 publicado permite contrastar el binario de YOLO con GitHub. Para Cascade, la suma calculada desde la descarga oficial se guarda como referencia local para detectar cambios posteriores y verificar copias.

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

Verifica los pesos conservados y prepara, si hace falta, la fuente que busca el validador de Ultralytics.

```bash
(cd "$BASELINE_WORK/checkpoints" && sha256sum -c SHA256SUMS)
if [ ! -s "$YOLO_CONFIG_DIR/Arial.ttf" ]; then
  curl -fL https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    -o "$YOLO_CONFIG_DIR/Arial.ttf"
fi
```

## GPU y comprobación CUDA

La asignación comprobada fue una NVIDIA A100 PCIe de 40 GB, 32 CPU, 64 GB de memoria y partición `short`. La petición de una A100 con 4 CPU fue rechazada. La de 32 CPU fue aceptada.

```bash
srun --partition=short --gres=gpu:a100:1 \
  --cpus-per-task=32 --mem=64G --time=01:30:00 --pty bash
```

El trabajo de comprobación terminó correctamente tras cargar los módulos, detectar la A100, ejecutar `nms_rotated` de Detectron2 en CUDA e importar Ultralytics y realizar un producto matricial de PyTorch en CUDA. Se validaron estas operaciones. La carga de checkpoints, la inferencia completa y el entrenamiento quedan pendientes.

Para repetir esas comprobaciones, carga los tres módulos dentro del nodo y ejecuta lo siguiente.

```bash
nvidia-smi
"$CASCADE_ENV/bin/python" - <<'PY'
import torch
from detectron2.layers import nms_rotated

print(torch.cuda.get_device_name(0))
boxes = torch.tensor([[10., 10., 5., 5., 0.]], device="cuda")
scores = torch.tensor([0.9], device="cuda")
assert nms_rotated(boxes, scores, 0.5).tolist() == [0]
PY

"$YOLO_ENV/bin/python" - <<'PY'
import torch
import ultralytics

print(ultralytics.__version__, torch.cuda.get_device_name(0))
matrix = torch.eye(2, device="cuda")
torch.testing.assert_close(matrix @ matrix, matrix)
PY
```

## Envío y recuperación de trabajos

Desde la raíz del repositorio, crea el directorio de logs **antes** de enviar. Las opciones de Bash `SLURM_LOGS` se usan en el [protocolo del baseline](../experiments/baseline/README.md#protocolo-previo-al-commit). Si omites esas opciones, los scripts escriben los logs en el directorio de envío, que ya existe.

```bash
mkdir -p "$BASELINE_RESULTS/slurm"
SLURM_LOGS=(--output="$BASELINE_RESULTS/slurm/%x-%j.out" --error="$BASELINE_RESULTS/slurm/%x-%j.err")
```

Los scripts batch cargan los módulos dentro del trabajo y usan los intérpretes instalados en `$STORE/pills-baseline/envs`. Piden 1 A100, 32 CPU, 64 GB y un máximo de seis horas en `short`. Para probar la inicialización, elige salidas independientes mediante un directorio nuevo para cada tanda. Crea el directorio padre, dejando que Python cree los subdirectorios de ejecución.

```bash
export VERIFY_ROOT="$(mktemp -d "$BASELINE_RESULTS/verify-XXXXXX")"
printf '%s\n' "$VERIFY_ROOT"
sbatch "${SLURM_LOGS[@]}" --time=01:00:00 slurm/cascade.sh \
  --data "$BASELINE_DATA" --weights "$CASCADE_WEIGHTS" --output "$VERIFY_ROOT/init-cascade" --initialize
```

Cuando `sbatch` devuelve el ID, puedes cerrar SSH y el trabajo batch sigue en cola o en ejecución. Una sesión interactiva de `srun --pty` está ligada a la conexión. Conserva el ID y la ruta de salida para volver a consultarlos. Los ficheros `%x-%j.out` y `%x-%j.err` contienen stdout y stderr, con el nombre del trabajo y su ID. [Comandos SLURM en FT3](https://cesga-docs.gitlab.io/ft3-user-guide/batch_basic_commands.html)

```bash
squeue --me
sacct -j ID_DEL_TRABAJO --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES
```

Cada `--output` contiene `status.json`, el historial y los artefactos de la ejecución. Cascade usa `last_checkpoint` para localizar el último peso y conserva `best/model_best.pth`. YOLO escribe `weights/last.pt` y `weights/best.pt`. `status.json` recoge las rutas relativas cuando se registra el cierre de la sesión.

Para interrumpir un entrenamiento, espera a que exista un checkpoint si quieres recuperar el progreso y usa `scancel ID_DEL_TRABAJO`. Cuando termine, vuelve a enviar el mismo comando con `--resume` y las mismas rutas de datos, pesos y salida, así como la misma configuración y presupuesto. Para una pausa prevista, usa `--pause-after EPOCH` al iniciar, con una época anterior al final, y retira esa opción al reanudar. Consulta el [smoke test y los comandos de evaluación](../experiments/baseline/README.md#protocolo-previo-al-commit).

Desde el equipo local puedes recuperar los resultados por SSH. Ajusta la ruta remota si el checkout o las salidas están en otra ubicación.

```bash
remote_results=$(ssh ft3.cesga.es 'printf "%s\n" "$STORE/pills-baseline/results"')
mkdir -p results/cesga
rsync -av --exclude='*.pt' --exclude='*.pth' \
  "ft3.cesga.es:$remote_results/" results/cesga/
```
