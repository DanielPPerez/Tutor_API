# Pipeline de Limpieza de Imágenes - Resumen

## Descripción del Pipeline (Primera Persona)

1. **Conversión a escala de grises y normalización**: Convierto la imagen a escala de grises y aplico histogram stretching (normalización) para estandarizar la iluminación y mejorar el contraste.

2. **Mejora de contraste local (CLAHE)**: Aplico CLAHE (Contrast Limited Adaptive Histogram Equalization) para mejorar localmente el contraste, especialmente útil cuando hay variaciones de iluminación en la imagen.

3. **Binarización adaptativa**: Genero una binarización adaptativa usando umbral adaptativo gaussiano, que detecta el trazo independientemente de la iluminación local de la imagen.

4. **Segmentación HSV con fallback**: Intento segmentar el grafito usando una máscara HSV que filtra el color del lápiz y elimina las líneas azules del cuaderno. Si la máscara HSV está vacía o no cubre suficiente trazo detectado, uso directamente la binarización adaptativa (mecanismo de fallback).

5. **Eliminación de líneas del cuaderno**: Elimino las líneas horizontales del cuaderno usando morfología matemática con un kernel horizontal que detecta y resta estas líneas del resultado binario.

6. **Clausura morfológica**: Aplico clausura morfológica para soldar trazos rotos o punteados, uniendo partes del trazo que deberían estar conectadas.

7. **Filtro geométrico**: Filtro componentes conectados por área, solidity y aspect ratio para eliminar ruido y conservar solo componentes que tienen características de letras.

**Resultado**: Devuelvo un mapa binario donde el fondo es 0 y el trazo es 255.

---

## Valores Ajustables

Todos los valores ajustables están centralizados en `app/core/config.py` y se importan en los módulos correspondientes.

### Validación de Máscara HSV

- **`MIN_HSV_PIXELS_RATIO`** = `0.005` (0.5% del área de la imagen)
  - Umbral mínimo de píxeles en la máscara HSV para considerarla válida (ratio del área total)

- **`MIN_HSV_COVERAGE_RATIO`** = `0.3` (30%)
  - Umbral mínimo de cobertura: la máscara HSV debe cubrir al menos este porcentaje del trazo detectado por binarización adaptativa

- **`MIN_HSV_PIXELS_ABSOLUTE`** = `100`
  - Mínimo absoluto de píxeles en la máscara HSV (independiente del tamaño de imagen)

### Estandarización de Iluminación (CLAHE)

- **`CLAHE_CLIP_LIMIT`** = `4.0`
  - Límite de contraste para CLAHE

- **`CLAHE_TILE_GRID_SIZE`** = `(8, 8)`
  - Tamaño de la cuadrícula para CLAHE

### Binarización Adaptativa

- **`ADAPTIVE_THRESH_BLOCK_SIZE`** = `35`
  - Tamaño del bloque para umbral adaptativo (debe ser impar)

- **`ADAPTIVE_THRESH_C`** = `7`
  - Constante restada de la media para ajuste fino

### Segmentación HSV del Grafito (Valores por Defecto)

- **`HSV_SAT_MAX_GRAPHITE`** = `80`
  - Saturación máxima para detectar grafito

- **`HSV_VAL_MIN_GRAPHITE`** = `30`
  - Valor mínimo para detectar grafito

- **`HSV_VAL_MAX_GRAPHITE`** = `200`
  - Valor máximo para detectar grafito

- **`HSV_BLUE_H_MIN`** = `90`
  - Hue mínimo para filtrar líneas azules del cuaderno

- **`HSV_BLUE_H_MAX`** = `140`
  - Hue máximo para filtrar líneas azules del cuaderno

- **`HSV_BLUE_SAT_MIN`** = `40`
  - Saturación mínima para filtrar líneas azules

- **`HSV_BLUE_VAL_MIN`** = `40`
  - Valor mínimo para filtrar líneas azules

### Operaciones Morfológicas

#### Limpieza de Máscara HSV de Grafito

- **`MORPH_GRAPHITE_KERNEL_SIZE`** = `(3, 3)`
  - Tamaño del kernel elíptico

- **`MORPH_GRAPHITE_ITERATIONS`** = `1`
  - Iteraciones para operación OPEN

#### Detección y Eliminación de Líneas Horizontales

- **`MORPH_LINES_KERNEL_SIZE`** = `(45, 1)`
  - Kernel rectangular horizontal

- **`MORPH_LINES_ITERATIONS`** = `2`
  - Iteraciones para detectar líneas

- **`MORPH_LINES_DILATE_SIZE`** = `(3, 3)`
  - Tamaño del kernel para dilatar líneas detectadas

#### Clausura Morfológica (Soldar Trazos)

- **`MORPH_CLOSE_KERNEL_SIZE`** = `(7, 7)`
  - Tamaño del kernel elíptico para clausura

- **`MORPH_CLOSE_ITERATIONS`** = `1`
  - Iteraciones para operación CLOSE

### Filtro Geométrico de Componentes

- **`FILTER_MIN_AREA`** = `80`
  - Área mínima en píxeles para considerar un componente válido

- **`FILTER_SOLIDITY_RANGE`** = `(0.25, 1.0)`
  - Rango de solidity (área/convex_hull) aceptable

- **`FILTER_ASPECT_RATIO_RANGE`** = `(0.15, 6.0)`
  - Rango de relación ancho/alto aceptable

### Procesamiento Post-Limpieza (preprocess_robust)

- **`CROP_MARGIN`** = `30`
  - Margen en píxeles al recortar la letra a un cuadrado

- **`BINARIZATION_THRESHOLD`** = `127`
  - Umbral para binarización final antes de esqueletizar

- **`MORPH_PRE_SKEL_KERNEL_SIZE`** = `(3, 3)`
  - Kernel para unir trazos punteados antes de esqueletizar

---

## Ubicación de los Archivos

- **Configuración**: `app/core/config.py`
- **Implementación del pipeline**: `app/core/vision.py`
- **Procesamiento robusto**: `app/core/processor.py`

## Notas

- Todos los valores ajustables están centralizados en `config.py` para facilitar el ajuste y la experimentación.
- Los valores pueden ser sobrescritos mediante el parámetro `hsv_range` en las funciones `clean_notebook()` y `segment_graphite_hsv()`.
- El mecanismo de fallback garantiza que siempre se obtenga un resultado, incluso cuando la máscara HSV falla.












import yaml

# 1. Definir la ruta del nuevo YAML que vamos a crear
KAGGE_YAML_PATH = '/kaggle/working/dataset_kaggle.yaml'

# 2. Configurar el contenido con las rutas de Kaggle que detectamos antes
# Usamos las variables que definimos en la celda anterior (_DATASET_ROOT)
data_config = {
    'path': str(_DATASET_ROOT),      # La raíz del dataset en Kaggle
    'train': 'images/train',         # Ruta relativa a la raíz
    'val': 'images/val',             # Ruta relativa a la raíz
    'nc': 62,                        # Número de clases (0-9, A-Z, a-z)
    'names': [
        '0','1','2','3','4','5','6','7','8','9',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
    ]
}

# 3. Guardar el archivo en la carpeta donde sí tenemos permiso de escritura
with open(KAGGE_YAML_PATH, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

# 4. ACTUALIZAR la variable que usa YOLO
DATASET_YAML = KAGGE_YAML_PATH

print(f"✅ Nuevo archivo YAML creado en: {DATASET_YAML}")
print(f"📍 Apuntando a imágenes en: {data_config['path']}")

# =============================================================
# CELDA OPTIMIZADA: Detector YOLO — Fix velocidad
# Problema: dataset en /kaggle/input/ (NFS lento) + cache RAM insuficiente
# Solución: copiar a /kaggle/working/ (SSD local) + epochs reducidas
# =============================================================
import shutil, yaml
from pathlib import Path
from ultralytics import YOLO

# ── 1. Copiar dataset a SSD local (/kaggle/working/) ──────────
FAST_DS_DIR = Path('/kaggle/working/yolo_dataset_local')

if not FAST_DS_DIR.exists():
    print("Copiando dataset a SSD local (solo la primera vez, ~2-3 min)...")
    shutil.copytree(str(_DATASET_ROOT), str(FAST_DS_DIR))
    print(f"  Copiado: {sum(1 for _ in FAST_DS_DIR.rglob('*.jpg')):,} imágenes")
else:
    print(f"Dataset local ya existe: {FAST_DS_DIR}")

# ── 2. Crear dataset.yaml apuntando a SSD local ───────────────
with open(DATASET_YAML, 'r') as f:
    ds_yaml = yaml.safe_load(f)

ds_yaml['path'] = str(FAST_DS_DIR)
ds_yaml['train'] = 'images/train'
ds_yaml['val']   = 'images/val'

DATASET_YAML_LOCAL = '/kaggle/working/dataset_local.yaml'
with open(DATASET_YAML_LOCAL, 'w') as f:
    yaml.dump(ds_yaml, f)
print(f"YAML local: {DATASET_YAML_LOCAL}")

# ── 3. Config YOLO optimizado ─────────────────────────────────
# El modelo ya alcanza mAP50=0.993 en época 1 con este dataset.
# Con 15 épocas y patience=5 termina en ~40 min en lugar de 20+ horas.
YOLO_CFG_FAST = dict(
    model_variant  = 'yolov8n.pt',
    epochs         = 10,          # mAP converge en 2-3 épocas; 15 es más que suficiente
    batch          = 64,          # 2× T4 = 2×14GB VRAM; batch 64/GPU es seguro con yolov8n
    device         = '0,1',
    workers        = 4,           # 4 workers es óptimo para SSD local en Kaggle
    img_size       = 640,
    lr0            = 0.01,
    lrf            = 0.01,
    momentum       = 0.937,
    weight_decay   = 5e-4,
    warmup_epochs  = 3.0,         # warmup más corto para pocas épocas
    amp            = True,
    cache          = False,       # SSD local es suficientemente rápido; RAM insuficiente
    patience       = 5,           # early stop rápido: mAP ya es 0.993 desde época 1
    project        = str(YOLO_DIR),
    name           = 'char_detector_t4_fast',
    exist_ok       = True,
    pretrained     = True,
)

n_train = sum(1 for _ in FAST_DS_DIR.glob('images/train/*.jpg'))
n_val   = sum(1 for _ in FAST_DS_DIR.glob('images/val/*.jpg'))
print(f"\nConfig entrenamiento:")
print(f"  Train/Val  : {n_train:,} / {n_val:,} imágenes")
print(f"  Épocas     : {YOLO_CFG_FAST['epochs']} (patience={YOLO_CFG_FAST['patience']})")
print(f"  Batch/GPU  : {YOLO_CFG_FAST['batch']}")
print(f"  Cache      : {YOLO_CFG_FAST['cache']} (SSD local = no necesario)")
print(f"  Tiempo est : ~{(n_train // YOLO_CFG_FAST['batch']) * YOLO_CFG_FAST['epochs'] // 120:.0f} min")

# ── 4. MLflow ─────────────────────────────────────────────────
mlflow.set_tracking_uri(CFG['mlflow_uri'])
mlflow.set_experiment(CFG['mlflow_experiment'])

class _MLflowYOLOCB:
    def __init__(self): self.run_id = None; self._ctx = None

    def on_train_start(self, trainer):
        run = mlflow.start_run(run_name='detector_yolov8n_fast')
        self.run_id = run.info.run_id; self._ctx = run
        mlflow.log_params({
            'model'       : YOLO_CFG_FAST['model_variant'],
            'epochs'      : YOLO_CFG_FAST['epochs'],
            'batch'       : YOLO_CFG_FAST['batch'],
            'device'      : YOLO_CFG_FAST['device'],
            'img_size'    : YOLO_CFG_FAST['img_size'],
            'cache'       : str(YOLO_CFG_FAST['cache']),
            'dataset_src' : 'ssd_local_copy',
            'n_train'     : n_train,
            'n_val'       : n_val,
        })

    def on_fit_epoch_end(self, trainer):
        if not self.run_id: return
        epoch   = trainer.epoch
        metrics = trainer.metrics or {}
        losses  = trainer.loss_items
        log = {}
        if losses is not None and len(losses) >= 3:
            log['train/box_loss'] = float(losses[0])
            log['train/cls_loss'] = float(losses[1])
            log['train/dfl_loss'] = float(losses[2])
        for uk, mk in [
            ('metrics/mAP50(B)',    'val/mAP50'),
            ('metrics/mAP50-95(B)', 'val/mAP50_95'),
            ('metrics/precision(B)','val/precision'),
            ('metrics/recall(B)',   'val/recall'),
        ]:
            if uk in metrics: log[mk] = float(metrics[uk])
        if log: mlflow.log_metrics(log, step=epoch)

    def on_train_end(self, trainer):
        if self._ctx: mlflow.end_run()

yolo_cb   = _MLflowYOLOCB()
det_model = YOLO(YOLO_CFG_FAST['model_variant'])
det_model.add_callback('on_train_start',   yolo_cb.on_train_start)
det_model.add_callback('on_fit_epoch_end', yolo_cb.on_fit_epoch_end)
det_model.add_callback('on_train_end',     yolo_cb.on_train_end)

print("\n=== Iniciando entrenamiento optimizado ===\n")
yolo_results = det_model.train(
    data          = DATASET_YAML_LOCAL,
    epochs        = YOLO_CFG_FAST['epochs'],
    batch         = YOLO_CFG_FAST['batch'],
    imgsz         = YOLO_CFG_FAST['img_size'],
    device        = YOLO_CFG_FAST['device'],
    workers       = YOLO_CFG_FAST['workers'],
    lr0           = YOLO_CFG_FAST['lr0'],
    lrf           = YOLO_CFG_FAST['lrf'],
    momentum      = YOLO_CFG_FAST['momentum'],
    weight_decay  = YOLO_CFG_FAST['weight_decay'],
    warmup_epochs = YOLO_CFG_FAST['warmup_epochs'],
    amp           = YOLO_CFG_FAST['amp'],
    cache         = YOLO_CFG_FAST['cache'],
    patience      = YOLO_CFG_FAST['patience'],
    project       = YOLO_CFG_FAST['project'],
    name          = YOLO_CFG_FAST['name'],
    exist_ok      = YOLO_CFG_FAST['exist_ok'],
    pretrained    = YOLO_CFG_FAST['pretrained'],
)

print('\nYOLO completado')
try:
    res = yolo_results.results_dict
    print(f'  mAP50    : {res.get("metrics/mAP50(B)",    0):.4f}')
    print(f'  mAP50-95 : {res.get("metrics/mAP50-95(B)", 0):.4f}')
    print(f'  Precision: {res.get("metrics/precision(B)",0):.4f}')
    print(f'  Recall   : {res.get("metrics/recall(B)",   0):.4f}')
except: pass