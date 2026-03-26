"""
training/config.py
==================
Configuración centralizada del pipeline de entrenamiento.

Contiene:
  · DataSources        — qué carpetas de data/ se usan en cada entrenamiento.
  · DetectorConfig     — hiperparámetros del detector YOLOv8n.
  · LOCAL_CPU          — preset para máquina sin GPU (entrenamiento local).
  · KAGGLE_T4_DUAL     — preset para 2× T4 en Kaggle (máximo rendimiento).

Uso rápido:
    from training.config import LOCAL_CPU, KAGGLE_T4_DUAL
    cfg = KAGGLE_T4_DUAL          # o LOCAL_CPU
    train_detector(cfg=cfg, ...)
"""

from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict


# =============================================================================
# DataSources — qué carpetas de data/ alimentan el entrenamiento
# =============================================================================

class DataSources(TypedDict):
    """
    Define qué fuentes de datos se usan en cada entrenamiento.

    Campos
    ------
    data_root : str
        Ruta base del proyecto (contiene raw/, processed/, augmented/).
    use_raw : bool
        Incluir datos originales de data/raw/ en el dataset.
    use_augmented : bool
        Incluir imágenes aumentadas de data/augmented/.
    use_synthetic_yolo : bool
        Incluir el dataset sintético generado por generate_synthetic_yolo.py
        (data/processed/yolo_dataset/).
    val_split : float
        Fracción de imágenes para validación (0–1). Default 0.15.
    """
    data_root:          str
    use_raw:            bool
    use_augmented:      bool
    use_synthetic_yolo: bool
    val_split:          float


# Preset de DataSources por defecto (usa todo)
DEFAULT_SOURCES: DataSources = {
    "data_root":          "./data",
    "use_raw":            True,
    "use_augmented":      True,
    "use_synthetic_yolo": True,
    "val_split":          0.15,
}


# =============================================================================
# DetectorConfig — hiperparámetros del detector YOLOv8n
# =============================================================================

@dataclass
class DetectorConfig:
    """
    Hiperparámetros y opciones de entrenamiento del detector YOLOv8n.

    Parámetros de Ultralytics
    -------------------------
    model_variant : str
        Variante YOLO: "yolov8n.pt" (nano, recomendado para móvil).
    epochs : int
        Número de épocas de entrenamiento.
    batch : int
        Batch size por GPU (o total si device="cpu").
        · CPU local : 8–16
        · T4 × 1    : 32
        · T4 × 2    : 32 por GPU (64 efectivo con DDP)
    img_size : int
        Tamaño de entrada: 640 (estándar YOLO, requerido por la API).
    device : str
        "cpu"  → entrenamiento en CPU
        "0"    → GPU 0 únicamente
        "0,1"  → DDP en 2× T4 (Kaggle)
    workers : int
        Número de workers del DataLoader.
        · CPU local : min(4, n_cores - 1) — dejar al menos 1 núcleo libre
        · Kaggle    : 8 (T4 tiene 2 CPUs virtuales × 4 workers = margen amplio)
    lr0 : float
        Learning rate inicial.
    lrf : float
        Fracción del lr final respecto al inicial (OneCycleLR).
    momentum : float
        Momentum del optimizador SGD.
    weight_decay : float
        Regularización L2.
    warmup_epochs : float
        Épocas de warm-up lineal del lr.
    box : float
        Peso de la pérdida de caja (box loss).
    cls : float
        Peso de la pérdida de clasificación (cls loss).
    dfl : float
        Peso de la pérdida DFL (distribution focal loss).
    patience : int
        Épocas sin mejora antes de early stopping (0 = desactivado).
    cache : str | bool
        "ram"  → cachear imágenes en RAM (requiere ~8 GB libres en Kaggle)
        "disk" → cachear en disco (más lento pero menos RAM)
        False  → sin caché (recomendado para CPU con poca RAM)
    amp : bool
        Entrenamiento en precisión mixta (FP16).  Requiere GPU CUDA.
        Ignorado automáticamente por Ultralytics si device="cpu".
    exist_ok : bool
        Sobreescribir experimento existente.
    pretrained : bool
        Inicializar desde pesos preentrenados de COCO (transfer learning).
    freeze : int | None
        Número de capas a congelar (útil si se hace fine-tuning pequeño).
        None = sin congelamiento (entrenar todo).

    Rutas de salida
    ---------------
    project : str
        Carpeta raíz de experimentos Ultralytics.
    name : str
        Nombre del run dentro de project/.

    MLflow
    ------
    mlflow_tracking_uri : str | None
        URI del servidor MLflow. None → mlruns/ local.
    mlflow_experiment : str
        Nombre del experimento MLflow.
    """

    # ── Modelo ───────────────────────────────────────────────────────────────
    model_variant:        str   = "yolov8n.pt"
    img_size:             int   = 640
    pretrained:           bool  = True
    freeze:               int | None = None

    # ── Entrenamiento ────────────────────────────────────────────────────────
    epochs:               int   = 50
    batch:                int   = 16
    device:               str   = "cpu"
    workers:              int   = field(default_factory=lambda: max(1, multiprocessing.cpu_count() - 1))
    amp:                  bool  = False
    cache:                str | bool = False
    patience:             int   = 15
    exist_ok:             bool  = True

    # ── Learning rate ────────────────────────────────────────────────────────
    lr0:                  float = 0.01
    lrf:                  float = 0.01
    momentum:             float = 0.937
    weight_decay:         float = 5e-4
    warmup_epochs:        float = 3.0

    # ── Pérdidas ─────────────────────────────────────────────────────────────
    box:                  float = 7.5
    cls:                  float = 0.5
    dfl:                  float = 1.5

    # ── Rutas ────────────────────────────────────────────────────────────────
    project:              str   = "./runs/detect"
    name:                 str   = "char_detector"

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri:  str | None = None
    mlflow_experiment:    str   = "yolo_char_detector"

    # ── Fuentes de datos ─────────────────────────────────────────────────────
    sources:              DataSources = field(default_factory=lambda: dict(DEFAULT_SOURCES))

    def as_ultralytics_kwargs(self) -> dict:
        """
        Devuelve los argumentos directamente pasables a model.train(**kwargs).
        Excluye campos propios de config (project paths, mlflow, sources).
        """
        return {
            "epochs":         self.epochs,
            "batch":          self.batch,
            "imgsz":          self.img_size,
            "device":         self.device,
            "workers":        self.workers,
            "lr0":            self.lr0,
            "lrf":            self.lrf,
            "momentum":       self.momentum,
            "weight_decay":   self.weight_decay,
            "warmup_epochs":  self.warmup_epochs,
            "box":            self.box,
            "cls":            self.cls,
            "dfl":            self.dfl,
            "patience":       self.patience,
            "cache":          self.cache,
            "amp":            self.amp,
            "exist_ok":       self.exist_ok,
            "pretrained":     self.pretrained,
            "project":        self.project,
            "name":           self.name,
            **({"freeze": self.freeze} if self.freeze is not None else {}),
        }


# =============================================================================
# Preset: LOCAL_CPU
# =============================================================================

def _local_workers() -> int:
    """Workers óptimos para CPU local: la mitad de los núcleos, mínimo 2."""
    return max(2, multiprocessing.cpu_count() // 2)


LOCAL_CPU = DetectorConfig(
    # ── Modelo ───────────────────────────────────────────────────────────────
    model_variant    = "yolov8n.pt",
    img_size         = 640,
    pretrained       = True,
    freeze           = None,

    # ── Entrenamiento ────────────────────────────────────────────────────────
    # Épocas reducidas para que termine en un tiempo razonable sin GPU.
    # Con ~5K imágenes sintéticas, 30 épocas dan mAP50 > 0.7 en CPU.
    epochs           = 30,
    batch            = 8,           # Batch pequeño: menos RAM requerida
    device           = "cpu",
    workers          = _local_workers(),
    amp              = False,       # FP16 no está soportado en CPU
    cache            = False,       # Sin caché: menos RAM
    patience         = 10,

    # ── Learning rate ────────────────────────────────────────────────────────
    lr0              = 0.01,
    lrf              = 0.01,
    momentum         = 0.937,
    weight_decay     = 5e-4,
    warmup_epochs    = 2.0,

    # ── Pérdidas ─────────────────────────────────────────────────────────────
    box              = 7.5,
    cls              = 0.5,
    dfl              = 1.5,

    # ── Rutas ────────────────────────────────────────────────────────────────
    project          = "./runs/detect",
    name             = "char_detector_cpu",

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri  = None,        # mlruns/ local
    mlflow_experiment    = "yolo_char_detector_local",

    # ── Fuentes de datos ─────────────────────────────────────────────────────
    sources = {
        "data_root":          "./data",
        "use_raw":            True,
        "use_augmented":      False,    # Omitir aumentados para ir más rápido en local
        "use_synthetic_yolo": True,
        "val_split":          0.15,
    },
)


# =============================================================================
# Preset: KAGGLE_T4_DUAL
# =============================================================================

KAGGLE_T4_DUAL = DetectorConfig(
    # ── Modelo ───────────────────────────────────────────────────────────────
    model_variant    = "yolov8n.pt",
    img_size         = 640,
    pretrained       = True,
    freeze           = None,

    # ── Entrenamiento ────────────────────────────────────────────────────────
    # 2× T4 vía DDP. Cada T4 tiene 16 GB VRAM.
    # batch=32 por GPU → 64 efectivo (Ultralytics ajusta automáticamente con DDP).
    # cache="ram" requiere ~8 GB libres; en Kaggle (30 GB RAM) es seguro.
    epochs           = 100,
    batch            = 32,
    device           = "0,1",       # DDP: ambas T4
    workers          = 8,           # 8 workers por DataLoader en Kaggle
    amp              = True,        # FP16: ~2× velocidad en T4
    cache            = "ram",       # Cachear dataset en RAM de Kaggle
    patience         = 20,

    # ── Learning rate ────────────────────────────────────────────────────────
    # lr0 más alto y warmup más largo para aprovechar el batch grande (DDP)
    lr0              = 0.02,
    lrf              = 0.01,
    momentum         = 0.937,
    weight_decay     = 5e-4,
    warmup_epochs    = 5.0,

    # ── Pérdidas ─────────────────────────────────────────────────────────────
    box              = 7.5,
    cls              = 0.5,
    dfl              = 1.5,

    # ── Rutas ────────────────────────────────────────────────────────────────
    project          = "/kaggle/working/runs/detect",
    name             = "char_detector_t4",

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri  = None,    # mlruns/ en /kaggle/working/
    mlflow_experiment    = "yolo_char_detector_kaggle",

    # ── Fuentes de datos ─────────────────────────────────────────────────────
    sources = {
        "data_root":          "/kaggle/working/data",
        "use_raw":            True,
        "use_augmented":      True,
        "use_synthetic_yolo": True,
        "val_split":          0.15,
    },
)