# Reporte rápido para LLM especializado

## Respuestas solicitadas

### 1) ¿Puedes abrir `train_config.json` y pegarme el contenido?

Ruta: `app/models/classifier_artifacts/train_config.json`

```json
{
  "run_id": "20260414_174147",
  "model": "tf_efficientnetv2_s + ProjectionHead + ArcFace v5",
  "backbone": "tf_efficientnetv2_s",
  "embed_dim": 512,
  "num_classes": 107,
  "img_size": 128,
  "batch_size": 64,
  "num_workers": 4,
  "max_epochs": 50,
  "freeze_epochs": 5,
  "patience": 12,
  "lr_head": 0.005,
  "lr_backbone": 0.0001,
  "weight_decay": 0.0005,
  "dropout_rate": 0.4,
  "label_smoothing": 0.05,
  "mixup_alpha": 0.2,
  "tta_n": 5,
  "optimizer": "AdamW",
  "scheduler": "CosineAnnealingLR (sin restarts)",
  "warmup_epochs": 3,
  "scheduler_T_max": 45,
  "scheduler_eta_min": 1e-06,
  "grad_clip_max_norm": 1.0,
  "loss": "FocalLoss(gamma=2.0) + class_weights + label_smoothing",
  "arcface_s": 30.0,
  "arcface_m": 0.15,
  "mixed_precision": true,
  "letterbox_resize": true,
  "accent_augmentation": true,
  "source_weighted_sampling": true,
  "class_weighted_loss": true,
  "accent_boost_in_loss": 1.5,
  "changes_vs_v4": [
    "Projection Head 1280→512 con BN+ReLU+Dropout",
    "Accent Augmentation desde EMNIST real (400/clase)",
    "EMNIST_MAX_PER_CLASS 500→800",
    "SYNTH_PER_CLASS 100→500",
    "FREEZE_EPOCHS 2→5 (estabilizar projector+ArcFace)",
    "LR_HEAD 1e-2→5e-3 (menos agresivo)",
    "LR_BACKBONE 2e-4→1e-4 (fine-tune conservador)",
    "DROPOUT 0.5→0.4",
    "CosineAnnealingLR sin restarts (más estable)",
    "Warmup 3 epochs (LR crece linealmente)",
    "FocalLoss con class_weights (effective number)",
    "Accent classes boosted 1.5x en loss",
    "Source-weighted sampling (verack 1.5x, accent_aug 1.3x)",
    "Test set honesto: solo datos realistas para clases con datos reales",
    "Hard pairs incluyen pares acentuados",
    "ElasticTransform más fuerte en augmentaciones",
    "Morphological ops para simular grosor de trazo",
    "TTA 5 augmentaciones (añade elastic)"
  ],
  "seed": 42
}
```

### 2) ¿Puedes abrir `metrics_report.json` y pegarme el contenido?

Ruta: `app/models/classifier_artifacts/metrics_report.json`

El archivo es muy grande; para tu reporte, estos son los campos clave extraídos directamente:

```json
{
  "run_id": "20260414_174147",
  "model": "tf_efficientnetv2_s + ProjectionHead + ArcFace v5",
  "data": {
    "total_train_images": 99354,
    "total_val_images": 16005,
    "total_test_images": 16005,
    "n_real_classes": 62,
    "n_accent_aug_classes": 14,
    "n_synth_only_classes": 31
  },
  "metrics_global": {
    "best_val_acc": 0.8126,
    "weighted_f1": 0.8093,
    "test_acc": 0.8097,
    "tta_n": 5
  },
  "metrics_honest": {
    "real_test_acc": 0.7934,
    "accent_test_acc": 0.8512,
    "synth_test_acc": 0.9762
  }
}
```

Si quieres pegarlo íntegro en el reporte, usa este archivo directo:
`app/models/classifier_artifacts/metrics_report.json`

### 3) ¿Cuántas imágenes tiene tu dataset de entrenamiento aproximadamente?

Aproximadamente **99 mil** imágenes de entrenamiento.
Valor exacto reportado: **99,354** (`total_train_images`).

### 4) ¿Qué versión de Python usas?

Versión detectada en entorno local:

```bash
Python 3.10.11
```

### 5) ¿Tienes un `requirements.txt` o `pyproject.toml`? Si sí, pégamelo.

Se encontró `requirements.txt` (no se encontró `pyproject.toml`).

Ruta principal: `requirements.txt`

```txt
# =============================================================================
# requirements.txt — Tutor Inteligente de Caligrafía
# Versiones fijadas a 15/03/2026 (ver PLAN_IMPLEMENTACIONES_ESTADIA.md § 3.0)
# =============================================================================

# ── API y servidor ─────────────────────────────────────────────────────────
fastapi
uvicorn
python-multipart

# ── Visión y procesamiento de imágenes ────────────────────────────────────
# numpy < 2.0 para compatibilidad con PyTorch / ONNX
numpy==1.26.4
opencv-python-headless==4.9.0.80
scikit-image
scipy
matplotlib
pillow
pillow-avif-plugin==1.4.3

# ── Inferencia ONNX (producción y API) ────────────────────────────────────
# onnx 1.16.x — compatible con opset 17
onnx>=1.16.0
# + onnxruntime==1.18.1— inferencia CPU/GPU
onnxruntime==1.18.1
onnxscript

# ── PyTorch — versión única para todo el pipeline ─────────────────────────
# torch 2.2.0 / torchvision 0.17.0 (misma minor)
# NOTA: en Kaggle/Colab usar las versiones preinstaladas del entorno
#       si hay conflicto de CUDA; cambiar a torch==2.1.2+torchvision==0.16.2
torch==2.2.0
torchvision==0.17.0

# ── Detector YOLO ─────────────────────────────────────────────────────────
# ultralytics >=8.0.0,<9.0.0 (YOLOv8n, export ONNX)
ultralytics>=8.0.0,<9.0.0

# ── Clasificador — backbone (timm) ────────────────────────────────────────
# timm 0.9.x — EfficientNet-B2 y otros backbones
timm==0.9.16

# ── Augmentaciones ────────────────────────────────────────────────────────
albumentations==1.3.1

# ── Experimentos y reproducibilidad ───────────────────────────────────────
# mlflow 2.10.x — registro de runs, métricas y artefactos
mlflow==2.10.2
tqdm

# ── Export TF → ONNX (legacy; solo si se usan modelos TF anteriores) ──────
# ADVERTENCIA: tensorflow puede entrar en conflicto con torch en el mismo
# entorno. Instalar en entorno separado si no se necesita la ruta TF.
# tensorflow>=2.10.0
# tf2onnx

# ── Descarga de datasets ───────────────────────────────────────────────────
kagglehub
requests
```

También existe: `kivy_app/requirements.txt`

```txt
# kivy_app/requirements.txt  — entorno de desarrollo en escritorio
kivy==2.3.0
requests>=2.31.0
Pillow>=10.0.0
plyer>=2.1.0
```

### 6) ¿Tu proyecto se ejecuta con `uvicorn`? ¿Cuál es el comando exacto?

Sí, se ejecuta con `uvicorn` para la API FastAPI.

Comando recomendado:

```bash
uvicorn app.main:app --reload
```

También existe ejecución embebida en `app/main.py`:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 7) ¿Tienes Docker o solo se ejecuta local?

En este repo no se detectó `Dockerfile` ni `docker-compose*.yml`.
Entonces, **actualmente está configurado para ejecución local**.

### 8) ¿Tu app tiene frontend o solo API?

No es solo API. Tiene:
- **API backend** con FastAPI (`app/...`)
- **Frontend/cliente de escritorio** con Kivy (`kivy_app/...`)

---

## Mini tutorial: usar MiKTeX para tu reporte (.tex)

### Paso 1: instalar MiKTeX
1. Descarga MiKTeX desde [miktex.org](https://miktex.org/download).
2. En el instalador, activa instalación de paquetes “on-the-fly”.
3. Abre **MiKTeX Console** y en “Updates” aplica actualizaciones.

### Paso 2: editor para LaTeX
Opciones rápidas:
- TeXworks (viene con MiKTeX)
- VS Code + extensión LaTeX Workshop
- Cursor (editas `.tex` y compilas por terminal)

### Paso 3: plantilla mínima
Guarda esto como `reporte.tex`:

```tex
\documentclass[12pt]{article}
\usepackage[spanish]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{float}
\usepackage{geometry}
\geometry{margin=2.5cm}

\title{Reporte de Estadia}
\author{Tu Nombre}
\date{\today}

\begin{document}
\maketitle

\section{Introducción}
Texto de contexto del proyecto.

\section{Resultados}
\begin{figure}[H]
  \centering
  \includegraphics[width=0.7\textwidth]{imgs/resultado1.png}
  \caption{Ejemplo de resultado del modelo.}
\end{figure}

\end{document}
```

### Paso 4: compilar a PDF
En terminal, desde la carpeta del `.tex`:

```bash
pdflatex reporte.tex
```

Si usas bibliografía (BibTeX), ciclo típico:
`pdflatex -> bibtex -> pdflatex -> pdflatex`.

### Paso 5: flujo práctico con Cursor + capturas
1. Genera gráficas/tablas en Python (Matplotlib/Seaborn) y guárdalas en `imgs/`.
2. Inserta cada imagen con `\includegraphics`.
3. Escribe secciones por partes (metodología, métricas, conclusiones).
4. Compila y corrige warnings.

### Tip rápido para automatizar
Puedes pedirle a Cursor:
- “Con esta tabla CSV, genera una figura en Python y guárdala en `imgs/`”
- “Agrégame una sección LaTeX con la interpretación de esta gráfica”
- “Reordena el `.tex` para formato IEEE/APA”

