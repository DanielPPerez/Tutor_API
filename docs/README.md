# Tutor Inteligente de Caligrafía

Sistema de visión por computadora y deep learning que evalúa la calidad de trazos manuscritos de niños en etapa de aprendizaje de escritura. Incluye una API backend (FastAPI), un pipeline de entrenamiento completo en Kaggle y un cliente de escritorio (Kivy).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.18.1-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![EfficientNetV2](https://img.shields.io/badge/EfficientNetV2--S-ArcFace-purple)
![Kivy](https://img.shields.io/badge/Kivy-2.3.0-cyan)

---

## Tabla de Contenidos

1. [Objetivo del Sistema](#1-objetivo-del-sistema)
2. [Problema que Resuelve](#2-problema-que-resuelve)
3. [Arquitectura General](#3-arquitectura-general)
4. [Flujo de Procesamiento](#4-flujo-de-procesamiento)
5. [Estructura del Proyecto](#5-estructura-del-proyecto)
6. [Tecnologías Utilizadas](#6-tecnologías-utilizadas)
7. [Instalación y Ejecución del Backend](#7-instalación-y-ejecución-del-backend)
8. [Endpoints de la API](#8-endpoints-de-la-api)
9. [Notebooks de Entrenamiento (Kaggle)](#9-notebooks-de-entrenamiento-kaggle)
   - [9.1 Detector YOLOv8](#91-detector-yolov8-spanish_char_detectoripynb)
   - [9.2 Clasificador OCR](#92-clasificador-ocr-clasificador-ocr-spanishipynb)
10. [Modelo Clasificador — Especificaciones](#10-modelo-clasificador--especificaciones)
11. [Métricas de Evaluación de Trazo](#11-métricas-de-evaluación-de-trazo)
12. [Pipeline de Procesamiento de Imagen](#12-pipeline-de-procesamiento-de-imagen)
13. [Cliente de Escritorio Kivy](#13-cliente-de-escritorio-kivy)
14. [Supuestos y Limitaciones](#14-supuestos-y-limitaciones)
15. [Troubleshooting](#15-troubleshooting)
16. [Reentrenamiento de Modelos](#16-reentrenamiento-de-modelos)
17. [Trabajo Futuro](#17-trabajo-futuro)
18. [Checklist de Entrega (Handoff)](#18-checklist-de-entrega-handoff)
19. [Enlaces](#19-enlaces)
20. [Autor](#20-autor)

---

## 1. Objetivo del Sistema

Desarrollar un tutor inteligente de caligrafía que permita a niños en etapa de aprendizaje de escritura recibir evaluación y retroalimentación automática sobre la calidad de sus trazos manuscritos.

El sistema analiza fotografías de cuadernos escolares tomadas con celular, detecta los caracteres escritos, los compara contra plantillas de referencia y genera una calificación numérica (0–100) junto con retroalimentación pedagógica en español.

---

## 2. Problema que Resuelve

Los profesores de primaria no tienen tiempo de revisar individualmente cada trazo de cada alumno en cada plana. Este sistema automatiza la evaluación formativa, permitiendo que el alumno practique y reciba retroalimentación inmediata sin depender de la disponibilidad del docente.

---

## 3. Arquitectura General

```
            Cliente Kivy (Escritorio)
           kivy_app/ — UI para el alumno
                  |
                  | HTTP (REST)
                  v
+-------------------------------------------------------+
|              FastAPI Backend (API)                     |
|-------------------------------------------------------|
|                                                       |
|   /evaluate      /evaluate_plana      /recognize      |
|                                                       |
|-------------------------------------------------------|
|                                                       |
|  YOLOv8           Processor v4.2     Normalizer v6   |
|  Detector         (orquesta          (máscara de      |
|  (best_           pipelines)          trazo para      |
|  detector.pt)                         métricas)       |
|                                                       |
|  image_cleaner    Classifier          8 Módulos de    |
|  (inpainting      EfficientNetV2-S    Métricas        |
|   líneas azules)  + ArcFace           (dt, geo, topo, |
|                   + SmartOCR          ssim, traj,     |
|                                       hausdorff,      |
|                                       cosine, qual)   |
|                                                       |
|              Scorer + Feedback                        |
|         (nota final 0–100 + retroalimentación)        |
+-------------------------------------------------------+
```

---

## 4. Flujo de Procesamiento

### Evaluación de un carácter individual (`POST /evaluate`)

```
Foto de cuaderno (JPG/PNG)
        |
        v
YOLO detecta el carácter ← Imagen ORIGINAL (con líneas)
        |
   +----+----+
   v         v
Pipeline A   Pipeline B
  (OCR)       (Máscara)
   |               |
   v               v
image_cleaner   image_cleaner
→ grayscale     → binarización
  limpio        → morfología
→ EfficientNet  → deskew
→ SmartOCR      → crop + center
   |               |
   v               v
 "a"           Máscara
conf: 0.95     128×128
               blanco=trazo
               negro=fondo
        |
        v
  Cargar plantilla .npy
   del carácter esperado
        |
        v
  Esqueletizar ambos trazos
  (alumno y plantilla)
        |
        v
  Calcular 8 métricas
        |
        v
  Scorer → nota 0–100
  + feedback pedagógico
        |
        v
  Visualización comparativa
```

### Por qué existen dos pipelines separados

| Pipeline | Propósito | Salida |
|----------|-----------|--------|
| **A (OCR)** | Clasificar qué letra escribió el alumno | Carácter + confianza |
| **B (Métricas)** | Generar máscara binaria para evaluar forma | Máscara binaria 128×128 |

El clasificador necesita escala de grises continua (256 niveles) para distinguir detalles finos. Las métricas necesitan una máscara binaria para calcular distancias, topología y esqueletos. Mezclar ambos produce resultados inferiores en las dos tareas.

---

## 5. Estructura del Proyecto

```
proyecto/
├── app/
│   ├── main.py                         # Punto de entrada FastAPI
│   ├── core/
│   │   ├── config.py                   # Configuración global (umbrales, rutas, pesos)
│   │   ├── processor.py                # Orquestador principal (v4.2)
│   │   ├── normalizer.py               # Generador de máscara binaria (v6)
│   │   ├── image_cleaner.py            # Limpieza de imagen (inpainting de líneas)
│   │   ├── preprocessing.py            # Preprocesamiento para modelo ONNX
│   │   ├── classifier.py               # Clasificador OCR + SmartOCR
│   │   ├── binarizer.py                # Binarización adaptativa (Otsu/adaptiva)
│   │   ├── illumination.py             # Normalización de iluminación
│   │   └── image_quality.py            # Análisis de calidad de imagen
│   ├── api/
│   │   └── endpoints.py                # Router: /evaluate, /evaluate_plana, /recognize
│   ├── metrics/
│   │   ├── distance_transform.py       # Fidelidad por distance transform
│   │   ├── geometric.py                # Proporciones geométricas
│   │   ├── topologic.py                # Topología (loops, endpoints, junctions)
│   │   ├── trajectory.py               # Similitud de trayectoria (DTW)
│   │   ├── quality.py                  # Calidad del trazo (grosor, suavidad)
│   │   ├── segment_cosine.py           # Similitud coseno por segmentos
│   │   └── scorer.py                   # Calculadora de nota final + feedback
│   ├── models/
│   │   └── classifier_artifacts/
│   │       ├── best_classifier.onnx    # Modelo clasificador (ONNX)
│   │       ├── best_classifier.onnx.data
│   │       ├── best_detector.pt        # Detector YOLO (PyTorch)
│   │       └── best_detector.onnx      # Detector YOLO (ONNX)
│   ├── scripts/
│   │   └── generate_templates.py       # Generador de plantillas + esqueletos
│   ├── templates/
│   │   ├── principiante/               # Plantillas .npy con tolerancia amplia
│   │   ├── intermedio/                 # Plantillas .npy con tolerancia media
│   │   └── avanzado/                   # Plantillas .npy con tolerancia estricta
│   └── utils/
│       └── visualizer.py               # Generador de imágenes de comparación
├── kivy_app/
│   ├── main.py                         # Cliente de escritorio Kivy
│   └── requirements.txt
└── requirements.txt
```

---

## 6. Tecnologías Utilizadas

| Componente | Tecnología | Versión |
|------------|------------|---------|
| Lenguaje | Python | 3.10.11 |
| Backend API | FastAPI | latest |
| Servidor ASGI | Uvicorn | latest |
| Deep Learning Runtime | ONNX Runtime | 1.18.1 |
| Detección de objetos | YOLOv8 (Ultralytics) | ≥8.0.0 |
| Backbone clasificador | EfficientNetV2-S (timm) | 0.9.16 |
| Framework de entrenamiento | PyTorch | 2.2.0 |
| Procesamiento de imagen | OpenCV (headless) | 4.9.0.80 |
| Cómputo numérico | NumPy | 1.26.4 |
| Esqueletización | scikit-image | latest |
| Alineamiento temporal | SciPy (DTW) | latest |
| Augmentaciones | Albumentations | 1.3.1 |
| Experimentos | MLflow | 2.10.2 |
| Cliente escritorio | Kivy | 2.3.0 |

---

## 7. Instalación y Ejecución del Backend

### Prerrequisitos

- Python 3.10+
- pip
- ~2 GB de espacio para modelos y dependencias
- (Opcional) GPU con CUDA para inferencia acelerada

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/DanielPPerez/Tutor_API.git
cd Tutor_API
```

### Paso 2: Crear entorno virtual

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

> Si hay conflictos con CUDA/PyTorch, instala PyTorch primero siguiendo las instrucciones en https://pytorch.org/get-started/locally/ y luego instala el resto.

### Paso 4: Verificar que los modelos existen

Los siguientes archivos deben estar presentes antes de levantar el servidor:

```
app/models/classifier_artifacts/best_classifier.onnx
app/models/classifier_artifacts/best_classifier.onnx.data
app/models/classifier_artifacts/best_detector.pt   (o best_detector.onnx)
```

Si faltan, la API levantará pero fallará en inferencia. Consulta la sección [Reentrenamiento de Modelos](#16-reentrenamiento-de-modelos).

### Paso 5: Generar plantillas (solo la primera vez)

```bash
python -m app.scripts.generate_templates
```

Genera archivos `.npy` en `app/templates/` para los tres niveles de dificultad y sus esqueletos de 1px.

### Paso 6: Levantar el servidor

```bash
# Modo desarrollo (con recarga automática)
uvicorn app.main:app --reload

# Modo producción
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Paso 7: Verificar funcionamiento

Abre en el navegador:

```
http://localhost:8000/docs
```

Se mostrará la documentación interactiva de Swagger con los tres endpoints disponibles.

---

## 8. Endpoints de la API

### `POST /evaluate`

Evalúa el trazo de un único carácter manuscrito contra la plantilla esperada.

**Parámetros (form-data):**

| Parámetro | Tipo | Requerido | Default | Descripción |
|-----------|------|-----------|---------|-------------|
| `file` | File (JPG/PNG) | Sí | — | Foto del carácter manuscrito |
| `target_char` | string | Sí | — | Carácter esperado (ej: `a`, `B`, `ñ`, `3`) |
| `level` | string | No | `intermedio` | Nivel: `principiante`, `intermedio`, `avanzado` |

**Ejemplo:**

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -F "file=@foto_letra_a.jpg" \
  -F "target_char=a" \
  -F "level=intermedio"
```

**Respuesta exitosa:**

```json
{
  "target_char": "a",
  "detected_char": "a",
  "confidence": 0.9234,
  "score_final": 78.5,
  "level": "intermedio",
  "scores_breakdown": {
    "dt_precision": 82.3,
    "dt_coverage": 75.1,
    "topology": 100.0,
    "ssim": 68.5,
    "procrustes": 80.0,
    "hausdorff": 72.1,
    "trajectory": 65.2,
    "cosine": 71.8
  },
  "weights_used": {
    "dt_precision": 0.30,
    "dt_coverage": 0.20,
    "topology": 0.20,
    "ssim": 0.12,
    "procrustes": 0.10,
    "hausdorff": 0.04,
    "trajectory": 0.02,
    "cosine": 0.02
  },
  "feedback": "Buen trazo. La forma general es correcta. Mejora la continuidad.",
  "metadata": {
    "mask_source": "normalizer",
    "pipeline_version": "v4.2_clean",
    "used_image_cleaner": true,
    "yolo_detected": true,
    "yolo_confidence": 0.89
  },
  "image_student_b64": "iVBORw0KGgo...",
  "template_b64": "iVBORw0KGgo...",
  "comparison_b64": "iVBORw0KGgo..."
}
```

---

### `POST /evaluate_plana`

Detecta todos los caracteres en una foto de plana escolar y los evalúa en conjunto. El primer carácter detectado se usa como plantilla de referencia; los demás se evalúan contra ella.

**Parámetros (form-data):**

| Parámetro | Tipo | Requerido | Default | Descripción |
|-----------|------|-----------|---------|-------------|
| `file` | File (JPG/PNG) | Sí | — | Foto de la plana completa |
| `target_char` | string | No | (vacío) | Carácter esperado. Si vacío, se infiere automáticamente |
| `level` | string | No | `intermedio` | Nivel: `principiante`, `intermedio`, `avanzado` |

**Ejemplo:**

```bash
curl -X POST "http://localhost:8000/evaluate_plana" \
  -F "file=@plana_letra_a.jpg" \
  -F "target_char=a" \
  -F "level=principiante"
```

**Respuesta (campos clave):**

```json
{
  "template_char": "a",
  "n_detected": 12,
  "n_evaluated": 11,
  "avg_score": 72.3,
  "detection_method": "preprocess_multi",
  "smart_ocr": {
    "recognized_text": "aaaaaaaaaaaa",
    "overall_confidence": 0.91
  },
  "results": [
    {
      "index": 1,
      "detected_char": "a",
      "confidence": 0.91,
      "score_final": 85.2,
      "feedback": "Muy buen trazo."
    }
  ]
}
```

> **Fallback automático:** Si `preprocess_multi` falla, el endpoint aplica detección YOLO directa con Ultralytics como respaldo.

---

### `POST /recognize`

Reconoce todos los caracteres en la imagen y devuelve el texto. Usa SmartOCR con agrupación de palabras, contexto posicional y diccionario. No evalúa calidad de trazo.

**Ejemplo:**

```bash
curl -X POST "http://localhost:8000/recognize" \
  -F "file=@texto_manuscrito.jpg"
```

**Respuesta:**

```json
{
  "text": "Hola mundo",
  "n_detected": 9,
  "confidence": 0.87,
  "words": [
    {
      "word": "Hola",
      "raw_word": "Ho1a",
      "corrected": true,
      "correction_method": "dictionary_levenshtein"
    }
  ]
}
```

---

## 9. Notebooks de Entrenamiento (Kaggle)

Los modelos se entrenaron en Kaggle. A continuación se documenta el proceso completo de cada notebook para que cualquier persona pueda reproducir el entrenamiento desde cero.

---

### 9.1 Detector YOLOv8 (`spanish_char_detector.ipynb`)

**URL:** https://www.kaggle.com/code/danielperegrinoperez/detector-train

Este notebook entrena el detector single-class (clase 0 = `trazo`) que localiza caracteres manuscritos en imágenes completas. Es la **Etapa 1** del pipeline de 2 etapas.

#### Datasets requeridos en Kaggle

| Dataset | Propósito | Ruta esperada |
|---------|-----------|---------------|
| `spanish-ocr-dataset` (propio) | Imágenes YOLO con bboxes | `/kaggle/input/spanish-ocr-dataset/yolo_dataset_final` |
| `crawford/emnist` | Crops de caracteres individuales | `/kaggle/input/emnist` |
| `verack/spanish-handwritten-characterswords` | Crops reales en español | `/kaggle/input/spanish-handwritten-characterswords` |

#### Estructura del notebook

**Celda D-0 — Setup del entorno**

Verifica GPU, descarga pesos base de YOLOv8 (`yolov8n.pt`) y crea la estructura de directorios en `/kaggle/working/detector/`.

**Celda D-1 — Inspección y validación del dataset**

Realiza una auditoría completa antes de entrenar:
- Inventario de pares imagen/etiqueta y detección de huérfanos
- Parseo de etiquetas YOLO y estadísticas de bounding boxes
- Deduplicación por hash MD5 (preferencia al split `train`)
- Verificación de calidad de imagen (imágenes corruptas, todo blanco, todo negro)
- Saneamiento de coordenadas (clip a [0.001, 0.999], eliminación de boxes con w/h < 0.01)

**Celda D-2 — Composición de imágenes multi-carácter** *(paso crítico)*

El detector debe aprender a manejar palabras y planas completas, no solo caracteres aislados. Este paso construye ese dataset sintético:

1. Extrae crops individuales de tres fuentes: dataset base YOLO, EMNIST, y dataset español verack.
2. Aplica augmentación a nivel de crop (brillo, ruido, rotación, perspectiva).
3. Compone imágenes sintéticas de múltiples líneas sobre fondos variados (blanco, ruidoso, degradado, cuadriculado, papel).
4. Mezcla samples reales de carácter único con composites sintéticos (80/10/10 train/val/test).
5. Genera el archivo `data.yaml` requerido por Ultralytics.

```python
# Configuración de augmentación a nivel de crop
CROP_AUG = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(5, 20), p=0.3),
    A.ElasticTransform(alpha=1, sigma=5, p=0.2),
    A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.4),
    A.Perspective(scale=(0.01, 0.03), p=0.2),
])
```

**Celda D-3 — Entrenamiento YOLOv8s**

```python
model = YOLO('yolov8s.pt')
results = model.train(
    data     = str(BASE / 'dataset' / 'data.yaml'),
    epochs   = 80,
    imgsz    = 640,
    batch    = 32,
    device   = 0,
    mosaic   = 0.3,   # ≤0.5 para no cortar caracteres en los bordes de los tiles
    fliplr   = 0.0,   # sin flip horizontal (espejea los caracteres)
    flipud   = 0.0,   # sin flip vertical (texto al revés)
    degrees  = 5,
    scale    = 0.3,
    patience = 20,
    amp      = True,
)
```

> **Razón de `fliplr=0` y `flipud=0`:** Los flips crean caracteres espejados o invertidos que no existen en escritura real. Esto causaría que el detector aprenda patrones inválidos.

**Celda D-4 — Evaluación**

Evalúa en el split de test con umbral `conf=0.25, iou=0.5` y reporta mAP@0.50, mAP@0.50:0.95, Precision y Recall.

**Celda D-5 — Exportación**

```python
# Exportar a ONNX (opset 17, simplificado)
onnx_path = model.export(format='onnx', imgsz=640, opset=17, simplify=True)
```

Genera:
- `/kaggle/working/detector/exports/best_detector.pt`
- `/kaggle/working/detector/exports/best_detector.onnx`

**Celda D-6 — Validación end-to-end en plana**

Ejecuta el detector sobre una plana sintética generada localmente y verifica que las detecciones se ordenan correctamente en orden de lectura (de arriba a abajo, de izquierda a derecha).

```python
def sort_boxes_reading_order(boxes_xyxy, line_tolerance=15):
    """Ordena bboxes en orden de lectura usando agrupación por línea."""
    boxes = sorted(boxes_xyxy, key=lambda b: ((b[1]+b[3])//2 // line_tolerance, b[0]))
    return boxes
```

#### Salidas del notebook

| Archivo | Descripción |
|---------|-------------|
| `best_detector.pt` | Pesos PyTorch — usar en la API con Ultralytics |
| `best_detector.onnx` | Modelo ONNX — alternativa para runtime sin PyTorch |

> **Dónde colocarlos:** Ambos archivos deben ir a `app/models/classifier_artifacts/` en el backend.

---

### 9.2 Clasificador OCR (`clasificador-ocr-spanish.ipynb`)

**URL:** https://www.kaggle.com/code/danielperegrinoperez/clasificador-ocr-spanish

Este notebook entrena el clasificador de 107 clases que identifica qué carácter contiene cada crop. Es la **Etapa 2** del pipeline. Actualmente en versión v5.

#### Datasets requeridos en Kaggle

| Dataset | Propósito |
|---------|-----------|
| `danielperegrinoperez/char-map` | `char_map.json` con mapeo índice↔carácter |
| `crawford/emnist` | Datos reales de caracteres latinos y dígitos |
| `verack/spanish-handwritten-characterswords` | Datos reales de caracteres en español |
| `sueiras/handwritting-characters-database` | Datos adicionales de caracteres |

#### Arquitectura del modelo

```
Input (128×128 RGB)
        ↓
EfficientNetV2-S (tf_efficientnetv2_s, timm)
  → features: 1280-dimensional
        ↓
Projection Head
  Linear(1280 → 512) + BatchNorm + ReLU + Dropout(0.4)
        ↓
ArcFace Loss (s=30.0, m=0.15)
  → 107 clases
```

La combinación EfficientNet + ArcFace es estándar en reconocimiento de caracteres con muchas clases similares entre sí, ya que ArcFace maximiza el margen angular entre clases en el espacio de embeddings.

#### Hiperparámetros clave

```python
IMG_SIZE       = 128
NUM_CLASSES    = 107
BATCH_SIZE     = 64
LR_HEAD        = 5e-3
LR_BACKBONE    = 1e-4
MAX_EPOCHS     = 50
FREEZE_EPOCHS  = 5      # Backbone congelado mientras el head se estabiliza
PATIENCE       = 12
ARCFACE_S      = 30.0
ARCFACE_M      = 0.15
DROPOUT_RATE   = 0.4
LABEL_SMOOTH   = 0.05
MIXUP_ALPHA    = 0.2
TTA_N          = 5
```

#### Proceso de construcción del dataset

El dataset de entrenamiento se construye en fases:

**Fase A — Datos reales (EMNIST + verack + sueiras)**

- EMNIST provee hasta 800 imágenes por clase para dígitos y letras latinas sin acentos.
- verack provee datos reales de caracteres en español incluyendo letras acentuadas (`á`, `é`, `í`, `ó`, `ú`, `ü`, `ñ`) y mayúsculas acentuadas. El mapeo de nombres de carpeta a carácter se hace con NFC normalization para evitar problemas de encoding Unicode.
- Las imágenes EMNIST se corrigen de orientación antes de usarse:

```python
def fix_emnist_orientation(arr):
    arr = cv2.transpose(arr)
    arr = cv2.flip(arr, flipCode=1)
    arr = cv2.bitwise_not(arr)  # EMNIST: fondo negro, trazo blanco → invertir
    return arr
```

**Fase B — Accent Augmentation** *(técnica clave para caracteres acentuados)*

Las letras acentuadas (`á`, `é`, `ñ`, etc.) tienen pocos datos reales. Para compensarlo, se generan 400 imágenes por clase acentuada superponiendo el diacrítico correspondiente (tilde, diéresis, virgulilla) sobre imágenes reales EMNIST de la letra base. Esto produce ejemplos realistas y variados sin necesidad de datos etiquetados adicionales.

```python
ACCENT_MAP = {
    'a': [('á', 'acute')],
    'e': [('é', 'acute')],
    'n': [('ñ', 'tilde')],
    # ... etc.
}
ACCENT_SAMPLES_PER_BASE = 400  # imágenes por clase acentuada
```

**Fase C — Datos sintéticos**

Para las 31 clases que no tienen datos reales suficientes (signos de puntuación, trazos básicos, dígrafos), se generan imágenes sintéticas usando fuentes manuscritas y variaciones morfológicas que simulan grosor de trazo.

**Pesos de muestreo (WeightedRandomSampler)**

Para balancear la distribución durante el entrenamiento:

| Fuente | Peso |
|--------|------|
| verack (datos verificados) | 1.5× |
| Accent augmentation | 1.3× |
| Sintéticos | 0.7× |

#### Proceso de entrenamiento

El entrenamiento usa un esquema de dos fases:

1. **Freeze phase (primeras 5 épocas):** El backbone EfficientNet está congelado. Solo se actualiza el projection head y la capa ArcFace. Esto estabiliza el espacio de embeddings antes de hacer fine-tuning.

2. **Unfreeze phase (épocas 6–50):** El backbone se descongela con learning rates diferenciados por profundidad de capa (capas tempranas × 0.1, capas medias × 0.3, capas tardías × 1.0). Se aplica warmup de 3 épocas y CosineAnnealingLR.

**Función de pérdida:**

```
FocalLoss(gamma=2.0) + class_weights (effective number of samples) + accent_boost(1.5×)
```

FocalLoss reduce el peso de ejemplos fáciles y concentra el entrenamiento en los difíciles, lo que es útil cuando hay clases con muchos ejemplos sintéticos fáciles.

**Mixup augmentation** (alpha=0.2) se aplica durante el entrenamiento para mejorar la generalización.

**TTA (Test-Time Augmentation)** (5 transformaciones) durante la evaluación final promedia predicciones sobre versiones augmentadas de cada imagen de test.

#### Evaluación honesta

Se distinguen tres tipos de clases según la disponibilidad de datos reales:

| Tipo | Clases | Test accuracy |
|------|--------|---------------|
| Datos reales | 62 | **79.34%** ← predictor de producción |
| Accent augmented | 14 | 85.12% |
| Solo sintéticas | 31 | 97.62% (inflado) |

> **Importante:** La métrica relevante para producción es `real_test_acc = 79.34%`. La accuracy global (80.97%) incluye clases sintéticas con accuracy artificialmente alta.

#### Clases soportadas (107 total)

| Categoría | Ejemplos | Cantidad |
|-----------|----------|----------|
| Letras mayúsculas | A–Z, Ñ | 27 |
| Letras minúsculas | a–z, ñ | 27 |
| Letras acentuadas | á é í ó ú ü Á É Í Ó Ú Ü | 12 |
| Dígitos | 0–9 | 10 |
| Dígrafos | ch, ll, CH, LL | 4 |
| Puntuación | . , ; : ¿ ? ¡ ! ( ) … | 13 |
| Trazos básicos | línea vertical, horizontal, curva, círculo, oblicuas | 14 |

#### Exportación

```python
# Exportar a ONNX con external data (modelo grande)
torch.onnx.export(
    model,
    dummy_input,
    str(BEST_ONNX_PATH),
    export_params=True,
    opset_version=17,
)
```

El archivo `.onnx.data` es parte del modelo exportado cuando los pesos superan el límite de 2 GB de ProtoBuf. Ambos archivos (`best_classifier.onnx` y `best_classifier.onnx.data`) deben estar juntos en el mismo directorio.

#### Salidas del notebook

| Archivo | Descripción |
|---------|-------------|
| `best_classifier.pt` | Pesos PyTorch del clasificador |
| `best_classifier.onnx` + `.onnx.data` | Modelo ONNX para producción |
| `metrics_report.json` | Métricas completas por clase |
| `confusion_matrix.png` | Matriz de confusión (107×107) |
| `training_curves.png` | Curvas de entrenamiento |
| `top10_confused_pairs.json` | Los 10 pares de clases más confundidos |
| `char_map.json` | Mapeo índice↔carácter (requerido en producción) |

> **Dónde colocarlos:** Los archivos `.onnx` y `.onnx.data` van a `app/models/classifier_artifacts/`. El `char_map.json` debe ser accesible desde `app/core/classifier.py`.

---

## 10. Modelo Clasificador — Especificaciones

| Atributo | Valor |
|----------|-------|
| Arquitectura | EfficientNetV2-S + Projection Head (1280→512) + ArcFace |
| Versión | v5 (run_id: 20260414_174147) |
| Número de clases | 107 |
| Tamaño de entrada | 128×128 (letterbox resize) |
| Formato de inferencia | ONNX |

### Dataset

| Tipo | Cantidad |
|------|----------|
| Imágenes de entrenamiento | 99,354 |
| Imágenes de validación | 16,005 |
| Imágenes de test | 16,005 |
| Clases con datos reales | 62 |
| Clases con accent augmentation | 14 |
| Clases solo sintéticas | 31 |

### Métricas

| Métrica | Valor |
|---------|-------|
| Best validation accuracy | 81.26% |
| Test accuracy (global) | 80.97% |
| Weighted F1-Score | 0.8093 |
| Test accuracy (datos reales) | **79.34%** |
| Test accuracy (acentuadas augmentadas) | 85.12% |
| Test accuracy (solo sintéticas) | 97.62% |

---

## 11. Métricas de Evaluación de Trazo

El sistema evalúa la calidad del trazo en 8 dimensiones independientes combinadas con pesos configurables.

### Componentes y pesos

| Métrica | Peso | Qué mide |
|---------|------|----------|
| `dt_precision` | 0.30 | Cercanía del trazo del alumno al trazo ideal |
| `dt_coverage` | 0.20 | Porcentaje de la plantilla cubierta |
| `topology` | 0.20 | Estructura correcta (loops, endpoints, junctions) |
| `ssim` | 0.12 | Similitud estructural global de la imagen |
| `procrustes` | 0.10 | Similitud de forma tras alineamiento óptimo |
| `hausdorff` | 0.04 | Distancia máxima entre contornos |
| `trajectory` | 0.02 | Similitud de trayectoria espacial (DTW) |
| `cosine` | 0.02 | Similitud angular entre segmentos |

### Tolerancias Distance Transform por nivel

| Nivel | Tolerancia (píxeles) | Descripción |
|-------|---------------------|-------------|
| `principiante` | 8.0 | Muy tolerante, acepta trazos aproximados |
| `intermedio` | 5.0 | Balance entre forma y precisión |
| `avanzado` | 3.0 | Estricto, requiere alta precisión |

### Descripción de cada métrica

**dt_precision:** Calcula el distance transform de la plantilla y evalúa los valores en los puntos del esqueleto del alumno. Valores bajos = trazo cerca del ideal.

**dt_coverage:** Porcentaje de puntos de la plantilla que tienen trazo del alumno dentro del radio de tolerancia según nivel.

**topology:** Esqueletiza ambos trazos y cuenta features topológicos (loops, endpoints, junctions). Binario: 100 puntos si coincide, 30 si no.

**ssim:** Structural Similarity Index entre la máscara del alumno y la plantilla. Mide similitud perceptual global.

**procrustes:** Alinea óptimamente ambos esqueletos (traslación, rotación, escala) y mide la distancia residual.

**hausdorff:** Distancia máxima entre los contornos de alumno y plantilla. Penaliza desviaciones grandes puntuales.

**trajectory:** Dynamic Time Warping (DTW) entre los esqueletos ordenados espacialmente. Mide similitud de secuencia de puntos.

**cosine:** Divide ambos esqueletos en N segmentos, calcula vectores dirección y compara con similitud coseno.

---

## 12. Pipeline de Procesamiento de Imagen

### Desafíos de fotos reales de cuaderno

| Desafío | Solución |
|---------|----------|
| Líneas azules del cuaderno (renglones) | Detección HSV + inpainting |
| Sombras e iluminación variable | Normalización de iluminación |
| Bajo contraste del lápiz | CLAHE + normalización de percentiles |
| Textura del papel (fibras, manchas) | Morfología (apertura/cierre) |
| Inclinación del cuaderno (±15°) | Deskew automático |

### image_cleaner.py

```
Imagen BGR original
  → Detección de líneas azules (HSV: H=85-130, S=40-255, V=50-255)
  → Dilatación de máscara (cubrir bordes difusos)
  → Inpainting (cv2.INPAINT_TELEA)
  → Conversión a grayscale
  → CLAHE (contraste adaptativo)
  → Normalización de fondo (~245) y trazo (~0-80)
```

### normalizer.py v6

```
ROI BGR (del image_cleaner)
  → ¿Es foto real o imagen digital?

DIGITAL:
  grayscale → binarizar → crop + center

FOTO REAL:
  1. image_cleaner elimina líneas azules
  2. Binarización por percentiles
     threshold = fg + (bg - fg) * 0.45
  3. Morfología ligera (close + open)
  4. Validación de máscara
     Si falla → Fallback legacy (HSV + Otsu)
     Si falla → Fallback emergency (blur fuerte + Otsu)

Post-procesamiento:
  → remove_specks → clean_noise → fill_internal_gaps
  → deskew → crop_and_center (128×128)
```

### Doble red de seguridad

```
Capa 1: normalizer v6
  ├─ Pipeline image_cleaner (principal)
  ├─ Fallback: pipeline legacy (HSV + Otsu)
  └─ Fallback: pipeline emergency (blur + Otsu agresivo)

Capa 2: processor v4.2
  └─ _is_mask_garbage() — 4 heurísticas de validación:
       • Muy pocos píxeles activos? (< 0.5%)
       • Demasiados píxeles? (> 60%)
       • Fragmentación alta? (> 15 componentes)
       • Densidad de hull baja? (< 0.05)
     Si basura → _emergency_mask_from_clean_gray()
```

---

## 13. Cliente de Escritorio Kivy

El proyecto incluye un cliente de escritorio para uso directo por el alumno, construido con Kivy 2.3.0.

### Instalación

```bash
cd kivy_app
pip install -r requirements.txt
python main.py
```

### Dependencias del cliente

```
kivy==2.3.0
requests>=2.31.0
Pillow>=10.0.0
plyer>=2.1.0
```

El cliente se comunica con el backend FastAPI vía HTTP. El servidor debe estar corriendo en `localhost:8000`.

---

## 14. Supuestos y Limitaciones

### Supuestos

- **Entrada:** Fotos tomadas con celular de cuadernos de escritura infantil.
- **Iluminación:** Ambiente interior con iluminación razonable.
- **Orientación:** Cuaderno aproximadamente horizontal (se corrige hasta ±15°).
- **Instrumento:** Lápiz grafito o pluma de tinta negra/azul oscuro sobre papel blanco o cuadriculado.
- **Plantillas:** Generadas previamente con `generate_templates.py`.
- **Un carácter por foto** en `/evaluate`, múltiples en `/evaluate_plana`.

### Limitaciones conocidas

- No soporta escritura cursiva continua; solo letras separadas.
- Sensible a oclusión: dedos o sombras sobre el carácter afectan la detección.
- 107 clases soportadas; caracteres fuera del set no son reconocibles.
- YOLO necesita separación entre caracteres; caracteres muy juntos pueden fusionarse.
- Trazos muy tenues (lápiz H, 2H) pueden no generar suficiente contraste.
- Sin plantilla `.npy` para un carácter, la evaluación no es posible.
- Evalúa forma geométrica, no legibilidad semántica.
- Latencia: 200–500ms por carácter en CPU, 50–100ms con GPU.
- Sin persistencia en base de datos; cada evaluación es independiente.
- Test accuracy en producción: ~79.34% en datos reales (~20% de error en clasificación).
- Sin Docker ni contenedores.

---

## 15. Troubleshooting

### Error: "Nivel inválido"

**Causa:** Se envió `basico` en lugar de `principiante`.
**Solución:** Usar uno de los tres niveles válidos: `principiante`, `intermedio`, `avanzado`.

### Error: "No existe plantilla..."

**Causa:** Plantillas no generadas.
**Solución:**

```bash
python -m app.scripts.generate_templates
```

### API levanta pero no detecta ni clasifica

Verificar:
- Rutas de artefactos ONNX en `app/core/config.py`.
- Presencia física de `best_classifier.onnx` y `best_classifier.onnx.data` en `app/models/classifier_artifacts/`.
- Que ambos archivos `.onnx` y `.onnx.data` estén en el mismo directorio.
- Calidad de imagen de entrada (enfoque, contraste, oclusión).

### `/evaluate_plana` con detecciones inconsistentes

El endpoint tiene fallback automático a YOLO directo. Verificar:
- Que los caracteres estén bien separados en la imagen.
- Iluminación sin sombras duras.
- Resolución suficiente.

### Máscara de trazo incorrecta (score = 0 en todas las métricas)

El módulo `processor.py` tiene heurísticas para detectar máscaras inválidas. Si falla:
- Verificar que la imagen tenga suficiente contraste (lápiz oscuro, fondo claro).
- Verificar que no haya objetos que tapen el carácter.
- Revisar logs del servidor por mensajes `mask_garbage=True`.

---

## 16. Reentrenamiento de Modelos

Si necesitas reentrenar los modelos (por ejemplo, para agregar nuevas clases o mejorar accuracy):

### Reentrenar el detector

1. Abre el notebook `spanish_char_detector.ipynb` en Kaggle.
2. Asegúrate de tener los datasets requeridos montados (ver [Sección 9.1](#91-detector-yolov8-spanish_char_detectoripynb)).
3. Ejecuta todas las celdas en orden (D-0 → D-6).
4. Descarga `best_detector.pt` y `best_detector.onnx` del output.
5. Copia ambos archivos a `app/models/classifier_artifacts/`.

### Reentrenar el clasificador

1. Abre el notebook `clasificador-ocr-spanish.ipynb` en Kaggle.
2. Asegúrate de tener todos los datasets requeridos montados (ver [Sección 9.2](#92-clasificador-ocr-clasificador-ocr-spanishipynb)).
3. Si agregas nuevas clases, actualiza `char_map.json` primero.
4. Ejecuta todas las celdas en orden (Cell 0 → Cell 24).
5. Descarga el ZIP generado en Cell 24 (`classifier_v5_<run_id>.zip`).
6. Extrae `best_classifier.onnx`, `best_classifier.onnx.data`, y `char_map.json`.
7. Copia a `app/models/classifier_artifacts/`.
8. Regenera las plantillas:

```bash
python -m app.scripts.generate_templates
```

### Notas importantes para el reentrenamiento

- El clasificador usa `char_map.json` para mapear índices a caracteres. Si agregas clases, este archivo debe actualizarse antes de entrenar.
- El notebook incluye un timer de 110 minutos (`MAX_TRAINING_MINUTES`) para que quepa dentro de los límites de sesión de Kaggle. Ajusta si necesitas más tiempo.
- Las semillas están fijadas (`SEED=42`) para reproducibilidad, pero los resultados pueden variar ligeramente entre ejecuciones en GPU por operaciones no deterministas de CUDA.
- El test accuracy honesto (`real_test_acc`) es el indicador correcto de rendimiento en producción. Ignora `synth_test_acc` para esta evaluación.

---

## 17. Trabajo Futuro

- **Escritura cursiva:** Modelo de segmentación semántica para separar letras conectadas.
- **Base de datos:** Persistencia de evaluaciones para tracking de progreso del alumno.
- **Dashboard web:** Interfaz React/Vue además del cliente Kivy.
- **Dockerización:** Contenedores para deployment reproducible.
- **App móvil nativa:** Android/iOS con cámara integrada.
- **Entrenamiento continuo:** Fine-tuning con datos de usuarios reales.
- **Gamificación:** Niveles, badges y recompensas.
- **Mejora del clasificador:** Más datos reales, semi-supervised learning, destilación de conocimiento.
- **API pública:** Autenticación, rate limiting, versionado de modelos.

---

## 18. Checklist de Entrega (Handoff)

Antes de entregar el proyecto a otra persona, valida:

- [ ] `uvicorn app.main:app --reload` inicia sin errores
- [ ] `/docs` visible y funcional en `http://localhost:8000/docs`
- [ ] `/evaluate` responde correctamente con imagen de prueba
- [ ] `/evaluate_plana` responde con `results` y `smart_ocr`
- [ ] `/recognize` devuelve `text` y `characters`
- [ ] Plantillas generadas en `app/templates/`
- [ ] `best_classifier.onnx` y `best_classifier.onnx.data` presentes y en el mismo directorio
- [ ] `best_detector.pt` presente en `app/models/classifier_artifacts/`
- [ ] Dependencias instalables limpiamente desde `requirements.txt`
- [ ] `char_map.json` accesible para el clasificador

**Documentación adicional recomendada:**

- Manual de pruebas con casos reales y criterios de aceptación.
- Guía de reentrenamiento reproducible (dataset, seeds, comandos, export ONNX).
- Versionado de artefactos de modelo (checksum MD5/SHA256 + fecha + run_id de origen).

---

## 19. Enlaces

| Recurso | URL |
|---------|-----|
| Repositorio backend | https://github.com/DanielPPerez/Tutor_API |
| Notebook detector (Kaggle) | https://www.kaggle.com/code/danielperegrinoperez/detector-train |
| Notebook clasificador (Kaggle) | https://www.kaggle.com/code/danielperegrinoperez/clasificador-ocr-spanish |

---

## 20. Autor

**Daniel Peregrino Pérez**

Proyecto desarrollado como trabajo de estadías profesionales.
