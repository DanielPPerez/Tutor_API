# Plan de implementaciones — Tutor inteligente (Estadía)

## 1. Resumen en lenguaje natural

El proyecto es un **tutor inteligente** que evalúa la caligrafía de alumnos: recibe una imagen de libreta, la normaliza, detecta el carácter o trazo con un modelo YOLO, lo clasifica con un OCR en español y lo compara contra una plantilla por nivel (principiante, intermedio, avanzado). Devuelve un JSON con la calificación según métricas (Distance Transform, topología, SSIM, Procrustes, Hausdorff, trayectoria, coseno) y una gráfica en base64 con la superposición del trazo del alumno sobre la guía. Los objetivos de negocio son extender el sistema con más datos y modelos (clasificador y detector entrenados con datasets estándar y aumentados), un modo de calificación por planas usando el primer carácter como plantilla, un pipeline de entrenamiento local reproducible con MLflow, y una aplicación Android (APK) con Kivy que consuma la API. Este documento detalla el plan para esas implementaciones y define la **interfaz esperada** del notebook y del código para pruebas y contratos claros.

---

## 2. Sección crítica: Interfaz esperada (Expected Interface)

Contrato que define cómo interactúan las pruebas y el resto del sistema con cada archivo, función, clase o endpoint. Cada elemento debe cumplir con los campos indicados.

---

### 2.1 API y endpoints

| Campo | Valor |
|-------|--------|
| **Path** | `app/api/endpoints.py` |
| **Name** | `evaluate` |
| **Type** | API Endpoint (FastAPI) |
| **Input** | `file: UploadFile` (imagen), `target_char: str` (Form), `level: str` (Form, default `"intermedio"`) — niveles: `principiante`, `intermedio`, `avanzado`. |
| **Output** | JSON: `target_char`, `detected_char`, `confidence`, `score_final`, `level`, `scores_breakdown`, `weights_used`, `feedback`, `metadata`, `metrics_extra`, `image_student_b64`, `template_b64`, `comparison_b64`. En error de detección: `error`, `target_char`, `detected_char`, `confidence`. |
| **Description** | Evalúa el trazo del alumno contra la plantilla del carácter indicado; orquesta detección YOLO, normalización, clasificación OCR, esqueletización, métricas y generación de imágenes base64. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/api/endpoints.py` |
| **Name** | `get_templates` |
| **Type** | Función |
| **Input** | `char: str`, `level: str`. |
| **Output** | `tuple[np.ndarray \| None, np.ndarray \| None]` — (carril del nivel, esqueleto 1px); ambos cacheados en memoria. |
| **Description** | Carga desde `app/templates/` los NPY del carácter y nivel; devuelve carril y esqueleto para comparación. |

---

### 2.2 Core: procesamiento y modelos

| Campo | Valor |
|-------|--------|
| **Path** | `app/core/processor.py` |
| **Name** | `preprocess_robust` |
| **Type** | Función |
| **Input** | `img_bytes: bytes` (imagen en bruto). |
| **Output** | `tuple` de 4 o 5 elementos: `(img_a, metadata, detected_char, confidence)` o `(img_a, metadata, detected_char, confidence, raw_crop_bgr)`. `img_a`: binaria 128×128 o None; `metadata`: dict; `raw_crop_bgr`: crop BGR de YOLO o None. |
| **Description** | Decodifica imagen, ejecuta YOLO, extrae crop, normaliza, clasifica con ONNX y devuelve máscara binaria, metadatos, carácter detectado y confianza. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/core/processor.py` |
| **Name** | `_detect_yolo` |
| **Type** | Función |
| **Input** | `img_bgr: np.ndarray` (BGR, cualquier tamaño). |
| **Output** | `list[tuple[int, int, int, int, float]]` — cajas en formato (x1, y1, x2, y2, confianza). |
| **Description** | Preprocesa a 640×640, ejecuta ONNX YOLO, postprocesa NMS y umbral; devuelve lista de detecciones. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/core/processor.py` |
| **Name** | `_classify` |
| **Type** | Función |
| **Input** | `img_normalized: np.ndarray` (binaria 128×128). |
| **Output** | `tuple[str, float]` — (carácter detectado, confianza). |
| **Description** | Prepara entrada para ONNX del clasificador, ejecuta inferencia y mapea logits a carácter con `char_map.json` / `class_map.json`. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/core/detector.py` |
| **Name** | `detect_character` |
| **Type** | Función |
| **Input** | `image_bgr` (imagen BGR). |
| **Output** | Formato definido por el módulo (cajas o resultado del detector). |
| **Description** | Wrapper o delegado del detector de caracteres/trazos usado por el pipeline. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/core/normalizer.py` |
| **Name** | `normalize_character` |
| **Type** | Función |
| **Input** | `image_bgr: np.ndarray`, `bbox_xyxy: tuple | None` (opcional, caja YOLO). |
| **Output** | `tuple[np.ndarray, dict]` — imagen binaria 128×128 y metadata (ángulo, escala, roi_refined, dimensiones, etc.). |
| **Description** | Pipeline de limpieza: CLAHE, binarización, eliminación líneas, morfología, deskew, calidad, remove_specks; centra en TARGET_SHAPE. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/core/config.py` |
| **Name** | (módulo) |
| **Type** | Módulo de configuración |
| **Input** | N/A (constantes y rutas). |
| **Output** | N/A. Define: `YOLO_MODEL_PATH`, `MOBILENET_MODEL_PATH`, `CLASS_MAP_PATH`, `TEMPLATE_OUTPUT_DIR`, `TARGET_SHAPE`, `SCORING_WEIGHTS`, `DT_TOLERANCE_BY_LEVEL`, etc. |
| **Description** | Punto único de parámetros del pipeline (detección, normalización, scoring, plantillas). |

---

### 2.3 Métricas y scoring

| Campo | Valor |
|-------|--------|
| **Path** | `app/metrics/scorer.py` |
| **Name** | `calculate_final_score` |
| **Type** | Función |
| **Input** | `geo_metrics`, `topo_match`, `traj_dist`, `dt_precision_score`, `dt_coverage`, `cosine_segment_score`, `level`. |
| **Output** | `dict` con `score_final`, `level`, `scores_breakdown`, `weights_used`. |
| **Description** | Combina métricas con pesos de config y tolerancia por nivel; devuelve nota y desglose. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/metrics/scorer.py` |
| **Name** | `get_feedback` |
| **Type** | Función |
| **Input** | `result: dict` (salida de `calculate_final_score`). |
| **Output** | `str` — texto pedagógico para el alumno. |
| **Description** | Genera mensaje de retroalimentación según score y nivel. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/metrics/distance_transform.py` |
| **Name** | `calculate_dt_fidelity` |
| **Type** | Función |
| **Input** | `skeleton_template`, `student_mass`, `level`. |
| **Output** | `tuple` — (score_precision, coverage, dist_map, heatmap_bgr). |
| **Description** | Calcula fidelidad del trazo del alumno respecto al carril de la plantilla (Distance Transform). |

| Campo | Valor |
|-------|--------|
| **Path** | `app/metrics/geometric.py` |
| **Name** | `calculate_geometric` |
| **Type** | Función |
| **Input** | `skel_p: np.ndarray`, `skel_a: np.ndarray`. |
| **Output** | `dict` con SSIM, Procrustes, Hausdorff u otras métricas geométricas. |
| **Description** | Comparación geométrica esqueleto plantilla vs esqueleto alumno. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/metrics/topologic.py` |
| **Name** | `get_topology` |
| **Type** | Función |
| **Input** | `skel: np.ndarray`. |
| **Output** | `dict` (p. ej. `loops`, conteo de agujeros). |
| **Description** | Extrae topología del esqueleto (bucles/agujeros) para comparación. |

---

### 2.4 Utilidades y visualización

| Campo | Valor |
|-------|--------|
| **Path** | `app/utils/visualizer.py` |
| **Name** | `generate_comparison_plot` |
| **Type** | Función |
| **Input** | `skel_p`, `skel_a`, `score`, `level`, `char`, `img_a` (opcional). |
| **Output** | `str` — imagen PNG codificada en base64. |
| **Description** | Genera overlay verde/rojo/amarillo (plantilla vs alumno), canvas 128×128 escalado a 512, devuelve base64. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/utils/visualizer.py` |
| **Name** | `build_raw_crop_image` |
| **Type** | Función |
| **Input** | `raw_crop_bgr: np.ndarray | None`, `img_a` (fallback). |
| **Output** | `np.ndarray` BGR para codificar a PNG/base64. |
| **Description** | Construye imagen de salida del crop del alumno (YOLO o normalizado). |

---

### 2.5 Modelos y datos

| Campo | Valor |
|-------|--------|
| **Path** | `app/models/char_map.json` |
| **Name** | (archivo) |
| **Type** | Recurso JSON |
| **Input** | N/A. |
| **Output** | Estructura: `idx2char`, `char2idx`, `num_classes` (101 clases: minúsculas, mayúsculas con ñ y tildes, dígitos, símbolos). |
| **Description** | Mapeo índice ↔ carácter para el clasificador OCR; debe ser la referencia única (unificar con `class_map.json` donde exista). |

---

## 3. Plan de implementaciones (detalle)

En todo el proyecto se usa un único nombre de archivo para la descarga de datos: **`dataset_downloads.py`** (en `app/scripts/`). No se utilizan otros nombres como `download_datasets.py`. Cada subsección es autocontenida e incluye la **Expected Interface** (Path, Name, Type, Input, Output, Description) de los archivos y funciones que se crearán, y los **parámetros que se intercambian** entre ellos.

---

### 3.0 Enfoque de la solución, flujo de entrenamiento y versiones

**Enfoque unificado:** El pipeline en producción es: **imagen completa → detector YOLO (bounding boxes) → recorte por caja → normalización → clasificador**. Por tanto, en entrenamiento debe reflejarse ese flujo: el **detector** se entrena con **imágenes completas** y etiquetas de cajas; el **clasificador** se entrena con **recortes ya normalizados** (cada recorte proviene de aplicar una caja a una imagen: ya sea caja ground-truth en datos o salida del detector sobre imágenes con anotación). Es decir: los modelos reciben imagen completa solo en el caso del detector; para el clasificador se generan primero las cajas (reales o simuladas), se recorta y se normaliza, y con ese crop normalizado se entrena.

**Estado actual del notebook:** El notebook `Notebook_OCR_español.ipynb` entrena hoy el **clasificador** sobre **imágenes ya recortadas por carácter** (rutas a archivos de un carácter por imagen), sin etapa “imagen completa → bbox → normalizar”. No entrena el detector. Por tanto hay que: (1) Añadir en el mismo notebook (o en el pipeline SOLID) el **entrenamiento del detector** sobre imágenes completas y anotaciones en formato YOLO. (2) Para el clasificador, alimentar el entrenamiento con **crops normalizados** obtenidos a partir de imagen completa + bbox (leyendo datasets que tengan bbox por carácter o generando crops desde imágenes de palabras con anotaciones); opcionalmente seguir usando también imágenes de carácter suelto ya recortadas, pero unificando el preprocesado (normalización) para que coincida con el que se aplica en inferencia tras la detección.

**YOLO elegido:** Se usa **solo YOLO** para detección. La variante es **YOLOv8 nano (YOLOv8n)** de **Ultralytics**: buen equilibrio tamaño/velocidad, exportación ONNX estable y adecuada para móviles y tabletas. Entrada estándar **640×640**; salida ONNX compatible con el postprocesado actual (formato xywh normalizado + confianza). La versión de librería a usar es la indicada en la tabla de versiones más abajo.

**Versiones de librerías y enfoque ante conflictos:** Se define un conjunto fijo de versiones para entorno reproducible y sin conflictos entre sí. Criterios: (1) Una sola versión de PyTorch y de ONNX/onnxruntime para todo el pipeline (entrenamiento + export + inferencia). (2) Ultralytics compatible con esa versión de PyTorch. (3) MLflow y dependencias de serialización compatibles con los checkpoints y artefactos que se guarden. (4) Resolución de conflictos: prioridad a las versiones listadas; si una dependencia transitoria exige una versión distinta, se documenta la excepción y se fija la versión transitoria en `requirements.txt` o en el archivo de entorno de entrenamiento. En la siguiente tabla se usan versiones conservadoras y probadas en 2025–2026:

| Paquete | Versión | Notas |
|---------|---------|--------|
| Python | 3.10 o 3.11 | Evitar 3.12+ por posibles incompatibilidades con ultralytics/torch. |
| torch | 2.1.x o 2.2.x | Ej.: 2.1.2 o 2.2.0. |
| torchvision | Misma minor que torch | Ej.: 0.16.2 con torch 2.1.2. |
| ultralytics | >=8.0.0, <9.0.0 | YOLOv8; export ONNX. |
| onnx | 1.14.x o 1.16.x | Compatible con opset 17. |
| onnxruntime | 1.16.x o 1.17.x | Inferencia ONNX CPU/GPU. |
| timm | 0.9.x | Backbone clasificador. |
| albumentations | 1.3.x | Augmentaciones. |
| mlflow | 2.9.x o 2.10.x | Registro de experimentos. |
| opencv-python-headless | 4.8.x o 4.9.x | Sin conflicto con GUI en servidor. |
| numpy | <2.0 o 1.26.x | Compatibilidad con PyTorch/ONNX. |

El archivo que concentra estas versiones en el proyecto es **`training/requirements-train.txt`** (o la ruta acordada dentro de la carpeta de entrenamiento). El enfoque ante un fallo de compatibilidad es: (1) Reproducir en un entorno aislado con solo esas versiones. (2) Si el error viene de una dependencia transitoria, fijar esa dependencia a una versión concreta en el archivo de requisitos. (3) Documentar en `docs/` cualquier excepción y la versión final usada.

---

### 3.1 Descarga de datasets

**Objetivo:** Un único script, **`dataset_downloads.py`**, descarga la versión más reciente disponible a 15/03/2026 de:  **EMNIST By Class**, **handwritting_characters_database** (p. ej. GitHub sueiras/handwritting_characters_database), **iam-handwriting-word-database** (p. ej. Kaggle nibinv23/iam-handwriting-word-database), **spanish-handwritten-characterswords** (p. ej. Kaggle verack/spanish-handwritten-characterswords). Todo se guarda bajo **`data/`** 

**Expected Interface — archivo y funciones:**

| Campo | Valor |
|-------|--------|
| **Path** | `app/scripts/dataset_downloads.py` |
| **Name** | `dataset_downloads` (módulo) |
| **Type** | Módulo / script ejecutable |
| **Input** | Ejecución: sin argumentos obligatorios; opcionalmente `--data-root` (ruta base, default `./data`). Variables de entorno: `KAGGLE_USERNAME`, `KAGGLE_KEY` si se usa API de Kaggle. |
| **Output** | Efecto lateral: crea y llena `data/raw/handwritting_characters_database`, `data/raw/iam_handwriting`, `data/raw/spanish_handwritten_characters_words` (o nombres estándar acordados). Retorno de la función principal: `dict` con claves por dataset y valor `{"path": str, "ok": bool, "message": str}`. |
| **Description** | Descarga y descomprime los tres datasets en `data/`; usa kagglehub o API Kaggle para los que vengan de Kaggle, y HTTP/Git para handwritting_characters_database; documenta en comentarios las URLs y la fecha 15/03/2026. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/scripts/dataset_downloads.py` |
| **Name** | `download_all` |
| **Type** | Función |
| **Input** | `data_root: str = "data"`. |
| **Output** | `dict[str, dict]`: por cada dataset una entrada con `path`, `ok`, `message`. |
| **Description** | Orquesta la descarga de los tres datasets hacia `data_root`; crea `data_root/raw` si no existe; devuelve el estado de cada uno. |

**Parámetros intercambiados:** `download_all` recibe `data_root` (string). No recibe ni devuelve datos de imágenes; solo rutas y estado. El resto del proyecto (verificación de clases, preparación YOLO, entrenamiento) lee desde `data/` usando las rutas documentadas en el script o en un pequeño `data/README.md` generado por el script.

---

### 3.2 Verificación de clases frente a `char_map.json`

**Objetivo:** Poder entrenar el clasificador para **todas** las clases de `app/models/char_map.json` (101 clases). Crea un **único archivo** que lee los README y/o la estructura de cada dataset en `data/`, extrae las clases que aporta cada uno, las cruza con las clases de `char_map.json` y escribe un reporte: qué clases cubre cada dataset y cuáles faltan.Deben de incluirse tambien las siguientes clases si no estan lineas rectas verticales, horizontales, oblicuas, curvas, circulares. Ese archivo es **`app/scripts/verify_dataset_classes.py`**; la salida se escribe en un JSON `data/dataset_classes_report.json` que el mismo script pueda consumir después.

**Expected Interface:**

| Campo | Valor |
|-------|--------|
| **Path** | `app/scripts/verify_dataset_classes.py` |
| **Name** | `verify_dataset_classes` (módulo) |
| **Type** | Módulo / script ejecutable |
| **Input** | Argumentos: `--data-root` (default `data`), `--char-map` (default `app/models/char_map.json`), `--out` (default `data/dataset_classes_report.json`). |
| **Output** | Efecto: escribe el reporte en `--out`. La función principal devuelve un `dict`: `{"char_map_classes": int, "by_dataset": {"nombre": {"classes": list, "covered": list, "missing": list}}, "global_missing": list}`. |
| **Description** | Lee `char_map.json` (idx2char/char2idx); para cada dataset en `data/raw/` inspecciona README y estructura de carpetas/nombres para inferir clases; cruza con las 101 clases y genera el reporte. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/scripts/verify_dataset_classes.py` |
| **Name** | `run_verification` |
| **Type** | Función |
| **Input** | `data_root: str`, `char_map_path: str`, `output_path: str`. |
| **Output** | `dict` con `char_map_classes`, `by_dataset`, `global_missing` como arriba. |
| **Description** | Ejecuta la lógica de verificación y escritura del reporte; usada por el script al ejecutarse desde CLI. |

**Parámetros intercambiados:** `run_verification` recibe rutas (strings); no recibe imágenes. Lee `char_map.json` (parámetro `char_map_path`) y las carpetas bajo `data_root`. La salida (`output_path`) la consumen humanos o scripts de preparación de datos para saber qué clases requieren datos sintéticos o más fuentes.

---

### 3.3 Detector de caracteres/trazos (un bbox por carácter)

**Objetivo:** Entrenar un modelo **YOLO** (YOLOv8n) que reciba **imagen completa** y devuelva **un bounding box por carácter o trazo**. Muchas imágenes en los datasets son palabras completas; el detector debe localizar cada carácter. Formato de anotaciones: **YOLO normalizado** (una línea por objeto: `class_id x_center y_center width height` en [0,1]). Los datos se preparan en `data/processed/yolo_dataset/` (o subcarpeta equivalente); se incluyen imágenes completas y labels. El modelo se entrena con imágenes completas; en inferencia: imagen completa → YOLO → cajas → por cada caja se recorta y se normaliza para el clasificador. Se usa **solo YOLO** (YOLOv8n, Ultralytics); exportación a ONNX 640×640 para móviles/tabletas.

**Expected Interface — archivos y funciones nuevos:**

| Campo | Valor |
|-------|--------|
| **Path** | `training/datasets/yolo_dataset.py` (o ruta acordada dentro de la carpeta SOLID de entrenamiento) |
| **Name** | `YOLOCharacterDataset` |
| **Type** | Clase (torch Dataset) |
| **Input** | Constructor: `root: str` (ruta a `data/processed/yolo_dataset`), `split: str` ("train" \| "val"), `img_size: int = 640`, `augment: bool`. |
| **Output** | `__getitem__(idx)` devuelve `(img_tensor: torch.Tensor, targets: dict)` donde `targets` contiene listas de cajas en formato interno del entrenamiento YOLO (normalizado). `img_tensor`: imagen completa de tamaño `img_size`. |
| **Description** | Dataset que carga imágenes completas y sus anotaciones YOLO; aplica augmentación si `augment=True`; devuelve tensores listos para el entrenamiento del detector. |

| Campo | Valor |
|-------|--------|
| **Path** | `training/train_detector.py` (o nombre acordado en la carpeta SOLID) |
| **Name** | `train_detector` |
| **Type** | Función |
| **Input** | `data_yaml: str` (ruta a `data/processed/yolo_dataset/dataset.yaml`), `epochs: int`, `batch: int`, `device: str`, `project: str`, `name: str`, `mlflow_tracking_uri: str \| None`. Hiperparámetros opcionales (lr, etc.) según interfaz acordada. |
| **Output** | `dict` con `best_weights_path`, `metrics` (mAP50, mAP50-95, loss, etc.). Registra en MLflow run, métricas por época e hiperparámetros; guarda mejor checkpoint y ONNX en carpeta de experimento. |
| **Description** | Entrena YOLOv8n con Ultralytics sobre el dataset definido en `data_yaml`; entrada = imágenes completas; exporta ONNX al final; registra todo en MLflow. |

**Parámetros intercambiados:** `YOLOCharacterDataset` recibe `root` que debe apuntar al mismo árbol que genera `dataset_downloads.py` + scripts de preparación (p. ej. `data/processed/yolo_dataset`). `train_detector` recibe `data_yaml`; ese YAML referencia `path`, `train`, `val`, `nc`, `names` (convención YOLO). El artefacto de salida (ONNX) se consume por `app/core/processor.py` (`YOLO_MODEL_PATH`) con la misma convención de entrada 640×640 y salida (xywh_norm, conf).

---

### 3.4 Uso de imágenes aumentadas y datos en `data/`

**Objetivo:** Tanto el **clasificador** como el **detector** se entrenan usando: (1) Datos crudos en `data/` (descargados con `dataset_downloads.py` y preparados en `data/processed/`), y (2) Conjunto de **imágenes aumentadas** existente en el proyecto. Todos los datasets viven bajo `data/` (raw, processed, augmented). Los pipelines de entrenamiento deben poder elegir fuentes: solo raw, solo aumentados, o ambos, para maximizar robustez.

**Expected Interface:**

| Campo | Valor |
|-------|--------|
| **Path** | `training/config.py` (dentro de la carpeta SOLID de entrenamiento) |
| **Name** | `DataSources` (o estructura equivalente) |
| **Type** | Clase o TypedDict de configuración |
| **Input** | Campos: `data_root: str`, `use_raw: bool`, `use_augmented: bool`, `use_synthetic_yolo: bool` (para detector). |
| **Output** | N/A (configuración). Las funciones de dataset leen esta config para construir listas de rutas y labels. |
| **Description** | Define qué carpetas de `data/` se usan en cada entrenamiento (clasificador vs detector) y si se mezclan raw y aumentados. |

**Parámetros intercambiados:** `DataSources`  es leído por los constructores de los datasets de clasificación y de detección. Esos datasets devuelven muestras (rutas o tensores) que los entrenadores consumen; no hay parámetros de imagen que crucen entre scripts más allá de las rutas definidas en `data/` y en el YAML del detector.

---

### 3.5 Modo calificación por planas (plantilla = primer carácter)

**Objetivo:** Ofrecer un modo donde el **primer carácter** detectado en la imagen (orden de lectura) se toma como **plantilla de referencia** y el resto de trazos/caracteres de la misma imagen se **califican contra esa referencia** con las mismas métricas (DT, topología, geométricas, etc.). El detector debe devolver **múltiples cajas** (una por carácter); la primera caja define la plantilla (se extrae crop, se normaliza, se esqueletiza); cada otra caja se normaliza y se compara contra ese esqueleto de referencia.

**Expected Interface:**

| Campo | Valor |
|-------|--------|
| **Path** | `app/api/endpoints.py` |
| **Name** | `evaluate_plana` |
| **Type** | API Endpoint (FastAPI) |
| **Input** | `file: UploadFile`, `level: str` (Form, opcional). No se envía `target_char`; el carácter de referencia lo define el primer carácter detectado. |
| **Output** | JSON: lista de resultados por carácter (desde el segundo en adelante), cada uno con `detected_char`, `confidence`, `score_final`, `scores_breakdown`, `feedback`, etc.; más un campo `template_char` (el carácter usado como plantilla) y las imágenes base64 que se acuerden (p. ej. comparación por carácter o resumen). |
| **Description** | Recibe imagen de plana; llama a `preprocess_multi` o equivalente que devuelve múltiples crops normalizados; el primero se usa como plantilla; para los demás se calcula score contra esa plantilla y se devuelve la lista de calificaciones. |

| Campo | Valor |
|-------|--------|
| **Path** | `app/core/processor.py` |
| **Name** | `preprocess_multi` |
| **Type** | Función |
| **Input** | `img_bytes: bytes`, `max_boxes: int \| None` (opcional). |
| **Output** | `list[tuple]`: cada elemento `(img_a, metadata, detected_char, confidence, raw_crop_bgr)` para un carácter; orden de lectura (izq–der, arriba–abajo). Si no hay detecciones, lista vacía. |
| **Description** | Decodifica imagen, ejecuta YOLO, obtiene todas las cajas ordenadas por lectura; por cada caja extrae crop, normaliza, clasifica; devuelve lista de resultados para modo plana. |

**Parámetros intercambiados:** `evaluate_plana` pasa `img_bytes` a `preprocess_multi`. `preprocess_multi` usa `_detect_yolo` (lista de cajas), luego por cada caja llama a `normalize_character(img_bgr, bbox_xyxy)` y `_classify(img_normalized)`; devuelve las tuplas que el endpoint agrupa para construir la respuesta (template = primer elemento; calificaciones = resto).

---

### 3.6 Notebook y pipeline SOLID con MLflow

**Objetivo:** El notebook **Notebook_OCR_español.ipynb** debe incluir también el **modelo de detección**: flujo de entrenamiento YOLOv8n sobre imágenes completas y exportación ONNX, además del clasificador. A partir del notebook se crea una **carpeta** (p. ej. **`training/`** en la raíz del repo) que sigue principios SOLID y contiene todo el pipeline de entrenamiento (clasificador + detector) en **modo local reproducible**, con **MLflow** para métricas, hiperparámetros y artefactos. El entrenamiento del clasificador debe alimentarse con **crops normalizados** (obtenidos a partir de imagen completa + bbox cuando los datos lo permitan), alineado con el flujo “imagen completa → bbox → normalizar → clasificar”.

**Expected Interface — carpeta `training/`:**

| Campo | Valor |
|-------|--------|
| **Path** | `training/` |
| **Name** | (carpeta de módulos) |
| **Type** | Paquete / conjunto de módulos |
| **Input** | N/A. Contiene: `config.py`, `datasets/` (clasificación y YOLO), `train_classifier.py`, `train_detector.py`, `run_experiment.py` (opcional), `requirements-train.txt`. |
| **Output** | Artefactos: checkpoints, ONNX (clasificador y detector), `char_map.json`; MLflow: experimentos, métricas por época, hiperparámetros. |
| **Description** | Pipeline de entrenamiento desacoplado de la app; ejecutable en cualquier máquina con las versiones fijadas en `requirements-train.txt`; registra en MLflow evolución de métricas y guarda hiperparámetros. |

| Campo | Valor |
|-------|--------|
| **Path** | `training/train_classifier.py` |
| **Name** | `train_classifier` |
| **Type** | Función |
| **Input** | `data_root: str`, `char_map_path: str`, `epochs: int`, `batch_size: int`, `img_size: int`, `device: str`, `mlflow_tracking_uri: str \| None`, y opcionales (lr, weight_decay, etc.). Opción de fuente: solo crops preexistentes o generación de crops desde imágenes completas + bbox. |
| **Output** | `dict` con `best_ckpt_path`, `char_map_path`, `metrics` (loss, accuracy top-1/top-5 por época). Registra en MLflow y guarda ONNX. |
| **Description** | Entrena el clasificador (EfficientNet-B2 o el acordado) sobre crops normalizados; los datos pueden venir de listas (path, label) o de un dataset que internamente genere crops desde imágenes completas y anotaciones; guarda checkpoint y ONNX. |

**Parámetros intercambiados:** `train_classifier` recibe `data_root` y `char_map_path`; lee `char_map.json` para `num_classes` y mapeos; los datasets internos reciben rutas que deben estar bajo `data/` (raw/processed/augmented). `train_detector` recibe `data_yaml` que apunta a `data/processed/yolo_dataset/dataset.yaml`. MLflow: ambos entrenamientos escriben en el mismo `tracking_uri` (o por defecto `./mlruns`); se registran `params`, `metrics` (por época) y artefactos (ONNX, checkpoint). Los ONNX y `char_map.json` generados se copian o referencian en `app/models/weights/` y `app/models/char_map.json` para que la API use los modelos actualizados.

---

### 3.7 APK Android con Kivy

**Objetivo:** Llevar el flujo a una **APK Android** con **Kivy**. La API se mantiene (FastAPI); la app móvil consume `POST /evaluate` y, si se implementa, el endpoint de modo plana. Se crea una **carpeta adicional fuera de `app/`**, por ejemplo **`kivy_app/`**, que contiene una UI simple: captura o selección de imagen, envío a la API (URL configurable), visualización del resultado (score, feedback, imágenes en base64).

**Expected Interface:**

| Campo | Valor |
|-------|--------|
| **Path** | `kivy_app/` |
| **Name** | (carpeta del proyecto Kivy) |
| **Type** | Proyecto Kivy (main.py, buildozer.spec, assets) |
| **Input** | Usuario: imagen (cámara o galería), carácter objetivo y nivel en pantalla. Config: URL base de la API (variable de entorno o archivo config). |
| **Output** | APK Android; en ejecución: pantalla de resultado con score, feedback e imágenes (student, template, comparison) decodificando los base64 de la respuesta. |
| **Description** | UI mínima en Kivy: pantalla de captura/selector, pantalla de resultado; módulo de red que llama a la API y parsea el JSON. |

| Campo | Valor |
|-------|--------|
| **Path** | `kivy_app/api_client.py` |
| **Name** | `evaluate_image` |
| **Type** | Función |
| **Input** | `image_path: str` (o bytes de imagen), `target_char: str`, `level: str`, `base_url: str`. |
| **Output** | `dict` con las claves del JSON de `POST /evaluate`: `score_final`, `feedback`, `image_student_b64`, `template_b64`, `comparison_b64`, etc.; en error, lanza o devuelve estructura de error acordada. |
| **Description** | Envía multipart/form-data a `base_url/evaluate` con `file`, `target_char`, `level`; devuelve el JSON de la respuesta. |

**Parámetros intercambiados:** `evaluate_image` recibe la imagen (ruta o bytes), `target_char`, `level` y `base_url`; devuelve el mismo esquema JSON que el endpoint. La UI lee de ese dict los campos `score_final`, `feedback` y las cadenas base64 para mostrarlas. No se pasan parámetros entre la app y la API más allá de ese contrato (form + JSON).

---

## 4. Estructura sugerida del notebook (solo MD)

El notebook debe contener al menos estas secciones documentales y de interfaz:

1. **Resumen en lenguaje natural**  
   Un párrafo introductorio con: contexto del tutor inteligente, uso de OCR y detector YOLO, objetivos de negocio (calificación de caligrafía, soporte español, edge/móvil).

2. **Interfaz esperada (Expected Interface)**  
   Para cada archivo, función o clase relevante generada o usada en el notebook:
   - **Path:** ruta exacta del archivo (o “celda del notebook” si aplica).
   - **Name:** nombre exacto de la función, clase o endpoint.
   - **Type:** tipo (función, clase, componente, API, recurso).
   - **Input:** parámetros y tipos aceptados.
   - **Output:** formato preciso de los datos devueltos.
   - **Description:** breve explicación del comportamiento.

Esto permite que las pruebas y la carpeta SOLID tengan un contrato claro con el código del notebook.

---

## 5. Resumen de entregables

| # | Entregable |
|---|------------|
| 1 | `app/scripts/dataset_downloads.py` con descarga de handwritting_characters_database, iam-handwriting-word-database, spanish-handwritten-characterswords (versión 15/03/2026), todo en `data/`. |
| 2 | `app/scripts/verify_dataset_classes.py` y salida `docs/dataset_classes_report.md` (o `data/dataset_classes_report.json`) cruzando clases de cada dataset con `char_map.json`. |
| 3 | Pipeline y modelo detector de caracteres/trazos con bbox por carácter; integración con imágenes de palabras. |
| 4 | Entrenamiento clasificador y detector usando también imágenes aumentadas; datos en `data/`. |
| 5 | Modo calificación por planas (primer carácter como plantilla, resto comparado contra él). |
| 6 | Notebook con detección incluida + sección “Interfaz esperada”; carpeta `training/` con pipeline SOLID local, reproducible, con MLflow. |
| 7 | Proyecto Kivy en carpeta `kivy_app/`: UI simple + `api_client.evaluate_image` + instrucciones para APK Android. |

Este documento sirve como **plan maestro** y **contrato de interfaz** para desarrollo y pruebas; el código debe respetar los Path, Name, Type, Input, Output y Description aquí definidos.
