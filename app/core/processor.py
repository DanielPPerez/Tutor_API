"""
app/core/processor.py
=====================
Responsabilidad única (SRP): orquestar la detección YOLO + clasificación
MobileNet sobre una imagen cruda de libreta.

ESTE ARCHIVO ES EL PUENTE entre los modelos ONNX y el pipeline de evaluación.
Es donde vive toda la lógica de inferencia. evaluate.py solo lo llama.

Responde a las 4 preguntas del diagnóstico:
--------------------------------------------

PROBLEMA 1 — Falsos positivos del detector (cajas rojas en el espiral,
             el margen, sombras):
  CAUSA: DETECTION_THRESHOLD=0.45 en config.py es demasiado bajo para un
  modelo recién entrenado. Además, NMS_THRESHOLD=0.45 deja pasar cajas
  duplicadas superpuestas.
  SOLUCIÓN: Subir a 0.55 en inferencia. Agregar filtro post-NMS que descarta
  cajas demasiado pequeñas (ruido) o demasiado grandes (toda la página).
  Filtrar por aspect ratio: una letra tiene AR entre 0.3 y 3.0.

PROBLEMA 2 — Clasificador no acierta la letra:
  CAUSA A: CLASS_NAMES en config.py tiene el orden INCORRECTO.
           El orden actual es A-Z a-z 0-9 (alfabético).
           EMNIST byclass tiene el orden 0-9 A-Z a-z (numérico primero).
           Cuando el modelo predice índice 10, config dice "K" pero EMNIST
           dice "A". Todo está desplazado.
  CAUSA B: El class_map.json generado en Kaggle usa los índices de EMNIST
           (correcto) pero processor.py nunca lo carga — usa CLASS_NAMES
           de config.py (incorrecto). El class_map.json estaba huérfano.
  SOLUCIÓN: Cargar class_map.json al inicializar el processor. Si no existe,
            construir el mapeo correcto de EMNIST en memoria. CLASS_NAMES
            en config.py queda como fallback de último recurso.

PROBLEMA 3 — class_map.json nunca se cargaba:
  CAUSA: No había ningún import de class_map.json en app/core/.
         El archivo existía en app/models/weights/ pero nadie lo leía.
  SOLUCIÓN: _load_class_map() busca class_map.json junto al modelo ONNX.
            Si lo encuentra, lo usa. Si no, usa EMNIST_CLASS_ORDER como
            fallback hardcoded (orden correcto de EMNIST byclass).

PROBLEMA 4 — Reemplazar modelos sin actualizar config:
  No es necesario cambiar config.py. El processor carga el modelo desde
  config.MOBILENET_MODEL_PATH y busca class_map.json en la misma carpeta.
  Solo tienes que poner los archivos en app/models/weights/ y listo.

Exports públicos
----------------
  preprocess_robust(img_bytes) -> tuple  (usado por evaluate.py)

Devuelve:
  img_a        : np.ndarray uint8 {0,255} 128×128 — imagen normalizada
  metadata     : dict
  detected_char: str  — letra detectada por el clasificador
  confidence   : float — confianza del clasificador [0-1]
  raw_crop_bgr : np.ndarray | None — recorte BGR original de YOLO
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.core import config
from app.core.normalizer import normalize_character


# =============================================================================
# Orden CORRECTO de clases EMNIST byclass
# (0-9 primero, luego A-Z, luego a-z)
# NOTA: Este es el orden que usa torchvision.datasets.EMNIST split='byclass'
# =============================================================================
EMNIST_CLASS_ORDER = list(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


# =============================================================================
# Carga de modelos (singleton — se cargan una sola vez al importar)
# =============================================================================

def _build_session(path: str) -> ort.InferenceSession | None:
    """Carga una sesión ONNX Runtime. Devuelve None si el archivo no existe."""
    if not os.path.exists(path):
        print(f"[processor] WARN Modelo no encontrado: {path}")
        return None
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(path, providers=providers)
        print(f"[processor] OK Modelo cargado: {Path(path).name}")
        return sess
    except Exception as e:
        print(f"[processor] ERROR Error cargando {path}: {e}")
        return None


def _load_class_map() -> dict[int, str]:
    """
    Carga el mapeo índice → clase para el clasificador.

    Orden de búsqueda:
      1. class_map.json en la misma carpeta que mobilenet_classifier.onnx
      2. EMNIST_CLASS_ORDER hardcoded (0-9, A-Z, a-z)
         → Este es el fallback correcto si no hay JSON.

    POR QUÉ NO SE USA config.CLASS_NAMES:
      config.CLASS_NAMES tiene el orden A-Z a-z 0-9 (alfabético).
      EMNIST byclass tiene el orden 0-9 A-Z a-z (numérico primero).
      Si usas config.CLASS_NAMES con un modelo entrenado en EMNIST byclass,
      todos los índices están desplazados y la clasificación es incorrecta.
    """
    def _parse_map_json(raw: object) -> dict[int, str] | None:
        """
        Acepta varios formatos:
          - {"0": "a", "1": "b", ...}
          - {"idx2char": {"0": "a", ...}, ...}
          - {"idx_to_char": {...}}
        """
        if not isinstance(raw, dict):
            return None

        # Formato "envuelto"
        for key in ("idx2char", "idx_to_char", "class_map"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break

        if not isinstance(raw, dict):
            return None

        out: dict[int, str] = {}
        for k, v in raw.items():
            try:
                idx = int(k)
            except Exception:
                continue
            out[idx] = str(v)
        return out or None

    # Buscar mapa de clases:
    # 1) config.CLASS_MAP_PATH (tu proyecto usa char_map.json)
    # 2) junto al modelo (class_map.json o char_map.json)
    candidates: list[Path] = []
    try:
        candidates.append(Path(config.CLASS_MAP_PATH))
    except Exception:
        pass
    model_dir = Path(config.MOBILENET_MODEL_PATH).parent
    candidates.extend([model_dir / "class_map.json", model_dir / "char_map.json"])

    for p in candidates:
        if not p.exists():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            parsed = _parse_map_json(raw)
            if parsed is not None:
                print(f"[processor] OK mapa de clases cargado: {p.name} ({len(parsed)} clases)")
                return parsed
            print(f"[processor] WARN {p.name} no tiene un formato reconocido. Ignorando.")
        except Exception as e:
            print(f"[processor] WARN Error leyendo {p.name}: {e}. Probando siguiente candidato.")

    # Fallback: orden correcto de EMNIST byclass
    print("[processor] INFO Usando EMNIST_CLASS_ORDER como mapa de clases (fallback)")
    return {i: c for i, c in enumerate(EMNIST_CLASS_ORDER)}


# Inicialización global (una sola vez por proceso)
_yolo_session       = _build_session(config.YOLO_MODEL_PATH)
_classifier_session = _build_session(config.MOBILENET_MODEL_PATH)
_class_map          = _load_class_map()


# =============================================================================
# Detección YOLO
# =============================================================================

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU entre 2 cajas xyxy."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _nms_xyxy(
    boxes: list[tuple[int, int, int, int, float]],
    iou_threshold: float,
    max_detections: int | None = None,
) -> list[tuple[int, int, int, int, float]]:
    """
    Non-Max Suppression (NMS) sobre cajas xyxy con confianza.
    Devuelve cajas ordenadas por confianza descendente.
    """
    if not boxes:
        return []

    arr = np.array([[x1, y1, x2, y2, conf] for (x1, y1, x2, y2, conf) in boxes], dtype=np.float32)
    order = np.argsort(-arr[:, 4])
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if max_detections is not None and len(keep) >= max_detections:
            break
        rest = order[1:]
        if rest.size == 0:
            break
        ious = []
        a = arr[i, :4]
        for j in rest:
            ious.append(_iou_xyxy(a, arr[int(j), :4]))
        ious = np.array(ious, dtype=np.float32)
        order = rest[ious <= iou_threshold]

    kept = arr[keep]
    return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])) for b in kept]


def _detect_yolo(img_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """
    Ejecuta el detector YOLO sobre la imagen y devuelve las cajas filtradas.

    Returns
    -------
    Lista de (x1, y1, x2, y2, confidence) ordenadas por confianza descendente.

    Filtros post-NMS aplicados:
      - conf >= DETECTION_THRESHOLD (0.55 recomendado, ajustable en config)
      - Área mínima: 0.05% del área total (descarta puntos de ruido)
      - Área máxima: 60% del área total (descarta cajas de toda la página)
      - Aspect ratio entre 0.25 y 4.0 (letras no son extremadamente alargadas)
    """
    if _yolo_session is None:
        return []

    H, W   = img_bgr.shape[:2]
    img_sz = config.YOLO_INPUT_SIZE  # 640

    # Preprocesar para YOLO: BGR → RGB, resize, normalizar [0,1], NCHW
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_sz, img_sz))
    tensor  = resized.astype(np.float32) / 255.0
    tensor  = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]  # (1,3,640,640)

    input_name = _yolo_session.get_inputs()[0].name
    outputs    = _yolo_session.run(None, {input_name: tensor})

    # YOLOv8 ONNX output shape: (1, 5, num_boxes) — formato xywh_norm + conf
    # o (1, num_boxes, 5) dependiendo del opset. Normalizar:
    preds = outputs[0]
    if preds.ndim == 3 and preds.shape[1] == 5:
        preds = preds[0].T  # (num_boxes, 5)
    elif preds.ndim == 3:
        preds = preds[0]    # (num_boxes, 5+)
    else:
        preds = preds

    # Escala de vuelta a píxeles de la imagen original
    scale_x = W / img_sz
    scale_y = H / img_sz
    img_area = H * W

    boxes: list[tuple[int, int, int, int, float]] = []
    for det in preds:
        # det = [cx, cy, w, h, conf] (coordenadas normalizadas [0,1])
        if len(det) < 5:
            continue
        cx, cy, bw, bh = det[0], det[1], det[2], det[3]
        # Compatibilidad: algunas exportaciones tienen objectness + class_conf
        conf = float(det[4]) if len(det) == 5 else float(det[4:].max())

        if conf < config.DETECTION_THRESHOLD:
            continue

        # Convertir a píxeles absolutos
        x1 = int((cx - bw / 2) * img_sz * scale_x)
        y1 = int((cy - bh / 2) * img_sz * scale_y)
        x2 = int((cx + bw / 2) * img_sz * scale_x)
        y2 = int((cy + bh / 2) * img_sz * scale_y)

        # Clampear a los bordes de la imagen
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))

        box_area = (x2 - x1) * (y2 - y1)
        box_w    = x2 - x1
        box_h    = y2 - y1
        ar       = box_w / max(box_h, 1)

        # ── Filtros post-NMS ──────────────────────────────────────────────
        # 1. Área mínima: elimina puntos de ruido
        if box_area < img_area * 0.0005:
            continue
        # 2. Área máxima: elimina cajas de toda la página o el espiral
        if box_area > img_area * 0.60:
            continue
        # 3. Aspect ratio: letras no son extremadamente alargadas
        if ar < 0.20 or ar > 5.0:
            continue

        boxes.append((x1, y1, x2, y2, conf))

    # Ordenar por confianza descendente
    boxes.sort(key=lambda b: b[4], reverse=True)
    # NMS para eliminar cajas duplicadas
    boxes = _nms_xyxy(boxes, iou_threshold=config.NMS_THRESHOLD, max_detections=50)
    return boxes


# =============================================================================
# Clasificación EfficientNet-B2
# =============================================================================

# Normalización ImageNet — misma que en el entrenamiento del notebook
_IMAGENET_MEAN    = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_IMAGENET_STD     = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
_IMAGENET_MEAN_1C = np.array([0.485], dtype=np.float32)[:, None, None]
_IMAGENET_STD_1C  = np.array([0.229], dtype=np.float32)[:, None, None]


def _get_model_input_shape() -> tuple[int, int, int]:
    """Lee (channels, height, width) del modelo ONNX en tiempo de ejecución."""
    meta  = _classifier_session.get_inputs()[0]
    shape = list(meta.shape) if meta.shape else []

    def _to_int(d):
        if isinstance(d, int): return d
        if isinstance(d, str): return int(d) if d.isdigit() else None
        return None

    c = _to_int(shape[1]) if len(shape) >= 2 else None
    h = _to_int(shape[2]) if len(shape) >= 3 else None
    w = _to_int(shape[3]) if len(shape) >= 4 else None
    return (c or 3, h or 64, w or 64)


def _prepare_tensor_from_raw_crop(
    raw_crop_bgr: np.ndarray,
    expected_c: int,
    expected_h: int,
    expected_w: int,
) -> np.ndarray:
    """
    Prepara tensor desde el crop BGR original de YOLO.

    DISTRIBUCIÓN IDENTICA AL ENTRENAMIENTO:
      Notebook: img_bgr[y1:y2, x1:x2] → BGR→RGB → resize(64,64) → /255 → ImageNet normalize
      Este path: raw_crop_bgr           → BGR→RGB → resize(H,W)  → /255 → ImageNet normalize

    El modelo ve fondo≈papel claro (+1.99σ) y trazo≈tinta oscura (-1.26σ),
    igual que durante el entrenamiento.
    """
    img_rgb = cv2.cvtColor(raw_crop_bgr, cv2.COLOR_BGR2RGB)
    interp  = cv2.INTER_AREA if (img_rgb.shape[0] > expected_h or img_rgb.shape[1] > expected_w)               else cv2.INTER_CUBIC
    img_rgb = cv2.resize(img_rgb, (expected_w, expected_h), interpolation=interp)

    img_f = img_rgb.astype(np.float32) / 255.0   # (H, W, 3)
    img_f = img_f.transpose(2, 0, 1)              # (3, H, W)

    if expected_c == 1:
        img_f = img_f[:1, ...]
        img_f = (img_f - _IMAGENET_MEAN_1C) / _IMAGENET_STD_1C
    else:
        img_f = (img_f - _IMAGENET_MEAN) / _IMAGENET_STD

    return img_f[np.newaxis, ...].astype(np.float32)  # (1, C, H, W)


def _prepare_tensor_from_binary_mask(
    img_normalized: np.ndarray,
    expected_c: int,
    expected_h: int,
    expected_w: int,
) -> np.ndarray | None:
    """
    Fallback: prepara tensor desde la máscara binaria cuando no hay raw_crop_bgr.

    PROBLEMA ORIGINAL (sin este fix):
      Entrenamiento: fondo=papel claro (~240px) → +1.99σ, trazo=tinta (~50px) → -1.26σ
      Inferencia:    fondo=negro binario (0px)   → -2.12σ, trazo=blanco (255px) → +2.25σ
      → Invertido completamente. Diferencia de 3.5-4.1σ → predicciones erráticas.

    CORRECCIÓN (este path):
      cv2.bitwise_not() invierte: trazo→0 (oscuro), fondo→255 (claro)
      → Diferencia residual < 0.9σ respecto al entrenamiento.
    """
    coords = cv2.findNonZero(img_normalized)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    margin = 4
    x1 = max(0, x - margin);  y1 = max(0, y - margin)
    x2 = min(img_normalized.shape[1], x + w + margin)
    y2 = min(img_normalized.shape[0], y + h + margin)

    # Invertir: trazo pasa de 255→0 (oscuro), fondo de 0→255 (claro)
    cropped_inv = cv2.bitwise_not(img_normalized[y1:y2, x1:x2])

    interp   = cv2.INTER_AREA if cropped_inv.shape[0] > expected_h else cv2.INTER_CUBIC
    img_crop = cv2.resize(cropped_inv, (expected_w, expected_h), interpolation=interp)

    img_f = img_crop.astype(np.float32) / 255.0

    if expected_c == 1:
        img_f = img_f[np.newaxis, ...]
        img_f = (img_f - _IMAGENET_MEAN_1C) / _IMAGENET_STD_1C
    else:
        img_f = np.stack([img_f, img_f, img_f], axis=0)  # (3, H, W)
        img_f = (img_f - _IMAGENET_MEAN) / _IMAGENET_STD

    return img_f[np.newaxis, ...].astype(np.float32)  # (1, C, H, W)


def _classify(
    img_normalized: np.ndarray,
    raw_crop_bgr:   np.ndarray | None = None,
) -> tuple[str, float]:
    """
    Clasifica un carácter y devuelve (letra, confianza).

    CAUSA RAIZ DEL PROBLEMA ORIGINAL:
    El modelo fue entrenado con crops BGR→RGB de imágenes reales (fondo=papel
    claro, trazo=tinta oscura).  La versión anterior recibía solo la máscara
    binaria del normalizer (trazo=255 BLANCO, fondo=0 NEGRO) — INVERTIDO —
    con una diferencia de 3.5-4.1 desviaciones estándar post-normalización.
    Eso explica las predicciones erráticas y la alta varianza de confianza.

    ESTRATEGIA (en orden de prioridad):
      1. raw_crop_bgr disponible → distribución IDENTICA al entrenamiento (óptimo).
      2. Solo máscara binaria    → invertir con bitwise_not, diferencia < 0.9σ (fallback).

    Parameters
    ----------
    img_normalized : np.ndarray  Máscara binaria uint8 (trazo=255, fondo=0). Fallback.
    raw_crop_bgr   : np.ndarray | None  Crop BGR de YOLO. Camino óptimo.
    """
    if _classifier_session is None:
        return "desconocido", 0.0

    expected_c, expected_h, expected_w = _get_model_input_shape()

    # ── Path 1: crop real BGR de YOLO (idéntico al entrenamiento) ────────────
    if raw_crop_bgr is not None and raw_crop_bgr.size > 0:
        tensor = _prepare_tensor_from_raw_crop(raw_crop_bgr, expected_c, expected_h, expected_w)

    # ── Path 2: fallback con máscara binaria invertida ────────────────────────
    else:
        tensor = _prepare_tensor_from_binary_mask(img_normalized, expected_c, expected_h, expected_w)
        if tensor is None:
            return "desconocido", 0.0

    # ── Inferencia ────────────────────────────────────────────────────────────
    input_name = _classifier_session.get_inputs()[0].name
    logits     = _classifier_session.run(None, {input_name: tensor})[0][0]  # (num_classes,)

    exp_l      = np.exp(logits - logits.max())
    probs      = exp_l / exp_l.sum()
    pred_idx   = int(np.argmax(probs))

    return _class_map.get(pred_idx, f"idx_{pred_idx}"), float(probs[pred_idx])
# =============================================================================
# Función principal — usada por evaluate.py
# =============================================================================

def preprocess_robust(
    img_bytes: bytes,
) -> tuple[np.ndarray, dict, str, float, np.ndarray | None]:
    """
    Pipeline completo: imagen cruda → carácter normalizado + clasificado.

    Pasos:
      1. Decodificar bytes → BGR
      2. Detectar con YOLO → mejor caja
      3. Extraer raw_crop_bgr (para mostrar al usuario)
      4. Normalizar con normalize_character()
      5. Clasificar con MobileNet

    Returns
    -------
    img_a        : np.ndarray uint8 {0,255} 128×128
    metadata     : dict (de normalize_character + detección)
    detected_char: str
    confidence   : float
    raw_crop_bgr : np.ndarray | None
    """
    # 1. Decodificar imagen
    nparr   = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("No se pudo decodificar la imagen. "
                         "Verifica que el archivo sea JPG/PNG válido.")

    # 2. Detección YOLO
    boxes = _detect_yolo(img_bgr)

    yolo_box    = None
    raw_crop_bgr = None

    if boxes:
        x1, y1, x2, y2, yolo_conf = boxes[0]  # La de mayor confianza
        yolo_box     = (x1, y1, x2, y2)
        raw_crop_bgr = img_bgr[y1:y2, x1:x2].copy()

    # 3. Normalizar carácter
    img_a, metadata = normalize_character(img_bgr, yolo_box=yolo_box)

    # Agregar info de detección al metadata
    metadata["yolo_detected"]   = yolo_box is not None
    metadata["yolo_confidence"] = float(boxes[0][4]) if boxes else 0.0
    metadata["n_detections"]    = len(boxes)

    # 4. Clasificar — pasar raw_crop_bgr para usar distribución idéntica al entrenamiento
    detected_char, confidence = _classify(img_a, raw_crop_bgr=raw_crop_bgr)
    metadata["classifier_confidence"] = confidence

    return img_a, metadata, detected_char, confidence, raw_crop_bgr


# =============================================================================
# Multi-caracter (detección múltiple + clasificación por caja)
# =============================================================================

def _sort_reading_order(
    boxes: list[tuple[int, int, int, int, float]],
    line_y_tol_ratio: float = 0.35,
) -> list[tuple[int, int, int, int, float]]:
    """
    Ordena cajas en orden de lectura: arriba→abajo y dentro de cada línea izq→der.
    """
    if not boxes:
        return []
    heights = np.array([(y2 - y1) for (_x1, y1, _x2, y2, _c) in boxes], dtype=np.float32)
    median_h = float(np.median(heights)) if len(heights) else 1.0
    y_tol = max(8.0, median_h * float(line_y_tol_ratio))

    # Agrupar por líneas usando el centro Y
    entries = []
    for x1, y1, x2, y2, conf in boxes:
        cy = (y1 + y2) / 2.0
        cx = (x1 + x2) / 2.0
        entries.append((x1, y1, x2, y2, conf, cx, cy))
    entries.sort(key=lambda e: e[6])  # por cy

    lines: list[list[tuple]] = []
    for e in entries:
        placed = False
        for line in lines:
            line_cy = float(np.mean([it[6] for it in line]))
            if abs(e[6] - line_cy) <= y_tol:
                line.append(e)
                placed = True
                break
        if not placed:
            lines.append([e])

    # Orden final
    out: list[tuple[int, int, int, int, float]] = []
    for line in lines:
        line.sort(key=lambda e: e[5])  # por cx
        for x1, y1, x2, y2, conf, _cx, _cy in line:
            out.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    return out


def preprocess_multi(
    img_bytes: bytes,
    max_boxes: int | None = None,
) -> list[tuple[np.ndarray, dict, str, float, np.ndarray]]:
    """
    Pipeline multi-carácter: detecta todas las cajas YOLO, normaliza y
    clasifica cada una.  Devuelve los resultados en orden de lectura
    (izquierda→derecha, arriba→abajo).

    Expected Interface (PLAN_IMPLEMENTACIONES_ESTADIA.md § 3.5)
    -----------------------------------------------------------
    Input  : img_bytes (bytes), max_boxes (int | None)
    Output : list[tuple] — cada elemento:
               (img_a, metadata, detected_char, confidence, raw_crop_bgr)
             · img_a        : np.ndarray uint8 128×128 — máscara binaria
             · metadata     : dict con info de normalización + detección
             · detected_char: str — clase predicha por el clasificador
             · confidence   : float — confianza del clasificador [0-1]
             · raw_crop_bgr : np.ndarray BGR — recorte original de YOLO
             Si no hay detecciones devuelve lista vacía.

    Parámetros intercambiados con evaluate_plana
    --------------------------------------------
    evaluate_plana pasa img_bytes; el primer elemento de la lista devuelta
    es la plantilla (template); los restantes se califican contra ella.
    El orden de lectura lo establece _sort_reading_order().
    """
    nparr   = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("No se pudo decodificar la imagen. Verifica JPG/PNG válido.")

    boxes = _detect_yolo(img_bgr)
    boxes = _sort_reading_order(boxes)

    # Limitar número de caracteres si se especifica
    if max_boxes is not None:
        boxes = boxes[: max(0, int(max_boxes))]

    n_total = len(boxes)
    results: list[tuple[np.ndarray, dict, str, float, np.ndarray]] = []

    for (x1, y1, x2, y2, conf) in boxes:
        yolo_box     = (x1, y1, x2, y2)
        raw_crop_bgr = img_bgr[y1:y2, x1:x2].copy()

        img_a, metadata = normalize_character(img_bgr, yolo_box=yolo_box)

        # Enriquecer metadata con info de detección
        metadata["yolo_detected"]          = True
        metadata["yolo_confidence"]        = float(conf)
        metadata["n_detections"]           = n_total
        metadata["bbox_xyxy"]             = [int(x1), int(y1), int(x2), int(y2)]

        detected_char, cls_conf = _classify(img_a, raw_crop_bgr=raw_crop_bgr)
        metadata["classifier_confidence"] = float(cls_conf)

        results.append((img_a, metadata, detected_char, float(cls_conf), raw_crop_bgr))

    return results