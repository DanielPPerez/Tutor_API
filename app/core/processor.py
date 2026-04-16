"""
app/core/processor.py  (v4.2 — compatible con normalizer v6)
==========================================
Orquesta detección YOLO + limpieza + clasificación.

CAMBIOS v4.2 vs v4.1:
  - COMPATIBLE con normalizer v6 (que integra image_cleaner).
  - Agregada validación de máscara + fallback de emergencia:
    Si el normalizer v6 produce una máscara basura (puede pasar
    en fotos muy difíciles), se genera una máscara desde el
    grayscale limpio del image_cleaner como red de seguridad.
  - Funciones nuevas: _is_mask_garbage(), _emergency_mask_from_clean_gray()
  - metadata incluye "mask_source" para debugging.

CAMBIOS v4.1 vs v4:
  - YOLO SIEMPRE recibe imagen ORIGINAL sin limpiar.
  - preprocess_robust() devuelve 6 valores.
  - preprocess_multi() genera display_crops separados de raw_crops.
  - Eliminadas llamadas a clean_for_detection() en flujo principal.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.core import config

# ── Pipeline nuevo ──
from app.core.image_cleaner import (
    clean_for_detection,
    clean_crop_for_classification,
    clean_crop_for_display,
)
from app.core.preprocessing import (
    prepare_for_model,
    IMG_SIZE,
)

# ── Normalizer: SOLO para máscara de trazo (métricas) ──
from app.core.normalizer import normalize_character

# ── Clasificador ──
from app.core.classifier import (
    classify_char_smart,
    classify_from_clean_gray,
    classify_word,
    classify_line,
    get_raw_top_k,
    CharContext,
    CLASS_MAP,
    DIGITS,
    ALL_LETTERS,
    PUNCTUATION,
    STROKE_NAMES,
    NUM_MODEL_CLASSES,
    _USE_LETTERBOX,
    _IS_NEW_MODEL,
    debug_check_image,
)

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Orden EMNIST byclass (fallback para modelo antiguo)
# ═════════════════════════════════════════════════════════════════════════════
EMNIST_CLASS_ORDER = list(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


# ═════════════════════════════════════════════════════════════════════════════
# Carga de modelos YOLO (singleton)
# ═════════════════════════════════════════════════════════════════════════════

_yolo_session: Optional[ort.InferenceSession] = None
_yolo_ultralytics = None
_USE_ULTRALYTICS = False


def _build_session(path: str) -> Optional[ort.InferenceSession]:
    if not os.path.exists(path):
        print(f"[processor] WARN Modelo no encontrado: {path}")
        return None
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(path, providers=providers)
        print(f"[processor] OK Modelo ONNX cargado: {Path(path).name}")
        return sess
    except Exception as e:
        print(f"[processor] ERROR Error cargando {path}: {e}")
        return None


def _load_yolo_detector():
    global _yolo_session, _yolo_ultralytics, _USE_ULTRALYTICS

    model_path = config.YOLO_MODEL_PATH
    pt_path = model_path.replace(".onnx", ".pt")

    try:
        from ultralytics import YOLO as UltralyticsYOLO

        if os.path.exists(pt_path):
            _yolo_ultralytics = UltralyticsYOLO(pt_path, task="detect")
            _USE_ULTRALYTICS = True
            print(
                f"[processor] ✅ YOLO cargado con Ultralytics: "
                f"{Path(pt_path).name}"
            )
            return
        elif os.path.exists(model_path):
            _yolo_ultralytics = UltralyticsYOLO(model_path, task="detect")
            _USE_ULTRALYTICS = True
            print(
                f"[processor] ✅ YOLO cargado con Ultralytics: "
                f"{Path(model_path).name}"
            )
            return
    except ImportError:
        print("[processor] INFO ultralytics no disponible, usando ONNX Runtime")
    except Exception as e:
        print(f"[processor] WARN Ultralytics falló: {e}, usando ONNX Runtime")

    _yolo_session = _build_session(model_path)
    _USE_ULTRALYTICS = False

    if _yolo_session is not None:
        out_shape = _yolo_session.get_outputs()[0].shape
        print(f"[processor] INFO YOLO ONNX output shape: {out_shape}")


_load_yolo_detector()


# ═════════════════════════════════════════════════════════════════════════════
# Validación del modelo clasificador
# ═════════════════════════════════════════════════════════════════════════════

def _validate_classifier_model() -> bool:
    model_path = Path(config.MOBILENET_MODEL_PATH)
    if not model_path.exists():
        print("[processor] ⚠️  Modelo clasificador NO encontrado")
        return False

    size_mb = model_path.stat().st_size / (1024 * 1024)

    external_data_candidates = [
        model_path.with_suffix('.onnx.data'),
        model_path.parent / (model_path.stem + '.onnx_data'),
        model_path.parent / (model_path.stem + '_external_data'),
        model_path.parent / 'model.onnx.data',
    ]

    has_external = False
    for ext_path in external_data_candidates:
        if ext_path.exists():
            ext_size_mb = ext_path.stat().st_size / (1024 * 1024)
            print(
                f"[processor] INFO Datos externos: "
                f"{ext_path.name} ({ext_size_mb:.1f} MB)"
            )
            has_external = True
            break

    if _IS_NEW_MODEL and size_mb < 5.0 and not has_external:
        print(f"\n{'=' * 70}")
        print(f"  ⚠️  ADVERTENCIA: Modelo ONNX sospechosamente pequeño")
        print(f"  Archivo:  {model_path.name} ({size_mb:.1f} MB)")
        print(f"  Esperado: ~84 MB (EfficientNetV2-S float32)")
        print(f"{'=' * 70}\n")
        return False

    print(
        f"[processor] Clasificador ONNX: {size_mb:.1f} MB "
        f"({'OK' if size_mb >= 5.0 or has_external else 'SOSPECHOSO'})"
    )
    return True


_classifier_model_ok = _validate_classifier_model()

print(
    f"[processor] Clasificador: {NUM_MODEL_CLASSES} clases | "
    f"Nuevo: {_IS_NEW_MODEL} | Letterbox: {_USE_LETTERBOX} | "
    f"YOLO: {'Ultralytics' if _USE_ULTRALYTICS else 'ONNX Runtime'} | "
    f"Model OK: {_classifier_model_ok}"
)


# ═════════════════════════════════════════════════════════════════════════════
# Detección YOLO
# ═════════════════════════════════════════════════════════════════════════════

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
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
    boxes: List[Tuple[int, int, int, int, float]],
    iou_threshold: float,
    max_detections: Optional[int] = None,
) -> List[Tuple[int, int, int, int, float]]:
    if not boxes:
        return []
    arr = np.array(
        [[x1, y1, x2, y2, conf] for (x1, y1, x2, y2, conf) in boxes],
        dtype=np.float32,
    )
    order = np.argsort(-arr[:, 4])
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if max_detections is not None and len(keep) >= max_detections:
            break
        rest = order[1:]
        if rest.size == 0:
            break
        ious = np.array(
            [_iou_xyxy(arr[i, :4], arr[int(j), :4]) for j in rest],
            dtype=np.float32,
        )
        order = rest[ious <= iou_threshold]
    kept = arr[keep]
    return [
        (int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4]))
        for b in kept
    ]


def _letterbox_yolo(
    img: np.ndarray,
    target_size: int = 640,
    fill_value: int = 114,
) -> Tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    ratio = min(target_size / h, target_size / w)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    canvas = np.full(
        (target_size, target_size, 3), fill_value, dtype=np.uint8
    )
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return canvas, ratio, pad_w, pad_h


def _detect_yolo_ultralytics(
    img_bgr: np.ndarray,
) -> List[Tuple[int, int, int, int, float]]:
    if _yolo_ultralytics is None:
        return []

    results = _yolo_ultralytics.predict(
        source=img_bgr,
        imgsz=config.YOLO_INPUT_SIZE,
        conf=config.DETECTION_THRESHOLD,
        iou=config.NMS_THRESHOLD,
        verbose=False,
    )

    H, W = img_bgr.shape[:2]
    img_area = H * W

    boxes: List[Tuple[int, int, int, int, float]] = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu())

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))

        box_area = (x2 - x1) * (y2 - y1)
        ar = (x2 - x1) / max(y2 - y1, 1)

        if box_area < img_area * 0.0005:
            continue
        if box_area > img_area * 0.60:
            continue
        if ar < 0.20 or ar > 5.0:
            continue

        boxes.append((x1, y1, x2, y2, conf))

    return boxes


def _detect_yolo_onnx(
    img_bgr: np.ndarray,
) -> List[Tuple[int, int, int, int, float]]:
    if _yolo_session is None:
        return []

    H, W = img_bgr.shape[:2]
    img_sz = config.YOLO_INPUT_SIZE

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    letterboxed, ratio, pad_w, pad_h = _letterbox_yolo(rgb, target_size=img_sz)

    tensor = letterboxed.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

    input_name = _yolo_session.get_inputs()[0].name
    outputs = _yolo_session.run(None, {input_name: tensor})

    preds = outputs[0]

    if preds.ndim == 3:
        if preds.shape[1] == 5 and preds.shape[2] > preds.shape[1]:
            preds = preds[0].T
        elif preds.shape[2] == 5:
            preds = preds[0]
        else:
            preds = preds[0]

    img_area = H * W

    boxes: List[Tuple[int, int, int, int, float]] = []
    for det in preds:
        if len(det) < 5:
            continue

        cx, cy, bw, bh = det[0], det[1], det[2], det[3]

        if len(det) == 5:
            conf = float(det[4])
        else:
            conf = float(det[4:].max())

        if conf < config.DETECTION_THRESHOLD:
            continue

        px1 = cx - bw / 2.0
        py1 = cy - bh / 2.0
        px2 = cx + bw / 2.0
        py2 = cy + bh / 2.0

        px1 = (px1 - pad_w) / ratio
        py1 = (py1 - pad_h) / ratio
        px2 = (px2 - pad_w) / ratio
        py2 = (py2 - pad_h) / ratio

        x1 = max(0, min(int(px1), W - 1))
        y1 = max(0, min(int(py1), H - 1))
        x2 = max(x1 + 1, min(int(px2), W))
        y2 = max(y1 + 1, min(int(py2), H))

        box_area = (x2 - x1) * (y2 - y1)
        ar = (x2 - x1) / max(y2 - y1, 1)

        if box_area < img_area * 0.0005:
            continue
        if box_area > img_area * 0.60:
            continue
        if ar < 0.20 or ar > 5.0:
            continue

        boxes.append((x1, y1, x2, y2, conf))

    boxes.sort(key=lambda b: b[4], reverse=True)
    boxes = _nms_xyxy(
        boxes, iou_threshold=config.NMS_THRESHOLD, max_detections=50
    )
    return boxes


def _detect_yolo(
    img_bgr: np.ndarray,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Detección YOLO — SIEMPRE recibe imagen ORIGINAL sin limpiar.
    """
    if _USE_ULTRALYTICS:
        return _detect_yolo_ultralytics(img_bgr)
    else:
        return _detect_yolo_onnx(img_bgr)


# ═════════════════════════════════════════════════════════════════════════════
# Validación de máscara + fallback de emergencia (NUEVO v4.2)
# ═════════════════════════════════════════════════════════════════════════════

def _is_mask_garbage(mask: np.ndarray) -> bool:
    """
    Detecta si una máscara del normalizer es basura (ruido disperso).

    Heurísticas:
    1. Muy pocos píxeles activos (< 0.5% del total)
    2. Demasiados píxeles activos (> 60% — capturó fondo/líneas)
    3. Muchos componentes pequeños dispersos (fragmentación alta)
    4. Densidad del convex hull muy baja (puntos dispersos vs trazo continuo)
    """
    if mask is None or mask.size == 0:
        return True

    total_pixels = mask.size
    active_pixels = int(np.sum(mask > 0))

    # Muy pocos píxeles → probablemente vacía o solo ruido
    if active_pixels < total_pixels * 0.005:
        return True

    # Demasiados píxeles → probablemente capturó el fondo/líneas
    if active_pixels > total_pixels * 0.60:
        return True

    # Verificar fragmentación
    mask_bin = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin, connectivity=8
    )
    n_components = n_labels - 1  # sin fondo

    if n_components == 0:
        return True

    # Si hay muchos componentes pequeños → dispersión = ruido
    if n_components > 15:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = float(areas.max())
        # Si el componente más grande tiene menos del 30% de los píxeles activos
        if largest_area < active_pixels * 0.30:
            logger.debug(
                f"_is_mask_garbage: fragmentación alta — "
                f"{n_components} componentes, largest={largest_area:.0f}, "
                f"active={active_pixels}"
            )
            return True

    # Verificar densidad del convex hull del componente más grande
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    largest_mask = (labels == largest_idx).astype(np.uint8)

    contours, _ = cv2.findContours(
        largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            density = float(areas[largest_idx - 1]) / hull_area
            # Un trazo real tiene densidad > 0.10 en su hull
            # Puntos dispersos tienen densidad < 0.05
            if density < 0.05:
                logger.debug(
                    f"_is_mask_garbage: baja densidad hull — "
                    f"density={density:.3f}"
                )
                return True

    return False


def _emergency_mask_from_clean_gray(
    gray_clean: np.ndarray,
    target_shape: tuple = None,
) -> np.ndarray:
    """
    Genera una máscara binaria de emergencia desde el grayscale limpio
    de image_cleaner, cuando la máscara del normalizer es basura.

    El gray_clean ya tiene:
    - Líneas azules eliminadas (inpainting)
    - Fondo ~blanco (245), trazo ~negro (0-80)
    - Contraste normalizado

    Solo necesitamos binarizar con un umbral simple y centrar.
    """
    from app.core.config import TARGET_SIZE
    ts = TARGET_SIZE

    if target_shape is None:
        target_shape = (ts, ts)

    if gray_clean is None or gray_clean.size == 0:
        return np.zeros(target_shape, dtype=np.uint8)

    # El gray_clean tiene fondo ~245 y trazo ~0-80
    bg_val = float(np.percentile(gray_clean, 90))
    fg_val = float(np.percentile(gray_clean, 10))

    if bg_val - fg_val < 30:
        # Sin contraste suficiente — intentar Otsu
        _, mask = cv2.threshold(
            gray_clean, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        # Umbral basado en percentiles
        threshold = fg_val + (bg_val - fg_val) * 0.45
        mask = np.zeros_like(gray_clean, dtype=np.uint8)
        mask[gray_clean < threshold] = 255

    # Limpiar ruido pequeño
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Recortar y centrar en target_shape (igual que normalizer.crop_and_center)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros(target_shape, dtype=np.uint8)

    x, y, w_obj, h_obj = cv2.boundingRect(coords)
    w_obj = max(1, w_obj)
    h_obj = max(1, h_obj)
    obj = mask[y:y + h_obj, x:x + w_obj]

    # Escalar para que quepa en el canvas con padding
    padding = getattr(config, 'NORMALIZER_PADDING', 10)
    inner = target_shape[0] - padding * 2
    scale = inner / max(w_obj, h_obj)
    new_w = max(1, int(w_obj * scale))
    new_h = max(1, int(h_obj * scale))

    interp = cv2.INTER_LANCZOS4 if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(obj, (new_w, new_h), interpolation=interp)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    # Centrar en canvas
    final = np.zeros(target_shape, dtype=np.uint8)
    ox = (target_shape[1] - new_w) // 2
    oy = (target_shape[0] - new_h) // 2
    final[oy:oy + new_h, ox:ox + new_w] = resized

    return final


def _validate_and_fix_mask(
    mask: np.ndarray,
    gray_clean: np.ndarray,
    source_label: str = "unknown",
) -> Tuple[np.ndarray, str]:
    """
    Valida una máscara del normalizer. Si es basura, genera una de emergencia.

    Returns:
        (mask_final, mask_source)
        mask_source: "normalizer" | "emergency_from_clean_gray" | "normalizer_low_quality"
    """
    if not _is_mask_garbage(mask):
        return mask, "normalizer"

    logger.warning(
        f"{source_label}: máscara del normalizer es basura, "
        f"generando máscara de emergencia desde clean_gray"
    )

    emergency_mask = _emergency_mask_from_clean_gray(gray_clean)

    if not _is_mask_garbage(emergency_mask):
        return emergency_mask, "emergency_from_clean_gray"
    else:
        logger.warning(
            f"{source_label}: máscara de emergencia también insuficiente, "
            f"usando máscara original del normalizer"
        )
        return mask, "normalizer_low_quality"


# ═════════════════════════════════════════════════════════════════════════════
# Utilidades de tipo esperado
# ═════════════════════════════════════════════════════════════════════════════

def _infer_expected_type(expected_char: Optional[str]) -> Optional[str]:
    if expected_char is None:
        return None
    if expected_char in STROKE_NAMES:
        return None
    if expected_char in ALL_LETTERS:
        if expected_char.isupper():
            return 'letter_upper'
        else:
            return 'letter_lower'
    elif expected_char in DIGITS:
        return 'digit'
    elif expected_char in PUNCTUATION:
        return 'punct'
    else:
        return 'letter'


# ═════════════════════════════════════════════════════════════════════════════
# Clasificación de un crop
# ═════════════════════════════════════════════════════════════════════════════

def _classify_crop(
    raw_crop_bgr: np.ndarray,
    context: str = CharContext.UNKNOWN,
    expected_type: Optional[str] = None,
    expected_char: Optional[str] = None,
    neighbors: Tuple[Optional[str], Optional[str]] = (None, None),
    use_smart: bool = True,
    use_tta: bool = True,
) -> Tuple[str, float, Dict]:
    """
    Clasifica un crop BGR de YOLO.

    Pipeline:
      1. image_cleaner.clean_crop_for_classification(crop_bgr)
         → grayscale continuo, fondo~blanco, trazo~negro
      2. classify_char_smart(gray_clean)
         → internamente: preprocessing.prepare_for_model → ONNX → SmartOCR

    NUNCA usa la máscara binaria del normalizer.
    """
    gray_clean = clean_crop_for_classification(raw_crop_bgr)

    if use_smart:
        result = classify_char_smart(
            gray_clean, context, expected_type,
            expected_char, neighbors,
            use_tta=use_tta,
        )
        return result['char'], result['confidence'], result
    else:
        char, conf = classify_from_clean_gray(gray_clean, use_tta=use_tta)
        return char, conf, {
            'char': char, 'confidence': conf,
            'raw_char': char, 'raw_confidence': conf,
            'method': 'raw_clean', 'alternatives': [],
        }


# ═════════════════════════════════════════════════════════════════════════════
# Orden de lectura
# ═════════════════════════════════════════════════════════════════════════════

def _sort_reading_order(
    boxes: List[Tuple[int, int, int, int, float]],
    line_y_tol_ratio: float = 0.35,
) -> Tuple[List[Tuple[int, int, int, int, float]], List[List[int]]]:
    if not boxes:
        return [], []

    heights = np.array(
        [(y2 - y1) for (_, y1, _, y2, _) in boxes], dtype=np.float32,
    )
    median_h = float(np.median(heights)) if len(heights) else 1.0
    y_tol = max(8.0, median_h * float(line_y_tol_ratio))

    entries = []
    for x1, y1, x2, y2, conf in boxes:
        cy = (y1 + y2) / 2.0
        cx = (x1 + x2) / 2.0
        entries.append((x1, y1, x2, y2, conf, cx, cy))
    entries.sort(key=lambda e: e[6])

    lines: List[List[tuple]] = []
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

    lines.sort(key=lambda line: np.mean([e[6] for e in line]))

    out_boxes: List[Tuple[int, int, int, int, float]] = []
    line_groups: List[List[int]] = []
    global_idx = 0

    for line in lines:
        line.sort(key=lambda e: e[5])
        group = []
        for x1, y1, x2, y2, conf, _, _ in line:
            out_boxes.append(
                (int(x1), int(y1), int(x2), int(y2), float(conf))
            )
            group.append(global_idx)
            global_idx += 1
        line_groups.append(group)

    return out_boxes, line_groups


def _group_into_words(
    boxes: List[Tuple[int, int, int, int, float]],
    line_indices: List[int],
    gap_ratio: float = 1.5,
) -> List[List[int]]:
    if len(line_indices) <= 1:
        return [line_indices] if line_indices else []

    widths = [(boxes[i][2] - boxes[i][0]) for i in line_indices]
    median_w = float(np.median(widths)) if widths else 20.0
    gap_threshold = median_w * gap_ratio

    words: List[List[int]] = [[line_indices[0]]]

    for k in range(1, len(line_indices)):
        prev_idx = line_indices[k - 1]
        curr_idx = line_indices[k]
        prev_x2 = boxes[prev_idx][2]
        curr_x1 = boxes[curr_idx][0]
        gap = curr_x1 - prev_x2

        if gap > gap_threshold:
            words.append([curr_idx])
        else:
            words[-1].append(curr_idx)

    return words


# ═════════════════════════════════════════════════════════════════════════════
# Función principal — un solo carácter
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_robust(
    img_bytes: bytes,
    use_smart: bool = True,
    expected_char: Optional[str] = None,
) -> Tuple[np.ndarray, dict, str, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Procesa una imagen de un solo carácter.

    Pipeline:
      1. Decodificar imagen
      2. YOLO detecta en imagen ORIGINAL (sin limpiar)
      3. Pipeline A: crop → clean → classify (con TTA)
      4. Pipeline B: normalizer → máscara de trazo (para métricas)
      5. NUEVO v4.2: Validar máscara + fallback de emergencia

    Returns:
        (mask, metadata, detected_char, confidence, raw_crop_bgr, display_crop)
    """

    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError(
            "No se pudo decodificar la imagen. "
            "Verifica que el archivo sea JPG/PNG válido."
        )

    # ══════════════════════════════════════════════════════════════
    # YOLO recibe imagen ORIGINAL — NO limpiar antes de detectar.
    # ══════════════════════════════════════════════════════════════
    boxes = _detect_yolo(img_bgr)

    yolo_box = None
    raw_crop_bgr = None

    if boxes:
        x1, y1, x2, y2, yolo_conf = boxes[0]
        yolo_box = (x1, y1, x2, y2)
        raw_crop_bgr = img_bgr[y1:y2, x1:x2].copy()

    # ── Pipeline B: Máscara de trazo para métricas ──
    mask, metadata = normalize_character(img_bgr, yolo_box=yolo_box)

    # ── Pipeline A: Clasificación (con TTA para evaluación individual) ──
    expected_type = _infer_expected_type(expected_char)
    context = CharContext.STANDALONE

    if expected_char is not None and expected_char in STROKE_NAMES:
        expected_type = None
        expected_char_for_cls = None
    else:
        expected_char_for_cls = expected_char

    if raw_crop_bgr is not None and raw_crop_bgr.size > 0:
        # Limpiar para clasificación (también necesario para emergency mask)
        gray_clean = clean_crop_for_classification(raw_crop_bgr)

        detected_char, confidence, detail = _classify_crop(
            raw_crop_bgr,
            context=context,
            expected_type=expected_type,
            expected_char=expected_char_for_cls,
            use_smart=use_smart,
            use_tta=True,
        )
        display_crop = clean_crop_for_display(raw_crop_bgr)
    else:
        gray_clean = clean_crop_for_classification(img_bgr)

        detected_char, confidence, detail = _classify_crop(
            img_bgr,
            context=context,
            expected_type=expected_type,
            expected_char=expected_char_for_cls,
            use_smart=use_smart,
            use_tta=True,
        )
        raw_crop_bgr = None
        display_crop = clean_crop_for_display(img_bgr)

    # ══════════════════════════════════════════════════════════════
    # NUEVO v4.2: Validación de máscara + fallback de emergencia
    # Si la máscara del normalizer es basura (puntos dispersos,
    # ruido de papel/líneas), generar una máscara de emergencia
    # desde el grayscale limpio de image_cleaner.
    # ══════════════════════════════════════════════════════════════
    mask, mask_source = _validate_and_fix_mask(
        mask, gray_clean, source_label="preprocess_robust"
    )
    metadata["mask_source"] = mask_source

    metadata["yolo_detected"] = yolo_box is not None
    metadata["yolo_confidence"] = float(boxes[0][4]) if boxes else 0.0
    metadata["n_detections"] = len(boxes)
    metadata["model_type"] = "arcface_smart" if _IS_NEW_MODEL else "legacy"
    metadata["smart_ocr"] = use_smart
    metadata["yolo_backend"] = (
        "ultralytics" if _USE_ULTRALYTICS else "onnxruntime"
    )
    metadata["expected_char"] = expected_char
    metadata["classifier_model_ok"] = _classifier_model_ok
    metadata["classifier_confidence"] = confidence
    metadata["classification_method"] = detail.get('method', 'raw')
    metadata["raw_prediction"] = detail.get('raw_char', detected_char)
    metadata["raw_confidence"] = detail.get('raw_confidence', confidence)
    metadata["expected_type_filter"] = expected_type
    metadata["pipeline_version"] = "v4.2_clean"

    return mask, metadata, detected_char, confidence, raw_crop_bgr, display_crop


# ═════════════════════════════════════════════════════════════════════════════
# Multi-carácter
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_multi(
    img_bytes: bytes,
    max_boxes: Optional[int] = None,
    use_smart: bool = True,
    group_words: bool = True,
    expected_chars: Optional[str] = None,
) -> Dict:
    """
    Procesa imagen con múltiples caracteres (plana).

    Pipeline v4.2:
      1. Decodificar imagen
      2. YOLO detecta en imagen ORIGINAL
      3. Para cada detección:
         a) raw_crop = crop ORIGINAL
         b) display_crop = crop limpio
         c) gray_clean = grayscale limpio (para clasificación Y emergency mask)
         d) mask = máscara de trazo (para métricas)
         e) NUEVO: validar máscara + fallback si es basura
      4. Clasificar con contexto de palabras
    """

    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(
            "No se pudo decodificar la imagen. Verifica JPG/PNG válido."
        )

    logger.info(
        f"preprocess_multi: imagen {img_bgr.shape[1]}x{img_bgr.shape[0]}, "
        f"backend={'ultralytics' if _USE_ULTRALYTICS else 'onnxruntime'}, "
        f"expected={expected_chars!r}"
    )

    # ══════════════════════════════════════════════════════════════
    # YOLO recibe imagen ORIGINAL
    # ══════════════════════════════════════════════════════════════
    boxes = _detect_yolo(img_bgr)
    logger.info(f"preprocess_multi: {len(boxes)} detecciones YOLO")

    if max_boxes is not None:
        boxes = boxes[:max(0, int(max_boxes))]

    if not boxes:
        return {
            'characters': [], 'text': '', 'words': [],
            'lines': [], 'n_detections': 0, 'confidence': 0.0,
            'detection_method': (
                'ultralytics' if _USE_ULTRALYTICS else 'onnxruntime'
            ),
            'classifier_model_ok': _classifier_model_ok,
        }

    # ── Ordenar y agrupar ──
    sorted_boxes, line_groups = _sort_reading_order(boxes)
    n_total = len(sorted_boxes)

    # ── Obtener crops, display crops, masks y clasificar ──
    raw_crops: List[np.ndarray] = []
    display_crops: List[np.ndarray] = []
    clean_grays: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    metadata_list: List[dict] = []

    for (x1, y1, x2, y2, conf) in sorted_boxes:
        yolo_box = (x1, y1, x2, y2)

        # Crop de la imagen ORIGINAL
        raw_crop = img_bgr[y1:y2, x1:x2].copy()
        raw_crops.append(raw_crop)

        # Display crop limpio
        display_crop = clean_crop_for_display(raw_crop)
        display_crops.append(display_crop)

        # Pipeline A: Limpiar para clasificación
        gray_clean = clean_crop_for_classification(raw_crop)
        clean_grays.append(gray_clean)

        # Pipeline B: Máscara de trazo para métricas
        mask, meta = normalize_character(img_bgr, yolo_box=yolo_box)

        # ── NUEVO v4.2: Validación de máscara + fallback ──
        mask, mask_source = _validate_and_fix_mask(
            mask, gray_clean,
            source_label=f"preprocess_multi[box_{x1},{y1}]"
        )
        meta["mask_source"] = mask_source

        masks.append(mask)

        meta["yolo_detected"] = True
        meta["yolo_confidence"] = float(conf)
        meta["n_detections"] = n_total
        meta["bbox_xyxy"] = [int(x1), int(y1), int(x2), int(y2)]
        meta["model_type"] = (
            "arcface_smart" if _IS_NEW_MODEL else "legacy"
        )
        meta["yolo_backend"] = (
            "ultralytics" if _USE_ULTRALYTICS else "onnxruntime"
        )
        meta["pipeline_version"] = "v4.2_clean"
        metadata_list.append(meta)

    # ── Inferir expected_char y tipo global ──
    expected_char = None
    global_expect_type = None

    if expected_chars:
        if len(expected_chars) == 1:
            expected_char = expected_chars
        else:
            expected_char = expected_chars[0]

        if expected_char in STROKE_NAMES:
            global_expect_type = None
            expected_char = None
        elif expected_char in ALL_LETTERS:
            global_expect_type = 'letter'
        elif expected_char in DIGITS:
            global_expect_type = 'digit'
        else:
            global_expect_type = None

    # ── Clasificar ──
    if use_smart and group_words:
        result = _classify_with_word_context(
            sorted_boxes, line_groups, raw_crops, display_crops,
            clean_grays, masks, metadata_list, n_total,
            global_expect_type=global_expect_type,
            expected_char=expected_char,
        )
    else:
        result = _classify_without_context(
            sorted_boxes, raw_crops, display_crops, clean_grays,
            masks, metadata_list, n_total, use_smart,
            expected_type=global_expect_type,
            expected_char=expected_char,
        )

    result['detection_method'] = (
        'ultralytics' if _USE_ULTRALYTICS else 'onnxruntime'
    )
    result['classifier_model_ok'] = _classifier_model_ok
    return result


def _classify_with_word_context(
    sorted_boxes, line_groups, raw_crops, display_crops,
    clean_grays, masks, metadata_list, n_total,
    global_expect_type: Optional[str] = None,
    expected_char: Optional[str] = None,
) -> Dict:
    """
    Clasificación con agrupación en palabras y contexto SmartOCR.
    """
    all_chars: List[Optional[Dict]] = [None] * n_total
    all_words: List[Dict] = []
    all_lines: List[Dict] = []

    for line_idx, line_indices in enumerate(line_groups):
        is_first_line = (line_idx == 0)
        word_groups = _group_into_words(sorted_boxes, line_indices)
        line_text_parts: List[str] = []

        for word_idx, word_indices in enumerate(word_groups):
            is_sentence_start = (is_first_line and word_idx == 0)

            word_grays = [clean_grays[i] for i in word_indices]

            if expected_char:
                char_results = []
                expected_type = _infer_expected_type(expected_char)

                for k, idx in enumerate(word_indices):
                    result = classify_char_smart(
                        word_grays[k],
                        context=CharContext.STANDALONE,
                        expected_type=expected_type,
                        expected_char=expected_char,
                        use_tta=False,
                    )
                    char_results.append(result)

                raw_word = ''.join(r['char'] for r in char_results)
                avg_conf = float(np.mean(
                    [r['confidence'] for r in char_results]
                )) if char_results else 0.0

                word_result = {
                    'word': raw_word,
                    'raw_word': raw_word,
                    'chars': char_results,
                    'confidence': avg_conf,
                    'corrected': False,
                    'correction_method': 'expected_char_mode',
                }

            else:
                if global_expect_type:
                    expect = global_expect_type
                else:
                    first_raw = get_raw_top_k(word_grays[0], top_k=3, use_tta=False)
                    first_char = first_raw['top1_char']
                    expect = 'digit' if first_char in DIGITS else 'letter'

                word_result = classify_word(
                    word_grays,
                    expect_type=expect,
                    is_sentence_start=is_sentence_start,
                    use_tta=False,
                )

            for k, idx in enumerate(word_indices):
                char_detail = (
                    word_result['chars'][k]
                    if k < len(word_result['chars'])
                    else {}
                )
                char_data = {
                    'char': char_detail.get('char', '?'),
                    'confidence': char_detail.get('confidence', 0.0),
                    'raw_char': char_detail.get('raw_char', '?'),
                    'raw_confidence': char_detail.get(
                        'raw_confidence', 0.0
                    ),
                    'method': char_detail.get('method', 'unknown'),
                    'bbox_xyxy': list(sorted_boxes[idx][:4]),
                    'yolo_confidence': sorted_boxes[idx][4],
                    'metadata': metadata_list[idx],
                    'normalized_mask': masks[idx],
                    'raw_crop_bgr': raw_crops[idx],
                    'display_crop': display_crops[idx],
                }
                all_chars[idx] = char_data

            word_data = {
                'word': word_result['word'],
                'raw_word': word_result['raw_word'],
                'confidence': word_result['confidence'],
                'corrected': word_result['corrected'],
                'correction_method': word_result['correction_method'],
                'char_indices': word_indices,
                'n_chars': len(word_indices),
            }
            all_words.append(word_data)
            line_text_parts.append(word_result['word'])

        line_data = {
            'text': ' '.join(line_text_parts),
            'word_count': len(word_groups),
            'char_count': len(line_indices),
        }
        all_lines.append(line_data)

    full_text = '\n'.join(line['text'] for line in all_lines)

    confidences = [
        c['confidence'] for c in all_chars if c is not None
    ]
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        'characters': [c for c in all_chars if c is not None],
        'text': full_text,
        'words': all_words,
        'lines': all_lines,
        'n_detections': n_total,
        'confidence': avg_conf,
    }


def _classify_without_context(
    sorted_boxes, raw_crops, display_crops, clean_grays,
    masks, metadata_list, n_total, use_smart,
    expected_type: Optional[str] = None,
    expected_char: Optional[str] = None,
) -> Dict:
    """
    Clasificación sin agrupación (backward compatible).
    """
    characters: List[Dict] = []

    for i in range(n_total):
        char, conf, detail = _classify_crop(
            raw_crops[i],
            expected_type=expected_type,
            expected_char=expected_char,
            use_smart=use_smart,
            use_tta=False,
        )

        metadata_list[i]["classifier_confidence"] = float(conf)
        metadata_list[i]["classification_method"] = detail.get(
            'method', 'raw'
        )

        characters.append({
            'char': char,
            'confidence': conf,
            'raw_char': detail.get('raw_char', char),
            'raw_confidence': detail.get('raw_confidence', conf),
            'method': detail.get('method', 'raw'),
            'bbox_xyxy': list(sorted_boxes[i][:4]),
            'yolo_confidence': sorted_boxes[i][4],
            'metadata': metadata_list[i],
            'normalized_mask': masks[i],
            'raw_crop_bgr': raw_crops[i],
            'display_crop': display_crops[i],
        })

    text = ''.join(c['char'] for c in characters)
    confidences = [c['confidence'] for c in characters]
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        'characters': characters,
        'text': text,
        'words': [],
        'lines': [],
        'n_detections': n_total,
        'confidence': avg_conf,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Backward compatible — lista de tuplas
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_multi_legacy(
    img_bytes: bytes,
    max_boxes: Optional[int] = None,
    expected_chars: Optional[str] = None,
) -> List[Tuple[np.ndarray, dict, str, float, np.ndarray]]:
    result = preprocess_multi(
        img_bytes, max_boxes=max_boxes,
        use_smart=True, group_words=True,
        expected_chars=expected_chars,
    )

    legacy_results = []
    for char_data in result['characters']:
        legacy_results.append((
            char_data.get('normalized_mask', np.array([])),
            char_data.get('metadata', {}),
            char_data['char'],
            char_data['confidence'],
            char_data.get('raw_crop_bgr', np.array([])),
        ))

    return legacy_results