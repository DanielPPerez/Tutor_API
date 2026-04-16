"""
app/core/normalizer.py  (v6 — integra image_cleaner para fotos reales)
=======================================================
Genera MÁSCARA BINARIA de trazo para métricas de evaluación.

⚠️  IMPORTANTE: Este módulo NO se usa para clasificación OCR.
    La clasificación usa: image_cleaner → preprocessing → classifier
    
    Este módulo SOLO se usa para:
    - Métricas de trazo (dt_fidelity, geometric, topologic, trajectory)
    - Esqueletización del trazo del alumno
    - Comparación visual trazo vs plantilla

CAMBIOS v6 vs v5:
  - INTEGRACIÓN DE image_cleaner: Para fotos reales, usa image_cleaner
    para eliminar líneas azules ANTES de binarizar, en lugar de depender
    solo de remove_color_lines() con HSV.
  - Nueva función _clean_with_image_cleaner() que aplica el pipeline de
    limpieza de image_cleaner al ROI antes de binarización.
  - Binarización mejorada para fotos: usa el grayscale limpio del
    image_cleaner (fondo ~245, trazo ~0-80) con umbral simple.
  - Fallback: si image_cleaner no está disponible, usa pipeline anterior.
  - normalize_character() ahora detecta si es foto y aplica pipeline mejorado.

Formatos de salida:
  - normalize_character()    → máscara (blanco=trazo, negro=fondo)
  - normalize_for_metrics()  → igual, alias semántico
"""

import math
import logging

import cv2
import numpy as np

from app.core import config
from app.core.image_quality import analyze, ImageQuality, PipelineParams
from app.core.illumination import normalize_illumination, to_lab_lightness
from app.core.binarizer import binarize

logger = logging.getLogger(__name__)

# ── Intentar importar image_cleaner ──
_HAS_IMAGE_CLEANER = False
try:
    from app.core.image_cleaner import clean_crop_for_classification
    _HAS_IMAGE_CLEANER = True
    logger.info("normalizer v6: image_cleaner disponible, pipeline mejorado activo")
except ImportError:
    logger.warning(
        "normalizer v6: image_cleaner NO disponible, "
        "usando pipeline legacy para fotos"
    )


# =============================================================================
# Utilidades
# =============================================================================

def _safe_odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def _is_empty_image(img: np.ndarray) -> bool:
    """Verifica si la imagen está vacía o tiene dimensión 0."""
    if img is None:
        return True
    if img.size == 0:
        return True
    if img.shape[0] == 0 or img.shape[1] == 0:
        return True
    return False


# =============================================================================
# 1. EXTRACCIÓN DE ROI
# =============================================================================

def _find_char_bbox(
    gray: np.ndarray,
    canny_low: int = 30,
    canny_high: int = 120,
) -> tuple[int, int, int, int] | None:
    H, W = gray.shape
    if H < 3 or W < 3:
        return None

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, k, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    frame_area = H * W
    valid = [
        c for c in contours
        if frame_area * 0.01 < cv2.contourArea(c) < frame_area * 0.90
    ]
    if not valid:
        valid = contours

    all_pts = np.vstack(valid)
    x, y, w, h = cv2.boundingRect(all_pts)

    if w < 2 or h < 2:
        return None

    m = config.ROI_CONTOUR_MARGIN
    return (max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m))


def extract_roi(
    image_bgr: np.ndarray,
    yolo_box=None,
    canny_low: int = 30,
    canny_high: int = 120,
) -> tuple[np.ndarray, bool]:
    H, W = image_bgr.shape[:2]

    if yolo_box is not None:
        x1, y1, x2, y2 = [int(v) for v in yolo_box]
        p = config.ROI_PADDING
        px1 = max(0, x1 - p)
        py1 = max(0, y1 - p)
        px2 = min(W, x2 + p)
        py2 = min(H, y2 + p)
        roi_pad = image_bgr[py1:py2, px1:px2]

        if _is_empty_image(roi_pad):
            return image_bgr.copy(), False

        gray_roi = cv2.cvtColor(roi_pad, cv2.COLOR_BGR2GRAY)
        bbox_rel = _find_char_bbox(gray_roi, canny_low, canny_high)
        if bbox_rel is not None:
            rx1, ry1, rx2, ry2 = bbox_rel
            ax1 = max(0, px1 + rx1)
            ay1 = max(0, py1 + ry1)
            ax2 = min(W, px1 + rx2)
            ay2 = min(H, py1 + ry2)
            roi = image_bgr[ay1:ay2, ax1:ax2]
        else:
            roi = roi_pad

        if _is_empty_image(roi):
            return image_bgr.copy(), False
        return roi.copy(), True
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        bbox = _find_char_bbox(gray, canny_low, canny_high)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi = image_bgr[y1:y2, x1:x2]
            if _is_empty_image(roi):
                return image_bgr.copy(), False
            return roi.copy(), False
        return image_bgr.copy(), False


# =============================================================================
# 2. ELIMINACIÓN DE LÍNEAS POR COLOR (HSV) — Legacy, usado como fallback
# =============================================================================

def remove_color_lines(image_bgr: np.ndarray) -> np.ndarray:
    """Borra líneas de libreta de colores. NO toca el grafito (gris oscuro)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    combined = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for (lo, hi) in config.HSV_LINE_RANGES:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        combined = cv2.bitwise_or(combined, mask)
    if config.HSV_MASK_DILATE > 0:
        d = config.HSV_MASK_DILATE * 2 + 1
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
        combined = cv2.dilate(combined, k, iterations=1)
    result = image_bgr.copy()
    result[combined > 0] = (255, 255, 255)
    return result


# =============================================================================
# 2b. LIMPIEZA CON image_cleaner (NUEVO v6)
# =============================================================================

def _clean_with_image_cleaner(roi_bgr: np.ndarray) -> np.ndarray | None:
    """
    Usa image_cleaner para obtener un grayscale limpio del ROI.
    
    El image_cleaner:
    - Elimina líneas azules por inpainting (mucho mejor que HSV simple)
    - Normaliza fondo a ~245 (blanco) y trazo a ~0-80 (negro)
    - Aplica CLAHE para contraste uniforme
    
    Returns:
        grayscale limpio (uint8) o None si falla
    """
    if not _HAS_IMAGE_CLEANER:
        return None
    
    try:
        gray_clean = clean_crop_for_classification(roi_bgr)
        if gray_clean is None or gray_clean.size == 0:
            return None
        return gray_clean
    except Exception as e:
        logger.warning(f"_clean_with_image_cleaner falló: {e}")
        return None


def _binarize_from_clean_gray(gray_clean: np.ndarray) -> np.ndarray:
    """
    Binariza un grayscale ya limpio del image_cleaner.
    
    Como el image_cleaner ya hizo:
    - Eliminación de líneas azules (inpainting)
    - Normalización de fondo (~245) y trazo (~0-80)
    - CLAHE para contraste
    
    La binarización es simple y precisa: solo necesitamos un umbral
    basado en los percentiles reales de la imagen.
    """
    if gray_clean is None or gray_clean.size == 0:
        return np.zeros((config.TARGET_SIZE, config.TARGET_SIZE), dtype=np.uint8)
    
    # Percentiles para determinar umbral
    bg_val = float(np.percentile(gray_clean, 90))   # fondo (~245)
    fg_val = float(np.percentile(gray_clean, 10))    # trazo (~0-80)
    
    # Si no hay suficiente contraste, intentar Otsu
    if bg_val - fg_val < 30:
        logger.debug(
            f"_binarize_from_clean_gray: bajo contraste "
            f"(bg={bg_val:.0f}, fg={fg_val:.0f}), usando Otsu"
        )
        _, binary = cv2.threshold(
            gray_clean, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary
    
    # Umbral basado en percentiles: punto medio entre fondo y trazo
    # Sesgamos ligeramente hacia el fondo para capturar trazos suaves
    threshold = fg_val + (bg_val - fg_val) * 0.45
    
    # Binarizar: píxeles más oscuros que el umbral = trazo (blanco en máscara)
    binary = np.zeros_like(gray_clean, dtype=np.uint8)
    binary[gray_clean < threshold] = 255
    
    logger.debug(
        f"_binarize_from_clean_gray: bg={bg_val:.0f}, fg={fg_val:.0f}, "
        f"threshold={threshold:.0f}, "
        f"stroke_pixels={np.sum(binary > 0)}/{binary.size}"
    )
    
    return binary


# =============================================================================
# 3. ELIMINACIÓN DE LÍNEAS RESIDUALES (morfológica)
# =============================================================================

def remove_grid_lines(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    line_w = max(config.GRID_LINE_MIN_WIDTH, w // 8)
    h_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (line_w, 1)),
    )
    v_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_w)),
    )
    grid_mask = cv2.add(h_lines, v_lines)
    if grid_mask.max() == 0:
        return binary
    return cv2.inpaint(binary, grid_mask, config.INPAINT_RADIUS, cv2.INPAINT_TELEA)


# =============================================================================
# 4. ELIMINACIÓN DE ISLAS DE RUIDO
# =============================================================================

def remove_specks(binary: np.ndarray, min_area: int | None = None) -> np.ndarray:
    """
    Elimina componentes conectados pequeños.
    Conserva SIEMPRE el componente más grande.
    """
    h, w = binary.shape

    if min_area is None:
        min_area = max(
            config.SPECK_MIN_AREA_PX,
            int(h * w * config.SPECK_AREA_RATIO),
        )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if n_labels <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1

    clean = np.zeros_like(binary)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if i == largest_label or area >= min_area:
            clean[labels == i] = 255

    return clean


# =============================================================================
# 5. LIMPIEZA MORFOLÓGICA
# =============================================================================

def clean_noise(binary: np.ndarray, morph_k: int = 2) -> np.ndarray:
    k = max(1, morph_k)
    if k <= 1:
        return binary

    open_k = np.ones((k, k), np.uint8)
    close_k = np.ones((k + 1, k + 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)
    return cleaned


# =============================================================================
# 6. RELLENO DE HUECOS INTERNOS
# =============================================================================

def _fill_internal_gaps(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape

    n_labels, _ = cv2.connectedComponents(binary, connectivity=8)
    if n_labels - 1 <= 3:
        return binary

    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    for seed in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if flood[seed] == 0:
            cv2.floodFill(flood, mask, (seed[1], seed[0]), 128)

    interior_gaps = (flood == 0)
    result = binary.copy()
    result[interior_gaps] = 255
    return result


# =============================================================================
# 7. DESKEW
# =============================================================================

def deskew(binary: np.ndarray) -> tuple[np.ndarray, float]:
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return binary, 0.0
    m = cv2.moments(binary)
    if abs(m["mu20"] - m["mu02"]) < 1e-5:
        return binary, 0.0
    angle_deg = math.degrees(
        0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])
    )
    if abs(angle_deg) > config.MAX_DESKEW_ANGLE:
        return binary, 0.0
    h, w = binary.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated, float(round(angle_deg, 2))


# =============================================================================
# 8. RECORTE Y CENTRADO
# =============================================================================

def crop_and_center(binary: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Recorta el trazo y lo centra en un canvas de TARGET_SIZE × TARGET_SIZE.

    FORMATO DE SALIDA: máscara binaria
      - Trazo = 255 (BLANCO)
      - Fondo = 0   (NEGRO)
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        empty = np.zeros(
            (config.TARGET_SIZE, config.TARGET_SIZE), dtype=np.uint8
        )
        return empty, {
            "w_obj": 0, "h_obj": 0, "scale_factor": 0.0,
            "aspect_ratio": 0.0, "centroid_x": 0.5, "centroid_y": 0.5,
        }

    x, y, w_obj, h_obj = cv2.boundingRect(coords)

    w_obj = max(1, w_obj)
    h_obj = max(1, h_obj)

    obj = binary[y:y + h_obj, x:x + w_obj]

    inner = config.TARGET_SIZE - config.NORMALIZER_PADDING * 2
    scale = inner / max(w_obj, h_obj)
    new_w = max(1, int(w_obj * scale))
    new_h = max(1, int(h_obj * scale))

    interp = cv2.INTER_LANCZOS4 if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(obj, (new_w, new_h), interpolation=interp)

    # Re-binarizar después del resize (ambas interpolaciones generan grises)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    final = np.zeros(
        (config.TARGET_SIZE, config.TARGET_SIZE), dtype=np.uint8
    )
    ox = (config.TARGET_SIZE - new_w) // 2
    oy = (config.TARGET_SIZE - new_h) // 2
    final[oy:oy + new_h, ox:ox + new_w] = resized

    pts = cv2.findNonZero(final)
    if pts is not None:
        cx = float(pts[:, 0, 0].mean()) / config.TARGET_SIZE
        cy = float(pts[:, 0, 1].mean()) / config.TARGET_SIZE
    else:
        cx, cy = 0.5, 0.5

    return final, {
        "w_obj": int(w_obj),
        "h_obj": int(h_obj),
        "scale_factor": float(round(scale, 4)),
        "aspect_ratio": float(round(w_obj / max(h_obj, 1), 4)),
        "centroid_x": round(cx, 4),
        "centroid_y": round(cy, 4),
    }


# =============================================================================
# 9. VALIDACIÓN DE MÁSCARA (NUEVO v6)
# =============================================================================

def _is_mask_valid(mask: np.ndarray) -> bool:
    """
    Verifica si una máscara binaria tiene un trazo real y coherente.
    
    Retorna False si:
    - Muy pocos píxeles activos (< 0.5% → probablemente vacía/ruido)
    - Demasiados píxeles activos (> 60% → capturó fondo/líneas)
    - Demasiados componentes pequeños dispersos (fragmentación = ruido)
    """
    if mask is None or mask.size == 0:
        return False
    
    total_pixels = mask.size
    active_pixels = int(np.sum(mask > 0))
    
    # Muy pocos píxeles
    if active_pixels < total_pixels * 0.005:
        return False
    
    # Demasiados píxeles
    if active_pixels > total_pixels * 0.60:
        return False
    
    # Verificar fragmentación
    mask_bin = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin, connectivity=8
    )
    n_components = n_labels - 1  # sin fondo
    
    if n_components == 0:
        return False
    
    # Si hay muchos componentes y el más grande es pequeño → ruido disperso
    if n_components > 15:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = float(areas.max())
        if largest_area < active_pixels * 0.30:
            logger.debug(
                f"_is_mask_valid: fragmentación alta: "
                f"{n_components} componentes, largest={largest_area:.0f}, "
                f"total_active={active_pixels}"
            )
            return False
    
    return True


# =============================================================================
# FUNCIÓN PRINCIPAL — Máscara para métricas
# =============================================================================

def normalize_character(
    image_crop: np.ndarray,
    yolo_box=None,
) -> tuple[np.ndarray, dict]:
    """
    Pipeline de normalización para generar MÁSCARA BINARIA de trazo.

    ⚠️  SOLO para métricas de evaluación de trazo.
    ⚠️  NO usar para clasificación OCR.

    FORMATO DE SALIDA:
      - Máscara binaria: trazo = 255 (BLANCO), fondo = 0 (NEGRO)
      - Shape: (TARGET_SIZE, TARGET_SIZE), dtype=uint8

    Pipeline v6 (fotos reales):
      ROI → image_cleaner (elimina líneas azules, normaliza) →
      binarización simple → specks → morphology → fill_gaps →
      deskew → crop_and_center
      
    Pipeline v6 (digitales):
      ROI → grayscale → illumination → binarize →
      grid_lines → specks → morphology → deskew → crop_and_center
    """
    # ── Protección contra imagen vacía ──
    if _is_empty_image(image_crop):
        empty = np.zeros(
            (config.TARGET_SIZE, config.TARGET_SIZE), dtype=np.uint8
        )
        return empty, _empty_metadata()

    # ── Paso 0: Medir calidad de la imagen completa ──
    gray_full = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    q, p = analyze(gray_full)

    # ── Paso 1: Extraer ROI ──
    roi, from_yolo = extract_roi(
        image_crop, yolo_box,
        canny_low=p.canny_low,
        canny_high=p.canny_high,
    )

    if _is_empty_image(roi):
        empty = np.zeros(
            (config.TARGET_SIZE, config.TARGET_SIZE), dtype=np.uint8
        )
        return empty, _empty_metadata()

    # ══════════════════════════════════════════════════════════════
    # DECISIÓN: ¿Es foto real o digital?
    # Para fotos reales, usamos image_cleaner que es MUCHO mejor
    # eliminando líneas azules (usa inpainting, no solo HSV).
    # ══════════════════════════════════════════════════════════════
    
    is_digital = q.is_digital
    used_image_cleaner = False
    used_lab = False
    
    # Re-analizar calidad del ROI
    gray_roi_raw = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    q_roi, p_roi = analyze(gray_roi_raw)
    
    if not is_digital and _HAS_IMAGE_CLEANER:
        # ──────────────────────────────────────────────────────
        # PIPELINE NUEVO: Foto real con image_cleaner
        # ──────────────────────────────────────────────────────
        binary = _pipeline_photo_with_cleaner(roi, p_roi)
        used_image_cleaner = True
        
        # Validar resultado
        if not _is_mask_valid(binary):
            logger.warning(
                "normalizer: pipeline image_cleaner produjo máscara inválida, "
                "intentando pipeline legacy"
            )
            binary = _pipeline_legacy(roi, q_roi, p_roi)
            used_image_cleaner = False
            
            # Si legacy también falla, intentar pipeline de emergencia
            if not _is_mask_valid(binary):
                logger.warning(
                    "normalizer: pipeline legacy también falló, "
                    "intentando binarización de emergencia"
                )
                binary = _pipeline_emergency(roi)
    
    elif not is_digital:
        # ──────────────────────────────────────────────────────
        # PIPELINE LEGACY: Foto real sin image_cleaner
        # ──────────────────────────────────────────────────────
        binary = _pipeline_legacy(roi, q_roi, p_roi)
    
    else:
        # ──────────────────────────────────────────────────────
        # PIPELINE DIGITAL: Imágenes limpias/templates
        # ──────────────────────────────────────────────────────
        binary = _pipeline_digital(roi, q_roi, p_roi)

    # ── Paso post: Eliminar manchas ──
    binary = remove_specks(binary, min_area=p_roi.speck_min_area)

    # ── Paso post: Limpieza morfológica ──
    binary = clean_noise(binary, morph_k=p_roi.morph_k)

    # ── Paso post: Relleno de huecos internos (solo fotos) ──
    if not is_digital:
        binary = _fill_internal_gaps(binary)

    # ── Verificar que quedó algo ──
    if cv2.countNonZero(binary) == 0:
        logger.warning("normalizer: máscara vacía después de todo el pipeline")
        binary = _pipeline_emergency(roi)
        binary = remove_specks(binary, min_area=p_roi.speck_min_area)

    # ── Paso final: Deskew ──
    binary, angle = deskew(binary)

    # ── Paso final: Recorte y centrado ──
    final_img, crop_metrics = crop_and_center(binary)

    # ── Metadata ──
    metadata = {
        "angle_corrected": float(round(angle, 2)),
        "original_aspect_ratio": crop_metrics["aspect_ratio"],
        "scale_factor": crop_metrics["scale_factor"],
        "char_width_px": crop_metrics["w_obj"],
        "char_height_px": crop_metrics["h_obj"],
        "roi_refined": from_yolo,
        "stroke_centroid_x": crop_metrics["centroid_x"],
        "stroke_centroid_y": crop_metrics["centroid_y"],
        "image_source": "digital" if is_digital else "photo",
        "output_format": "mask_white_on_black",
        "output_purpose": "metrics_only",
        "used_image_cleaner": used_image_cleaner,  # NUEVO v6
        "quality": {
            "blur_score": q_roi.blur_score,
            "contrast": q_roi.contrast,
            "brightness": q_roi.brightness,
            "ink_ratio": q_roi.ink_ratio,
            "shadow_score": q_roi.shadow_score,
            "is_blurry": q_roi.is_blurry,
            "is_dark": q_roi.is_dark,
            "has_shadow": q_roi.has_shadow,
            "is_low_contrast": q_roi.is_low_contrast,
            "is_digital": q_roi.is_digital,
        },
        "pipeline_params": {
            "block_size": p_roi.block_size,
            "adaptive_c": p_roi.adaptive_c,
            "morph_k": p_roi.morph_k,
            "clahe_clip": p_roi.clahe_clip,
            "used_bg_division": p_roi.use_bg_division,
            "used_otsu": p_roi.use_otsu,
            "used_lab": used_lab,
            "skipped_illumination": p_roi.skip_illumination,
        },
    }

    return final_img, metadata


# =============================================================================
# PIPELINES DE BINARIZACIÓN SEPARADOS (NUEVO v6)
# =============================================================================

def _pipeline_photo_with_cleaner(
    roi_bgr: np.ndarray,
    p_roi: PipelineParams,
) -> np.ndarray:
    """
    Pipeline para fotos reales USANDO image_cleaner.
    
    1. image_cleaner elimina líneas azules por inpainting
    2. Resultado: grayscale con fondo ~245, trazo ~0-80
    3. Binarización simple por umbral de percentiles
    4. Limpieza morfológica ligera
    """
    gray_clean = _clean_with_image_cleaner(roi_bgr)
    
    if gray_clean is None:
        logger.warning("_pipeline_photo_with_cleaner: image_cleaner retornó None")
        return np.zeros((100, 100), dtype=np.uint8)
    
    # Binarizar desde el grayscale limpio
    binary = _binarize_from_clean_gray(gray_clean)
    
    # Limpieza morfológica ligera para cerrar gaps en el trazo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Eliminar ruido pequeño
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary


def _pipeline_legacy(
    roi_bgr: np.ndarray,
    q_roi: ImageQuality,
    p_roi: PipelineParams,
) -> np.ndarray:
    """
    Pipeline legacy para fotos sin image_cleaner.
    
    HSV color removal → grayscale → illumination → binarize → grid_lines
    """
    # Borrar líneas de libreta (HSV)
    roi_clean = remove_color_lines(roi_bgr)

    # Escala de grises
    roi_hsv = cv2.cvtColor(roi_clean, cv2.COLOR_BGR2HSV)
    sat_mean = float(roi_hsv[:, :, 1].mean())

    if sat_mean > 20:
        gray = to_lab_lightness(roi_clean)
    else:
        gray = cv2.cvtColor(roi_clean, cv2.COLOR_BGR2GRAY)

    # Normalización de iluminación
    if not p_roi.skip_illumination:
        enhanced = normalize_illumination(
            gray,
            use_bg_division=p_roi.use_bg_division,
            bg_blur_k=p_roi.bg_blur_k,
            clahe_clip=p_roi.clahe_clip,
            clahe_tile=p_roi.clahe_tile,
        )
    else:
        enhanced = gray

    # Binarización
    binary = binarize(
        enhanced,
        use_otsu=p_roi.use_otsu,
        block_size=p_roi.block_size,
        adaptive_c=p_roi.adaptive_c,
        contrast=q_roi.contrast,
    )

    # Eliminar líneas de cuadrícula
    binary = remove_grid_lines(binary)

    return binary


def _pipeline_digital(
    roi_bgr: np.ndarray,
    q_roi: ImageQuality,
    p_roi: PipelineParams,
) -> np.ndarray:
    """
    Pipeline para imágenes digitales/templates limpias.
    
    Simple: grayscale → binarize (no necesita limpieza de líneas ni illumination)
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Iluminación (normalmente se skipea para digitales)
    if p_roi.skip_illumination:
        enhanced = gray
    else:
        enhanced = normalize_illumination(
            gray,
            use_bg_division=p_roi.use_bg_division,
            bg_blur_k=p_roi.bg_blur_k,
            clahe_clip=p_roi.clahe_clip,
            clahe_tile=p_roi.clahe_tile,
        )

    # Binarización
    binary = binarize(
        enhanced,
        use_otsu=p_roi.use_otsu,
        block_size=p_roi.block_size,
        adaptive_c=p_roi.adaptive_c,
        contrast=q_roi.contrast,
    )

    return binary


def _pipeline_emergency(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Pipeline de emergencia: último recurso cuando todo falla.
    
    Estrategia agresiva:
    1. Grayscale directo
    2. Blur fuerte para suavizar ruido
    3. Otsu invertido
    4. Apertura morfológica agresiva para quitar ruido
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    # Blur fuerte
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Otsu invertido (trazo oscuro → blanco)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # Apertura agresiva para quitar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Si quedó demasiado → probablemente capturó el fondo
    total = binary.size
    active = np.sum(binary > 0)
    if active > total * 0.50:
        # Invertir y re-intentar
        binary = cv2.bitwise_not(binary)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return binary


# =============================================================================
# METADATA VACÍA
# =============================================================================

def _empty_metadata() -> dict:
    """Metadata por defecto para imágenes vacías/fallidas."""
    return {
        "angle_corrected": 0.0,
        "original_aspect_ratio": 0.0,
        "scale_factor": 0.0,
        "char_width_px": 0,
        "char_height_px": 0,
        "roi_refined": False,
        "stroke_centroid_x": 0.5,
        "stroke_centroid_y": 0.5,
        "image_source": "unknown",
        "output_format": "mask_white_on_black",
        "output_purpose": "metrics_only",
        "used_image_cleaner": False,
        "quality": {
            "blur_score": 0, "contrast": 0, "brightness": 0,
            "ink_ratio": 0, "shadow_score": 0,
            "is_blurry": False, "is_dark": False,
            "has_shadow": False, "is_low_contrast": False,
            "is_digital": False,
        },
        "pipeline_params": {
            "block_size": 0, "adaptive_c": 0, "morph_k": 0,
            "clahe_clip": 0, "used_bg_division": False,
            "used_otsu": False, "used_lab": False,
            "skipped_illumination": False,
        },
    }


# =============================================================================
# ALIAS SEMÁNTICO — Más claro sobre el propósito
# =============================================================================

def normalize_for_metrics(
    image_crop: np.ndarray,
    yolo_box=None,
) -> tuple[np.ndarray, dict]:
    """
    Genera máscara binaria de trazo para métricas de evaluación.

    Alias semántico de normalize_character().
    Hace explícito que el resultado es para métricas, NO para clasificación.

    Returns:
        (mask, metadata)
        - mask: grayscale uint8, trazo=255(BLANCO), fondo=0(NEGRO)
        - metadata: dict con info del pipeline
    """
    return normalize_character(image_crop, yolo_box)


# =============================================================================
# FUNCIONES DEPRECADAS — Mantenidas por backward compatibility
# =============================================================================

def mask_to_classifier_image(mask: np.ndarray) -> np.ndarray:
    """
    DEPRECADA: Ya no se necesita.
    """
    logger.warning(
        "mask_to_classifier_image() está DEPRECADA. "
        "La clasificación ahora usa image_cleaner + preprocessing."
    )
    inverted = cv2.bitwise_not(mask)
    bgr = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    return bgr


def normalize_for_classifier(
    image_crop: np.ndarray,
    yolo_box=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    DEPRECADA: Ya no se necesita.
    """
    logger.warning(
        "normalize_for_classifier() está DEPRECADA. "
        "Usar image_cleaner.clean_crop_for_classification() "
        "→ preprocessing.prepare_for_model()"
    )
    mask, metadata = normalize_character(image_crop, yolo_box)
    classifier_img = mask_to_classifier_image(mask)
    return mask, classifier_img, metadata


def build_display_crop(
    roi_bgr: np.ndarray, target_size: int | None = None
) -> np.ndarray:
    """
    DEPRECADA: Usar image_cleaner.clean_crop_for_display() en su lugar.
    """
    logger.warning(
        "build_display_crop() está DEPRECADA. "
        "Usar image_cleaner.clean_crop_for_display()"
    )
    ts = target_size or config.TARGET_SIZE
    h, w = roi_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((ts, ts, 3), dtype=np.uint8)

    scale = ts / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_LANCZOS4 if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(roi_bgr, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((ts, ts, 3), dtype=np.uint8)
    ox = (ts - new_w) // 2
    oy = (ts - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = (
        resized if resized.ndim == 3
        else cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    )
    return canvas