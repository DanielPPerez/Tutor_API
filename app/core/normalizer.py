"""
app/core/normalizer.py  (v3)
============================
CORRECCIONES en esta versión:

  Bug 1 — Imágenes digitales destruidas:
    El pipeline detecta automáticamente si la entrada es una imagen generada
    por computadora (fuente tipográfica) y aplica un camino corto:
      → Otsu directo (histograma ya es bimodal perfecto)
      → Sin CLAHE agresivo
      → Sin corrección de fondo por división
      → Kernel morfológico mínimo (=1) para no romper bordes perfectos
    Esto preserva la forma exacta de la letra digital.

  Bug 2 — Puntos residuales en imagen normalizada:
    - remove_specks() ahora usa p.speck_min_area calculado por image_quality
      en lugar de SPECK_MIN_AREA_PX fijo. Escala correctamente con resolución.
    - clean_noise() usa morph_k=1 en imágenes digitales (no hace nada
      destructivo en bordes perfectos).
    - Se agrega un paso final de convex fill SOLO si el trazo es muy
      fragmentado (componentes > 3) para rellenar huecos internos antes
      de skeletonizar sin destruir la forma.

  Bug 3 — Comparación rota en papel (se traslada a visualizer.py):
    El normalizer ahora exporta `stroke_centroid` en metadata para que
    visualizer pueda alinear por centroide real en lugar de bounding box.
"""

import math

import cv2
import numpy as np

from app.core import config
from app.core.image_quality  import analyze, ImageQuality, PipelineParams
from app.core.illumination   import normalize_illumination, to_lab_lightness
from app.core.binarizer      import binarize


# =============================================================================
# Utilidades
# =============================================================================

def _safe_odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


# =============================================================================
# 1. EXTRACCIÓN DE ROI
# =============================================================================

def _find_char_bbox(
    gray:       np.ndarray,
    canny_low:  int = 30,
    canny_high: int = 120,
) -> tuple[int, int, int, int] | None:
    H, W = gray.shape
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges    = cv2.Canny(blurred, canny_low, canny_high)
    k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges    = cv2.dilate(edges, k, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = H * W
    valid = [c for c in contours
             if frame_area * 0.01 < cv2.contourArea(c) < frame_area * 0.90]
    if not valid:
        valid = contours

    all_pts = np.vstack(valid)
    x, y, w, h = cv2.boundingRect(all_pts)
    m = config.ROI_CONTOUR_MARGIN
    return (max(0, x-m), max(0, y-m), min(W, x+w+m), min(H, y+h+m))


def extract_roi(
    image_bgr:  np.ndarray,
    yolo_box    = None,
    canny_low:  int = 30,
    canny_high: int = 120,
) -> tuple[np.ndarray, bool]:
    H, W = image_bgr.shape[:2]

    if yolo_box is not None:
        x1, y1, x2, y2 = [int(v) for v in yolo_box]
        p   = config.ROI_PADDING
        px1 = max(0, x1-p);  py1 = max(0, y1-p)
        px2 = min(W, x2+p);  py2 = min(H, y2+p)
        roi_pad = image_bgr[py1:py2, px1:px2]
        if roi_pad.size == 0:
            return image_bgr.copy(), False

        gray_roi = cv2.cvtColor(roi_pad, cv2.COLOR_BGR2GRAY)
        bbox_rel = _find_char_bbox(gray_roi, canny_low, canny_high)
        if bbox_rel is not None:
            rx1, ry1, rx2, ry2 = bbox_rel
            ax1 = max(0, px1+rx1);  ay1 = max(0, py1+ry1)
            ax2 = min(W, px1+rx2);  ay2 = min(H, py1+ry2)
            roi = image_bgr[ay1:ay2, ax1:ax2]
        else:
            roi = roi_pad
        return (roi.copy() if roi.size > 0 else image_bgr.copy()), True
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        bbox = _find_char_bbox(gray, canny_low, canny_high)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi = image_bgr[y1:y2, x1:x2]
            return (roi.copy() if roi.size > 0 else image_bgr.copy()), False
        return image_bgr.copy(), False


# =============================================================================
# 2. ELIMINACIÓN DE LÍNEAS POR COLOR (HSV)
# =============================================================================

def remove_color_lines(image_bgr: np.ndarray) -> np.ndarray:
    """Borra líneas de libreta de colores. NO toca el grafito (gris oscuro)."""
    hsv      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    combined = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for (lo, hi) in config.HSV_LINE_RANGES:
        mask     = cv2.inRange(hsv, np.array(lo), np.array(hi))
        combined = cv2.bitwise_or(combined, mask)
    if config.HSV_MASK_DILATE > 0:
        d = config.HSV_MASK_DILATE * 2 + 1
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
        combined = cv2.dilate(combined, k, iterations=1)
    result = image_bgr.copy()
    result[combined > 0] = (255, 255, 255)
    return result


# =============================================================================
# 5. ELIMINACIÓN DE LÍNEAS RESIDUALES (morfológica)
# =============================================================================

def remove_grid_lines(binary: np.ndarray) -> np.ndarray:
    h, w   = binary.shape
    line_w = max(config.GRID_LINE_MIN_WIDTH, w // 8)
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (line_w, 1)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_w)))
    grid_mask = cv2.add(h_lines, v_lines)
    if grid_mask.max() == 0:
        return binary
    return cv2.inpaint(binary, grid_mask, config.INPAINT_RADIUS, cv2.INPAINT_TELEA)


# =============================================================================
# 6. ELIMINACIÓN DE ISLAS DE RUIDO (CORREGIDO)
# =============================================================================

def remove_specks(binary: np.ndarray, min_area: int | None = None) -> np.ndarray:
    """
    Elimina componentes conectados pequeños.
    
    CORRECCIÓN: min_area ahora viene de PipelineParams.speck_min_area
    (calculado en image_quality.py según la resolución real de la imagen)
    en lugar del valor fijo SPECK_MIN_AREA_PX de config que no escalaba bien.
    
    Además: conserva SIEMPRE el componente más grande, aunque sea pequeño.
    Esto evita que imágenes de trazos muy finos queden completamente vacías.
    """
    h, w = binary.shape

    if min_area is None:
        # Fallback al valor de config si no se pasó parámetro
        min_area = max(
            config.SPECK_MIN_AREA_PX,
            int(h * w * config.SPECK_AREA_RATIO),
        )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if n_labels <= 1:
        return binary  # Solo fondo, nada que limpiar

    # Encontrar el componente más grande (excluyendo fondo=0)
    areas = stats[1:, cv2.CC_STAT_AREA]  # áreas sin el fondo
    largest_label = int(np.argmax(areas)) + 1  # +1 porque saltamos el fondo

    clean = np.zeros_like(binary)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Conservar: componente más grande SIEMPRE + los que superan min_area
        if i == largest_label or area >= min_area:
            clean[labels == i] = 255

    return clean


# =============================================================================
# 7. LIMPIEZA MORFOLÓGICA (CORREGIDA)
# =============================================================================

def clean_noise(binary: np.ndarray, morph_k: int = 2) -> np.ndarray:
    """
    Apertura + cierre morfológico.
    
    CORRECCIÓN: Con morph_k=1 (imágenes digitales) no hace nada destructivo
    — un kernel 1×1 es la identidad. Con morph_k=2-3 (fotos) limpia
    ruido sin fragmentar el trazo.
    
    El cierre se hace con kernel más grande que la apertura para rellenar
    huecos del trazo sin borrar puntos legítimos.
    """
    k = max(1, morph_k)

    if k <= 1:
        # Identidad: no modificar la imagen (para imágenes digitales perfectas)
        return binary

    open_k  = np.ones((k,     k),     np.uint8)
    close_k = np.ones((k + 1, k + 1), np.uint8)  # cierre ligeramente mayor
    cleaned = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  open_k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)
    return cleaned


# =============================================================================
# 7b. RELLENO DE HUECOS INTERNOS (solo para trazos muy fragmentados)
# =============================================================================

def _fill_internal_gaps(binary: np.ndarray) -> np.ndarray:
    """
    Rellena huecos internos del trazo usando flood fill desde las esquinas
    (técnica de flood fill inverso).

    Solo se aplica cuando el trazo está fragmentado en muchos componentes
    (típico de imágenes de papel con binarización imperfecta).

    NO modifica el exterior del trazo — solo rellena agujeros encerrados.
    """
    h, w = binary.shape

    # Contar componentes para decidir si aplicar
    n_labels, _ = cv2.connectedComponents(binary, connectivity=8)
    if n_labels - 1 <= 3:
        return binary  # Trazo continuo, no necesita relleno

    # Flood fill desde el borde para marcar todo el fondo accesible
    # Lo que queda sin marcar = hueco interior = debería ser trazo
    flood = binary.copy()
    mask  = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Semillas en las 4 esquinas y bordes
    for seed in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
        if flood[seed] == 0:
            cv2.floodFill(flood, mask, (seed[1], seed[0]), 128)

    # Píxeles que quedaron a 0 = huecos interiores → rellenarlos
    interior_gaps = (flood == 0)
    result = binary.copy()
    result[interior_gaps] = 255
    return result


# =============================================================================
# 8. DESKEW
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
    M    = cv2.getRotationMatrix2D((w//2, h//2), angle_deg, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated, float(round(angle_deg, 2))


# =============================================================================
# 9. RECORTE Y CENTRADO
# =============================================================================

def crop_and_center(binary: np.ndarray) -> tuple[np.ndarray, dict]:
    coords = cv2.findNonZero(binary)
    if coords is None:
        empty = np.zeros((config.TARGET_SIZE, config.TARGET_SIZE), dtype=np.uint8)
        return empty, {"w_obj": 0, "h_obj": 0, "scale_factor": 0.0,
                       "aspect_ratio": 0.0, "centroid_x": 0.5, "centroid_y": 0.5}

    x, y, w_obj, h_obj = cv2.boundingRect(coords)
    obj = binary[y:y+h_obj, x:x+w_obj]

    inner  = config.TARGET_SIZE - config.NORMALIZER_PADDING * 2
    scale  = inner / max(w_obj, h_obj)
    new_w  = max(1, int(w_obj * scale))
    new_h  = max(1, int(h_obj * scale))

    interp  = cv2.INTER_LANCZOS4 if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(obj, (new_w, new_h), interpolation=interp)

    # Re-binarizar después del resize (LANCZOS introduce grises en los bordes)
    if scale >= 1.0:
        _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    final = np.zeros((config.TARGET_SIZE, config.TARGET_SIZE), dtype=np.uint8)
    ox = (config.TARGET_SIZE - new_w) // 2
    oy = (config.TARGET_SIZE - new_h) // 2
    final[oy:oy+new_h, ox:ox+new_w] = resized

    # Centroide normalizado [0,1] del trazo en el canvas final
    pts = cv2.findNonZero(final)
    if pts is not None:
        cx = float(pts[:, 0, 0].mean()) / config.TARGET_SIZE
        cy = float(pts[:, 0, 1].mean()) / config.TARGET_SIZE
    else:
        cx, cy = 0.5, 0.5

    return final, {
        "w_obj":        int(w_obj),
        "h_obj":        int(h_obj),
        "scale_factor": float(round(scale, 4)),
        "aspect_ratio": float(round(w_obj / max(h_obj, 1), 4)),
        "centroid_x":   round(cx, 4),
        "centroid_y":   round(cy, 4),
    }


# =============================================================================
# UTILIDAD: crop para mostrar al usuario
# =============================================================================

def build_display_crop(roi_bgr: np.ndarray, target_size: int | None = None) -> np.ndarray:
    ts   = target_size or config.TARGET_SIZE
    h, w = roi_bgr.shape[:2]
    scale  = ts / max(h, w)
    new_w  = max(1, int(w * scale))
    new_h  = max(1, int(h * scale))
    interp = cv2.INTER_LANCZOS4 if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(roi_bgr, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((ts, ts, 3), dtype=np.uint8)
    ox = (ts - new_w) // 2
    oy = (ts - new_h) // 2
    canvas[oy:oy+new_h, ox:ox+new_w] = (
        resized if resized.ndim == 3
        else cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    )
    return canvas


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def normalize_character(
    image_crop: np.ndarray,
    yolo_box    = None,
) -> tuple[np.ndarray, dict]:
    """
    Pipeline de normalización con detección automática de origen de imagen.

    Camino A — Imagen digital (fuente tipográfica, captura de pantalla):
      1. extract_roi
      2. remove_color_lines (no-op en digitales pero inofensivo)
      3. Convertir a gris
      4. Binarización Otsu directa (histograma ya bimodal)
      5. remove_specks mínimo (solo antialiasing)
      6. clean_noise con k=1 (identidad — no destruir bordes perfectos)
      7. deskew
      8. crop_and_center con re-binarización post-resize

    Camino B — Fotografía de papel (celular, escáner):
      1. extract_roi (Canny adaptativo)
      2. remove_color_lines (HSV)
      3. LAB si hay color / gris si papel blanco
      4. normalize_illumination (bg_division + CLAHE adaptativo)
      5. binarize (Otsu o Adaptativo según contraste)
      6. remove_grid_lines
      7. remove_specks (umbral adaptativo a resolución)
      8. clean_noise (kernel adaptativo)
      9. _fill_internal_gaps (si trazo fragmentado)
     10. deskew
     11. crop_and_center
    """
    # ── Paso 0: Medir calidad de la imagen completa ───────────────────────────
    gray_full = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    q, p      = analyze(gray_full)

    # ── Paso 1: Extraer ROI ───────────────────────────────────────────────────
    roi, from_yolo = extract_roi(
        image_crop, yolo_box,
        canny_low  = p.canny_low,
        canny_high = p.canny_high,
    )

    # ── Paso 2: Borrar líneas de libreta (HSV) ────────────────────────────────
    roi_clean = remove_color_lines(roi)

    # ── Paso 3: Escala de grises ──────────────────────────────────────────────
    roi_hsv  = cv2.cvtColor(roi_clean, cv2.COLOR_BGR2HSV)
    sat_mean = float(roi_hsv[:, :, 1].mean())
    used_lab = False

    if q.is_digital:
        # Digital: gris clásico directo, LAB no aporta nada
        gray = cv2.cvtColor(roi_clean, cv2.COLOR_BGR2GRAY)
    elif sat_mean > 20:
        # Papel con color (azul/amarillo): LAB es más robusto
        gray     = to_lab_lightness(roi_clean)
        used_lab = True
    else:
        gray = cv2.cvtColor(roi_clean, cv2.COLOR_BGR2GRAY)

    # Re-medir en el ROI para parámetros más precisos
    q_roi, p_roi = analyze(gray)

    # ── Paso 4: Normalización de iluminación (SOLO para fotos) ───────────────
    if p_roi.skip_illumination or q_roi.is_digital:
        enhanced = gray   # Digital: no tocar la imagen
    else:
        enhanced = normalize_illumination(
            gray,
            use_bg_division = p_roi.use_bg_division,
            bg_blur_k       = p_roi.bg_blur_k,
            clahe_clip      = p_roi.clahe_clip,
            clahe_tile      = p_roi.clahe_tile,
        )

    # ── Paso 5: Binarización ─────────────────────────────────────────────────
    binary = binarize(
        enhanced,
        use_otsu   = p_roi.use_otsu,
        block_size = p_roi.block_size,
        adaptive_c = p_roi.adaptive_c,
        contrast   = q_roi.contrast,
    )

    # ── Pasos 6-7: Solo para fotos (las digitales no tienen líneas de libreta)
    if not q_roi.is_digital:
        binary = remove_grid_lines(binary)

    # ── Paso 8: Eliminar manchas (umbral adaptativo) ─────────────────────────
    binary = remove_specks(binary, min_area=p_roi.speck_min_area)

    # ── Paso 9: Limpieza morfológica ─────────────────────────────────────────
    binary = clean_noise(binary, morph_k=p_roi.morph_k)

    # ── Paso 9b: Relleno de huecos internos (solo fotos fragmentadas) ─────────
    if not q_roi.is_digital:
        binary = _fill_internal_gaps(binary)

    # ── Paso 10: Deskew ───────────────────────────────────────────────────────
    binary, angle = deskew(binary)

    # ── Paso 11: Recorte y centrado ───────────────────────────────────────────
    final_img, crop_metrics = crop_and_center(binary)

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        "angle_corrected":       float(round(angle, 2)),
        "original_aspect_ratio": crop_metrics["aspect_ratio"],
        "scale_factor":          crop_metrics["scale_factor"],
        "char_width_px":         crop_metrics["w_obj"],
        "char_height_px":        crop_metrics["h_obj"],
        "roi_refined":           from_yolo,
        "stroke_centroid_x":     crop_metrics["centroid_x"],
        "stroke_centroid_y":     crop_metrics["centroid_y"],
        "image_source":          "digital" if q_roi.is_digital else "photo",
        "quality": {
            "blur_score":       q_roi.blur_score,
            "contrast":         q_roi.contrast,
            "brightness":       q_roi.brightness,
            "ink_ratio":        q_roi.ink_ratio,
            "shadow_score":     q_roi.shadow_score,
            "is_blurry":        q_roi.is_blurry,
            "is_dark":          q_roi.is_dark,
            "has_shadow":       q_roi.has_shadow,
            "is_low_contrast":  q_roi.is_low_contrast,
            "is_digital":       q_roi.is_digital,
        },
        "pipeline_params": {
            "block_size":        p_roi.block_size,
            "adaptive_c":        p_roi.adaptive_c,
            "morph_k":           p_roi.morph_k,
            "clahe_clip":        p_roi.clahe_clip,
            "used_bg_division":  p_roi.use_bg_division,
            "used_otsu":         p_roi.use_otsu,
            "used_lab":          used_lab,
            "skipped_illumination": p_roi.skip_illumination,
        },
    }

    return final_img, metadata