"""
app/core/image_quality.py  (v2)
================================
CORRECCIONES en esta versión:
  - detect_image_source() detecta si la imagen es DIGITAL (generada por
    computadora/fuente) o FOTOGRAFÍA (celular/escáner). El pipeline aplica
    pasos muy distintos según el origen.
  - Las imágenes digitales tienen blur_score altísimo (>5000), contraste >100
    y brightness bimodal (fondo=255, trazo=0). El pipeline anterior las
    destruía porque CLAHE + bg_division no tienen nada que mejorar.
  - Se agrega is_digital al ImageQuality y el flag se propaga a PipelineParams
    para que normalizer.py cortocircuite los pasos innecesarios.
"""

from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass(frozen=True)
class ImageQuality:
    blur_score:      float
    contrast:        float
    brightness:      float
    ink_ratio:       float
    shadow_score:    float
    resolution_mp:   float
    # Diagnósticos booleanos
    is_blurry:       bool
    is_dark:         bool
    is_overexposed:  bool
    has_shadow:      bool
    is_low_contrast: bool
    is_digital:      bool   # ← NUEVO: imagen generada por computadora


@dataclass(frozen=True)
class PipelineParams:
    block_size:       int
    adaptive_c:       int
    morph_k:          int
    clahe_clip:       float
    clahe_tile:       int
    use_bg_division:  bool
    bg_blur_k:        int
    use_otsu:         bool
    canny_low:        int
    canny_high:       int
    skip_illumination: bool  # ← NUEVO: saltar CLAHE/bg_division en imágenes digitales
    speck_min_area:   int    # ← NUEVO: umbral mínimo de mancha calculado por imagen


# =============================================================================
# Detección de origen de imagen
# =============================================================================

def _detect_digital(gray: np.ndarray) -> bool:
    """
    Determina si la imagen fue generada por computadora (fuente tipográfica,
    captura de pantalla) en lugar de ser una fotografía de papel.

    Una imagen digital tiene:
      1. Bordes perfectamente nítidos → blur_score muy alto (>3000)
      2. Histograma bimodal: fondo blanco puro (255) + trazo negro puro (0-30)
         → el 70%+ de píxeles están en los extremos del histograma
      3. Contraste muy alto (std > 80)
      4. Prácticamente cero píxeles en tonos medios (grises 40-220 < 5%)

    Una foto de papel siempre tiene grises intermedios por:
      - Bordes suavizados por la óptica del celular
      - Textura del papel
      - Sombras y variaciones de iluminación
    """
    # Criterio 1: nitidez extrema
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(lap.var())
    if blur_score < 3000:
        return False  # Demasiado borroso para ser digital

    # Criterio 2: histograma bimodal (pocos píxeles en tonos medios)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = gray.size
    extremes = float(hist[:30].sum() + hist[230:].sum()) / total
    midtones  = float(hist[40:220].sum()) / total

    # Digital: >65% en extremos, <10% en tonos medios
    if extremes > 0.65 and midtones < 0.10:
        return True

    # Criterio 3: contraste muy alto + píxeles casi binarios
    std = float(gray.std())
    if std > 90 and midtones < 0.08:
        return True

    return False


# =============================================================================
# Medición de calidad
# =============================================================================

def measure_image_quality(gray: np.ndarray) -> ImageQuality:
    h, w = gray.shape
    f    = gray.astype(np.float32)

    lap        = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(lap.var())
    contrast   = float(f.std())
    brightness = float(f.mean())
    ink_ratio  = float((gray < 128).mean())

    bg_k       = _safe_odd(max(31, min(h, w) // 4))
    bg_blur    = cv2.GaussianBlur(f, (bg_k, bg_k), 0)
    shadow_map = np.abs(f - bg_blur) / (bg_blur + 1e-6)
    shadow_score = float(shadow_map.mean())

    resolution_mp = float(h * w / 1_000_000)
    is_digital    = _detect_digital(gray)

    return ImageQuality(
        blur_score     = round(blur_score,    2),
        contrast       = round(contrast,      2),
        brightness     = round(brightness,    2),
        ink_ratio      = round(ink_ratio,     4),
        shadow_score   = round(shadow_score,  4),
        resolution_mp  = round(resolution_mp, 4),
        is_blurry      = blur_score   < 50.0  and not is_digital,
        is_dark        = brightness   < 60.0  and not is_digital,
        is_overexposed = brightness   > 210.0 and not is_digital,
        has_shadow     = shadow_score > 0.25  and not is_digital,
        is_low_contrast= contrast     < 20.0  and not is_digital,
        is_digital     = is_digital,
    )


# =============================================================================
# Derivación de parámetros adaptativos
# =============================================================================

def derive_pipeline_params(q: ImageQuality, img_shape: tuple[int, int]) -> PipelineParams:
    h, w  = img_shape
    side  = min(h, w)

    # ── Imágenes digitales: pipeline mínimo ──────────────────────────────────
    # No necesitan CLAHE, corrección de fondo ni umbral adaptativo complejo.
    # Solo binarización Otsu directa (ya son casi binarias) + limpieza mínima.
    if q.is_digital:
        return PipelineParams(
            block_size       = 11,
            adaptive_c       = 2,
            morph_k          = 1,          # kernel mínimo — no destruir bordes
            clahe_clip       = 1.0,        # CLAHE casi desactivado
            clahe_tile       = 8,
            use_bg_division  = False,      # Sin corrección de fondo
            bg_blur_k        = 31,
            use_otsu         = True,       # Otsu directo (histograma bimodal perfecto)
            canny_low        = 30,
            canny_high       = 120,
            skip_illumination = True,      # Saltar paso 4 completo
            speck_min_area   = 5,          # Manchas muy pequeñas (antialiasing)
        )

    # ── Fotografías de papel: pipeline completo adaptativo ───────────────────
    base_bs = max(3, int(side * 0.08))
    if q.has_shadow:
        base_bs = int(base_bs * 1.5)
    block_size = _safe_odd(base_bs)

    if q.is_low_contrast or q.is_dark:
        adaptive_c = 4
    elif q.has_shadow:
        adaptive_c = 3
    else:
        adaptive_c = 2

    morph_k = max(1, int(side * 0.018))

    if q.is_dark or q.is_low_contrast:
        clahe_clip = 5.0
    elif q.is_overexposed:
        clahe_clip = 1.5
    elif q.has_shadow:
        clahe_clip = 4.0
    else:
        clahe_clip = 3.0

    clahe_tile      = max(2, min(8, side // 16))
    use_bg_division = q.has_shadow or q.is_dark
    bg_blur_k       = _safe_odd(max(31, side // 3))

    use_otsu = (
        q.contrast >= 35.0
        and not q.has_shadow
        and not q.is_dark
        and not q.is_overexposed
    )

    if q.is_blurry or q.is_low_contrast:
        canny_low, canny_high = 15, 60
    elif q.is_dark:
        canny_low, canny_high = 20, 80
    else:
        canny_low, canny_high = 30, 120

    # Área mínima de mancha escala con resolución de la imagen
    speck_min_area = max(10, int(side * side * 0.0008))

    return PipelineParams(
        block_size       = block_size,
        adaptive_c       = adaptive_c,
        morph_k          = morph_k,
        clahe_clip       = clahe_clip,
        clahe_tile       = clahe_tile,
        use_bg_division  = use_bg_division,
        bg_blur_k        = bg_blur_k,
        use_otsu         = use_otsu,
        canny_low        = canny_low,
        canny_high       = canny_high,
        skip_illumination = False,
        speck_min_area   = speck_min_area,
    )


def analyze(gray: np.ndarray) -> tuple[ImageQuality, PipelineParams]:
    q = measure_image_quality(gray)
    p = derive_pipeline_params(q, gray.shape)
    return q, p


def _safe_odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1