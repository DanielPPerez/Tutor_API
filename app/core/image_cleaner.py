"""
app/core/image_cleaner.py
=========================
Limpieza de imágenes reales (fotos de cuaderno) para OCR.

Responsabilidades:
  1. Eliminar líneas de color (azules, rojas, verdes) del cuaderno
  2. Normalizar iluminación sin destruir gradientes
  3. Preparar imagen para YOLO (menos falsos positivos) [utilidad, NO en flujo principal]
  4. Preparar crop para clasificación (grayscale continuo, NO binario)

Principio fundamental:
  NUNCA binarizar. El modelo fue entrenado con imágenes de gradientes
  continuos (0-255). La binarización destruye información que el modelo
  necesita para distinguir caracteres.

Formatos de salida:
  - clean_for_detection()           → BGR 3ch, sin líneas de color
                                      ⚠ UTILIDAD SOLAMENTE — NO usar en flujo principal.
                                      YOLO debe recibir la imagen ORIGINAL.
  - clean_crop_for_classification() → Grayscale uint8, fondo~blanco, trazo~negro
                                      Valores CONTINUOS (no binarios)
  - clean_crop_for_display()        → BGR 3ch, limpio para UI

Cambios respecto a la versión anterior:
  - _build_color_line_mask() ahora EXCLUYE píxeles de tinta (grafito)
  - clean_crop_for_classification() SIEMPRE usa inpainting (nunca blanco puro)
  - clean_crop_for_classification() valida post-limpieza (fallback a original)
  - _normalize_background_to_white() protege contra borrar todo el contraste
  - Parámetro de agresividad configurable
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE RANGOS HSV PARA LÍNEAS DE COLOR
# ═════════════════════════════════════════════════════════════════════════════

# Cada rango es (lower_hsv, upper_hsv)
# Cubren líneas azules, rojas y verdes típicas de cuadernos escolares
#
# IMPORTANTE: Los rangos son CONSERVADORES para NO capturar grafito.
# El grafito/lápiz tiene saturación muy baja (< 30), estos rangos
# empiezan en saturación >= 40 para evitar borrar trazos.

HSV_LINE_RANGES: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = [
    # ── Azul claro (líneas de cuaderno típicas) ──
    ((90, 40, 60), (135, 255, 255)),

    # ── Azul oscuro ──
    ((100, 30, 30), (130, 255, 200)),

    # ── Rojo (cuadernos con margen rojo) — rango bajo ──
    ((0, 50, 50), (10, 255, 255)),

    # ── Rojo — rango alto ──
    ((165, 50, 50), (180, 255, 255)),

    # ── Verde (algunos cuadernos) ──
    ((35, 40, 50), (85, 255, 255)),
]

# ── Rangos HSV para detectar TINTA / GRAFITO (lápiz) ──
# El grafito tiene saturación muy baja y luminosidad baja-media.
# Estos píxeles deben PROTEGERSE y nunca borrarse como línea.
INK_SAT_MAX = 40       # Saturación máxima para considerar grafito
INK_VALUE_MAX = 150     # Value máximo para considerar grafito (oscuro)

# Dilatación de la máscara de líneas para cubrir bordes difusos
HSV_MASK_DILATE_K = 3  # kernel size (0 = sin dilatación)

# Tamaño mínimo de componente para considerar como línea (no ruido)
MIN_LINE_COMPONENT_AREA = 200

# ── Parámetro de agresividad de limpieza ──
# 0.0 = sin limpieza, 1.0 = máxima agresividad
# Controla la dilatación de la máscara de líneas y el umbral de aplicación.
# Default conservador: proteger el trazo es más importante que eliminar líneas.
CLEANING_AGGRESSIVENESS: float = 0.5

# ── Validación post-limpieza ──
# Si el contraste (std) del grayscale resultante es menor que este umbral,
# la limpieza borró demasiado y se usa el original.
MIN_CONTRAST_AFTER_CLEAN = 10.0


# ═════════════════════════════════════════════════════════════════════════════
# 1. DETECCIÓN Y ELIMINACIÓN DE LÍNEAS DE COLOR
# ═════════════════════════════════════════════════════════════════════════════

def _build_color_line_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Construye máscara de líneas de color EXCLUYENDO píxeles de tinta.

    Pasos:
      1. Convertir a HSV
      2. Crear máscara de píxeles de color (azul, rojo, verde) por rangos HSV
      3. Crear máscara de píxeles de tinta (baja saturación + oscuros)
      4. Restar: mask_final = mask_color AND NOT mask_ink
      5. Dilatar ligeramente (según agresividad)
      6. Filtrar componentes pequeños (ruido)

    La máscara de tinta protege el trazo del alumno incluso cuando
    escribe directamente sobre una línea azul.

    Args:
        img_bgr: imagen BGR original

    Returns:
        Máscara binaria uint8 (255 = línea de color segura de borrar, 0 = no)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ── Paso 2: Máscara de píxeles de color (candidatos a línea) ──
    color_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for (lo, hi) in HSV_LINE_RANGES:
        mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                           np.array(hi, dtype=np.uint8))
        color_mask = cv2.bitwise_or(color_mask, mask)

    # ── Paso 3: Máscara de píxeles de tinta/grafito (PROTEGER) ──
    # Grafito = baja saturación + baja-media luminosidad
    # Estos píxeles son trazo del alumno y NO deben borrarse
    h_ch, s_ch, v_ch = cv2.split(hsv)
    ink_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    ink_mask[(s_ch < INK_SAT_MAX) & (v_ch < INK_VALUE_MAX)] = 255

    # ── Paso 4: Restar tinta de la máscara de color ──
    # Solo borrar donde HAY color Y NO HAY tinta
    safe_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(ink_mask))

    # ── Paso 5: Dilatar para cubrir bordes difusos de las líneas ──
    # La dilatación se escala con la agresividad
    effective_dilate_k = max(1, int(HSV_MASK_DILATE_K * CLEANING_AGGRESSIVENESS * 2))
    if effective_dilate_k > 0 and effective_dilate_k % 2 == 0:
        effective_dilate_k += 1  # Asegurar impar

    if effective_dilate_k >= 3:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (effective_dilate_k, effective_dilate_k)
        )
        safe_mask = cv2.dilate(safe_mask, kernel, iterations=1)

        # Después de dilatar, volver a excluir tinta para no invadir trazos
        safe_mask = cv2.bitwise_and(safe_mask, cv2.bitwise_not(ink_mask))

    # ── Paso 6: Filtrar componentes pequeños (ruido de color, no líneas) ──
    if MIN_LINE_COMPONENT_AREA > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            safe_mask, connectivity=8
        )
        filtered = np.zeros_like(safe_mask)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_LINE_COMPONENT_AREA:
                filtered[labels == i] = 255
        safe_mask = filtered

    return safe_mask


def remove_color_lines(
    img_bgr: np.ndarray,
    use_inpaint: bool = True,
    inpaint_radius: int = 5,
) -> np.ndarray:
    """
    Elimina líneas de color de la imagen usando inpainting.

    SIEMPRE usa inpainting por defecto (en lugar de reemplazo por blanco)
    para preservar la continuidad del trazo cuando el alumno escribe
    sobre una línea azul.

    CONSERVADOR: Solo elimina píxeles con saturación significativa.
    El grafito a lápiz (saturación ~0-20) NO se toca gracias a la
    máscara de tinta en _build_color_line_mask().

    Args:
        img_bgr: imagen BGR original
        use_inpaint: usar inpainting (True) o reemplazo blanco (False)
        inpaint_radius: radio del inpainting

    Returns:
        Imagen BGR con líneas de color eliminadas
    """
    mask = _build_color_line_mask(img_bgr)

    if cv2.countNonZero(mask) == 0:
        return img_bgr.copy()

    if use_inpaint:
        result = cv2.inpaint(img_bgr, mask, inpaintRadius=inpaint_radius,
                             flags=cv2.INPAINT_TELEA)
    else:
        result = img_bgr.copy()
        result[mask > 0] = (255, 255, 255)

    return result


def _has_significant_color_lines(img_bgr: np.ndarray) -> bool:
    """
    Detecta si la imagen tiene líneas de color significativas.
    Útil para decidir si aplicar la limpieza.

    Returns:
        True si > 1% de píxeles son líneas de color
    """
    mask = _build_color_line_mask(img_bgr)
    ratio = float(cv2.countNonZero(mask)) / max(mask.size, 1)
    return ratio > 0.01


# ═════════════════════════════════════════════════════════════════════════════
# 2. NORMALIZACIÓN DE ILUMINACIÓN (SUAVE, SIN BINARIZAR)
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_illumination_soft(
    gray: np.ndarray,
    bg_blur_k: int = 51,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
) -> np.ndarray:
    """
    Normaliza iluminación desigual sin binarizar.

    Pipeline:
      1. Estimar fondo con blur grande (Gaussian)
      2. Dividir imagen por fondo → compensa iluminación desigual
      3. CLAHE suave → mejora contraste local sin saturar
      4. Resultado: grayscale continuo con iluminación uniforme

    IMPORTANTE: Mantiene gradientes del trazo — NO produce valores binarios.

    Args:
        gray: grayscale uint8
        bg_blur_k: kernel del blur para estimar fondo (impar, grande)
        clahe_clip: clip limit de CLAHE (menor = más suave)
        clahe_tile: tamaño del tile de CLAHE

    Returns:
        Grayscale uint8 con iluminación normalizada
    """
    h, w = gray.shape[:2]
    if h < 10 or w < 10:
        return gray.copy()

    # Asegurar kernel impar
    bg_blur_k = bg_blur_k if bg_blur_k % 2 == 1 else bg_blur_k + 1
    # El kernel no puede ser mayor que la imagen
    bg_blur_k = min(bg_blur_k, min(h, w) // 2 * 2 + 1)
    bg_blur_k = max(bg_blur_k, 3)

    # 1. Estimar fondo (superficie de iluminación)
    background = cv2.GaussianBlur(gray, (bg_blur_k, bg_blur_k), 0)

    # 2. Dividir imagen por fondo → compensa gradientes de iluminación
    # Evitar división por cero
    background_safe = np.maximum(background.astype(np.float32), 1.0)
    normalized = (gray.astype(np.float32) / background_safe) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # 3. CLAHE suave para mejorar contraste local
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(clahe_tile, clahe_tile)
    )
    enhanced = clahe.apply(normalized)

    return enhanced


def _normalize_background_to_white(
    gray: np.ndarray,
    target_bg: int = 245,
    percentile_bg: float = 90.0,
) -> np.ndarray:
    """
    Ajusta el fondo para que sea ~blanco sin tocar el trazo.

    Calcula el valor del percentil alto (fondo) y escala linealmente
    para que el fondo quede cerca de 'target_bg'.

    Protección: Si no hay suficiente contraste entre foreground y
    background (< 30 niveles de diferencia), NO escala para evitar
    producir una imagen completamente blanca sin trazo visible.

    Args:
        gray: grayscale uint8
        target_bg: valor objetivo para el fondo (default: 245 ≈ blanco)
        percentile_bg: percentil para estimar el valor del fondo

    Returns:
        Grayscale uint8 con fondo normalizado a ~blanco
    """
    bg_value = float(np.percentile(gray, percentile_bg))
    fg_value = float(np.percentile(gray, 10.0))  # percentil bajo = trazo

    if bg_value < 10:
        # Imagen muy oscura — probablemente invertida o vacía
        return gray.copy()

    # ── PROTECCIÓN: Si no hay contraste suficiente, no escalar ──
    # Esto evita que una imagen donde todo es ~gris (trazo borrado)
    # se escale a todo blanco, perdiendo cualquier resto de trazo.
    if bg_value - fg_value < 30:
        logger.debug(
            f"_normalize_background_to_white: contraste insuficiente "
            f"(bg={bg_value:.0f}, fg={fg_value:.0f}, diff={bg_value - fg_value:.0f}). "
            f"Saltando normalización."
        )
        return gray.copy()

    # Factor de escala para llevar el fondo a target_bg
    scale = target_bg / max(bg_value, 1.0)

    # Aplicar escala lineal (mantiene proporciones de gradiente)
    result = np.clip(gray.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 3. DETECCIÓN DE POLARIDAD (SIMPLIFICADA)
# ═════════════════════════════════════════════════════════════════════════════

def _detect_polarity(gray: np.ndarray) -> str:
    """
    Detecta si la imagen es dark-on-light (normal) o light-on-dark (invertida).

    Método: Comparar la media de los bordes (que deberían ser fondo)
    con la media del centro (que debería tener trazo).

    Returns:
        'dark_on_light' — trazo oscuro sobre fondo claro (normal/correcto)
        'light_on_dark' — trazo claro sobre fondo oscuro (necesita invertir)
        'ambiguous'     — no se puede determinar con certeza
    """
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return 'ambiguous'

    # Media del borde (2px alrededor)
    border = np.concatenate([
        gray[0:2, :].ravel(),       # top
        gray[-2:, :].ravel(),       # bottom
        gray[:, 0:2].ravel(),       # left
        gray[:, -2:].ravel(),       # right
    ])
    border_mean = float(border.mean())

    # Media general
    overall_mean = float(gray.mean())

    # En una imagen normal (fondo blanco, trazo negro):
    #   border_mean >> overall_mean (bordes son fondo = claro)
    #   overall_mean > 127 (mayoría es fondo claro)

    if overall_mean > 170 and border_mean > 180:
        return 'dark_on_light'

    if overall_mean < 80 and border_mean < 60:
        return 'light_on_dark'

    # Ratio: si el borde es mucho más claro que el promedio → dark_on_light
    if border_mean > overall_mean + 30 and border_mean > 150:
        return 'dark_on_light'

    # Si el borde es mucho más oscuro que el promedio → light_on_dark
    if border_mean < overall_mean - 30 and border_mean < 100:
        return 'light_on_dark'

    return 'ambiguous'


def ensure_dark_on_light(gray: np.ndarray) -> np.ndarray:
    """
    Garantiza que la imagen tenga trazo oscuro sobre fondo claro.
    Solo invierte si está MUY seguro de que es light-on-dark.
    En caso de duda, NO invierte (es menos dañino).

    Args:
        gray: grayscale uint8

    Returns:
        Grayscale uint8, garantizado dark-on-light
    """
    polarity = _detect_polarity(gray)

    if polarity == 'light_on_dark':
        logger.debug("ensure_dark_on_light: invirtiendo (light_on_dark detectado)")
        return cv2.bitwise_not(gray)

    # 'dark_on_light' o 'ambiguous' → no invertir
    return gray.copy()


# ═════════════════════════════════════════════════════════════════════════════
# 4. FUNCIONES PÚBLICAS PRINCIPALES
# ═════════════════════════════════════════════════════════════════════════════

def clean_for_detection(img_bgr: np.ndarray) -> np.ndarray:
    """
    Limpia una imagen completa para mejorar la detección YOLO.

    ⚠ NOTA: Esta función se mantiene como UTILIDAD pero NO debe usarse
    en el flujo principal del pipeline. YOLO fue entrenado con fotos de
    cuaderno CON líneas — las líneas no son un problema para YOLO.
    Limpiar antes de YOLO REDUCE detecciones (de 6 a 4 en pruebas).

    YOLO debe recibir SIEMPRE la imagen ORIGINAL sin limpiar.
    La limpieza solo se aplica a los crops individuales DESPUÉS de detección.

    Args:
        img_bgr: imagen BGR original (foto completa del cuaderno)

    Returns:
        Imagen BGR limpia, mismas dimensiones
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    # Solo limpiar si hay líneas de color significativas
    if _has_significant_color_lines(img_bgr):
        cleaned = remove_color_lines(img_bgr, use_inpaint=True)
        logger.debug("clean_for_detection: líneas de color eliminadas (inpainting)")
        return cleaned

    logger.debug("clean_for_detection: sin líneas de color significativas")
    return img_bgr.copy()


def clean_crop_for_classification(
    crop_bgr: np.ndarray,
    remove_lines: bool = True,
    normalize_illumination: bool = True,
    normalize_background: bool = True,
    fix_polarity: bool = True,
    aggressiveness: Optional[float] = None,
) -> np.ndarray:
    """
    Limpia un crop de carácter (de YOLO) para alimentar al clasificador.

    Pipeline:
      1. Quitar líneas de color residuales con INPAINTING (preserva trazos)
      2. Convertir a grayscale
      3. Normalizar iluminación (sin binarizar)
      4. Asegurar polaridad dark-on-light
      5. Normalizar fondo a ~blanco
      6. Validar que el resultado tiene contenido (fallback a original)
      7. Resultado: grayscale continuo, fondo~245, trazo~0-80

    SIEMPRE usa inpainting en vez de reemplazo por blanco.
    Esto preserva la continuidad del trazo cuando el alumno escribe
    sobre una línea azul del cuaderno.

    Si la imagen no tiene líneas de color significativas (< 0.5%),
    skip la limpieza completamente para evitar artefactos innecesarios.

    CRÍTICO: Este formato coincide con las imágenes de entrenamiento
    EMNIST (grayscale, fondo blanco, trazo negro, gradientes suaves).

    Args:
        crop_bgr: crop BGR del carácter detectado por YOLO
        remove_lines: quitar líneas de color
        normalize_illumination: normalizar iluminación desigual
        normalize_background: ajustar fondo a ~blanco
        fix_polarity: asegurar dark-on-light
        aggressiveness: override de agresividad [0.0-1.0] (None = usar global)

    Returns:
        Grayscale uint8, fondo~blanco(245), trazo~negro(0-80),
        valores CONTINUOS (no binarios)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        logger.warning("clean_crop_for_classification: crop vacío")
        return np.full((128, 128), 245, dtype=np.uint8)

    h, w = crop_bgr.shape[:2]
    if h < 3 or w < 3:
        logger.warning(f"clean_crop_for_classification: crop muy pequeño ({w}x{h})")
        return np.full((128, 128), 245, dtype=np.uint8)

    img = crop_bgr.copy()

    # ── Paso 1: Quitar líneas de color residuales con INPAINTING ──
    if remove_lines:
        mask = _build_color_line_mask(img)
        line_ratio = float(cv2.countNonZero(mask)) / max(mask.size, 1)

        if line_ratio >= 0.005:
            # SIEMPRE usar inpainting — rellena con textura circundante,
            # preservando el trazo incluso donde cruza una línea azul.
            # El radio de inpainting se ajusta según agresividad.
            eff_aggr = aggressiveness if aggressiveness is not None else CLEANING_AGGRESSIVENESS
            inpaint_r = max(3, int(3 + 4 * eff_aggr))  # 3-7 según agresividad
            img = cv2.inpaint(img, mask, inpaintRadius=inpaint_r,
                              flags=cv2.INPAINT_TELEA)

            logger.debug(
                f"clean_crop: líneas eliminadas con inpainting "
                f"({line_ratio:.1%} de píxeles, radius={inpaint_r})"
            )
        else:
            logger.debug(
                f"clean_crop: líneas de color insignificantes "
                f"({line_ratio:.3%}), saltando limpieza"
            )

    # ── Paso 2: Convertir a grayscale ──
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # ── Paso 3: Normalizar iluminación ──
    if normalize_illumination:
        # Solo aplicar si hay variación significativa de iluminación
        blur_k = min(31, max(3, min(h, w) // 2 * 2 + 1))
        if blur_k % 2 == 0:
            blur_k += 1
        local_std = cv2.GaussianBlur(
            cv2.absdiff(gray, cv2.GaussianBlur(gray, (blur_k, blur_k), 0)),
            (blur_k, blur_k), 0
        )
        illumination_variance = float(local_std.std())

        if illumination_variance > 15:
            gray = _normalize_illumination_soft(gray)
            logger.debug(
                f"clean_crop: iluminación normalizada "
                f"(variance={illumination_variance:.1f})"
            )

    # ── Paso 4: Asegurar polaridad dark-on-light ──
    if fix_polarity:
        gray = ensure_dark_on_light(gray)

    # ── Paso 5: Normalizar fondo a ~blanco ──
    if normalize_background:
        gray = _normalize_background_to_white(gray, target_bg=245)

    # ── Paso 6: Validación post-limpieza ──
    # Si la limpieza destruyó el contenido (todo gris uniforme),
    # volver al grayscale original sin limpieza como fallback.
    contrast = float(gray.std())
    if contrast < MIN_CONTRAST_AFTER_CLEAN:
        logger.warning(
            f"clean_crop: limpieza borró el trazo (contrast={contrast:.1f} < "
            f"{MIN_CONTRAST_AFTER_CLEAN}). Usando grayscale original como fallback."
        )
        # Fallback: grayscale del crop original, con polaridad y fondo corregidos
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr.copy()
        if fix_polarity:
            gray = ensure_dark_on_light(gray)
        if normalize_background:
            gray = _normalize_background_to_white(gray, target_bg=245)

    return gray


def clean_crop_for_display(
    crop_bgr: np.ndarray,
    target_size: int = 128,
) -> np.ndarray:
    """
    Limpia un crop para mostrar en la UI.
    Usa inpainting para eliminar líneas, devuelve BGR para visualización.

    Args:
        crop_bgr: crop BGR del carácter
        target_size: tamaño de salida

    Returns:
        BGR uint8, limpio, para mostrar al usuario
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.full((target_size, target_size, 3), 255, dtype=np.uint8)

    # Limpiar líneas de color con inpainting (preserva trazos)
    cleaned = remove_color_lines(crop_bgr, use_inpaint=True)

    # Resize manteniendo aspect ratio
    h, w = cleaned.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_size, target_size, 3), 255, dtype=np.uint8)

    scale = target_size / max(h, w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(cleaned, (new_w, new_h), interpolation=interp)

    canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas


# ═════════════════════════════════════════════════════════════════════════════
# 5. UTILIDADES PÚBLICAS
# ═════════════════════════════════════════════════════════════════════════════

def set_cleaning_aggressiveness(value: float) -> None:
    """
    Ajusta la agresividad de limpieza global sin recompilar.

    Args:
        value: 0.0 (mínima) a 1.0 (máxima). Default: 0.5
    """
    global CLEANING_AGGRESSIVENESS
    CLEANING_AGGRESSIVENESS = max(0.0, min(1.0, float(value)))
    logger.info(f"Agresividad de limpieza ajustada a {CLEANING_AGGRESSIVENESS:.2f}")


def get_cleaning_info(img_bgr: np.ndarray) -> dict:
    """
    Devuelve información diagnóstica sobre la imagen.
    Útil para debugging.

    Returns:
        dict con métricas de la imagen
    """
    if img_bgr is None or img_bgr.size == 0:
        return {"error": "imagen vacía"}

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mask = _build_color_line_mask(img_bgr)
    line_ratio = float(cv2.countNonZero(mask)) / max(mask.size, 1)

    polarity = _detect_polarity(gray)

    return {
        "width": w,
        "height": h,
        "gray_mean": round(float(gray.mean()), 1),
        "gray_std": round(float(gray.std()), 1),
        "color_line_ratio": round(line_ratio, 4),
        "has_color_lines": line_ratio > 0.01,
        "polarity": polarity,
        "percentile_5": round(float(np.percentile(gray, 5)), 1),
        "percentile_95": round(float(np.percentile(gray, 95)), 1),
        "cleaning_aggressiveness": CLEANING_AGGRESSIVENESS,
    }