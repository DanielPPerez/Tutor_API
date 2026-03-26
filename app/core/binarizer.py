"""
app/core/binarizer.py
=====================
Responsabilidad única (SRP): convertir una imagen en escala de grises
normalizada en una máscara binaria (trazo=255, fondo=0).

Por qué va en app/core/:
  - Es una etapa central y reutilizable del pipeline de preprocesamiento.
  - normalizer.py y debug_and_refine_roi.py la usan.
  - Al estar separada se puede cambiar el algoritmo (ej. agregar método
    Niblack o Sauvola) sin tocar normalizer.py → OCP.

Exports públicos
----------------
  binarize(gray, params)  -> np.ndarray  (función principal adaptativa)
  binarize_otsu(gray)     -> np.ndarray  (Otsu puro, para debug)
  binarize_adaptive(gray, block_size, c) -> np.ndarray
  binarize_sauvola(gray, window, k)      -> np.ndarray  (robusto en papel texturizado)
"""

from __future__ import annotations

import cv2
import numpy as np

# Importación opcional de scikit-image para Sauvola
try:
    from skimage.filters import threshold_sauvola
    _SAUVOLA_OK = True
except ImportError:
    _SAUVOLA_OK = False


# =============================================================================
# Métodos individuales
# =============================================================================

def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Umbral global de Otsu.
    Mejor cuando: buena luz, alto contraste, sin sombras.
    Devuelve THRESH_BINARY_INV: trazo=255, fondo=0.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def binarize_adaptive(
    gray:       np.ndarray,
    block_size: int   = 11,
    c:          int   = 2,
) -> np.ndarray:
    """
    Umbral adaptativo Gaussiano.
    Mejor cuando: iluminación desigual, sombras moderadas.
    block_size debe ser impar y >= 3.
    Devuelve THRESH_BINARY_INV: trazo=255, fondo=0.
    """
    bs = max(3, block_size if block_size % 2 == 1 else block_size + 1)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs,
        c,
    )


def binarize_sauvola(
    gray:   np.ndarray,
    window: int   = 25,
    k:      float = 0.2,
) -> np.ndarray:
    """
    Umbral local de Sauvola.
    Mejor cuando: papel texturizado (cuaderno cuadriculado), iluminación muy
    variable, presión de lápiz muy irregular.

    Requiere scikit-image. Si no está instalado, cae a adaptativo.

    El método de Sauvola calcula un umbral local como:
        T = mean * (1 + k * (std/R - 1))
    donde R es el rango dinámico (típicamente 128). Esto lo hace más robusto
    al ruido de papel que el adaptativo Gaussiano.
    """
    if not _SAUVOLA_OK:
        return binarize_adaptive(gray, block_size=window, c=2)

    thresh  = threshold_sauvola(gray, window_size=window, k=k)
    binary  = (gray < thresh).astype(np.uint8) * 255   # trazo oscuro → 255
    return binary


# =============================================================================
# Función principal adaptativa
# =============================================================================

def binarize(
    gray:       np.ndarray,
    use_otsu:   bool  = False,
    block_size: int   = 11,
    adaptive_c: int   = 2,
    use_sauvola: bool = False,
    contrast:   float = 0.0,   # std de la imagen; si 0, se calcula aquí
) -> np.ndarray:
    """
    Elige automáticamente el mejor método de binarización según las
    condiciones de la imagen.

    Jerarquía de decisión:
      1. Si use_sauvola y scikit-image disponible → Sauvola
         (papel texturizado, ruido de cuadrícula muy pronunciado)
      2. Si use_otsu → Otsu
         (alto contraste, sin sombras: condición ideal)
      3. Por defecto → Adaptativo Gaussiano con parámetros adaptativos

    Parameters vienen de PipelineParams (image_quality.py), no hardcodeados.

    Returns
    -------
    np.ndarray uint8 {0,255} — trazo=255, fondo=0
    """
    # Calcular contraste si no se pasó
    if contrast <= 0:
        contrast = float(gray.std())

    if use_sauvola and _SAUVOLA_OK:
        return binarize_sauvola(gray)

    if use_otsu:
        return binarize_otsu(gray)

    return binarize_adaptive(gray, block_size=block_size, c=adaptive_c)