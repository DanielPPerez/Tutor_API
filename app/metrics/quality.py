"""
quality.py
==========
Métricas de calidad intrínseca del trazo del alumno.

calculate_quality_metrics(img_a) -> dict con:
  stroke_density    : float  [0-1]  — fracción de píxeles activos / total del canvas
  stroke_continuity : float  [0-1]  — qué tan continuo es el trazo (1 componente = 1.0)
  thickness_mean    : float         — grosor medio del trazo en píxeles
  thickness_std     : float         — variación del grosor (alta = presión irregular)
  bounding_fill     : float  [0-1]  — fracción del bounding box ocupada por el trazo
  smoothness        : float  [0-1]  — suavidad de los bordes (alta = trazo limpio)

Compatibilidad:
  - Entrada: np.ndarray uint8 {0,255} — masa binaria del trazo normalizado
  - Salida:  dict JSON-serializable (todos float / int nativos de Python)
  - Tolerante a imagen vacía
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


# =============================================================================
# Sub-métricas
# =============================================================================

def _stroke_density(bin_img: np.ndarray) -> float:
    """Fracción de píxeles de trazo sobre el total del canvas."""
    total = bin_img.size
    if total == 0:
        return 0.0
    return float(round(np.sum(bin_img > 0) / total, 4))


def _stroke_continuity(bin_img: np.ndarray) -> float:
    """
    Mide qué tan continuo es el trazo.

    Si hay 1 componente conectado → 1.0 (trazo perfecto).
    Si hay N componentes           → 1 / N (fragmentado).
    Penaliza letras trazadas en pedazos.
    """
    n_labels, _ = cv2.connectedComponents(bin_img, connectivity=8)
    n_components = max(1, n_labels - 1)   # restar fondo
    return float(round(1.0 / n_components, 4))


def _thickness_stats(bin_img: np.ndarray) -> tuple[float, float]:
    """
    Estima el grosor local del trazo usando la Distancia Transform sobre el
    fondo y muestreando los valores en los píxeles activos.

    thickness_mean ≈ radio medio del trazo en píxeles.
    thickness_std  ≈ variación de la presión del lápiz.
    """
    if np.sum(bin_img > 0) == 0:
        return 0.0, 0.0

    # distanceTransform sobre la IMAGEN BINARIA (1 = trazo, 0 = fondo)
    # Da el radio máximo de una circunferencia inscrita en el trazo en cada px.
    dist = distance_transform_edt(bin_img > 0).astype(np.float32)

    # Solo los píxeles activos
    vals = dist[bin_img > 0]

    mean = float(round(float(np.mean(vals)),  3))
    std  = float(round(float(np.std(vals)),   3))
    return mean, std


def _bounding_fill(bin_img: np.ndarray) -> float:
    """
    Proporción del bounding box ocupada por el trazo.
    Un valor cercano a 1 indica que el trazo rellena bien la letra.
    Un valor muy bajo indica un trazo muy delgado o letra incompleta.
    """
    coords = cv2.findNonZero(bin_img)
    if coords is None:
        return 0.0
    _, _, w, h = cv2.boundingRect(coords)
    bbox_area = w * h
    if bbox_area == 0:
        return 0.0
    stroke_area = float(np.sum(bin_img > 0))
    return float(round(stroke_area / bbox_area, 4))


def _smoothness(bin_img: np.ndarray) -> float:
    """
    Suavidad de los bordes del trazo.

    Método: perimeter / (2 * sqrt(pi * area)).
    Para un círculo perfecto el resultado es 1.0 (máximo suavidad).
    Cuanto más irregular el borde, más alto es el valor → invertimos para
    que 1.0 = trazo muy suave y 0.0 = trazo muy rugoso.

    Retorna float [0-1] clampado.
    """
    area = float(np.sum(bin_img > 0))
    if area < 4:
        return 0.0

    # Calcular el perímetro usando el número de transiciones fondo→trazo
    # (erosión y resta — equivale al contorno interior)
    k       = np.ones((3, 3), dtype=np.uint8)
    eroded  = cv2.erode(bin_img, k, iterations=1)
    border  = bin_img - eroded
    perimeter = float(np.sum(border > 0))

    if perimeter == 0:
        return 1.0

    # Índice de circularidad: ratio compacidad (=1 para círculo, <1 para formas irregulares)
    circularity = (4.0 * np.pi * area) / (perimeter ** 2)
    # Clampamos entre 0 y 1 (puede superar 1 en formas muy compactas por cuantización)
    return float(round(min(1.0, max(0.0, circularity)), 4))


# =============================================================================
# API pública
# =============================================================================

def calculate_quality_metrics(img_a: np.ndarray) -> dict:
    """
    Calcula métricas de calidad intrínseca del trazo del alumno.

    Parameters
    ----------
    img_a : np.ndarray  uint8 {0,255}
        Masa binaria del trazo normalizado (salida de normalize_character).
        NO debe ser el esqueleto.

    Returns
    -------
    dict con claves float/int JSON-serializables:
        stroke_density    : fracción de canvas cubierta por el trazo
        stroke_continuity : continuidad (1 componente → 1.0)
        thickness_mean    : radio medio del trazo (px)
        thickness_std     : variación de grosor (baja = presión uniforme)
        bounding_fill     : fracción del bounding box cubierta
        smoothness        : suavidad de bordes [0-1]
    """
    if img_a is None or img_a.size == 0 or np.sum(img_a > 0) == 0:
        return {
            "stroke_density":    0.0,
            "stroke_continuity": 0.0,
            "thickness_mean":    0.0,
            "thickness_std":     0.0,
            "bounding_fill":     0.0,
            "smoothness":        0.0,
        }

    bin_img = (img_a > 0).astype(np.uint8) * 255

    t_mean, t_std = _thickness_stats(bin_img)

    return {
        "stroke_density":    _stroke_density(bin_img),
        "stroke_continuity": _stroke_continuity(bin_img),
        "thickness_mean":    t_mean,
        "thickness_std":     t_std,
        "bounding_fill":     _bounding_fill(bin_img),
        "smoothness":        _smoothness(bin_img),
    }