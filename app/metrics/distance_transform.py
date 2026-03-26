"""
dt_fidelity.py
==============
Metrica de fidelidad basada en Distance Transform (DT).

Logica del pipeline completo:

  Plantilla (esqueleto 1 px)  ──► distanceTransform ──► mapa de distancias
                                                              │
  Trazo del alumno (masa binaria) ────────────────────────────┘
       │                                                      │
       └──► pixeles donde el alumno escribio ────► distancias en esos puntos
                                                              │
                                                    TOLERANCIA (px)
                                                              │
                                              error = max(0, dist - tolerancia)
                                                              │
                                                    score 0-100

Ademas genera:
  - heatmap_bgr : mapa visual de calor (verde=dentro del carril, rojo=fuera)
  - coverage    : fraccion del esqueleto "cubierto" por el trazo del alumno

Todos los parametros vienen de config.py para que el docente pueda
ajustar la dificultad sin tocar codigo.
"""

import cv2
import numpy as np
from app.core import config


# =============================================================================
# Mapa de distancias (cacheado externamente si se llama varias veces)
# =============================================================================

def build_distance_map(skeleton_template: np.ndarray) -> np.ndarray:
    """
    Construye el mapa de distancias euclidianas desde el esqueleto de la plantilla.

    Cada pixel del mapa indica cuantos pixeles de distancia hay hasta la
    linea guia mas cercana. Pixeles sobre la linea => distancia 0.

    Parameters
    ----------
    skeleton_template : np.ndarray  uint8 {0,255}
        Esqueleto de 1 px de la plantilla ideal.

    Returns
    -------
    np.ndarray  float32  — mismo tamano que la entrada.
    """
    # distanceTransform trabaja sobre el FONDO (pixeles a 0).
    # La linea guia (255) debe ser el objeto: invertimos para que el fondo
    # sea blanco y la linea negra, como pide la funcion.
    line_bin = (skeleton_template > 0).astype(np.uint8) * 255
    inv      = cv2.bitwise_not(line_bin)
    dist_map = cv2.distanceTransform(inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist_map


# =============================================================================
# Cobertura del esqueleto
# =============================================================================

def _coverage_ratio(skeleton: np.ndarray, student_mass: np.ndarray,
                    tolerance: float) -> float:
    """
    Fraccion del esqueleto de la plantilla que queda "cubierta" por el trazo
    del alumno dentro de un radio de tolerancia.

    Un valor alto (>0.8) indica que el alumno siguio todo el trayecto de la letra.
    Un valor bajo (<0.5) indica que hay partes de la letra que no trazo.
    """
    skel_pts = np.argwhere(skeleton > 0)
    if len(skel_pts) == 0:
        return 0.0

    # Dilatar la masa del alumno con radio = tolerancia para simular la zona valida
    radius = max(1, int(tolerance))
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate((student_mass > 0).astype(np.uint8) * 255, k, iterations=1)

    covered = np.sum(dilated[skel_pts[:, 0], skel_pts[:, 1]] > 0)
    return float(round(covered / len(skel_pts), 4))


# =============================================================================
# Heatmap visual
# =============================================================================

def _build_heatmap(dist_map: np.ndarray, student_mass: np.ndarray,
                   tolerance: float) -> np.ndarray:
    """
    Genera un mapa de calor BGR para visualizacion:
      - Verde  : trazo del alumno dentro de la tolerancia (bien)
      - Amarillo: trazo del alumno ligeramente fuera (advertencia)
      - Rojo   : trazo del alumno muy fuera (error)
      - Gris   : zona de la plantilla no cubierta
    """
    h, w   = dist_map.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    user_mask = student_mass > 0
    distances = dist_map[user_mask]

    # Clasificar por zona de error
    inside   = user_mask & (dist_map <= tolerance)
    warning  = user_mask & (dist_map > tolerance) & (dist_map <= tolerance * 2)
    error_z  = user_mask & (dist_map > tolerance * 2)

    heatmap[inside]  = (0,   200, 0)    # Verde
    heatmap[warning] = (0,   200, 220)  # Amarillo (BGR)
    heatmap[error_z] = (0,   0,   220)  # Rojo

    return heatmap


# =============================================================================
# Metrica principal
# =============================================================================

def calculate_dt_fidelity(
    skeleton_template: np.ndarray,
    student_mass: np.ndarray,
    level: str = "intermedio"
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Calcula la fidelidad del trazo del alumno respecto al esqueleto ideal.

    Estrategia dual de puntuacion:
      - score_precision : penaliza cada pixel del alumno que cae FUERA del carril
      - score_coverage  : penaliza las zonas del esqueleto que el alumno NO trazo

    La nota final combina ambas para premiar tanto la precision como la
    completitud del trazo (importante en pedagogia infantil).

    Parameters
    ----------
    skeleton_template : np.ndarray  uint8 {0,255}
        Esqueleto de 1 px de la plantilla.
    student_mass : np.ndarray  uint8 {0,255}
        Trazo binarizado del alumno (masa, NO esqueleto).
    level : str
        Nivel de dificultad para elegir la tolerancia desde config.
        Valores: "principiante", "intermedio", "avanzado".

    Returns
    -------
    score_final : float   0-100  (nota combinada)
    coverage    : float   0-1    (fraccion del esqueleto cubierto)
    dist_map    : np.ndarray float32  (mapa de distancias, para debug)
    heatmap_bgr : np.ndarray uint8 BGR (visualizacion de errores)
    """
    # ── Tolerancia segun nivel ────────────────────────────────────────────────
    tolerance = config.DT_TOLERANCE_BY_LEVEL.get(level, config.DT_TOLERANCE_DEFAULT)

    # ── Mapa de distancias ────────────────────────────────────────────────────
    dist_map = build_distance_map(skeleton_template)

    # ── Puntuacion de precision ───────────────────────────────────────────────
    user_idx = np.where(student_mass > 0)
    if len(user_idx[0]) == 0:
        empty_heatmap = np.zeros((*skeleton_template.shape, 3), dtype=np.uint8)
        return 0.0, 0.0, dist_map, empty_heatmap

    distances = dist_map[user_idx].astype(np.float64)
    errors    = np.maximum(0.0, distances - tolerance)
    avg_error = float(np.mean(errors))

    # Factor de castigo: un error promedio de DT_MAX_AVG_ERROR => score 0
    max_err       = config.DT_MAX_AVG_ERROR
    score_precision = max(0.0, 100.0 * (1.0 - avg_error / max_err))

    # ── Cobertura del esqueleto ───────────────────────────────────────────────
    coverage      = _coverage_ratio(skeleton_template, student_mass, tolerance)
    score_coverage = coverage * 100.0

    # ── Nota final: combinacion ponderada ─────────────────────────────────────
    # Precision (donde escribe el alumno) + Cobertura (que trazo completo)
    w_prec = config.DT_WEIGHT_PRECISION
    w_cov  = config.DT_WEIGHT_COVERAGE
    score_final = w_prec * score_precision + w_cov * score_coverage

    # ── Heatmap visual ────────────────────────────────────────────────────────
    heatmap_bgr = _build_heatmap(dist_map, student_mass, tolerance)

    return (
        float(round(score_final,    2)),
        float(round(coverage,       4)),
        dist_map,
        heatmap_bgr,
    )


# =============================================================================
# Metrica de comparacion esqueleto vs esqueleto (linea contra linea)
# =============================================================================

def calculate_skeleton_fidelity(
    skeleton_template: np.ndarray,
    skeleton_student: np.ndarray,
    level: str = "intermedio"
) -> float:
    """
    Variante de DT donde AMBAS imagenes son esqueletos de 1 px.
    Proporciona la mayor precision posible para usuarios avanzados.

    Usa el mismo mapa de distancias pero sobre el esqueleto del alumno.

    Parameters
    ----------
    skeleton_template : uint8 {0,255} — esqueleto plantilla
    skeleton_student  : uint8 {0,255} — esqueleto del trazo del alumno
    level             : str

    Returns
    -------
    float  0-100
    """
    tolerance = config.DT_TOLERANCE_BY_LEVEL.get(level, config.DT_TOLERANCE_DEFAULT)
    dist_map  = build_distance_map(skeleton_template)

    skel_pts  = np.where(skeleton_student > 0)
    if len(skel_pts[0]) == 0:
        return 0.0

    distances = dist_map[skel_pts].astype(np.float64)
    errors    = np.maximum(0.0, distances - tolerance)
    avg_error = float(np.mean(errors))

    score = max(0.0, 100.0 * (1.0 - avg_error / config.DT_MAX_AVG_ERROR))
    return float(round(score, 2))