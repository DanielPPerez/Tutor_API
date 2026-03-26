"""
geometric.py
============
Metricas geometricas entre dos esqueletos: SSIM, Procrustes y Hausdorff.

CAMBIO RESPECTO A LA VERSION ANTERIOR
--------------------------------------
calculate_geometric() ahora espera ESQUELETOS (1 px de grosor) en AMBAS
entradas, no masas binarias. Esto es coherente con el nuevo pipeline:

  normalizer  -> normalize_character()       -> masa binaria del alumno
  templates   -> skeletonize_student_char()  -> esqueleto del alumno
  templates   -> skeleton/<nombre>.npy * 255 -> esqueleto de la plantilla

Comparar esqueleto vs esqueleto da precision maxima para detectar trazos
erraticos y ausencia de partes de la letra.
"""

import cv2
import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim

from app.core.config import PROCRUSTES_N_POINTS, HAUSDORFF_TOLERANCE, HAUSDORFF_FACTOR


# =============================================================================
# Utilidades
# =============================================================================

def _resample_sequence_to_n(points: np.ndarray, n: int) -> np.ndarray:
    """Remuestrea una secuencia de puntos a exactamente n puntos por interpolacion lineal."""
    if len(points) == 0:
        return np.array([]).reshape(0, 2)
    if len(points) == 1:
        return np.tile(points, (n, 1))

    pts    = np.vstack([points, points[0]])   # Cerrar el contorno
    cumlen = np.zeros(len(pts))
    cumlen[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    total  = cumlen[-1]

    if total == 0:
        return np.tile(points.mean(axis=0), (n, 1))

    target = np.linspace(0, total * (n - 1) / n, n, endpoint=False)
    idx    = np.clip(np.searchsorted(cumlen, target, side="right") - 1, 0, len(pts) - 2)
    t      = (target - cumlen[idx]) / (cumlen[idx + 1] - cumlen[idx] + 1e-9)
    return (1 - t)[:, None] * pts[idx] + t[:, None] * pts[idx + 1]


def align_skeletons(
    skel_p: np.ndarray, skel_a: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Alinea dos esqueletos por sus centroides para compensar pequenos
    desplazamientos de posicion introducidos en el preprocesamiento.

    Si el desplazamiento es menor a 1 px no aplica ninguna transformacion.
    """
    pts_p = np.argwhere(skel_p > 0)
    pts_a = np.argwhere(skel_a > 0)

    if len(pts_p) == 0 or len(pts_a) == 0:
        return skel_p, skel_a

    offset = pts_p.mean(axis=0) - pts_a.mean(axis=0)
    if np.linalg.norm(offset) < 1.0:
        return skel_p, skel_a

    rows, cols = skel_a.shape
    M = np.float32([[1, 0, offset[1]], [0, 1, offset[0]]])
    skel_a_aligned = cv2.warpAffine(
        skel_a.astype(np.uint8), M, (cols, rows),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return skel_p, skel_a_aligned.astype(skel_p.dtype)


# =============================================================================
# Sub-metricas
# =============================================================================

def calculate_procrustes(
    skel_p: np.ndarray, skel_a: np.ndarray,
    seq_p: np.ndarray,  seq_a: np.ndarray
) -> tuple[float, float]:
    """
    Alineacion Procrustes: escala, rotacion y traslacion optimas para minimizar
    la suma de cuadrados de diferencias entre las secuencias de puntos.
    """
    if len(seq_p) < 3 or len(seq_a) < 3:
        return 999.0, 0.0

    pts_p = _resample_sequence_to_n(seq_p, PROCRUSTES_N_POINTS)
    pts_a = _resample_sequence_to_n(seq_a, PROCRUSTES_N_POINTS)

    try:
        _, _, disparity = procrustes(pts_p, pts_a)
    except (ValueError, np.linalg.LinAlgError):
        return 999.0, 0.0

    score = max(0.0, 100.0 - disparity * 50.0)
    return float(round(disparity, 4)), float(round(score, 2))


# =============================================================================
# Metrica principal
# =============================================================================

def calculate_geometric(
    skel_p: np.ndarray,
    skel_a: np.ndarray,
    tolerance_radius: int = 2,
    align: bool = True
) -> dict:
    """
    Calcula metricas geometricas comparando ESQUELETO de plantilla vs
    ESQUELETO del alumno.

    Ambas entradas deben ser uint8 {0,255} con 1 px de grosor (salida de
    skeletonize_binary / skeletonize_student_char).

    Parameters
    ----------
    skel_p           : esqueleto de la plantilla ideal
    skel_a           : esqueleto del trazo del alumno
    tolerance_radius : radio de tolerancia para metricas (px)
    align            : si True, alinea los centroides antes de calcular

    Returns
    -------
    dict con: ssim, ssim_score, procrustes_disparity, procrustes_score,
              hausdorff, score (hausdorff score)
    """
    from app.metrics.trajectory import get_sequence_from_skel

    # Asegurar mismo tamano
    if skel_p.shape != skel_a.shape:
        skel_a = cv2.resize(
            skel_a.astype(np.uint8),
            (skel_p.shape[1], skel_p.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    if align:
        skel_p, skel_a = align_skeletons(skel_p, skel_a)

    # ── SSIM ─────────────────────────────────────────────────────────────────
    # Convertir a float [0,1] para SSIM
    img_p = (skel_p > 0).astype(np.float32)
    img_a = (skel_a > 0).astype(np.float32)

    ssim_val = ssim(img_p, img_a, data_range=1.0)
    if np.isnan(ssim_val):
        ssim_val = 0.0

    # SSIM en [-1, 1] -> score 0-100
    ssim_score = float(round((ssim_val + 1.0) / 2.0 * 100.0, 2))
    ssim_val   = float(round(ssim_val, 4))

    # ── Procrustes ────────────────────────────────────────────────────────────
    seq_p = get_sequence_from_skel(skel_p)
    seq_a = get_sequence_from_skel(skel_a)
    proc_disparity, proc_score = calculate_procrustes(skel_p, skel_a, seq_p, seq_a)

    # ── Hausdorff ─────────────────────────────────────────────────────────────
    pts_p = np.argwhere(skel_p > 0).astype(np.float32)
    pts_a = np.argwhere(skel_a > 0).astype(np.float32)

    if len(pts_p) == 0 or len(pts_a) == 0:
        haus_dist = 999.0
    else:
        d1 = directed_hausdorff(pts_p, pts_a)[0]
        d2 = directed_hausdorff(pts_a, pts_p)[0]
        haus_dist = float(max(d1, d2))

    if np.isinf(haus_dist) or np.isnan(haus_dist):
        haus_dist = 999.0

    adjusted_h  = max(0.0, haus_dist - HAUSDORFF_TOLERANCE)
    score_haus  = max(0.0, 100.0 - adjusted_h * HAUSDORFF_FACTOR)

    return {
        "ssim":                 ssim_val,
        "ssim_score":           ssim_score,
        "procrustes_disparity": proc_disparity,
        "procrustes_score":     proc_score,
        "hausdorff":            float(round(haus_dist, 2)),
        "score":                float(round(score_haus, 2)),
    }