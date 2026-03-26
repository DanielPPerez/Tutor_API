import numpy as np
from app.core.config import MAX_POINTS_TRAJECTORY, DTW_BAND_RATIO


def get_sequence_from_skel(skel):
    """Convierte el esqueleto en una lista de puntos (row, col) ordenados por ángulo."""
    points = np.argwhere(skel > 0)
    if len(points) == 0:
        return np.array([]).reshape(0, 2)
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 0] - center[0], points[:, 1] - center[1])
    return points[np.argsort(angles)]


def _subsample_sequence(points, max_points):
    """Submuestrea una secuencia a como máximo max_points (uniforme por índices)."""
    n = len(points)
    if n <= max_points:
        return points
    indices = np.linspace(0, n - 1, max_points, dtype=int)
    return points[indices]


def _dtw_band(seq_p, seq_a, band_ratio):
    """
    DTW con ventana de Sakoe-Chiba: solo se rellena una banda |i - j| <= band.
    Memoria O(n * band). Devuelve distancia DTW normalizada (media por paso).
    """
    n, m = len(seq_p), len(seq_a)
    if n == 0 or m == 0:
        return 999.0
    band = max(1, int(max(n, m) * band_ratio))
    inf_val = 1e9
    D = {}
    D[0, 0] = float(np.linalg.norm(seq_p[0] - seq_a[0]))
    for i in range(n):
        for j in range(max(0, i - band), min(m, i + band + 1)):
            if i == 0 and j == 0:
                continue
            d_ij = float(np.linalg.norm(seq_p[i] - seq_a[j]))
            candidates = []
            if (i - 1, j) in D:
                candidates.append(D[i - 1, j])
            if (i, j - 1) in D:
                candidates.append(D[i, j - 1])
            if (i - 1, j - 1) in D:
                candidates.append(D[i - 1, j - 1])
            D[i, j] = d_ij + (min(candidates) if candidates else inf_val)
    if (n - 1, m - 1) not in D:
        best = inf_val
        for j in range(max(0, (n - 1) - band), min(m, (n - 1) + band + 1)):
            if (n - 1, j) in D:
                best = min(best, D[n - 1, j])
        for i in range(max(0, (m - 1) - band), min(n, (m - 1) + band + 1)):
            if (i, m - 1) in D:
                best = min(best, D[i, m - 1])
        if best >= inf_val:
            return 999.0
        return best / max(n, m)
    return D[n - 1, m - 1] / max(n, m)


def calculate_trajectory_dist(skel_p, skel_a):
    """
    Distancia de trayectoria con submuestreo (menos puntos) y DTW con banda,
    sin materializar la matriz N×M completa.
    """
    seq_p = get_sequence_from_skel(skel_p)
    seq_a = get_sequence_from_skel(skel_a)
    if len(seq_p) == 0 or len(seq_a) == 0:
        return 999.0

    seq_p = _subsample_sequence(seq_p, MAX_POINTS_TRAJECTORY)
    seq_a = _subsample_sequence(seq_a, MAX_POINTS_TRAJECTORY)
    dtw_dist = _dtw_band(seq_p, seq_a, DTW_BAND_RATIO)

    if np.isnan(dtw_dist) or np.isinf(dtw_dist):
        return 999.0
    return float(round(dtw_dist, 2))