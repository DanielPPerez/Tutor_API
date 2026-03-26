import numpy as np
from app.metrics.trajectory import get_sequence_from_skel

# Número de segmentos en que se divide el esqueleto (ordenado por ángulo)
N_SEGMENTS = 12

def _segment_direction(points):
    """
    Dado un array de puntos (n, 2), devuelve un vector dirección unitario
    (de inicio a fin del segmento). Si el segmento es degenerado, devuelve None.
    """
    if len(points) < 2:
        return None
    start = points[0]
    end = points[-1]
    v = end - start
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return None
    return v / norm


def get_segment_vectors(skel, n_segments=N_SEGMENTS):
    """
    Ordena los puntos del esqueleto por ángulo respecto al centroide,
    los divide en n_segments segmentos y devuelve un vector dirección (2,) por segmento.
    """
    points = get_sequence_from_skel(skel)
    if len(points) < 2:
        return []
    n = len(points)
    vectors = []
    seg_size = max(1, n // n_segments)
    for i in range(n_segments):
        lo = i * seg_size
        hi = min((i + 1) * seg_size, n)
        if hi <= lo:
            continue
        seg = points[lo:hi]
        d = _segment_direction(seg)
        if d is not None:
            vectors.append(d)
    return vectors


def calculate_segment_cosine_similarity(skel_p, skel_a, n_segments=N_SEGMENTS):
    """
    Divide ambos esqueletos en n_segments y compara el ángulo de cada segmento
    con similitud de coseno: S(A,B) = cos(θ) = (A·B)/(||A|| ||B||).
    Devuelve un valor en [0, 100] (promedio de cosenos mapeado de [-1,1] a [0,100]).
    """
    vecs_p = get_segment_vectors(skel_p, n_segments)
    vecs_a = get_segment_vectors(skel_a, n_segments)
    if not vecs_p or not vecs_a:
        return 0.0, 50.0  # neutral si no hay segmentos
    # Ajustar cantidad: usar el mínimo de segmentos válidos
    k = min(len(vecs_p), len(vecs_a))
    vecs_p = np.array(vecs_p[:k])
    vecs_a = np.array(vecs_a[:k])
    # S(A,B) = (A·B)/(||A|| ||B||); ya son unitarios -> A·B
    cosines = np.sum(vecs_p * vecs_a, axis=1)
    cosines = np.clip(cosines, -1.0, 1.0)
    mean_cos = float(np.mean(cosines))
    # [-1, 1] -> [0, 100]
    score = (mean_cos + 1) / 2 * 100
    return float(round(mean_cos, 4)), float(round(score, 2))
