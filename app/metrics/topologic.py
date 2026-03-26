"""
topologic.py
============
Métricas topológicas de un esqueleto de carácter.

get_topology(skel) -> dict con:
  loops       : int   — número de bucles/agujeros (componentes interiores)
  endpoints   : int   — puntas del trazo (vecinos == 1)
  junctions   : int   — bifurcaciones/cruces (vecinos >= 3)
  components  : int   — componentes conectados del esqueleto

Compatibilidad con el resto del pipeline:
  - Entrada: np.ndarray uint8 {0,255} — esqueleto de 1 px
  - Salida:  dict con claves enteras, serializables a JSON sin conversión extra
  - Tolerante a imagen vacía (devuelve dict de ceros)

Nota sobre loops:
  La versión anterior usaba findContours con RETR_CCOMP y contaba los contornos
  hijo (h[3] != -1). Eso es correcto para MASA binaria pero poco fiable en
  esqueletos de 1 px porque el contorno de un esqueleto no tiene "interior".
  La nueva implementación usa connectedComponents sobre el FONDO de la imagen
  binaria para contar cavidades reales, que es el método canónico para esqueletos.
"""

import cv2
import numpy as np
from scipy.ndimage import generic_filter


# =============================================================================
# Conteo de vecinos (Crossing Number)
# =============================================================================

def _neighbor_count_map(skel: np.ndarray) -> np.ndarray:
    """
    Devuelve un mapa donde cada píxel de esqueleto tiene el número de
    vecinos de 8-conectividad también activos.

    Usa generic_filter de scipy con una ventana 3×3:
      center pixel value == 1  →  suma de vecinos = sum(P) - 1
      center pixel value == 0  →  0 (ignorado)
    """
    bin_skel = (skel > 0).astype(np.float32)

    def _count(P):
        # P es la ventana 3×3 aplanada; P[4] es el centro
        return float(np.sum(P) - P[4]) if P[4] > 0 else 0.0

    return generic_filter(bin_skel, _count, size=(3, 3), mode="constant", cval=0)


# =============================================================================
# Conteo de bucles por componentes conectados del fondo
# =============================================================================

def _count_loops(skel: np.ndarray) -> int:
    """
    Cuenta los bucles/agujeros en el esqueleto usando la topología del FONDO.

    Algoritmo:
      1. Dilatar ligeramente el esqueleto para cerrar huecos de 1 px.
      2. Invertir la imagen (fondo → primer plano).
      3. Contar los componentes conectados del fondo que NO tocan el borde.
         Cada uno de esos componentes es un "hueco" encerrado → un bucle.

    Esto es más fiable que RETR_CCOMP para esqueletos de 1 px.
    """
    h, w = skel.shape

    # Dilatar el esqueleto para cerrar posibles microgaps
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate((skel > 0).astype(np.uint8) * 255, k, iterations=1)

    # Binarizar y añadir borde de 1 px de fondo para que el fondo exterior
    # sea siempre una única componente
    padded = cv2.copyMakeBorder(dilated, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Fondo = píxeles a 0
    fondo = (padded == 0).astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(fondo, connectivity=8)

    # La componente que toca el borde (la más grande, generalmente label 1)
    # es el fondo exterior; las demás son huecos encerrados.
    if n_labels <= 1:
        return 0

    # Identificar el label del fondo exterior (el que toca las esquinas)
    corner_labels = {
        int(labels[0, 0]),
        int(labels[0, -1]),
        int(labels[-1, 0]),
        int(labels[-1, -1]),
    }

    # Bucles = componentes de fondo que NO son el fondo exterior
    interior = [l for l in range(1, n_labels) if l not in corner_labels]
    return len(interior)


# =============================================================================
# API pública
# =============================================================================

def get_topology(skel: np.ndarray) -> dict:
    """
    Analiza la topología del esqueleto de un carácter.

    Parameters
    ----------
    skel : np.ndarray  uint8 {0,255}
        Esqueleto de 1 px proveniente de skeletonize_binary() o
        skeletonize_student_char().

    Returns
    -------
    dict con:
        loops       : int  — bucles/agujeros encerrados (ej: 1 para 'A', 2 para 'B')
        endpoints   : int  — puntas del trazo (vecinos == 1)
        junctions   : int  — bifurcaciones / cruces (vecinos >= 3)
        components  : int  — componentes conectados del esqueleto
    """
    # Protección: entrada vacía
    if skel is None or skel.size == 0 or np.sum(skel > 0) == 0:
        return {"loops": 0, "endpoints": 0, "junctions": 0, "components": 0}

    # ── Bucles ───────────────────────────────────────────────────────────────
    loops = _count_loops(skel)

    # ── Puntas y bifurcaciones (Crossing Number) ──────────────────────────────
    nmap      = _neighbor_count_map(skel)
    endpoints  = int(np.sum(nmap == 1))
    junctions  = int(np.sum(nmap >= 3))

    # ── Componentes conectados del propio esqueleto ────────────────────────────
    bin_skel   = (skel > 0).astype(np.uint8)
    n_comp, _  = cv2.connectedComponents(bin_skel, connectivity=8)
    components = max(0, int(n_comp) - 1)   # restar el fondo (label 0)

    return {
        "loops":       int(loops),
        "endpoints":   int(endpoints),
        "junctions":   int(junctions),
        "components":  int(components),
    }