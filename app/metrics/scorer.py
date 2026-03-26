"""
scoring.py
==========
Calculo de la nota final del trazo del alumno.

Metricas que intervienen
------------------------
1. DT Precision   (distance_transform) — Fidelidad pixel a pixel al carril
2. DT Coverage                         — Fraccion del esqueleto cubierto
3. Topologia                           — Coincidencia de bucles/agujeros
4. SSIM                                — Similitud estructural global
5. Procrustes                          — Ajuste geometrico de forma
6. Hausdorff                           — Penalizacion por trazos muy alejados
7. Trayectoria DTW                     — Dinamica del trazo (orden de puntos)
8. Coseno de segmentos                 — Coherencia de angulos

Ponderacion
-----------
El peso de cada metrica esta en config.SCORING_WEIGHTS para que el
equipo pedagogico pueda ajustarlo sin tocar codigo.

Niveles de dificultad
---------------------
Cada nivel activa tolerancias distintas en el DT (config.DT_TOLERANCE_BY_LEVEL).
El calculo de score es identico; solo cambia con cuanta distancia se "perdona"
al alumno en el Distance Transform.
"""

from __future__ import annotations

import numpy as np
from app.core import config


# =============================================================================
# Score de cada metrica individual (normalizadas a 0-100)
# =============================================================================

def _score_dt_precision(dt_score: float) -> float:
    return max(0.0, min(100.0, float(dt_score)))


def _score_dt_coverage(coverage: float) -> float:
    """
    Penaliza fuertemente la falta de cobertura del esqueleto.
    Un alumno que solo trazo la mitad de la letra no puede sacar nota alta.
    """
    return max(0.0, min(100.0, float(coverage) * 100.0))


def _score_topo(topo_match: bool) -> float:
    """
    Penaliza la topologia incorrecta (bucles / agujeros).
    Una 'A' sin triangulo interior o una 'O' abierta bajan la nota.
    """
    return config.SCORING_TOPO_HIT if topo_match else config.SCORING_TOPO_MISS


def _score_ssim(geo_metrics: dict) -> float:
    return max(0.0, min(100.0, geo_metrics.get("ssim_score", 0.0)))


def _score_procrustes(geo_metrics: dict) -> float:
    return max(0.0, min(100.0, geo_metrics.get("procrustes_score", 0.0)))


def _score_hausdorff(geo_metrics: dict) -> float:
    h_dist   = geo_metrics.get("hausdorff", 999.0)
    adjusted = max(0.0, float(h_dist) - config.HAUSDORFF_TOLERANCE)
    return max(0.0, 100.0 - adjusted * config.HAUSDORFF_FACTOR)


def _score_trajectory(traj_dist: float) -> float:
    """
    Convierte la distancia DTW de trayectoria en score.
    Factor: cada unidad de distancia descuenta config.SCORING_TRAJ_FACTOR puntos.
    """
    return max(0.0, 100.0 - float(traj_dist) * config.SCORING_TRAJ_FACTOR)


def _score_cosine(cosine_segment_score: float) -> float:
    return max(0.0, min(100.0, float(cosine_segment_score)))


# =============================================================================
# Nota final ponderada
# =============================================================================

def calculate_final_score(
    geo_metrics: dict,
    topo_match: bool,
    traj_dist: float,
    dt_precision_score: float,
    dt_coverage: float,
    cosine_segment_score: float = 50.0,
    level: str = "intermedio",
) -> dict:
    """
    Calcula la nota final del trazo del alumno.

    Parameters
    ----------
    geo_metrics : dict
        Salida de geometric.calculate_geometric() con claves:
        ssim_score, procrustes_score, hausdorff.
    topo_match : bool
        True si el numero de bucles/agujeros coincide con la plantilla.
    traj_dist : float
        Distancia DTW de trayectoria (trajectory.py).
    dt_precision_score : float  [0-100]
        Score de precision del Distance Transform (dt_fidelity.calculate_dt_fidelity).
    dt_coverage : float  [0-1]
        Fraccion del esqueleto cubierto por el trazo del alumno.
    cosine_segment_score : float  [0-100]
        Coherencia de angulos de segmentos (opcional, default 50).
    level : str
        Nivel de dificultad: "principiante", "intermedio", "avanzado".
        Usado solo para logging / metadata; la tolerancia ya fue aplicada en DT.

    Returns
    -------
    dict con:
        "score_final"      : float  [0-100]  — nota combinada
        "scores_breakdown" : dict            — detalle de cada componente
        "weights_used"     : dict            — pesos aplicados
        "level"            : str
    """
    w = config.SCORING_WEIGHTS

    # Calcular cada componente
    s_dt_prec = _score_dt_precision(dt_precision_score)
    s_dt_cov  = _score_dt_coverage(dt_coverage)
    s_topo    = _score_topo(topo_match)
    s_ssim    = _score_ssim(geo_metrics)
    s_proc    = _score_procrustes(geo_metrics)
    s_haus    = _score_hausdorff(geo_metrics)
    s_traj    = _score_trajectory(traj_dist)
    s_cos     = _score_cosine(cosine_segment_score)

    # Nota ponderada
    final = (
        s_dt_prec * w["dt_precision"]  +
        s_dt_cov  * w["dt_coverage"]   +
        s_topo    * w["topology"]       +
        s_ssim    * w["ssim"]           +
        s_proc    * w["procrustes"]     +
        s_haus    * w["hausdorff"]      +
        s_traj    * w["trajectory"]     +
        s_cos     * w["cosine"]
    )

    return {
        "score_final": float(round(final, 2)),
        "level":       level,
        "scores_breakdown": {
            "dt_precision":  round(s_dt_prec, 2),
            "dt_coverage":   round(s_dt_cov,  2),
            "topology":      round(s_topo,     2),
            "ssim":          round(s_ssim,     2),
            "procrustes":    round(s_proc,     2),
            "hausdorff":     round(s_haus,     2),
            "trajectory":    round(s_traj,     2),
            "cosine":        round(s_cos,      2),
        },
        "weights_used": dict(w),
    }


# =============================================================================
# Retroalimentacion pedagogica (texto para mostrar al alumno/docente)
# =============================================================================

def get_feedback(result: dict) -> str:
    """
    Genera un mensaje de retroalimentacion en lenguaje simple basado en el
    desglose de scores. Pensado para docentes y ninos.

    Parameters
    ----------
    result : dict
        Salida de calculate_final_score().

    Returns
    -------
    str  — mensaje de retroalimentacion.
    """
    score   = result["score_final"]
    bd      = result["scores_breakdown"]
    msgs    = []

    # Nota global
    if score >= 85:
        msgs.append("Excelente trazo.")
    elif score >= 65:
        msgs.append("Buen intento, sigue practicando.")
    else:
        msgs.append("Necesitas practicar mas esta letra.")

    # Retroalimentacion especifica
    if bd["dt_coverage"] < 50:
        msgs.append("Parece que no completaste toda la letra.")
    if bd["dt_precision"] < 50:
        msgs.append("Intenta mantenerte dentro del carril de la letra.")
    if bd["topology"] < 50:
        msgs.append("Revisa los bucles o circulos de la letra (ej: el hueco de la 'A').")
    if bd["hausdorff"] < 40:
        msgs.append("Hay partes del trazo muy alejadas de la forma correcta.")

    return " ".join(msgs)