"""
evaluate.py  —  Router FastAPI  POST /evaluate  |  POST /evaluate_plana
========================================================================
Recibe una imagen de libreta y devuelve:
  - image_student_b64 : crop RAW de YOLO (foto original sin normalizar)
  - template_b64      : carril del nivel seleccionado
  - comparison_b64    : overlay verde/rojo/amarillo (de visualizer.py)
  - scores_breakdown  : dict plano, compatible con index.html
  - weights_used      : pesos reales usados
  - feedback          : texto pedagógico
  - metadata          : ángulo, escala, roi_refined, dimensiones

POST /evaluate_plana
  - Recibe imagen de plana (múltiples caracteres repetidos)
  - El PRIMER carácter detectado actúa como plantilla de referencia
  - Los demás se califican contra ese primer carácter
  - No requiere target_char: la referencia es la propia escritura
"""

import base64
import os

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core import config
from app.core.processor import preprocess_robust, preprocess_multi

# ── Métricas ──────────────────────────────────────────────────────────────────
from app.metrics.distance_transform import calculate_dt_fidelity
from app.metrics.geometric        import calculate_geometric
from app.metrics.topologic        import get_topology
from app.metrics.trajectory       import calculate_trajectory_dist
from app.metrics.quality          import calculate_quality_metrics
from app.metrics.segment_cosine   import calculate_segment_cosine_similarity
from app.metrics.scorer          import calculate_final_score, get_feedback

# ── Esqueletización + Visualización ───────────────────────────────────────────
from app.scripts.generate_templates import skeletonize_student_char
from app.utils.visualizer           import generate_comparison_plot, build_raw_crop_image

router = APIRouter()
_TEMPLATE_CACHE: dict[str, dict] = {}


# =============================================================================
# Carga de plantillas
# =============================================================================

def _safe_name(char: str) -> str:
    if char.isdigit():
        return f"digit_{char}"
    base   = "N_tilde" if char.upper() in ("Ñ", "N\u0303") else char
    suffix = "upper" if char.isupper() else "lower"
    return f"{base}_{suffix}"


def _load_npy(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    img = (arr > 0).astype(np.uint8) * 255
    if img.shape != config.TARGET_SHAPE:
        img = cv2.resize(
            img,
            (config.TARGET_SHAPE[1], config.TARGET_SHAPE[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return img


def get_templates(char: str, level: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Devuelve (carril_nivel, esqueleto_1px). Ambos cacheados en memoria."""
    key = f"{char}:{level}"
    if key in _TEMPLATE_CACHE:
        e = _TEMPLATE_CACHE[key]
        return e["carril"], e["skeleton"]

    base     = _safe_name(char)
    carril   = _load_npy(os.path.join(config.TEMPLATE_OUTPUT_DIR, level,      f"{base}_{level}.npy"))
    skeleton = _load_npy(os.path.join(config.TEMPLATE_OUTPUT_DIR, "skeleton", f"{base}_skeleton.npy"))

    # Fallback legacy (raíz de templates/)
    if carril is None:
        carril = _load_npy(os.path.join(config.TEMPLATE_OUTPUT_DIR, f"{base}.npy"))
    if skeleton is None:
        skeleton = carril

    _TEMPLATE_CACHE[key] = {"carril": carril, "skeleton": skeleton}
    return carril, skeleton


# =============================================================================
# Utilidad: numpy → PNG base64
# =============================================================================

def _to_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8") if ok else ""


# =============================================================================
# Endpoint
# =============================================================================

@router.post("/evaluate")
async def evaluate(
    file:        UploadFile = File(...),
    target_char: str        = Form(...),
    level:       str        = Form("intermedio"),
):
    """
    Evalúa el trazo del alumno contra la plantilla del carácter pedido.

    Form params
    -----------
    file        : imagen de libreta (jpg, png, webp…)
    target_char : letra/dígito esperado ("A", "b", "3"…)
    level       : "principiante" | "intermedio" | "avanzado"

    Imágenes devueltas
    ------------------
    image_student_b64 : crop RAW de YOLO — foto real de la libreta recortada
    template_b64      : carril del nivel  — guía visual para el alumno
    comparison_b64    : overlay verde/rojo/amarillo generado por visualizer.py
    """

    # ── 1. Validar nivel ──────────────────────────────────────────────────────
    valid_levels = set(config.TEMPLATE_DIFFICULTY_KERNELS.keys())
    if level not in valid_levels:
        raise HTTPException(
            status_code=422,
            detail=f"Nivel inválido '{level}'. Opciones: {sorted(valid_levels)}",
        )

    # ── 2. Cargar plantillas ──────────────────────────────────────────────────
    carril, skel_p = get_templates(target_char, level)
    if carril is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No existe plantilla para '{target_char}' (nivel: {level}). "
                f"Ejecuta generate_templates.py primero."
            ),
        )

    # ── 3. Preprocesar imagen del alumno ──────────────────────────────────────
    # preprocess_robust puede devolver 4 ó 5 valores.
    # El 5.º (raw_crop_bgr) es el recorte BGR original de YOLO — NUEVO.
    # Si tu versión actual sólo devuelve 4, raw_crop_bgr quedará como None
    # y build_raw_crop_image() usará img_a normalizado como fallback.
    img_bytes = await file.read()
    result    = preprocess_robust(img_bytes)

    if len(result) == 5:
        img_a, metadata, detected_char, confidence, raw_crop_bgr = result
    else:
        img_a, metadata, detected_char, confidence = result
        raw_crop_bgr = None

    if img_a is None or np.sum(img_a) == 0:
        return {
            "error":         "No se detectó trazo válido en la imagen.",
            "target_char":   target_char,
            "detected_char": None,
            "confidence":    0.0,
        }

    # ── 4. Esqueletizar trazo del alumno ──────────────────────────────────────
    skel_a = skeletonize_student_char(img_a)

    # ── 5. Distance Transform ─────────────────────────────────────────────────
    # skel_p  = esqueleto plantilla (guía de 1 px)
    # img_a   = masa binaria del alumno
    # Devuelve: (score_precision, coverage, dist_map, heatmap_bgr)
    dt_score, dt_coverage, _dist_map, _heatmap = calculate_dt_fidelity(
        skel_p, img_a, level=level
    )

    # ── 6. Métricas geométricas (esqueleto vs esqueleto) ─────────────────────
    geo = calculate_geometric(skel_p, skel_a)

    # ── 7. Topología ─────────────────────────────────────────────────────────
    topo_p     = get_topology(skel_p)
    topo_a     = get_topology(skel_a)
    topo_match = bool(topo_p.get("loops", 0) == topo_a.get("loops", 0))

    # ── 8. Trayectoria DTW ────────────────────────────────────────────────────
    traj_dist = calculate_trajectory_dist(skel_p, skel_a)

    # ── 9. Calidad intrínseca + coseno de segmentos ───────────────────────────
    quality         = calculate_quality_metrics(img_a)       # sobre la masa
    _, cosine_score = calculate_segment_cosine_similarity(skel_p, skel_a)

    # ── 10. Nota final ────────────────────────────────────────────────────────
    score_result = calculate_final_score(
        geo_metrics          = geo,
        topo_match           = topo_match,
        traj_dist            = traj_dist,
        dt_precision_score   = dt_score,
        dt_coverage          = dt_coverage,
        cosine_segment_score = cosine_score,
        level                = level,
    )

    feedback = get_feedback(score_result)

    # ── 11. Imágenes ──────────────────────────────────────────────────────────

    # a) Crop RAW de YOLO (contexto real de la libreta)
    raw_img = build_raw_crop_image(raw_crop_bgr, img_a)

    # b) Carril del nivel (plantilla visual para el alumno)
    template_img = cv2.cvtColor(carril, cv2.COLOR_GRAY2BGR)

    # c) Overlay de comparación — delegado completamente a visualizer.py
    comparison_b64 = generate_comparison_plot(
        skel_p = skel_p,
        skel_a = skel_a,
        score  = score_result["score_final"],
        level  = level,
        char   = target_char,
        img_a  = img_a,   # ← masa binaria del alumno para overlay más continuo
    )

    # ── 12. Respuesta ─────────────────────────────────────────────────────────
    return {
        # ── Identificación ────────────────────────────────────────────────
        "target_char":   target_char,
        "detected_char": detected_char or "?",
        "confidence":    float(round(confidence, 4)),

        # ── Nota (aplanada del dict de scoring.py) ────────────────────────
        "score_final":      score_result["score_final"],
        "level":            score_result["level"],
        "scores_breakdown": score_result["scores_breakdown"],
        "weights_used":     score_result["weights_used"],

        # ── Retroalimentación ─────────────────────────────────────────────
        "feedback": feedback,

        # ── Metadata del preprocesado ─────────────────────────────────────
        "metadata": {
            **(metadata or {}),
            "angle_corrected": (metadata or {}).get("angle_corrected", 0.0),
            "scale_factor":    (metadata or {}).get("scale_factor",    1.0),
            "roi_refined":     (metadata or {}).get("roi_refined",     False),
            "char_width_px":   (metadata or {}).get("char_width_px",   0),
            "char_height_px":  (metadata or {}).get("char_height_px",  0),
        },

        # ── Métricas auxiliares para debug/expansión ──────────────────────
        "metrics_extra": {
            "geometric":            geo,
            "topology": {
                "match":   topo_match,
                "student": topo_a,
                "pattern": topo_p,
            },
            "quality":              quality,
            "trajectory_error":     float(round(traj_dist,    4)),
            "segment_cosine_score": float(round(cosine_score, 4)),
            "dt_coverage_ratio":    float(round(dt_coverage,  4)),
        },

        # ── Las 3 imágenes que espera index.html ──────────────────────────
        "image_student_b64": _to_b64(raw_img),          # Crop RAW de YOLO
        "template_b64":      _to_b64(template_img),     # Carril del nivel
        "comparison_b64":    comparison_b64,             # Overlay verde/rojo (base64 ya)
    }


# =============================================================================
# Endpoint — Modo plana (plantilla = primer carácter detectado)
# =============================================================================

@router.post("/evaluate_plana")
async def evaluate_plana(
    file:  UploadFile = File(...),
    level: str        = Form("intermedio"),
):
    """
    Califica una plana completa usando el PRIMER carácter como plantilla.

    El endpoint no recibe target_char.  El primer carácter detectado en
    orden de lectura (izq→der, arriba→abajo) actúa como referencia; los
    caracteres siguientes se califican contra ese esqueleto de referencia
    con las mismas métricas que /evaluate.

    Form params
    -----------
    file  : imagen de plana (jpg, png, webp…)
    level : "principiante" | "intermedio" | "avanzado"

    Response
    --------
    {
      "template_char"   : str,          # carácter del primer bbox (plantilla)
      "template_b64"    : str,          # imagen del primer crop (base64 PNG)
      "n_evaluated"     : int,          # número de chars calificados (sin template)
      "results"         : [             # un dict por cada char calificado
        {
          "index"          : int,       # posición en orden de lectura (1-based)
          "detected_char"  : str,
          "confidence"     : float,
          "score_final"    : float,
          "level"          : str,
          "scores_breakdown": dict,
          "weights_used"   : dict,
          "feedback"       : str,
          "metadata"       : dict,
          "metrics_extra"  : dict,
          "image_student_b64": str,     # crop RAW del carácter calificado
          "comparison_b64"   : str,     # overlay vs plantilla
        }, ...
      ]
    }
    """

    # ── 1. Validar nivel ──────────────────────────────────────────────────────
    valid_levels = set(config.TEMPLATE_DIFFICULTY_KERNELS.keys())
    if level not in valid_levels:
        raise HTTPException(
            status_code=422,
            detail=f"Nivel inválido '{level}'. Opciones: {sorted(valid_levels)}",
        )

    # ── 2. Detectar todos los caracteres de la plana ──────────────────────────
    img_bytes  = await file.read()
    all_chars  = preprocess_multi(img_bytes)   # list[tuple] en orden de lectura

    if len(all_chars) == 0:
        raise HTTPException(
            status_code=422,
            detail="No se detectaron caracteres en la imagen. "
                   "Verifica que la imagen contenga trazos legibles.",
        )

    if len(all_chars) == 1:
        raise HTTPException(
            status_code=422,
            detail="Solo se detectó 1 carácter. "
                   "La plana debe tener al menos 2 caracteres para calificar.",
        )

    # ── 3. Plantilla = primer carácter detectado ──────────────────────────────
    tmpl_img_a, tmpl_meta, tmpl_char, tmpl_conf, tmpl_raw_bgr = all_chars[0]

    # Esqueletizar plantilla (1 px de grosor)
    skel_p = skeletonize_student_char(tmpl_img_a)

    # Imagen base64 del template (crop RAW del primer bbox)
    tmpl_display = build_raw_crop_image(tmpl_raw_bgr, tmpl_img_a)
    template_b64 = _to_b64(tmpl_display)

    # ── 4. Calificar cada carácter restante contra la plantilla ──────────────
    char_results: list[dict] = []

    for position, (img_a, metadata, detected_char, confidence, raw_crop_bgr) in \
            enumerate(all_chars[1:], start=1):

        # ── 4a. Verificar trazo válido ────────────────────────────────────────
        if img_a is None or np.sum(img_a) == 0:
            char_results.append({
                "index":           position,
                "detected_char":   detected_char or "?",
                "confidence":      float(round(confidence, 4)),
                "score_final":     0.0,
                "level":           level,
                "scores_breakdown": {},
                "weights_used":    {},
                "feedback":        "No se detectó trazo válido en este carácter.",
                "metadata":        metadata or {},
                "metrics_extra":   {},
                "image_student_b64": "",
                "comparison_b64":    "",
            })
            continue

        # ── 4b. Esqueletizar carácter del alumno ──────────────────────────────
        skel_a = skeletonize_student_char(img_a)

        # ── 4c. Métricas (idénticas a /evaluate) ──────────────────────────────
        dt_score, dt_coverage, _dist_map, _heatmap = calculate_dt_fidelity(
            skel_p, img_a, level=level,
        )
        geo        = calculate_geometric(skel_p, skel_a)
        topo_p     = get_topology(skel_p)
        topo_a     = get_topology(skel_a)
        topo_match = bool(topo_p.get("loops", 0) == topo_a.get("loops", 0))
        traj_dist  = calculate_trajectory_dist(skel_p, skel_a)
        quality    = calculate_quality_metrics(img_a)
        _, cosine_score = calculate_segment_cosine_similarity(skel_p, skel_a)

        # ── 4d. Nota final ────────────────────────────────────────────────────
        score_result = calculate_final_score(
            geo_metrics          = geo,
            topo_match           = topo_match,
            traj_dist            = traj_dist,
            dt_precision_score   = dt_score,
            dt_coverage          = dt_coverage,
            cosine_segment_score = cosine_score,
            level                = level,
        )
        feedback = get_feedback(score_result)

        # ── 4e. Imágenes del carácter calificado ──────────────────────────────
        raw_img        = build_raw_crop_image(raw_crop_bgr, img_a)
        comparison_b64 = generate_comparison_plot(
            skel_p = skel_p,
            skel_a = skel_a,
            score  = score_result["score_final"],
            level  = level,
            char   = tmpl_char,   # referencia = carácter plantilla
            img_a  = img_a,
        )

        char_results.append({
            # Identificación
            "index":           position,
            "detected_char":   detected_char or "?",
            "confidence":      float(round(confidence, 4)),
            # Nota
            "score_final":      score_result["score_final"],
            "level":            score_result["level"],
            "scores_breakdown": score_result["scores_breakdown"],
            "weights_used":     score_result["weights_used"],
            # Retroalimentación
            "feedback": feedback,
            # Metadata
            "metadata": {
                **(metadata or {}),
                "angle_corrected": (metadata or {}).get("angle_corrected", 0.0),
                "scale_factor":    (metadata or {}).get("scale_factor",    1.0),
                "roi_refined":     (metadata or {}).get("roi_refined",     False),
                "char_width_px":   (metadata or {}).get("char_width_px",   0),
                "char_height_px":  (metadata or {}).get("char_height_px",  0),
            },
            # Métricas auxiliares
            "metrics_extra": {
                "geometric":            geo,
                "topology": {
                    "match":   topo_match,
                    "student": topo_a,
                    "pattern": topo_p,
                },
                "quality":              quality,
                "trajectory_error":     float(round(traj_dist,    4)),
                "segment_cosine_score": float(round(cosine_score, 4)),
                "dt_coverage_ratio":    float(round(dt_coverage,  4)),
            },
            # Imágenes
            "image_student_b64": _to_b64(raw_img),
            "comparison_b64":    comparison_b64,
        })

    # ── 5. Estadísticas agregadas de la plana ─────────────────────────────────
    valid_scores = [r["score_final"] for r in char_results if r["score_final"] > 0]
    avg_score    = round(float(np.mean(valid_scores)), 4) if valid_scores else 0.0

    # ── 6. Respuesta ──────────────────────────────────────────────────────────
    return {
        # Plantilla usada como referencia
        "template_char":       tmpl_char or "?",
        "template_confidence": float(round(tmpl_conf, 4)),
        "template_b64":        template_b64,
        # Resumen de la plana
        "n_detected":   len(all_chars),
        "n_evaluated":  len(char_results),
        "avg_score":    avg_score,
        "level":        level,
        # Resultados por carácter (sin la plantilla)
        "results":      char_results,
    }