"""
evaluate.py  —  Router FastAPI  POST /evaluate  |  POST /evaluate_plana  |  POST /recognize
============================================================================================
Compatible con:
  - Modelo NUEVO: EfficientNetV2-S + ArcFace (107 clases) + SmartOCR post-processing
  - Modelo ANTIGUO: MobileNet/EMNIST (62 clases, logits directos)

Cambios v4.2:
  - build_raw_crop_image() ahora recibe display_crop como tercer parámetro
  - /evaluate: pasa display_crop a build_raw_crop_image()
  - /evaluate_plana: pasa display_crop tanto para template como para cada carácter
  - _crop_and_classify: devuelve raw_crop_bgr (original) Y display_crop (limpio)
  - YOLO siempre recibe imagen ORIGINAL (no limpiada)
"""

import base64
import logging
import os

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core import config
from app.core.processor import (
    preprocess_robust,
    preprocess_multi,
    preprocess_multi_legacy,
    _classify_crop,
)
from app.core.image_cleaner import (
    clean_crop_for_display,
)
from app.core.normalizer import normalize_character

# ── Métricas ──
from app.metrics.distance_transform import calculate_dt_fidelity
from app.metrics.geometric import calculate_geometric
from app.metrics.topologic import get_topology
from app.metrics.trajectory import calculate_trajectory_dist
from app.metrics.quality import calculate_quality_metrics
from app.metrics.segment_cosine import calculate_segment_cosine_similarity
from app.metrics.scorer import calculate_final_score, get_feedback

# ── Esqueletización + Visualización ──
from app.scripts.generate_templates import skeletonize_student_char
from app.utils.visualizer import (
    generate_comparison_plot,
    build_raw_crop_image,
)

logger = logging.getLogger(__name__)

router = APIRouter()
_TEMPLATE_CACHE: dict[str, dict] = {}


# =============================================================================
# Detección YOLO con Ultralytics (fallback para evaluate_plana y recognize)
# =============================================================================

_YOLO_DETECTOR = None
_YOLO_MODEL_PATH = os.path.join(
    "app", "models", "classifier_artifacts", "best_detector.onnx"
)
_YOLO_IMG_SIZE = 640
_YOLO_CONF = 0.25
_YOLO_IOU = 0.45


def _get_yolo_detector():
    """Carga lazy del detector YOLO usando Ultralytics."""
    global _YOLO_DETECTOR
    if _YOLO_DETECTOR is not None:
        return _YOLO_DETECTOR

    pt_path = _YOLO_MODEL_PATH.replace(".onnx", ".pt")

    try:
        from ultralytics import YOLO as UltralyticsYOLO

        if os.path.exists(pt_path):
            logger.info(f"Cargando detector YOLO desde {pt_path}")
            _YOLO_DETECTOR = UltralyticsYOLO(pt_path, task="detect")
        elif os.path.exists(_YOLO_MODEL_PATH):
            logger.info(f"Cargando detector YOLO desde {_YOLO_MODEL_PATH}")
            _YOLO_DETECTOR = UltralyticsYOLO(_YOLO_MODEL_PATH, task="detect")
        else:
            logger.error(
                f"No se encontró modelo detector en "
                f"{pt_path} ni {_YOLO_MODEL_PATH}"
            )
            return None

        logger.info("✅ Detector YOLO cargado con Ultralytics")
        return _YOLO_DETECTOR

    except ImportError:
        logger.error("ultralytics no instalado. pip install ultralytics")
        return None
    except Exception as e:
        logger.error(f"Error cargando detector YOLO: {e}")
        return None


def _detect_characters_ultralytics(
    img_bgr: np.ndarray,
    conf: float = _YOLO_CONF,
    iou: float = _YOLO_IOU,
    line_tolerance: float = 0.5,
) -> list[dict]:
    """
    Detecta caracteres usando YOLO Ultralytics.
    Recibe la imagen ORIGINAL sin limpiar.
    """
    detector = _get_yolo_detector()
    if detector is None:
        return []

    # YOLO recibe imagen ORIGINAL
    results = detector.predict(
        source=img_bgr,
        imgsz=_YOLO_IMG_SIZE,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        c = float(box.conf[0].cpu())
        detections.append({
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "confidence": round(c, 4),
        })

    if not detections:
        return []

    # Reading-order sort
    heights = [d["y2"] - d["y1"] for d in detections]
    median_h = sorted(heights)[len(heights) // 2]
    tol = line_tolerance * median_h

    detections.sort(key=lambda d: (d["y1"] + d["y2"]) / 2)

    lines = []
    current_line = [detections[0]]
    current_y = (detections[0]["y1"] + detections[0]["y2"]) / 2

    for d in detections[1:]:
        y_center = (d["y1"] + d["y2"]) / 2
        if abs(y_center - current_y) <= tol:
            current_line.append(d)
        else:
            lines.append(current_line)
            current_line = [d]
            current_y = y_center
    lines.append(current_line)

    ordered = []
    for line_idx, line_group in enumerate(lines):
        line_group.sort(key=lambda d: d["x1"])
        for d in line_group:
            d["line"] = line_idx
            ordered.append(d)

    return ordered


# =============================================================================
# Carga de plantillas
# =============================================================================

_TRAZO_SAFE_NAMES = {
    "|": "linea_vertical",
    "―": "linea_horizontal",
    "\\": "linea_oblicua_derecha",
    "~": "curva",
    "○": "circulo",
    "línea_vertical": "linea_vertical",
    "línea_horizontal": "linea_horizontal",
    "línea_oblicua_derecha": "linea_oblicua_derecha",
    "línea_oblicua_izquierda": "linea_oblicua_izquierda",
    "curva": "curva",
    "círculo": "circulo",
}


def _safe_name(char: str) -> str:
    """Convierte un carácter a nombre seguro para archivos."""
    if char in _TRAZO_SAFE_NAMES:
        return _TRAZO_SAFE_NAMES[char]

    if char.isdigit():
        return f"digit_{char}"

    if char.upper() in ("Ñ", "N\u0303"):
        suffix = "upper" if char.isupper() else "lower"
        return f"N_tilde_{suffix}"

    _ACCENT_MAP = {
        'á': 'a_acute', 'é': 'e_acute', 'í': 'i_acute',
        'ó': 'o_acute', 'ú': 'u_acute', 'ü': 'u_umlaut',
        'Á': 'A_acute', 'É': 'E_acute', 'Í': 'I_acute',
        'Ó': 'O_acute', 'Ú': 'U_acute', 'Ü': 'U_umlaut',
    }
    if char in _ACCENT_MAP:
        return _ACCENT_MAP[char]

    _PUNCT_MAP = {
        '.': 'period', ',': 'comma', ';': 'semicolon', ':': 'colon',
        '¿': 'question_open', '?': 'question_close',
        '¡': 'excl_open', '!': 'excl_close',
        '(': 'lparen', ')': 'rparen',
        '-': 'hyphen', '_': 'underscore',
        "'": 'apostrophe', '"': 'quote',
        '/': 'slash', '@': 'at', '#': 'hash', '$': 'dollar',
        '%': 'percent', '&': 'ampersand', '*': 'asterisk',
        '+': 'plus', '=': 'equals', '<': 'less', '>': 'greater',
    }
    if char in _PUNCT_MAP:
        return _PUNCT_MAP[char]

    suffix = "upper" if char.isupper() else "lower"
    return f"{char}_{suffix}"


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


def get_templates(
    char: str, level: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Devuelve (carril_nivel, esqueleto_1px). Cacheados en memoria."""
    key = f"{char}:{level}"
    if key in _TEMPLATE_CACHE:
        e = _TEMPLATE_CACHE[key]
        return e["carril"], e["skeleton"]

    base = _safe_name(char)
    carril = _load_npy(os.path.join(
        config.TEMPLATE_OUTPUT_DIR, level, f"{base}_{level}.npy"
    ))
    skeleton = _load_npy(os.path.join(
        config.TEMPLATE_OUTPUT_DIR, "skeleton", f"{base}_skeleton.npy"
    ))

    if carril is None:
        carril = _load_npy(os.path.join(
            config.TEMPLATE_OUTPUT_DIR, level, f"{base}.npy"
        ))
    if carril is None:
        carril = _load_npy(os.path.join(
            config.TEMPLATE_OUTPUT_DIR, f"{base}.npy"
        ))
    if skeleton is None:
        skeleton = carril

    _TEMPLATE_CACHE[key] = {"carril": carril, "skeleton": skeleton}
    return carril, skeleton


# =============================================================================
# Utilidades
# =============================================================================

def _to_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8") if ok else ""


_MIN_CONFIDENCE = 0.05


def _display_char(char: str | None, confidence: float) -> str:
    """Devuelve el carácter para mostrar. Evita 'desconocido' o None."""
    if not char or char == "desconocido":
        return "?"
    if confidence < _MIN_CONFIDENCE:
        return "?"
    return char


def _crop_and_classify(
    img_bgr: np.ndarray,
    bbox: dict,
    expected_char: str | None = None,
) -> dict:
    """
    Recorta un carácter detectado por YOLO y lo clasifica.

    Devuelve raw_crop_bgr (original) Y display_crop (limpio) por separado.
    """
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    crop_bgr = img_bgr[y1:y2, x1:x2].copy()

    if crop_bgr.size == 0:
        return {
            "normalized_mask": None,
            "metadata": {},
            "char": "?",
            "confidence": 0.0,
            "raw_crop_bgr": None,
            "display_crop": None,
            "raw_char": "?",
            "raw_confidence": 0.0,
            "method": "failed",
            "bbox_xyxy": [x1, y1, x2, y2],
        }

    try:
        from app.core.processor import _classify_crop as classify_crop_fn

        expected_type = None
        if expected_char:
            from app.core.processor import _infer_expected_type
            expected_type = _infer_expected_type(expected_char)

        detected_char, confidence, detail = classify_crop_fn(
            crop_bgr,
            expected_type=expected_type,
            expected_char=expected_char,
            use_smart=True,
            use_tta=True,
        )

        yolo_box = (x1, y1, x2, y2)
        mask, metadata = normalize_character(img_bgr, yolo_box=yolo_box)

        display_crop = clean_crop_for_display(crop_bgr)

    except Exception as e:
        logger.warning(
            f"Error clasificando crop en ({x1},{y1},{x2},{y2}): {e}"
        )
        return {
            "normalized_mask": None,
            "metadata": {},
            "char": "?",
            "confidence": 0.0,
            "raw_crop_bgr": crop_bgr,
            "display_crop": None,
            "raw_char": "?",
            "raw_confidence": 0.0,
            "method": "error",
            "bbox_xyxy": [x1, y1, x2, y2],
        }

    return {
        "normalized_mask": mask,
        "metadata": metadata or {},
        "char": detected_char or "?",
        "confidence": confidence or 0.0,
        "raw_crop_bgr": crop_bgr,
        "display_crop": display_crop,
        "raw_char": detail.get('raw_char', detected_char or "?"),
        "raw_confidence": detail.get('raw_confidence', confidence or 0.0),
        "method": detail.get('method', 'raw'),
        "bbox_xyxy": [x1, y1, x2, y2],
    }


# =============================================================================
# POST /evaluate — un solo carácter
# =============================================================================

@router.post("/evaluate")
async def evaluate(
    file: UploadFile = File(...),
    target_char: str = Form(...),
    level: str = Form("intermedio"),
):
    """
    Evalúa el trazo del alumno contra la plantilla del carácter pedido.
    """

    # ── 1. Validar nivel ──
    valid_levels = set(config.TEMPLATE_DIFFICULTY_KERNELS.keys())
    if level not in valid_levels:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Nivel inválido '{level}'. "
                f"Opciones: {sorted(valid_levels)}"
            ),
        )

    # ── 2. Cargar plantillas ──
    carril, skel_p = get_templates(target_char, level)
    if carril is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No existe plantilla para '{target_char}' "
                f"(nivel: {level}). Ejecuta generate_templates.py primero."
            ),
        )

    # ── 3. Preprocesar imagen del alumno ──
    img_bytes = await file.read()

    mask, metadata, detected_char, confidence, raw_crop_bgr, display_crop = (
        preprocess_robust(
            img_bytes,
            use_smart=True,
            expected_char=target_char,
        )
    )
    if mask is None or np.sum(mask) == 0:
        return {
            "error": "No se detectó trazo válido en la imagen.",
            "target_char": target_char,
            "detected_char": None,
            "confidence": 0.0,
        }

    detected_char = _display_char(detected_char, confidence)

    # ── 4. Esqueletizar trazo del alumno ──
    skel_a = skeletonize_student_char(mask)

    # ── 5. Distance Transform ──
    dt_score, dt_coverage, _dist_map, _heatmap = calculate_dt_fidelity(
        skel_p, mask, level=level
    )

    # ── 6. Métricas geométricas ──
    geo = calculate_geometric(skel_p, skel_a)

    # ── 7. Topología ──
    topo_p = get_topology(skel_p)
    topo_a = get_topology(skel_a)
    topo_match = bool(
        topo_p.get("loops", 0) == topo_a.get("loops", 0)
    )

    # ── 8. Trayectoria DTW ──
    traj_dist = calculate_trajectory_dist(skel_p, skel_a)

    # ── 9. Calidad + coseno ──
    quality = calculate_quality_metrics(mask)
    _, cosine_score = calculate_segment_cosine_similarity(skel_p, skel_a)

    # ── 10. Nota final ──
    score_result = calculate_final_score(
        geo_metrics=geo,
        topo_match=topo_match,
        traj_dist=traj_dist,
        dt_precision_score=dt_score,
        dt_coverage=dt_coverage,
        cosine_segment_score=cosine_score,
        level=level,
    )
    feedback = get_feedback(score_result)

    # ── 11. Imágenes ──
    # "Tu trazo": prioriza display_crop (limpio de image_cleaner)
    raw_img = build_raw_crop_image(
        raw_crop_bgr=raw_crop_bgr,
        mask=mask,
        display_crop=display_crop,
    )
    template_img = cv2.cvtColor(carril, cv2.COLOR_GRAY2BGR)
    comparison_b64 = generate_comparison_plot(
        skel_p=skel_p, skel_a=skel_a,
        score=score_result["score_final"],
        level=level, char=target_char, img_a=mask,
    )

    # ── 12. Respuesta ──
    return {
        "target_char": target_char,
        "detected_char": detected_char,
        "confidence": float(round(confidence, 4)),

        "score_final": score_result["score_final"],
        "level": score_result["level"],
        "scores_breakdown": score_result["scores_breakdown"],
        "weights_used": score_result["weights_used"],

        "feedback": feedback,

        "metadata": {
            **(metadata or {}),
            "angle_corrected": (metadata or {}).get(
                "angle_corrected", 0.0
            ),
            "scale_factor": (metadata or {}).get("scale_factor", 1.0),
            "roi_refined": (metadata or {}).get("roi_refined", False),
            "char_width_px": (metadata or {}).get("char_width_px", 0),
            "char_height_px": (metadata or {}).get("char_height_px", 0),
            "model_type": (metadata or {}).get("model_type", "unknown"),
            "classification_method": (metadata or {}).get(
                "classification_method", "raw"
            ),
            "raw_prediction": (metadata or {}).get(
                "raw_prediction", detected_char
            ),
            "raw_confidence": (metadata or {}).get(
                "raw_confidence", confidence
            ),
            "smart_ocr": (metadata or {}).get("smart_ocr", False),
            "pipeline_version": (metadata or {}).get(
                "pipeline_version", "v4.2_clean"
            ),
        },

        "metrics_extra": {
            "geometric": geo,
            "topology": {
                "match": topo_match,
                "student": topo_a,
                "pattern": topo_p,
            },
            "quality": quality,
            "trajectory_error": float(round(traj_dist, 4)),
            "segment_cosine_score": float(round(cosine_score, 4)),
            "dt_coverage_ratio": float(round(dt_coverage, 4)),
        },

        "image_student_b64": _to_b64(raw_img),
        "template_b64": _to_b64(template_img),
        "comparison_b64": comparison_b64,
    }


# =============================================================================
# POST /evaluate_plana — plana completa
# =============================================================================

@router.post("/evaluate_plana")
async def evaluate_plana(
    file: UploadFile = File(...),
    target_char: str = Form(""),
    level: str = Form("intermedio"),
):
    """
    Califica una plana completa.
    """

    # ── 1. Validar nivel ──
    valid_levels = set(config.TEMPLATE_DIFFICULTY_KERNELS.keys())
    if level not in valid_levels:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Nivel inválido '{level}'. "
                f"Opciones: {sorted(valid_levels)}"
            ),
        )

    # ── 2. Leer imagen ──
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(
            status_code=422,
            detail="No se pudo decodificar la imagen.",
        )

    logger.info(
        f"evaluate_plana: imagen {img_bgr.shape[1]}x{img_bgr.shape[0]}, "
        f"target_char={target_char!r}"
    )

    expected_chars = target_char if target_char else None

    # ── 3. Detectar caracteres con preprocess_multi ──
    smart_result = None
    characters = []

    try:
        smart_result = preprocess_multi(
            img_bytes,
            use_smart=True,
            group_words=True,
            expected_chars=expected_chars,
        )
        characters = smart_result.get('characters', [])
        logger.info(
            f"preprocess_multi detectó {len(characters)} caracteres"
        )
    except Exception as e:
        logger.warning(f"preprocess_multi falló: {e}")
        characters = []

    # ── 3b. FALLBACK: YOLO directo ──
    if len(characters) == 0:
        logger.info(
            "Fallback: usando Ultralytics YOLO directo para detección"
        )
        yolo_detections = _detect_characters_ultralytics(img_bgr)
        logger.info(
            f"YOLO Ultralytics detectó {len(yolo_detections)} caracteres"
        )

        if len(yolo_detections) == 0:
            raise HTTPException(
                status_code=422,
                detail="No se detectaron caracteres en la imagen.",
            )

        characters = []
        for det in yolo_detections:
            char_data = _crop_and_classify(
                img_bgr, det,
                expected_char=target_char if target_char else None,
            )
            char_data["line"] = det.get("line", 0)
            characters.append(char_data)

        recognized_text = "".join(
            c.get("char", "?") for c in characters
        )
        smart_result = {
            "characters": characters,
            "text": recognized_text,
            "words": [],
            "lines": [],
            "confidence": float(np.mean([
                c.get("confidence", 0.0) for c in characters
            ])) if characters else 0.0,
            "n_detections": len(characters),
            "detection_method": "ultralytics_yolo_fallback",
        }

        logger.info(
            f"Fallback exitoso: {len(characters)} caracteres clasificados"
        )

    # ── Validar mínimo de caracteres ──
    if len(characters) == 0:
        raise HTTPException(
            status_code=422,
            detail="No se detectaron caracteres en la imagen.",
        )

    if len(characters) == 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "Solo se detectó 1 carácter. "
                "La plana necesita al menos 2."
            ),
        )

    # ── 4. Plantilla = primer carácter ──
    tmpl = characters[0]
    tmpl_mask = tmpl.get('normalized_mask')
    tmpl_meta = tmpl.get('metadata', {})
    tmpl_char = tmpl.get('char', '?')
    tmpl_conf = tmpl.get('confidence', 0.0)
    tmpl_raw_crop = tmpl.get('raw_crop_bgr')
    tmpl_display = tmpl.get('display_crop')

    if target_char:
        tmpl_char = target_char
    else:
        tmpl_char = _display_char(tmpl_char, tmpl_conf)

    if tmpl_mask is None or (
        isinstance(tmpl_mask, np.ndarray) and tmpl_mask.size == 0
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "No se pudo procesar el carácter plantilla "
                "(primer carácter)."
            ),
        )

    skel_p = skeletonize_student_char(tmpl_mask)

    # Template "Tu trazo": prioriza display_crop
    tmpl_img = build_raw_crop_image(
        raw_crop_bgr=tmpl_raw_crop,
        mask=tmpl_mask,
        display_crop=tmpl_display,
    )
    template_b64 = _to_b64(tmpl_img)

    # ── 5. Calificar cada carácter restante ──
    char_results: list[dict] = []

    for position, char_data in enumerate(characters[1:], start=1):
        img_a = char_data.get('normalized_mask')
        metadata = char_data.get('metadata', {})
        detected_char = char_data.get('char', '?')
        confidence = char_data.get('confidence', 0.0)
        raw_crop = char_data.get('raw_crop_bgr')
        display_crop = char_data.get('display_crop')

        raw_char = char_data.get('raw_char', detected_char)
        raw_confidence = char_data.get('raw_confidence', confidence)
        method = char_data.get('method', 'raw')

        if img_a is None or (
            isinstance(img_a, np.ndarray) and np.sum(img_a) == 0
        ):
            char_results.append({
                "index": position,
                "detected_char": _display_char(
                    detected_char, confidence
                ),
                "confidence": float(round(confidence, 4)),
                "score_final": 0.0,
                "level": level,
                "scores_breakdown": {},
                "weights_used": {},
                "feedback": (
                    "No se detectó trazo válido en este carácter."
                ),
                "metadata": metadata,
                "metrics_extra": {},
                "image_student_b64": "",
                "comparison_b64": "",
                "smart_ocr": {
                    "raw_prediction": raw_char,
                    "raw_confidence": float(round(raw_confidence, 4)),
                    "method": method,
                },
            })
            continue

        skel_a = skeletonize_student_char(img_a)

        dt_score, dt_coverage, _dist_map, _heatmap = (
            calculate_dt_fidelity(skel_p, img_a, level=level)
        )
        geo = calculate_geometric(skel_p, skel_a)
        topo_p = get_topology(skel_p)
        topo_a = get_topology(skel_a)
        topo_match = bool(
            topo_p.get("loops", 0) == topo_a.get("loops", 0)
        )
        traj_dist = calculate_trajectory_dist(skel_p, skel_a)
        quality = calculate_quality_metrics(img_a)
        _, cosine_score = calculate_segment_cosine_similarity(
            skel_p, skel_a
        )

        score_result = calculate_final_score(
            geo_metrics=geo,
            topo_match=topo_match,
            traj_dist=traj_dist,
            dt_precision_score=dt_score,
            dt_coverage=dt_coverage,
            cosine_segment_score=cosine_score,
            level=level,
        )
        feedback = get_feedback(score_result)

        # "Tu trazo": prioriza display_crop (limpio de image_cleaner)
        raw_img = build_raw_crop_image(
            raw_crop_bgr=raw_crop,
            mask=img_a,
            display_crop=display_crop,
        )
        comparison_b64 = generate_comparison_plot(
            skel_p=skel_p, skel_a=skel_a,
            score=score_result["score_final"],
            level=level, char=tmpl_char, img_a=img_a,
        )

        display_char = _display_char(detected_char, confidence)

        char_results.append({
            "index": position,
            "detected_char": display_char,
            "confidence": float(round(confidence, 4)),
            "score_final": score_result["score_final"],
            "level": score_result["level"],
            "scores_breakdown": score_result["scores_breakdown"],
            "weights_used": score_result["weights_used"],
            "feedback": feedback,
            "metadata": {
                **(metadata or {}),
                "angle_corrected": (metadata or {}).get(
                    "angle_corrected", 0.0
                ),
                "scale_factor": (metadata or {}).get(
                    "scale_factor", 1.0
                ),
                "roi_refined": (metadata or {}).get(
                    "roi_refined", False
                ),
                "char_width_px": (metadata or {}).get(
                    "char_width_px", 0
                ),
                "char_height_px": (metadata or {}).get(
                    "char_height_px", 0
                ),
                "model_type": (metadata or {}).get(
                    "model_type", "unknown"
                ),
                "pipeline_version": (metadata or {}).get(
                    "pipeline_version", "v4.2_clean"
                ),
            },
            "smart_ocr": {
                "raw_prediction": raw_char,
                "raw_confidence": float(round(raw_confidence, 4)),
                "method": method,
            },
            "metrics_extra": {
                "geometric": geo,
                "topology": {
                    "match": topo_match,
                    "student": topo_a,
                    "pattern": topo_p,
                },
                "quality": quality,
                "trajectory_error": float(round(traj_dist, 4)),
                "segment_cosine_score": float(
                    round(cosine_score, 4)
                ),
                "dt_coverage_ratio": float(round(dt_coverage, 4)),
            },
            "image_student_b64": _to_b64(raw_img),
            "comparison_b64": comparison_b64,
        })

    # ── 6. Estadísticas agregadas ──
    valid_scores = [
        r["score_final"] for r in char_results if r["score_final"] > 0
    ]
    avg_score = (
        round(float(np.mean(valid_scores)), 4) if valid_scores else 0.0
    )

    # ── 7. Info SmartOCR de la plana ──
    recognized_text = (
        smart_result.get('text', '') if smart_result else ''
    )
    words_info = []
    for w in (smart_result or {}).get('words', []):
        words_info.append({
            "word": w.get('word', ''),
            "raw_word": w.get('raw_word', ''),
            "confidence": float(round(w.get('confidence', 0.0), 4)),
            "corrected": w.get('corrected', False),
            "correction_method": w.get('correction_method', 'none'),
            "n_chars": w.get('n_chars', 0),
        })

    lines_info = []
    for ln in (smart_result or {}).get('lines', []):
        lines_info.append({
            "text": ln.get('text', ''),
            "word_count": ln.get('word_count', 0),
            "char_count": ln.get('char_count', 0),
        })

    detection_method = (smart_result or {}).get(
        "detection_method", "preprocess_multi"
    )

    # ── 8. Respuesta ──
    return {
        "template_char": tmpl_char,
        "template_confidence": float(round(tmpl_conf, 4)),
        "template_b64": template_b64,

        "n_detected": len(characters),
        "n_evaluated": len(char_results),
        "avg_score": avg_score,
        "level": level,

        "detection_method": detection_method,

        "smart_ocr": {
            "recognized_text": recognized_text,
            "words": words_info,
            "lines": lines_info,
            "overall_confidence": float(round(
                (smart_result or {}).get('confidence', 0.0), 4
            )),
        },

        "results": char_results,
    }


# =============================================================================
# POST /recognize — Reconocimiento de texto puro con SmartOCR
# =============================================================================

@router.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
):
    """
    Reconoce todos los caracteres en la imagen y devuelve el texto.
    Usa SmartOCR con agrupación de palabras, contexto y diccionario.
    SIN expected_char (reconocimiento libre).
    """
    img_bytes = await file.read()

    result = None
    try:
        result = preprocess_multi(
            img_bytes,
            use_smart=True,
            group_words=True,
        )
    except Exception as e:
        logger.warning(f"preprocess_multi falló en /recognize: {e}")

    characters = (result or {}).get('characters', [])

    # Fallback a YOLO directo
    if not characters:
        logger.info("/recognize: fallback a Ultralytics YOLO")
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            yolo_dets = _detect_characters_ultralytics(img_bgr)
            if yolo_dets:
                characters = []
                for det in yolo_dets:
                    char_data = _crop_and_classify(img_bgr, det)
                    characters.append(char_data)

                result = {
                    "characters": characters,
                    "text": "".join(
                        c.get("char", "?") for c in characters
                    ),
                    "words": [],
                    "lines": [],
                    "confidence": float(np.mean([
                        c.get("confidence", 0.0)
                        for c in characters
                    ])) if characters else 0.0,
                    "n_detections": len(characters),
                }

    if not characters:
        return {
            "text": "",
            "n_detected": 0,
            "confidence": 0.0,
            "words": [],
            "lines": [],
            "characters": [],
        }

    chars_simple = []
    for c in characters:
        chars_simple.append({
            "char": c.get('char', '?'),
            "confidence": float(
                round(c.get('confidence', 0.0), 4)
            ),
            "raw_char": c.get('raw_char', '?'),
            "raw_confidence": float(
                round(c.get('raw_confidence', 0.0), 4)
            ),
            "method": c.get('method', 'raw'),
            "bbox_xyxy": c.get('bbox_xyxy', []),
        })

    words_simple = []
    for w in (result or {}).get('words', []):
        words_simple.append({
            "word": w.get('word', ''),
            "raw_word": w.get('raw_word', ''),
            "confidence": float(
                round(w.get('confidence', 0.0), 4)
            ),
            "corrected": w.get('corrected', False),
            "correction_method": w.get('correction_method', 'none'),
        })

    lines_simple = []
    for ln in (result or {}).get('lines', []):
        lines_simple.append({
            "text": ln.get('text', ''),
            "word_count": ln.get('word_count', 0),
            "char_count": ln.get('char_count', 0),
        })

    return {
        "text": (result or {}).get('text', ''),
        "n_detected": (result or {}).get(
            'n_detections', len(characters)
        ),
        "confidence": float(
            round((result or {}).get('confidence', 0.0), 4)
        ),
        "words": words_simple,
        "lines": lines_simple,
        "characters": chars_simple,
    }