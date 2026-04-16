"""
app/core/classifier.py  (v4.1 — TTA + refactorizado)
=====================================================
Clasificación de caracteres manuscritos con post-procesamiento inteligente.

CAMBIOS v4.1 vs v4:
  - NUEVO: Test-Time Augmentation (TTA) con 5 variantes
    (original, rotación ±3°, escala 95%/105%).
    Promedio de LOGITS (no probabilidades) antes de softmax.
    Mejora precisión ~2-3% sin re-entrenar el modelo.

  - _run_inference() acepta parámetro use_tta: bool
  - classify_char_smart() usa TTA por defecto (evaluación individual)
  - classify_word() puede desactivar TTA para velocidad

CAMBIOS v4 vs v3:
  - Eliminada TODA lógica de polaridad (_is_white_on_black, _ensure_dark_on_light,
    doble intento de polaridad). La polaridad ahora se resuelve en image_cleaner.py
    ANTES de llegar aquí.

  - Preprocesamiento delegado a preprocessing.py (prepare_for_model).
    Ya no hay funciones duplicadas de resize/normalize.

  - Una ÚNICA función de inferencia _run_inference() que acepta:
    a) Grayscale limpio (de image_cleaner) — CAMINO PRINCIPAL
    b) BGR crop (legacy/fallback)
    c) Tensor ya preparado

  - SmartOCR post-processing se mantiene INTACTO (boost, type_filter,
    confusion resolution, dictionary, etc.)

  - debug_check_image() simplificado (ya no necesita probar polaridades)

Compatible con:
  - Modelo NUEVO: EfficientNetV2-S + ArcFace (107 clases)
  - Modelo ANTIGUO: MobileNet/EMNIST (62 clases)
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict

import logging

from app.core import config
from app.core.preprocessing import (
    prepare_for_model,
    prepare_for_model_grayscale_1ch,
    IMG_SIZE,
)
from app.core.image_cleaner import (
    clean_crop_for_classification,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1. CARGA DEL MODELO ONNX
# ═════════════════════════════════════════════════════════════════════════════

session_cls = ort.InferenceSession(
    config.MOBILENET_MODEL_PATH,
    providers=['CPUExecutionProvider']
)


def _load_class_map() -> Dict[int, str]:
    """Carga idx→char desde char_map.json junto al modelo."""
    model_dir = Path(config.MOBILENET_MODEL_PATH).parent
    candidates = [
        Path(getattr(config, 'CLASS_MAP_PATH', '')),
        model_dir / 'char_map.json',
        model_dir / 'class_map.json',
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            with open(p, encoding='utf-8') as f:
                raw = json.load(f)
            idx2char = raw.get('idx2char', raw)
            if isinstance(idx2char, dict):
                return {int(k): str(v) for k, v in idx2char.items()}
        except Exception:
            continue
    print('[classifier] WARNING: char_map no encontrado, usando fallback')
    return {i: c for i, c in enumerate(
        getattr(config, 'EMNIST_CLASS_ORDER', [])
    )}


CLASS_MAP: Dict[int, str] = _load_class_map()
CHAR2IDX: Dict[str, int] = {v: k for k, v in CLASS_MAP.items()}
NUM_MODEL_CLASSES = len(CLASS_MAP)
print(f'[classifier] {NUM_MODEL_CLASSES} clases cargadas')


# ═════════════════════════════════════════════════════════════════════════════
# 2. DETECCIÓN AUTOMÁTICA DEL TIPO DE MODELO
# ═════════════════════════════════════════════════════════════════════════════

_input_meta = session_cls.get_inputs()[0]
_input_shape = _input_meta.shape
_output_meta = session_cls.get_outputs()[0]
_output_shape = _output_meta.shape


def _get_dim(shape, idx, fallback):
    if shape is None or idx >= len(shape):
        return fallback
    d = shape[idx]
    return int(d) if isinstance(d, int) else fallback


INPUT_C = _get_dim(_input_shape, 1, 3)
INPUT_H = _get_dim(_input_shape, 2, 128)
INPUT_W = _get_dim(_input_shape, 3, 128)
NUM_OUTPUTS = _get_dim(_output_shape, 1, 107)

# Test para detectar rango de salida
_test_input = np.random.randn(1, INPUT_C, INPUT_H, INPUT_W).astype(np.float32)
_test_output = session_cls.run(None, {_input_meta.name: _test_input})[0][0]
_max_abs_output = float(np.abs(_test_output).max())

_NEEDS_ARCFACE_SCALE = bool(_max_abs_output <= 1.5)
_IS_NEW_MODEL = (NUM_OUTPUTS >= 100)
_USE_LETTERBOX = _IS_NEW_MODEL
_ARCFACE_S = 30.0

_model_type_str = (
    ("ArcFace" if _NEEDS_ARCFACE_SCALE else "logits-escalados")
    if _IS_NEW_MODEL else "legacy"
)
print(
    f'[classifier] Input: ({INPUT_C}, {INPUT_H}, {INPUT_W}) '
    f'| Output: {NUM_OUTPUTS} clases '
    f'| Tipo: {_model_type_str} '
    f'| Letterbox: {_USE_LETTERBOX} '
    f'| NeedsScale: {_NEEDS_ARCFACE_SCALE}'
)


# ═════════════════════════════════════════════════════════════════════════════
# 3. CONJUNTOS DE CARACTERES Y MAPAS DE CONFUSIÓN
# ═════════════════════════════════════════════════════════════════════════════

LETTERS_LOWER = set('abcdefghijklmnopqrstuvwxyzñ')
LETTERS_UPPER = set('ABCDEFGHIJKLMNOPQRSTUVWXYZÑ')
LETTERS_ACCENTED_LOWER = set('áéíóúü')
LETTERS_ACCENTED_UPPER = set('ÁÉÍÓÚÜ')
ALL_LETTERS = (
    LETTERS_LOWER | LETTERS_UPPER
    | LETTERS_ACCENTED_LOWER | LETTERS_ACCENTED_UPPER
)
DIGITS = set('0123456789')
PUNCTUATION = set('.,;:¿?¡!()-_\'"/@#$%&*+=<>')
STROKE_NAMES = {
    'línea_vertical', 'línea_horizontal',
    'línea_oblicua_derecha', 'línea_oblicua_izquierda',
    'curva', 'círculo',
}

ACCENT_TO_BASE: Dict[str, str] = {
    'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u',
    'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U',
}

BASE_TO_ACCENTED: Dict[str, List[str]] = defaultdict(list)
for _acc, _base in ACCENT_TO_BASE.items():
    BASE_TO_ACCENTED[_base].append(_acc)

KNOWN_CONFUSIONS: Dict[str, Dict] = {
    '¡': {'alternatives': ['i', 'j', 'l', 'r', '1'], 'group': 'punct_vertical'},
    '!': {'alternatives': ['i', 'j', 'l', 'r', '1'], 'group': 'punct_vertical'},
    '+': {'alternatives': ['t', 'T', 'H', '4'], 'group': 'punct_cross'},
    '?': {'alternatives': ['2', '3', 'g', 's', '5'], 'group': 'punct_curve'},
    '/': {'alternatives': ['l', '1', 'I', 'i'], 'group': 'punct_slash'},
    '"': {'alternatives': ['n', 'm', 'H', 'M', 'h'], 'group': 'punct_double'},
    "'": {'alternatives': ['v', 'r', 'i'], 'group': 'punct_quote'},
    '<': {'alternatives': ['z', 'Z', 'c', 'v'], 'group': 'punct_angle'},
    '>': {'alternatives': ['z', 'Z', 's'], 'group': 'punct_angle'},
    '&': {'alternatives': ['8', 'B', 'S'], 'group': 'punct_complex'},
    '#': {'alternatives': ['H', 'M'], 'group': 'punct_complex'},
    '$': {'alternatives': ['S', 's', '5'], 'group': 'punct_complex'},
    '_': {'alternatives': ['-', 'I', 'l', 'e'], 'group': 'punct_line'},
    ')': {'alternatives': ['Á', 'Ü', ',', 'c'], 'group': 'punct_paren'},
    '(': {'alternatives': ['C', 'c', 'G'], 'group': 'punct_paren'},
    '-': {'alternatives': ['_', 'I', 'l'], 'group': 'punct_line'},
    ',': {'alternatives': ['.', '9', 'i'], 'group': 'punct_dot'},
    '.': {'alternatives': [',', 'o', 'c'], 'group': 'punct_dot'},
    'p': {'alternatives': ['b', 'd', 'q', '9', 'P'], 'group': 'round_letter'},
    'P': {'alternatives': ['B', 'D', 'R', 'p'], 'group': 'round_letter'},
    '0': {'alternatives': ['O', 'o', 'Q', 'D'], 'group': 'zero_oh'},
    'O': {'alternatives': ['0', 'o', 'Q', 'D'], 'group': 'zero_oh'},
    'o': {'alternatives': ['0', 'O', 'c'], 'group': 'zero_oh'},
    '1': {'alternatives': ['l', 'I', '7', 'T', 'Z', 'i'], 'group': 'one_el'},
    'l': {'alternatives': ['1', 'I', '!', 'i', '|'], 'group': 'one_el'},
    'I': {'alternatives': ['1', 'l', 'i', '!', '-', '_'], 'group': 'one_el'},
    'W': {'alternatives': ['w', 'M', 'N'], 'group': 'wide_letter'},
    'w': {'alternatives': ['W', 'M', '%'], 'group': 'wide_letter'},
    'M': {'alternatives': ['W', 'w', 'N', 'm'], 'group': 'wide_letter'},
    'U': {'alternatives': ['u', 'V', 'Y', 'J'], 'group': 'u_shape'},
    'Y': {'alternatives': ['U', 'y', 'V', '!'], 'group': 'u_shape'},
    'y': {'alternatives': ['Y', 'U', 'v', '!'], 'group': 'u_shape'},
    'ñ': {'alternatives': ['n', 'm', 'h'], 'group': 'tilde_letter'},
    'Ñ': {'alternatives': ['N', 'M', 'W'], 'group': 'tilde_letter'},
    'c': {'alternatives': ['C', 'e', '(', 'G'], 'group': 'c_shape'},
    'C': {'alternatives': ['c', 'G', '(', 'Q'], 'group': 'c_shape'},
    'F': {'alternatives': ['f', 'E', 'T'], 'group': 'f_shape'},
    'E': {'alternatives': ['F', 'e', 'É'], 'group': 'f_shape'},
    'á': {'base': 'a', 'alternatives': ['a'], 'group': 'accent'},
    'é': {'base': 'e', 'alternatives': ['e'], 'group': 'accent'},
    'í': {'base': 'i', 'alternatives': ['i'], 'group': 'accent'},
    'ó': {'base': 'o', 'alternatives': ['o', '0', '6'], 'group': 'accent'},
    'ú': {'base': 'u', 'alternatives': ['u'], 'group': 'accent'},
    'Á': {'base': 'A', 'alternatives': ['A', '4'], 'group': 'accent'},
    'É': {'base': 'E', 'alternatives': ['E'], 'group': 'accent'},
    'Í': {'base': 'I', 'alternatives': ['I'], 'group': 'accent'},
    'Ó': {'base': 'O', 'alternatives': ['O', '0'], 'group': 'accent'},
    'Ú': {'base': 'U', 'alternatives': ['U'], 'group': 'accent'},
}


# ═════════════════════════════════════════════════════════════════════════════
# 4. DICCIONARIO ESPAÑOL
# ═════════════════════════════════════════════════════════════════════════════

SPANISH_COMMON_WORDS: Set[str] = {
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'de', 'en', 'a', 'por', 'para', 'con', 'sin', 'sobre', 'entre',
    'hacia', 'desde', 'hasta', 'según', 'durante', 'ante', 'bajo',
    'yo', 'tú', 'él', 'ella', 'nosotros', 'ellos', 'ellas',
    'me', 'te', 'se', 'nos', 'le', 'lo', 'les',
    'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
    'mi', 'tu', 'su', 'mis', 'tus', 'sus', 'nuestro', 'nuestra',
    'es', 'son', 'está', 'están', 'ser', 'estar', 'hay', 'tiene',
    'ha', 'han', 'fue', 'era', 'haber', 'hacer', 'ir', 'ver',
    'dar', 'saber', 'poder', 'querer', 'decir', 'venir', 'tener',
    'poner', 'salir', 'llegar', 'pasar', 'quedar', 'creer', 'dejar',
    'llamar', 'llevar', 'encontrar', 'pensar', 'seguir', 'hablar',
    'conocer', 'vivir', 'sentir', 'tratar', 'mirar', 'contar',
    'deber', 'trabajar', 'leer', 'escribir', 'jugar', 'comer',
    'dormir', 'correr', 'abrir', 'cerrar',
    'y', 'o', 'pero', 'que', 'como', 'si', 'cuando', 'donde',
    'porque', 'aunque', 'ni', 'sino', 'mientras', 'pues',
    'no', 'más', 'ya', 'muy', 'también', 'así', 'bien', 'aquí',
    'ahora', 'después', 'entonces', 'antes', 'siempre', 'nunca',
    'sí', 'hoy', 'mañana', 'ayer', 'mucho', 'poco', 'todo',
    'casa', 'nombre', 'parte', 'mundo', 'país', 'lugar', 'cosa',
    'forma', 'agua', 'tierra', 'ciudad', 'pueblo', 'calle',
    'escuela', 'trabajo', 'familia', 'padre', 'madre', 'hijo', 'hija',
    'hombre', 'mujer', 'día', 'año', 'tiempo', 'vida', 'vez',
    'mano', 'ojo', 'niño', 'niña', 'libro', 'carta', 'mesa',
    'puerta', 'ventana', 'camino', 'noche', 'gente', 'punto',
    'bueno', 'malo', 'grande', 'pequeño', 'nuevo', 'viejo', 'largo',
    'primero', 'último', 'mejor', 'mayor', 'menor', 'mismo', 'otro',
    'alto', 'bajo', 'solo', 'cada', 'poco', 'mucho', 'tanto',
    'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete',
    'ocho', 'nueve', 'diez', 'cien', 'mil',
}

SPANISH_COMMON_BIGRAMS: Set[str] = {
    'de', 'en', 'el', 'la', 'es', 'er', 'an', 'al', 'on', 'ar',
    'os', 'as', 'or', 'ue', 'ad', 'ci', 'do', 'le', 'ra', 'se',
    'ta', 'te', 'co', 'ca', 'io', 'da', 'ma', 'pa', 'ro', 'to',
    'na', 'no', 'un', 'in', 'me', 'ti', 'st', 'ne', 'lo', 're',
    'qu', 'po', 'tr', 'pr', 'mi', 'su', 'ha', 'pe', 'ie', 'ia',
    'mo', 'ri', 'li', 'di', 'si', 'so', 'ba', 'ni', 'nt', 'nd',
    'ch', 'll', 'ab', 'am', 'ac', 'ec', 'ed', 'em', 'ho', 'hu',
}

SPANISH_IMPOSSIBLE_SEQS: Set[str] = {
    'aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'hhh',
    'iii', 'jjj', 'kkk', 'lll', 'mmm', 'nnn', 'ooo', 'ppp',
    'qqq', 'rrr', 'sss', 'ttt', 'uuu', 'vvv', 'www', 'xxx',
    'yyy', 'zzz', 'kk', 'ww', 'yy', 'zx', 'xz', 'qw', 'wq',
    'zz', 'xx', 'vv', 'jj', 'qq', 'jk', 'kj', 'zq', 'qz',
}


# ═════════════════════════════════════════════════════════════════════════════
# 5. CONTEXTO DE CLASIFICACIÓN
# ═════════════════════════════════════════════════════════════════════════════

class CharContext:
    UNKNOWN = 'unknown'
    WORD_START = 'word_start'
    WORD_MIDDLE = 'word_middle'
    WORD_END = 'word_end'
    STANDALONE = 'standalone'
    DIGIT_SEQUENCE = 'digit_seq'
    SENTENCE_START = 'sentence_start'


# ═════════════════════════════════════════════════════════════════════════════
# 6. INFERENCIA ONNX + TTA
# ═════════════════════════════════════════════════════════════════════════════

# ── Configuración TTA ──
TTA_N = 5  # Número de augmentaciones (incluye original)
TTA_ROTATION_DEG = 3.0   # Grados de rotación para variantes
TTA_SCALE_DOWN = 0.95     # Factor de escala zoom-out
TTA_SCALE_UP = 1.05       # Factor de escala zoom-in


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _run_onnx(tensor: np.ndarray) -> np.ndarray:
    """Ejecuta ONNX y devuelve logits raw."""
    logits = session_cls.run(None, {_input_meta.name: tensor})[0][0]
    if _NEEDS_ARCFACE_SCALE:
        logits = logits * _ARCFACE_S
    return logits


def _logits_to_result(logits: np.ndarray, top_k: int = 10) -> Dict:
    """Convierte logits a resultado con probs y top-K."""
    probs = _softmax(logits)
    top_indices = np.argsort(probs)[::-1][:top_k]
    return {
        'probs': probs,
        'top_k': [
            (CLASS_MAP.get(int(i), f'?{i}'), float(probs[i]))
            for i in top_indices
        ],
        'top1_char': CLASS_MAP.get(
            int(top_indices[0]), f'?{top_indices[0]}'
        ),
        'top1_conf': float(probs[top_indices[0]]),
        'top1_idx': int(top_indices[0]),
    }


def _extract_clean_gray(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Extrae grayscale limpio de cualquier formato de entrada.

    Usado por TTA para obtener la imagen base antes de generar variantes.
    Retorna None si la imagen ya es un tensor preparado (no se puede hacer TTA).

    Args:
        img: imagen en cualquier formato

    Returns:
        Grayscale uint8 limpio, o None si ya es tensor
    """
    # Tensor ya preparado → no se puede hacer TTA
    if img.dtype == np.float32 and img.ndim == 4:
        return None

    # BGR → limpiar
    if len(img.shape) == 3 and img.shape[2] == 3:
        return clean_crop_for_classification(img)

    # Grayscale → asumir ya limpio
    if len(img.shape) == 2:
        return img.copy()

    # BGRA → convertir y limpiar
    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return clean_crop_for_classification(bgr)

    # 1 canal → extraer
    if len(img.shape) == 3 and img.shape[2] == 1:
        return img[:, :, 0].copy()

    # Fallback
    logger.warning(
        f"_extract_clean_gray: formato inesperado {img.shape}, "
        "intentando como grayscale"
    )
    try:
        return img.reshape(img.shape[0], img.shape[1]).copy()
    except Exception:
        return None


def _gray_to_tensor(gray: np.ndarray) -> np.ndarray:
    """
    Convierte grayscale limpio a tensor para ONNX.
    Wrapper sobre prepare_for_model / prepare_for_model_grayscale_1ch.
    """
    if INPUT_C == 1:
        return prepare_for_model_grayscale_1ch(
            gray, use_letterbox=_USE_LETTERBOX, target_size=INPUT_H
        )
    else:
        return prepare_for_model(
            gray, use_letterbox=_USE_LETTERBOX, target_size=INPUT_H
        )


def generate_tta_variants(gray_clean: np.ndarray) -> List[np.ndarray]:
    """
    Genera N variantes de una imagen grayscale limpia para TTA.

    Variantes (TTA_N = 5):
      1. Original (sin cambios)
      2. Rotación +3° alrededor del centro
      3. Rotación -3° alrededor del centro
      4. Escala 95% (zoom out — carácter más pequeño, más padding)
      5. Escala 105% (zoom in — carácter más grande, menos padding)

    Las augmentaciones son SUAVES para no distorsionar el carácter.
    El fondo de relleno usa el valor de fondo real de la imagen
    (típicamente ~245 = blanco) para no introducir artefactos.

    Args:
        gray_clean: grayscale uint8, ya limpiado por image_cleaner.
                    Fondo ~blanco, trazo ~negro, valores continuos.

    Returns:
        Lista de N imágenes grayscale uint8, mismas dimensiones.
    """
    h, w = gray_clean.shape[:2]

    # Determinar valor de fondo para padding (usar percentil alto = fondo)
    bg_value = int(np.percentile(gray_clean, 90))
    bg_value = max(bg_value, 200)  # Al menos gris claro

    variants = []

    # ── Variante 1: Original ──
    variants.append(gray_clean.copy())

    # ── Variante 2: Rotación +3° ──
    center = (w / 2.0, h / 2.0)
    M_pos = cv2.getRotationMatrix2D(center, TTA_ROTATION_DEG, 1.0)
    rot_pos = cv2.warpAffine(
        gray_clean, M_pos, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(bg_value),
    )
    variants.append(rot_pos)

    # ── Variante 3: Rotación -3° ──
    M_neg = cv2.getRotationMatrix2D(center, -TTA_ROTATION_DEG, 1.0)
    rot_neg = cv2.warpAffine(
        gray_clean, M_neg, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(bg_value),
    )
    variants.append(rot_neg)

    # ── Variante 4: Escala 95% (zoom out) ──
    new_h_down = max(1, int(h * TTA_SCALE_DOWN))
    new_w_down = max(1, int(w * TTA_SCALE_DOWN))
    interp_down = cv2.INTER_AREA  # Mejor para reducción
    scaled_down = cv2.resize(
        gray_clean, (new_w_down, new_h_down), interpolation=interp_down
    )
    canvas_down = np.full((h, w), bg_value, dtype=np.uint8)
    y0 = (h - new_h_down) // 2
    x0 = (w - new_w_down) // 2
    canvas_down[y0:y0 + new_h_down, x0:x0 + new_w_down] = scaled_down
    variants.append(canvas_down)

    # ── Variante 5: Escala 105% (zoom in) ──
    new_h_up = max(1, int(h * TTA_SCALE_UP))
    new_w_up = max(1, int(w * TTA_SCALE_UP))
    interp_up = cv2.INTER_LINEAR  # Mejor para ampliación
    scaled_up = cv2.resize(
        gray_clean, (new_w_up, new_h_up), interpolation=interp_up
    )
    # Recortar el centro para volver al tamaño original
    y0 = (new_h_up - h) // 2
    x0 = (new_w_up - w) // 2
    # Protección contra bordes
    y_end = min(y0 + h, new_h_up)
    x_end = min(x0 + w, new_w_up)
    cropped_up = scaled_up[y0:y_end, x0:x_end]
    # Si por redondeo quedó ligeramente diferente, ajustar
    if cropped_up.shape[0] != h or cropped_up.shape[1] != w:
        canvas_up = np.full((h, w), bg_value, dtype=np.uint8)
        ch, cw = cropped_up.shape[:2]
        canvas_up[:ch, :cw] = cropped_up
        variants.append(canvas_up)
    else:
        variants.append(cropped_up)

    return variants


def _run_inference_with_tta(
    img: np.ndarray, top_k: int = 10
) -> Dict:
    """
    Test-Time Augmentation: crea N versiones de la imagen,
    ejecuta inferencia en cada una, y promedia los LOGITS.

    El promedio de logits antes de softmax da un resultado
    más robusto que una sola pasada, mejorando ~2-3% de precisión
    sin re-entrenar el modelo.

    Pipeline:
      1. Extraer grayscale limpio de la imagen de entrada
      2. Generar N variantes (original, rotaciones, escalas)
      3. Cada variante pasa por prepare_for_model() independientemente
      4. Ejecutar ONNX en cada variante → obtener logits
      5. Promediar logits (NO probabilidades)
      6. Softmax sobre logits promediados → resultado final

    Args:
        img: imagen en cualquier formato (gray, BGR, tensor)
        top_k: número de predicciones a devolver

    Returns:
        Dict con top_k, top1_char, top1_conf, probs, top1_idx
    """
    # Extraer grayscale limpio
    gray_clean = _extract_clean_gray(img)

    if gray_clean is None:
        # Es un tensor ya preparado → no se puede hacer TTA, inferencia normal
        logger.debug("TTA: imagen ya es tensor, fallback a inferencia simple")
        logits = _run_onnx(img)
        return _logits_to_result(logits, top_k)

    # Generar variantes
    variants = generate_tta_variants(gray_clean)

    # Ejecutar inferencia en cada variante y recopilar logits
    all_logits = []
    for i, variant in enumerate(variants):
        tensor = _gray_to_tensor(variant)
        logits = _run_onnx(tensor)
        all_logits.append(logits)

    # Promediar LOGITS (no probabilidades)
    avg_logits = np.mean(all_logits, axis=0)

    logger.debug(
        f"TTA: {len(variants)} variantes procesadas, "
        f"logits promediados"
    )

    return _logits_to_result(avg_logits, top_k)


def _prepare_image(img: np.ndarray) -> np.ndarray:
    """
    Prepara una imagen para inferencia (sin TTA).

    Acepta:
      a) Grayscale uint8 (H, W) — de image_cleaner (CAMINO PRINCIPAL)
      b) BGR uint8 (H, W, 3) — crop directo
      c) Tensor float32 (1, C, H, W) — ya preparado

    En los casos (a) y (b), limpia con image_cleaner y luego
    aplica preprocessing.prepare_for_model().

    Args:
        img: imagen en cualquier formato aceptado

    Returns:
        Tensor float32 (1, C, H, W) listo para ONNX
    """
    # Caso C: ya es tensor preparado
    if img.dtype == np.float32 and img.ndim == 4:
        return img

    # Extraer grayscale limpio
    gray_clean = _extract_clean_gray(img)
    if gray_clean is None:
        # Fallback extremo
        gray_clean = np.full((INPUT_H, INPUT_W), 245, dtype=np.uint8)

    return _gray_to_tensor(gray_clean)


def _run_inference(
    img: np.ndarray, top_k: int = 10, use_tta: bool = False
) -> Dict:
    """
    Inferencia completa: imagen → resultado con top-K.

    Pipeline:
      1. Si TTA habilitado: generar variantes, promediar logits
      2. Si TTA deshabilitado: preparar imagen, ejecutar ONNX directo
      3. Softmax → top-K

    Args:
        img: imagen en cualquier formato (gray, BGR, tensor)
        top_k: número de predicciones a devolver
        use_tta: usar Test-Time Augmentation (mejora ~2-3% de precisión,
                 ~5x más lento). Recomendado para evaluación individual,
                 opcional para reconocimiento multi-carácter.

    Returns:
        Dict con top_k, top1_char, top1_conf, probs, top1_idx
    """
    if use_tta:
        return _run_inference_with_tta(img, top_k)

    tensor = _prepare_image(img)
    logits = _run_onnx(tensor)
    return _logits_to_result(logits, top_k)


# ═════════════════════════════════════════════════════════════════════════════
# 7. FUNCIONES PÚBLICAS — Inferencia raw
# ═════════════════════════════════════════════════════════════════════════════

def get_raw_top_k(
    img: np.ndarray, top_k: int = 10, use_tta: bool = False
) -> Dict:
    """
    Inferencia raw con limpieza automática.
    Acepta BGR, grayscale o máscara.
    """
    return _run_inference(img, top_k, use_tta=use_tta)


def classify_character(
    normalized_mask: np.ndarray, use_tta: bool = False
) -> Tuple[str, float]:
    """
    Clasifica desde máscara binaria (backward compatible).
    La imagen pasa por image_cleaner internamente.
    """
    result = _run_inference(normalized_mask, top_k=1, use_tta=use_tta)
    return result['top1_char'], result['top1_conf']


def classify_from_bgr(
    img_bgr: np.ndarray, use_tta: bool = True
) -> Tuple[str, float]:
    """
    Clasifica desde imagen BGR.
    Limpieza + preprocesamiento automático.
    TTA habilitado por defecto (evaluación individual).
    """
    result = _run_inference(img_bgr, top_k=1, use_tta=use_tta)
    return result['top1_char'], result['top1_conf']


def classify_from_clean_gray(
    gray_clean: np.ndarray, use_tta: bool = True
) -> Tuple[str, float]:
    """
    Clasifica desde grayscale ya limpiado por image_cleaner.
    Camino más directo y eficiente.
    TTA habilitado por defecto (evaluación individual).
    """
    result = _run_inference(gray_clean, top_k=1, use_tta=use_tta)
    return result['top1_char'], result['top1_conf']


# ═════════════════════════════════════════════════════════════════════════════
# 8. POST-PROCESAMIENTO CONTEXTUAL (SmartOCR) — SIN CAMBIOS
# ═════════════════════════════════════════════════════════════════════════════

def _get_prob_from_top_k(
    char: str, top_k: List[Tuple[str, float]]
) -> float:
    for ch, prob in top_k:
        if ch == char:
            return prob
    return 0.0


def _filter_by_type(
    top_k: List[Tuple[str, float]], expected_type: str
) -> List[Tuple[str, float]]:
    """Filtra top-K dejando solo caracteres del tipo esperado."""
    type_sets = {
        'letter': ALL_LETTERS,
        'letter_lower': LETTERS_LOWER | LETTERS_ACCENTED_LOWER,
        'letter_upper': LETTERS_UPPER | LETTERS_ACCENTED_UPPER,
        'digit': DIGITS,
        'punct': PUNCTUATION,
        'letter_or_digit': ALL_LETTERS | DIGITS,
    }
    allowed = type_sets.get(expected_type)
    if allowed is None:
        return top_k
    filtered = [(ch, p) for ch, p in top_k if ch in allowed]
    return filtered if filtered else top_k


def _resolve_confusion(
    raw_char: str, raw_conf: float,
    top_k: List[Tuple[str, float]],
    context: str,
    neighbors: Tuple[Optional[str], Optional[str]],
) -> Optional[Dict]:
    """Resuelve confusiones conocidas usando contexto posicional."""
    left_char, right_char = neighbors

    # Regla 1: Puntuación dentro de palabra → probablemente letra
    if context in (CharContext.WORD_MIDDLE, CharContext.WORD_START,
                   CharContext.WORD_END):
        if raw_char in PUNCTUATION and raw_char not in {"'", '-'}:
            for ch, prob in top_k:
                if ch in ALL_LETTERS and prob > 0.01:
                    return {
                        'char': ch, 'confidence': prob,
                        'raw_char': raw_char,
                        'raw_confidence': raw_conf,
                        'method': 'punct_in_word→letter',
                        'alternatives': top_k[:5],
                    }
            if raw_char in KNOWN_CONFUSIONS:
                alts = KNOWN_CONFUSIONS[raw_char].get('alternatives', [])
                for alt in alts:
                    if alt in ALL_LETTERS:
                        alt_prob = _get_prob_from_top_k(alt, top_k)
                        if alt_prob > 0.001:
                            return {
                                'char': alt, 'confidence': alt_prob,
                                'raw_char': raw_char,
                                'raw_confidence': raw_conf,
                                'method': 'confusion_table_in_word',
                                'alternatives': top_k[:5],
                            }
                if alts:
                    first_letter = next(
                        (a for a in alts if a in ALL_LETTERS), alts[0]
                    )
                    return {
                        'char': first_letter,
                        'confidence': raw_conf * 0.5,
                        'raw_char': raw_char,
                        'raw_confidence': raw_conf,
                        'method': 'forced_confusion_remap',
                        'alternatives': top_k[:5],
                    }

    # Regla 2: Acentuada con baja confianza → preferir base
    if raw_char in ACCENT_TO_BASE:
        base = ACCENT_TO_BASE[raw_char]
        base_prob = _get_prob_from_top_k(base, top_k)
        if raw_conf < 0.90 and base_prob > raw_conf * 0.15:
            combined = raw_conf + base_prob
            return {
                'char': base, 'confidence': min(combined, 1.0),
                'raw_char': raw_char, 'raw_confidence': raw_conf,
                'method': 'accent_uncertain→base',
                'alternatives': top_k[:5],
            }
        if context in (CharContext.WORD_MIDDLE, CharContext.WORD_START,
                       CharContext.WORD_END) and raw_conf < 0.95:
            return {
                'char': base, 'confidence': raw_conf * 0.9,
                'raw_char': raw_char, 'raw_confidence': raw_conf,
                'method': 'accent_in_word→base',
                'alternatives': top_k[:5],
            }

    # Regla 3: Dígito en contexto de palabra → letra
    if context in (CharContext.WORD_MIDDLE, CharContext.WORD_START,
                   CharContext.WORD_END):
        if raw_char in DIGITS:
            for ch, prob in top_k:
                if ch in ALL_LETTERS and prob > 0.01:
                    return {
                        'char': ch, 'confidence': prob,
                        'raw_char': raw_char,
                        'raw_confidence': raw_conf,
                        'method': 'digit_in_word→letter',
                        'alternatives': top_k[:5],
                    }
            if raw_char in KNOWN_CONFUSIONS:
                alts = KNOWN_CONFUSIONS[raw_char].get('alternatives', [])
                for alt in alts:
                    if alt in ALL_LETTERS:
                        return {
                            'char': alt,
                            'confidence': raw_conf * 0.5,
                            'raw_char': raw_char,
                            'raw_confidence': raw_conf,
                            'method': 'digit_confusion→letter',
                            'alternatives': top_k[:5],
                        }

    # Regla 4: Mayúscula en medio de palabra → minúscula
    if context == CharContext.WORD_MIDDLE:
        if raw_char in LETTERS_UPPER or raw_char in LETTERS_ACCENTED_UPPER:
            lower = raw_char.lower()
            lower_prob = _get_prob_from_top_k(lower, top_k)
            if lower_prob > raw_conf * 0.05:
                return {
                    'char': lower,
                    'confidence': max(lower_prob, raw_conf * 0.8),
                    'raw_char': raw_char,
                    'raw_confidence': raw_conf,
                    'method': 'upper_in_middle→lower',
                    'alternatives': top_k[:5],
                }
            if lower in CHAR2IDX:
                return {
                    'char': lower, 'confidence': raw_conf * 0.7,
                    'raw_char': raw_char,
                    'raw_confidence': raw_conf,
                    'method': 'force_lower_in_middle',
                    'alternatives': top_k[:5],
                }

    # Regla 5: Letra en secuencia de dígitos → dígito
    if context == CharContext.DIGIT_SEQUENCE:
        if raw_char in ALL_LETTERS:
            for ch, prob in top_k:
                if ch in DIGITS and prob > 0.01:
                    return {
                        'char': ch, 'confidence': prob,
                        'raw_char': raw_char,
                        'raw_confidence': raw_conf,
                        'method': 'letter_in_digits→digit',
                        'alternatives': top_k[:5],
                    }
            if raw_char in KNOWN_CONFUSIONS:
                alts = KNOWN_CONFUSIONS[raw_char].get('alternatives', [])
                for alt in alts:
                    if alt in DIGITS:
                        return {
                            'char': alt,
                            'confidence': raw_conf * 0.5,
                            'raw_char': raw_char,
                            'raw_confidence': raw_conf,
                            'method': 'letter_confusion→digit',
                            'alternatives': top_k[:5],
                        }

    # Regla 6: Trazo en contexto de texto → buscar alternativa
    if context != CharContext.STANDALONE:
        if raw_char in STROKE_NAMES:
            for ch, prob in top_k:
                if ch not in STROKE_NAMES and prob > 0.01:
                    return {
                        'char': ch, 'confidence': prob,
                        'raw_char': raw_char,
                        'raw_confidence': raw_conf,
                        'method': 'stroke_in_text→char',
                        'alternatives': top_k[:5],
                    }

    return None


def _boost_expected_char(
    top_k: List[Tuple[str, float]],
    expected_char: str,
    boost_factor: float = 3.0,
) -> List[Tuple[str, float]]:
    """
    Cuando sabemos qué carácter espera el usuario (evaluación),
    boostar la probabilidad de ese carácter y sus variantes cercanas.
    """
    if not expected_char or not top_k:
        return top_k

    acceptable = {expected_char}

    if expected_char.isalpha():
        acceptable.add(expected_char.lower())
        acceptable.add(expected_char.upper())

    if expected_char in ACCENT_TO_BASE:
        acceptable.add(ACCENT_TO_BASE[expected_char])
    base_lower = expected_char.lower()
    if base_lower in BASE_TO_ACCENTED:
        for acc in BASE_TO_ACCENTED[base_lower]:
            acceptable.add(acc)
            acceptable.add(acc.upper())

    for pred_char, conf_info in KNOWN_CONFUSIONS.items():
        alts = conf_info.get('alternatives', [])
        if expected_char in alts or expected_char.lower() in alts:
            acceptable.add(pred_char)

    boosted = []
    for ch, prob in top_k:
        if ch in acceptable:
            boosted.append((ch, min(prob * boost_factor, 1.0)))
        else:
            boosted.append((ch, prob))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


# ═════════════════════════════════════════════════════════════════════════════
# 9. CLASIFICACIÓN INTELIGENTE (SmartOCR)
# ═════════════════════════════════════════════════════════════════════════════

def classify_char_smart(
    img: np.ndarray,
    context: str = CharContext.UNKNOWN,
    expected_type: Optional[str] = None,
    expected_char: Optional[str] = None,
    neighbors: Tuple[Optional[str], Optional[str]] = (None, None),
    use_tta: bool = True,
) -> Dict:
    """
    Clasificación inteligente con:
    1. Limpieza automática (image_cleaner)
    2. Preprocesamiento exacto del entrenamiento (preprocessing)
    3. Inferencia ONNX (con TTA opcional — habilitado por defecto)
    4. Boost del carácter esperado (evaluación de trazo)
    5. Filtro por tipo esperado
    6. Resolución de confusiones contextuales

    Args:
        img: imagen del carácter (gray, BGR, o tensor)
        context: contexto posicional (word_start, word_middle, etc.)
        expected_type: tipo esperado (letter, digit, punct, etc.)
        expected_char: carácter esperado (para boost en evaluación)
        neighbors: (char_izquierdo, char_derecho) para contexto
        use_tta: usar Test-Time Augmentation (default: True)
    """
    raw = _run_inference(img, top_k=15, use_tta=use_tta)

    raw_char = raw['top1_char']
    raw_conf = raw['top1_conf']
    top_k = raw['top_k']

    def _make_result(char, conf, method):
        return {
            'char': char,
            'confidence': min(conf, 1.0),
            'raw_char': raw_char,
            'raw_confidence': raw_conf,
            'method': method,
            'alternatives': top_k[:5],
        }

    # ── PASO 0: Boost del carácter esperado (evaluación) ──
    working_top_k = top_k
    if expected_char:
        working_top_k = _boost_expected_char(
            top_k, expected_char, boost_factor=3.0
        )
        if working_top_k and working_top_k[0][0] != raw_char:
            boosted_char, boosted_conf = working_top_k[0]
            original_conf = _get_prob_from_top_k(boosted_char, top_k)
            if original_conf > 0.005:
                return _make_result(
                    boosted_char, boosted_conf, 'expected_boost'
                )

    # ── PASO 1: Filtro por tipo esperado ──
    if expected_type:
        filtered = _filter_by_type(working_top_k, expected_type)
        if filtered:
            best_char, best_conf = filtered[0]
            if best_char != raw_char:
                return _make_result(
                    best_char, best_conf, 'type_filter'
                )

    # ── PASO 2: Resolver confusiones con contexto ──
    resolved = _resolve_confusion(
        raw_char, raw_conf, working_top_k, context, neighbors
    )
    if resolved:
        return resolved

    # ── PASO 3: Confianza alta sin contexto → aceptar ──
    if raw_conf > 0.97 and context == CharContext.UNKNOWN:
        return _make_result(raw_char, raw_conf, 'high_confidence')

    return _make_result(raw_char, raw_conf, 'raw')


def classify_mask_smart(
    normalized_mask: np.ndarray,
    context: str = CharContext.UNKNOWN,
    expected_type: Optional[str] = None,
    expected_char: Optional[str] = None,
    neighbors: Tuple[Optional[str], Optional[str]] = (None, None),
    use_tta: bool = True,
) -> Dict:
    """Clasificación inteligente desde máscara binaria (backward compatible)."""
    return classify_char_smart(
        normalized_mask, context, expected_type, expected_char, neighbors,
        use_tta=use_tta,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 10. CLASIFICACIÓN A NIVEL DE PALABRA
# ═════════════════════════════════════════════════════════════════════════════

def classify_word(
    char_images: List[np.ndarray],
    expect_type: str = 'letter',
    is_sentence_start: bool = False,
    use_tta: bool = False,
) -> Dict:
    """
    Clasifica una secuencia de imágenes como PALABRA.

    Args:
        char_images: lista de imágenes de caracteres individuales
        expect_type: tipo esperado ('letter', 'digit', 'mixed')
        is_sentence_start: True si es inicio de oración
        use_tta: usar TTA por carácter (default: False para velocidad)
    """
    n = len(char_images)
    if n == 0:
        return {
            'word': '', 'raw_word': '', 'chars': [],
            'confidence': 0.0, 'corrected': False,
            'correction_method': 'none',
        }    
    char_results = []
    for i, img in enumerate(char_images):
        if n == 1:
            ctx = CharContext.STANDALONE
        elif i == 0:
            ctx = (CharContext.SENTENCE_START if is_sentence_start
                   else CharContext.WORD_START)
        elif i == n - 1:
            ctx = CharContext.WORD_END
        else:
            ctx = CharContext.WORD_MIDDLE

        if expect_type == 'digit':
            etype = 'digit'
        elif expect_type == 'letter':
            if i == 0 and is_sentence_start:
                etype = 'letter_upper'
            elif i == 0:
                etype = 'letter'
            else:
                etype = 'letter_lower'
        elif expect_type == 'mixed':
            etype = 'letter_or_digit'
        else:
            etype = None

        left = char_results[i - 1]['char'] if i > 0 else None

        result = classify_char_smart(
            img, context=ctx, expected_type=etype,
            neighbors=(left, None),
            use_tta=use_tta,
        )
        char_results.append(result)

    # Segundo pase para caracteres de baja confianza
    for i in range(1, n - 1):
        left = char_results[i - 1]['char']
        right = char_results[i + 1]['char'] if i + 1 < n else None

        if char_results[i]['confidence'] < 0.7:
            ctx = CharContext.WORD_MIDDLE
            if expect_type == 'letter':
                etype = 'letter_lower'
            elif expect_type == 'digit':
                etype = 'digit'
            else:
                etype = None

            new_result = classify_char_smart(
                char_images[i], context=ctx,
                expected_type=etype, neighbors=(left, right),
                use_tta=use_tta,
            )
            if new_result['confidence'] > char_results[i]['confidence']:
                char_results[i] = new_result

    raw_word = ''.join(r['char'] for r in char_results)
    word = raw_word
    corrected = False
    correction_method = 'none'

    if expect_type in ('letter', 'mixed') and len(word) >= 2:
        dict_word = _dictionary_correct(word, char_results)
        if dict_word and dict_word.lower() != word.lower():
            word = dict_word
            corrected = True
            correction_method = 'dictionary'

    if not corrected and len(word) >= 2:
        seq_word = _sequence_correct(word, char_results)
        if seq_word and seq_word != word:
            word = seq_word
            corrected = True
            correction_method = 'sequence_fix'

    avg_conf = float(np.mean([r['confidence'] for r in char_results]))

    return {
        'word': word,
        'raw_word': raw_word,
        'chars': char_results,
        'confidence': avg_conf,
        'corrected': corrected,
        'correction_method': correction_method,
    }


def classify_word_from_masks(
    masks: List[np.ndarray],
    expect_type: str = 'letter',
    is_sentence_start: bool = False,
    use_tta: bool = False,
) -> Dict:
    """Igual que classify_word pero con máscaras binarias."""
    return classify_word(masks, expect_type, is_sentence_start, use_tta=use_tta)


def _dictionary_correct(
    word: str, char_results: List[Dict]
) -> Optional[str]:
    word_lower = word.lower()
    if word_lower in SPANISH_COMMON_WORDS:
        return word

    best_match = None
    best_score = -1
    variants = _generate_word_variants(word, char_results, max_changes=2)

    for variant in variants:
        if variant.lower() in SPANISH_COMMON_WORDS:
            score = sum(
                1 for a, b in zip(word.lower(), variant.lower())
                if a == b
            )
            if score > best_score:
                best_score = score
                best_match = variant

    return best_match


def _generate_word_variants(
    word: str, char_results: List[Dict], max_changes: int = 2
) -> List[str]:
    variants: Set[str] = set()
    chars = list(word)
    n = len(chars)

    for i in range(n):
        alts = char_results[i].get('alternatives', [])
        for alt_char, alt_prob in alts[:6]:
            if alt_char != chars[i] and alt_prob > 0.005:
                v = chars.copy()
                v[i] = alt_char
                variants.add(''.join(v))

        ch = chars[i]
        if ch in ACCENT_TO_BASE:
            v = chars.copy()
            v[i] = ACCENT_TO_BASE[ch]
            variants.add(''.join(v))

        base_lower = ch.lower()
        if base_lower in BASE_TO_ACCENTED:
            for acc in BASE_TO_ACCENTED[base_lower]:
                v = chars.copy()
                v[i] = acc if ch.islower() else acc.upper()
                variants.add(''.join(v))

        if ch.isalpha():
            v = chars.copy()
            v[i] = ch.swapcase()
            variants.add(''.join(v))

    if max_changes >= 2 and n <= 8:
        for i in range(n):
            for j in range(i + 1, n):
                alts_i = char_results[i].get('alternatives', [])
                alts_j = char_results[j].get('alternatives', [])
                for ai, _ in alts_i[:3]:
                    for aj, _ in alts_j[:3]:
                        v = chars.copy()
                        v[i] = ai
                        v[j] = aj
                        variants.add(''.join(v))

    return list(variants)


def _sequence_correct(
    word: str, char_results: List[Dict]
) -> Optional[str]:
    word_lower = word.lower()
    if len(word_lower) < 2:
        return None

    for i in range(len(word_lower) - 1):
        seq2 = word_lower[i:i + 2]
        seq3 = (
            word_lower[i:i + 3]
            if i + 3 <= len(word_lower) else ''
        )

        is_bad = (
            seq2 in SPANISH_IMPOSSIBLE_SEQS
            or seq3 in SPANISH_IMPOSSIBLE_SEQS
        )

        if is_bad:
            positions = [i, i + 1]
            positions.sort(
                key=lambda p: char_results[p]['confidence']
            )
            for pos in positions:
                alts = char_results[pos].get('alternatives', [])
                for alt_char, alt_prob in alts[:5]:
                    if alt_char != word[pos] and alt_prob > 0.005:
                        v = list(word)
                        v[pos] = alt_char
                        new_word = ''.join(v)
                        new_lower = new_word.lower()
                        new_seq2 = new_lower[i:i + 2]
                        new_seq3 = (
                            new_lower[i:i + 3]
                            if i + 3 <= len(new_lower) else ''
                        )
                        if (new_seq2 not in SPANISH_IMPOSSIBLE_SEQS
                                and new_seq3
                                not in SPANISH_IMPOSSIBLE_SEQS):
                            return new_word
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 11. CLASIFICACIÓN A NIVEL DE LÍNEA
# ═════════════════════════════════════════════════════════════════════════════

def classify_line(
    word_groups: List[List[np.ndarray]],
    is_first_line: bool = False,
    use_tta: bool = False,
) -> Dict:
    """
    Clasifica una línea completa de texto (múltiples palabras).

    Args:
        word_groups: lista de grupos de imágenes (cada grupo = una palabra)
        is_first_line: True si es la primera línea (inicio de oración)
        use_tta: usar TTA por carácter (default: False para velocidad)
    """
    words = []
    for i, char_images in enumerate(word_groups):
        if not char_images:
            continue

        is_sentence_start = (i == 0 and is_first_line)
        first_raw = get_raw_top_k(char_images[0], top_k=3, use_tta=False)
        first_char = first_raw['top1_char']
        expect = 'digit' if first_char in DIGITS else 'letter'

        result = classify_word(
            char_images,
            expect_type=expect,
            is_sentence_start=is_sentence_start,
            use_tta=use_tta,
        )
        words.append(result)

    text = ' '.join(w['word'] for w in words)
    avg_conf = (
        float(np.mean([w['confidence'] for w in words]))
        if words else 0.0
    )

    return {
        'text': text,
        'words': words,
        'confidence': avg_conf,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 12. UTILIDADES PÚBLICAS
# ═════════════════════════════════════════════════════════════════════════════

def get_confusion_info(char: str) -> Dict:
    """Devuelve información de confusiones conocidas para un carácter."""
    info: Dict = {'char': char, 'known_confusions': []}
    for pred_char, conf_data in KNOWN_CONFUSIONS.items():
        alts = conf_data.get('alternatives', [])
        base = conf_data.get('base')
        if char == pred_char or char in alts or char == base:
            info['known_confusions'].append({
                'predicted_as': pred_char,
                'alternatives': alts,
                'base': base,
            })
    return info


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax público (backward compatible)."""
    return _softmax(x)


def get_tta_config() -> Dict:
    """Devuelve la configuración actual de TTA (para debugging)."""
    return {
        'tta_n': TTA_N,
        'rotation_deg': TTA_ROTATION_DEG,
        'scale_down': TTA_SCALE_DOWN,
        'scale_up': TTA_SCALE_UP,
        'model_type': _model_type_str,
        'input_shape': (INPUT_C, INPUT_H, INPUT_W),
        'num_classes': NUM_OUTPUTS,
        'needs_arcface_scale': _NEEDS_ARCFACE_SCALE,
        'use_letterbox': _USE_LETTERBOX,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 13. DEBUG (SIMPLIFICADO)
# ═════════════════════════════════════════════════════════════════════════════

def debug_check_image(
    img: np.ndarray, label: str = "", use_tta: bool = True
) -> Dict:
    """
    Debug: verifica qué ve el modelo.
    Simplificado — ya no necesita probar polaridades porque
    image_cleaner se encarga.

    Ahora incluye resultado con y sin TTA para comparar.
    """
    if len(img.shape) == 2:
        gray = img
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape[:2]
    overall_mean = float(gray.mean())

    border_pixels = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ]) if h > 2 and w > 2 else gray.ravel()

    border_mean = float(border_pixels.mean())

    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    center_mean = float(center.mean()) if center.size > 0 else 0

    # Resultado sin TTA (rápido)
    result_no_tta = _run_inference(img, top_k=5, use_tta=False)

    # Resultado con TTA (más preciso)
    result_with_tta = _run_inference(img, top_k=5, use_tta=True) if use_tta else None

    debug_info = {
        'label': label,
        'shape': img.shape,
        'dtype': str(img.dtype),
        'border_mean': round(border_mean, 1),
        'center_mean': round(center_mean, 1),
        'overall_mean': round(overall_mean, 1),
        'result_no_tta': {
            'char': result_no_tta['top1_char'],
            'conf': round(result_no_tta['top1_conf'], 4),
            'top5': result_no_tta['top_k'][:5],
        },
    }

    if result_with_tta is not None:
        debug_info['result_with_tta'] = {
            'char': result_with_tta['top1_char'],
            'conf': round(result_with_tta['top1_conf'], 4),
            'top5': result_with_tta['top_k'][:5],
        }
        debug_info['tta_changed_prediction'] = (
            result_no_tta['top1_char'] != result_with_tta['top1_char']
        )

    # Backward compatible: 'result' key apunta al resultado con TTA si disponible
    debug_info['result'] = debug_info.get('result_with_tta', debug_info['result_no_tta'])

    return debug_info