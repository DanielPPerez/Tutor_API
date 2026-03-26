"""
generate_templates.py
=====================
Genera plantillas ideales leyendo las clases desde app/models/char_map.json
"""

import argparse
import os
import sys
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configurar paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.core import config

try:
    from skimage.morphology import skeletonize as ski_skeletonize
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False
    print("AVISO: scikit-image no encontrado. Se omitirá la esqueletización.")

# =============================================================================
# Carga de Clases desde JSON
# =============================================================================

def load_chars_from_map():
    path = os.path.join(os.path.dirname(__file__), "../../app/models/char_map.json")
    if not os.path.exists(path):
        print(f"ERROR: No se encontró el mapa en {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        content = data.get("root", data)
        idx2char = content.get("idx2char", {})
        if not idx2char:
            print("ERROR: No se encontraron clases en 'idx2char'.")
            return []
        sorted_keys = sorted(idx2char.keys(), key=lambda x: int(x))
        return [idx2char[k] for k in sorted_keys]
    except Exception as e:
        print(f"Error procesando JSON: {e}")
        return []

# =============================================================================
# Utilidades de nombre de archivo
# =============================================================================

def get_safe_filename(char: str) -> str:
    mapping = {
        '.': 'period', ',': 'comma', ';': 'semicolon', ':': 'colon',
        '¿': 'question_open', '?': 'question', '¡': 'excl_open', '!': 'excl',
        '(': 'lparen', ')': 'rparen', '-': 'hyphen', '_': 'underscore',
        "'": 'quote', '"': 'dquote', '/': 'slash', '@': 'at',
        '#': 'hash', '$': 'dollar', '%': 'percent', '&': 'ampersand',
        '*': 'asterisk', '+': 'plus', '=': 'equals', '<': 'lt', '>': 'gt',
        'á': 'a_tilde', 'é': 'e_tilde', 'í': 'i_tilde', 'ó': 'o_tilde', 'ú': 'u_tilde',
        'Á': 'A_tilde_upper', 'É': 'E_tilde_upper', 'Í': 'I_tilde_upper', 
        'Ó': 'O_tilde_upper', 'Ú': 'U_tilde_upper', 'ñ': 'enie', 'Ñ': 'ENIE_upper',
        'ü': 'u_diaeresis', 'Ü': 'U_diaeresis_upper'
    }
    if char in mapping: return mapping[char]
    if len(char) > 1:
        s = char.lower().replace(" ", "_")
        for a, b in zip("áéíóúü", "aeiouu"): s = s.replace(a, b)
        return s
    suffix = "upper" if char.isupper() else "lower"
    return f"{char}_{suffix}"

# =============================================================================
# Procesamiento de Imagen (Renders y Morfología)
# =============================================================================

def render_char_hires(char: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    size = config.TEMPLATE_RENDER_SIZE
    img  = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    render_text = char if len(char) == 1 else char[0] 
    left, top, right, bottom = font.getbbox(render_text)
    w, h = right - left, bottom - top
    draw.text(((size - w) / 2 - left, (size - h) / 2 - top), render_text, font=font, fill=255)
    return np.array(img)

def crop_and_letterbox(canvas: np.ndarray) -> np.ndarray | None:
    coords = cv2.findNonZero(canvas)
    if coords is None: return None
    x, y, wc, hc = cv2.boundingRect(coords)
    crop = canvas[y:y + hc, x:x + wc]
    inner = config.TARGET_SIZE - 2 * config.TEMPLATE_MARGIN
    scale = inner / max(wc, hc)
    new_w, new_h = max(1, int(wc * scale)), max(1, int(hc * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    final = np.zeros((config.TARGET_SIZE, config.TARGET_SIZE), dtype=np.uint8)
    final[(config.TARGET_SIZE-new_h)//2 : (config.TARGET_SIZE-new_h)//2 + new_h,
          (config.TARGET_SIZE-new_w)//2 : (config.TARGET_SIZE-new_w)//2 + new_w] = resized
    _, binary = cv2.threshold(cv2.GaussianBlur(final, (3, 3), 0), 100, 255, cv2.THRESH_BINARY)
    return binary

def skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    if not SKIMAGE_OK: return binary
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    return (ski_skeletonize(clean > 0)).astype(np.uint8) * 255

def skeletonize_student_char(image_array: np.ndarray) -> np.ndarray:
    """
    Función exportada para la API (endpoints.py).
    """
    if image_array is None or image_array.size == 0:
        return np.zeros((config.TARGET_SIZE, config.TARGET_SIZE), dtype=np.uint8)
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
    return skeletonize_binary(binary)

def dilate_skeleton(skeleton: np.ndarray, kernel_size: int) -> np.ndarray:
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    return cv2.dilate(skeleton, k, iterations=config.TEMPLATE_DILATE_ITERATIONS)

def save_template(array: np.ndarray, out_dir: str, name: str, suffix: str) -> None:
    base = os.path.join(out_dir, f"{name}_{suffix}")
    np.save(f"{base}.npy", (array > 0).astype(np.uint8))
    cv2.imwrite(f"{base}.png", array)

# =============================================================================
# Pipeline principal
# =============================================================================

def generate_clean_templates(filter_char: str | None = None) -> None:
    alphabet = load_chars_from_map()
    if not alphabet: return
    out_dir = config.TEMPLATE_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    for level in config.TEMPLATE_DIFFICULTY_KERNELS:
        os.makedirs(os.path.join(out_dir, level), exist_ok=True)
    if config.TEMPLATE_SAVE_SKELETON:
        os.makedirs(os.path.join(out_dir, "skeleton"), exist_ok=True)

    try:
        font = ImageFont.truetype(config.FONT_PATH, config.TEMPLATE_FONT_SIZE)
    except Exception as e:
        print(f"Error cargando fuente: {e}"); return

    if filter_char:
        alphabet = [filter_char] if filter_char in alphabet else []

    for i, char in enumerate(alphabet, 1):
        name = get_safe_filename(char)
        canvas = render_char_hires(char, font)
        binary = crop_and_letterbox(canvas)
        if binary is None: continue
        skeleton = skeletonize_binary(binary)
        if config.TEMPLATE_SAVE_SKELETON:
            save_template(skeleton, os.path.join(out_dir, "skeleton"), name, "skeleton")
        for level_name, ks in config.TEMPLATE_DIFFICULTY_KERNELS.items():
            carril = dilate_skeleton(skeleton, ks)
            save_template(carril, os.path.join(out_dir, level_name), name, level_name)
        print(f"  [{i:03d}/{len(alphabet)}] '{char}' -> {name} OK")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", default=None)
    args = parser.parse_args()
    generate_clean_templates(filter_char=args.char)