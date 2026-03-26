"""
app/scripts/generate_synthetic_yolo.py
=======================================
Genera el dataset sintético YOLO para el detector de trazos caligráficos.

FUENTES DE IMÁGENES — TODAS LAS CLASES DEL CHAR_MAP
-----------------------------------------------------
El script itera sobre TODAS las clases definidas en char_map.json (107 clases),
no solo las 62 de EMNIST.  Para cada clase busca imágenes reales en TODOS los
datasets disponibles:

  Dataset                               Clases aportadas (aprox.)
  ─────────────────────────────────────────────────────────────────
  EMNIST By Class (crawford/emnist)     0–9, A–Z, a–z  (62 clases)
  handwritting_characters_database      a–z, A–Z, 0–9, símbolos
  spanish_handwritten_characters_words  a–z, A–Z, ñ, Ñ, á é í ó ú, etc.
  IAM Handwriting                       palabras completas → se omite para
                                        composición por carácter individual

ÍNDICE UNIFICADO _build_class_image_index()
-------------------------------------------
Construye un dict  { char: list[source] }  donde cada source es:
  · ("emnist", class_idx, dataset_obj) — imagen desde torchvision EMNIST
  · Path                               — ruta a imagen .png/.jpg/.bmp

CONTEOS POR CLASE (leídos desde dataset_classes_report.json)
------------------------------------------------------------
  · Trazo primitivo (línea_*, curva, círculo) → PRIMITIVE_CLASS_COUNT  = 150
    (siempre dibujados con OpenCV; ningún dataset los contiene)
  · Clase faltante  (global_missing)          → MISSING_CLASS_COUNT   = 100
    (sin datos reales → composición con fuentes sintéticas / fallback)
  · Clase existente                           → EXISTING_CLASS_COUNT  = 50
    (tiene imágenes reales en ≥1 dataset → se usan como fuente)

FONDOS: .webp · .avif · .jpg · .jpeg · .png  (+ sintéticos si no hay)

Estructura de salida
--------------------
data/processed/yolo_dataset/
  images/train/    ← imágenes compuestas (640×640)
  labels/train/    ← etiquetas YOLO  "0 xc yc w h"
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torchvision
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    import pillow_avif  # noqa: F401
    _AVIF_OK = True
except ImportError:
    _AVIF_OK = False


# =============================================================================
# Configuración global
# =============================================================================

DATA_ROOT                = "./data"
OUTPUT_PATH              = "./data/processed/yolo_dataset"
BG_PATH                  = "./data/backgrounds"
REPORT_PATH              = "./data/dataset_classes_report.json"
CHAR_MAP_PATH            = "./app/models/char_map.json"

# Rutas de cada dataset (deben coincidir con dataset_downloads.py)
EMNIST_ROOT              = "./data/raw/emnist_byclass"
HWC_ROOT                 = "./data/raw/handwritting_characters_database"
SPANISH_ROOT             = "./data/raw/spanish_handwritten_characters_words"
# IAM se omite para composición por carácter (imágenes de palabras completas)

IMG_SIZE                 = 640
LETTER_SIZE_MIN          = 55
LETTER_SIZE_MAX          = 160

# ── Conteos por tipo de clase ────────────────────────────────────────────────
MISSING_CLASS_COUNT      = 200
EXISTING_CLASS_COUNT     = 10
PRIMITIVE_CLASS_COUNT    = 250
IMAGES_PER_CHAR_FALLBACK = 80

# ── Trazos primitivos — OpenCV; ningún dataset los contiene ─────────────────
PRIMITIVE_STROKES: list[str] = [
    "línea_vertical",
    "línea_horizontal",
    "línea_oblicua_derecha",
    "línea_oblicua_izquierda",
    "curva",
    "círculo",
]

# ── Augmentación ─────────────────────────────────────────────────────────────
INCLUDE_SYNTHETIC_BG     = True
SYNTHETIC_BG_FRACTION    = 0.15
PROB_SHADOW              = 0.45
PROB_GRADIENT_LIGHT      = 0.35
PROB_PENCIL_TEXTURE      = 0.50
PROB_INK_VARIATION       = 0.40
PROB_BLUR                = 0.25
PROB_NOISE               = 0.30
PROB_ROTATION            = 0.80
MAX_ROTATION_DEG         = 18

# Extensiones de imagen soportadas para datasets de archivos
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# =============================================================================
# Carga del char_map
# =============================================================================

def _load_char_map(path: str = CHAR_MAP_PATH) -> dict[str, Any]:
    """Carga char_map.json y devuelve dict con idx2char, char2idx, num_classes."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"char_map.json no encontrado en '{path}'.\n"
            "Ejecuta verify_dataset_classes.py primero."
        )
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        idx2char = {str(i): c for i, c in enumerate(raw)}
    elif "idx2char" in raw:
        idx2char = {str(k): v for k, v in raw["idx2char"].items()}
    else:
        idx2char = {str(k): v for k, v in raw.items() if str(k).isdigit()}

    char2idx = {v: int(k) for k, v in idx2char.items()}
    return {"idx2char": idx2char, "char2idx": char2idx, "num_classes": len(idx2char)}


# =============================================================================
# Lectura del reporte de cobertura
# =============================================================================

def load_coverage_report(report_path: str = REPORT_PATH) -> set[str]:
    """
    Lee dataset_classes_report.json (generado por verify_dataset_classes.py).

    Returns
    -------
    set[str]  — clases en global_missing; set vacío si no existe el archivo.
    """
    path = Path(report_path)
    if not path.exists():
        print(
            f"  [WARN] Reporte no encontrado en '{report_path}'.\n"
            f"         Usando fallback: {IMAGES_PER_CHAR_FALLBACK} imgs/clase."
        )
        return set()

    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    missing: list[str] = report.get("global_missing", [])
    prims_ok = all(p in missing for p in PRIMITIVE_STROKES)

    print(
        f"  Reporte: {report.get('char_map_classes', '?')} clases, "
        f"{len(missing)} faltantes."
    )
    if not prims_ok:
        absent = [p for p in PRIMITIVE_STROKES if p not in missing]
        print(f"  [INFO] Trazos primitivos no en global_missing (se generan igual): {absent}")
    else:
        print(f"  [OK] Trazos primitivos en global_missing → {PRIMITIVE_CLASS_COUNT} imgs c/u.")

    return set(missing)


def get_images_per_char(char: str, missing_classes: set[str]) -> int:
    """Número de imágenes a generar para una clase (no primitiva)."""
    if not missing_classes:
        return IMAGES_PER_CHAR_FALLBACK
    return MISSING_CLASS_COUNT if char in missing_classes else EXISTING_CLASS_COUNT


# =============================================================================
# Índice unificado de imágenes por clase (TODOS los datasets)
# =============================================================================

# Tipo de fuente: tupla EMNIST o Path de archivo
EmnistSource = tuple[str, int, Any]   # ("emnist", class_idx, dataset_obj)
FileSource   = Path
ImageSource  = EmnistSource | FileSource


def _index_emnist(
    emnist_root: str,
    char2idx: dict[str, int],
) -> dict[str, list[ImageSource]]:
    """
    Indexa EMNIST By Class.

    Mapeo estándar byclass: 0–9=0–9, A–Z=10–35, a–z=36–61
    """
    result: dict[str, list[ImageSource]] = {}

    try:
        ds = torchvision.datasets.EMNIST(
            root=emnist_root, split="byclass", train=True, download=False
        )
    except Exception as e:
        print(f"  [WARN] EMNIST no disponible: {e}")
        return result

    print("    Indexando EMNIST (≈30 s) ...")

    # Construir mapeo bidireccional: índice EMNIST → carácter
    emnist_chars = (
        [str(d) for d in range(10)]                              # 0-9
        + [chr(c) for c in range(ord("A"), ord("Z") + 1)]       # A-Z
        + [chr(c) for c in range(ord("a"), ord("z") + 1)]       # a-z
    )

    # Agrupar índices del dataset por clase
    class_indices: dict[int, list[int]] = {}
    for i, label in enumerate(ds.targets):
        class_indices.setdefault(int(label), []).append(i)

    for emnist_idx, char in enumerate(emnist_chars):
        if emnist_idx not in class_indices:
            continue
        indices = class_indices[emnist_idx]
        sources: list[ImageSource] = [("emnist", i, ds) for i in indices]
        result.setdefault(char, []).extend(sources)

    print(f"    EMNIST: {len(result)} clases indexadas, "
          f"{sum(len(v) for v in result.values()):,} imágenes totales.")
    return result


def _index_file_dataset(
    dataset_root: str,
    label: str,
) -> dict[str, list[ImageSource]]:
    """
    Indexa un dataset basado en archivos de imagen.

    Estrategias (en orden de prioridad):
      1. Carpetas cuyo nombre es un único carácter  →  nombre = clase
      2. Carpetas con nombre "class_X", "char_X"   →  X = clase
      3. Archivos cuyo nombre empieza por "X_"     →  X = clase
      4. Annotation JSON (0annotation.json)        →  char de la transcripción
    """
    result: dict[str, list[ImageSource]] = {}
    root   = Path(dataset_root)

    if not root.exists():
        print(f"  [WARN] {label}: carpeta no encontrada '{root}'")
        return result

    # ── Estrategia 1 & 2: carpetas por clase ────────────────────────────────
    found_via_folders = False
    for folder in sorted(root.rglob("*")):
        if not folder.is_dir():
            continue
        name = folder.name

        char: str | None = None
        if len(name) == 1:
            char = name
        else:
            import re
            m = re.match(r"^(?:class|char|label|sample)[_\-]?(.+)$", name, re.IGNORECASE)
            if m and len(m.group(1)) == 1:
                char = m.group(1)

        if char is None:
            continue

        imgs = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        ]
        if imgs:
            result.setdefault(char, []).extend(imgs)
            found_via_folders = True

    # ── Estrategia 3: nombre de archivo "X_NNN.ext" ─────────────────────────
    if not found_via_folders:
        import re
        for f in root.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in IMG_EXTS:
                continue
            m = re.match(r"^(.)[\-_]", f.stem)
            if m:
                result.setdefault(m.group(1), []).append(f)

    # ── Estrategia 4: 0annotation.json ──────────────────────────────────────
    for ann_path in root.rglob("0annotation.json"):
        try:
            with open(ann_path, "r", encoding="utf-8") as af:
                ann = json.load(af)
            img_dir = ann_path.parent
            for filename, transcription in ann.items():
                img_file = img_dir / filename
                if not img_file.exists():
                    continue
                for char in transcription:
                    if char.strip():
                        result.setdefault(char, []).append(img_file)
        except Exception as e:
            print(f"  [WARN] {label}: error leyendo {ann_path.name}: {e}")

    n_classes = len(result)
    n_images  = sum(len(v) for v in result.values())
    print(f"    {label}: {n_classes} clases, {n_images:,} imágenes.")
    return result


def _build_class_image_index(
    emnist_root:  str = EMNIST_ROOT,
    hwc_root:     str = HWC_ROOT,
    spanish_root: str = SPANISH_ROOT,
) -> dict[str, list[ImageSource]]:
    """
    Construye un índice unificado { char → list[ImageSource] }
    leyendo TODOS los datasets disponibles.

    Para cada clase el índice puede contener fuentes de múltiples datasets;
    al generar imágenes se muestrea aleatoriamente de la lista combinada.
    """
    print("\n  Construyendo índice unificado de imágenes por clase ...")
    unified: dict[str, list[ImageSource]] = {}

    def _merge(d: dict[str, list[ImageSource]]) -> None:
        for char, sources in d.items():
            unified.setdefault(char, []).extend(sources)

    _merge(_index_emnist(emnist_root, {}))
    _merge(_index_file_dataset(hwc_root,     "handwritting_characters_database"))
    _merge(_index_file_dataset(spanish_root, "spanish_handwritten_characters_words"))

    total_chars  = len(unified)
    total_images = sum(len(v) for v in unified.values())
    print(f"\n  Índice unificado: {total_chars} clases, {total_images:,} imágenes totales.")
    return unified


# =============================================================================
# Carga de una imagen desde cualquier fuente del índice
# =============================================================================

def _load_source_image(source: ImageSource) -> np.ndarray | None:
    """
    Carga una imagen desde una fuente del índice unificado.

    · Fuente EMNIST: extrae del dataset torchvision y corrige orientación.
    · Fuente archivo: lee con cv2/Pillow según extensión.

    Devuelve imagen en escala de grises (uint8) con trazo OSCURO sobre fondo
    CLARO, lista para composición.  None si falla la carga.
    """
    try:
        if isinstance(source, tuple):
            # ("emnist", sample_idx, dataset_obj)
            _, idx, ds = source
            img_pil, _ = ds[idx]
            arr = np.array(img_pil)
            # EMNIST byclass: transponer + flip para orientación correcta
            arr = cv2.flip(cv2.transpose(arr), flipCode=1)
            # byclass: trazo claro / fondo oscuro → invertir
            return cv2.bitwise_not(arr)

        else:
            # Path a archivo de imagen
            path: Path = source
            ext = path.suffix.lower()
            if ext in {".avif", ".webp"}:
                pil_img = Image.open(path).convert("L")
                arr     = np.array(pil_img)
            else:
                arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            if arr is None:
                return None

            # Normalizar a trazo oscuro / fondo claro
            # Heurística: si la media es oscura, el fondo es oscuro → invertir
            if arr.mean() < 128:
                arr = cv2.bitwise_not(arr)

            return arr

    except Exception as e:
        print(f"  [WARN] No se pudo cargar imagen ({e})")
        return None


# =============================================================================
# Generación de imagen de clase sin datos reales (fuente sintética)
# =============================================================================

def _render_char_fallback(char: str, size: int) -> np.ndarray:
    """
    Renderiza un carácter con Pillow cuando no hay imágenes reales.

    Útil para clases faltantes que no son trazos primitivos (ej. tildes
    o símbolos sin datos en ningún dataset).

    Intenta usar una fuente del sistema; si no hay ninguna disponible
    dibuja el carácter con la fuente default de Pillow.
    """
    img_pil = Image.new("L", (size, size), color=255)
    draw    = ImageDraw.Draw(img_pil)

    font_size  = int(size * 0.72)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None

    # Fuentes candidatas (rutas comunes en Windows, Linux y macOS)
    font_candidates = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # macOS
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]

    for fp in font_candidates:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()

    # Centrar el carácter en el canvas
    bbox = draw.textbbox((0, 0), char, font=font)
    x    = (size - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y    = (size - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), char, fill=random.randint(0, 40), font=font)

    return np.array(img_pil)


# =============================================================================
# Fondos
# =============================================================================

def load_backgrounds(bg_path: str = BG_PATH) -> list[np.ndarray]:
    """
    Carga fondos desde bg_path. Soporta .webp, .avif, .jpg, .jpeg, .png.
    Devuelve lista vacía si no hay fondos (se usarán sintéticos).
    """
    pil_exts = {".webp", ".avif"} if _AVIF_OK else {".webp"}
    cv2_exts = {".jpg", ".jpeg", ".png"}
    all_exts = cv2_exts | pil_exts

    bgs: list[np.ndarray] = []
    bg_dir = Path(bg_path)
    if not bg_dir.exists():
        return bgs

    for f in sorted(bg_dir.rglob("*")):
        if f.suffix.lower() not in all_exts:
            continue
        try:
            if f.suffix.lower() in pil_exts:
                img = cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(str(f))
            if img is not None and img.size > 0:
                bgs.append(img)
        except Exception as e:
            print(f"  [WARN] Fondo '{f.name}': {e}")

    return bgs


def make_synthetic_bg(size: int = IMG_SIZE) -> np.ndarray:
    """Genera fondo sintético de papel: blanco, cuadriculado, rayado o envejecido."""
    t = random.choice(["white", "grid", "lined", "aged"])
    if t == "white":
        bg = np.full((size, size, 3), 245, dtype=np.uint8)
        return np.clip(bg.astype(np.int16) + np.random.normal(0, 4, bg.shape).astype(np.int16), 200, 255).astype(np.uint8)
    elif t == "grid":
        bg = np.full((size, size, 3), 250, dtype=np.uint8)
        sp = random.randint(20, 35)
        for x in range(0, size, sp): cv2.line(bg, (x, 0), (x, size), (195, 200, 225), 1)
        for y in range(0, size, sp): cv2.line(bg, (0, y), (size, y), (195, 200, 225), 1)
        cv2.line(bg, (random.randint(60, 100), 0), (random.randint(60, 100), size), (180, 180, 230), 1)
        return bg
    elif t == "lined":
        bg = np.full((size, size, 3), 248, dtype=np.uint8)
        sp = random.randint(22, 32)
        for y in range(sp, size, sp): cv2.line(bg, (0, y), (size, y), (190, 205, 235), 1)
        return bg
    else:
        base = random.randint(225, 240)
        return np.clip(
            np.full((size, size, 3), [base, base + 5, base - 15], dtype=np.int16)
            + np.random.normal(0, 6, (size, size, 3)).astype(np.int16),
            180, 255,
        ).astype(np.uint8)


# =============================================================================
# Dibujado de trazos primitivos con OpenCV
# =============================================================================

def _safe_ri(a: int, b: int) -> int:
    """random.randint seguro: si a >= b retorna a."""
    return a if a >= b else random.randint(a, b)


def _draw_primitive_stroke(stroke_name: str, size: int) -> np.ndarray:
    """
    Dibuja un trazo primitivo en canvas blanco (fondo 255, tinta ~0).
    Todos los rangos de randint están protegidos para evitar ValueError.
    """
    canvas    = np.full((size, size), 255, dtype=np.uint8)
    margin    = int(size * 0.12)
    thickness = _safe_ri(max(1, size // 55), max(2, size // 22))
    gray_ink  = random.randint(0, 45)
    line_len  = min(
        _safe_ri(int(size * 0.55), max(int(size * 0.55) + 1, int(size * 0.85))),
        size - 2 * margin,
    )
    cx, cy = size // 2, size // 2

    if stroke_name == "línea_vertical":
        x  = _safe_ri(margin, size - margin)
        y1 = _safe_ri(margin, max(margin, size - margin - line_len))
        y2 = min(y1 + line_len, size - margin)
        cv2.line(canvas, (x, y1), (x, y2), gray_ink, thickness)

    elif stroke_name == "línea_horizontal":
        y  = _safe_ri(margin, size - margin)
        x1 = _safe_ri(margin, max(margin, size - margin - line_len))
        x2 = min(x1 + line_len, size - margin)
        cv2.line(canvas, (x1, y), (x2, y), gray_ink, thickness)

    elif stroke_name == "línea_oblicua_derecha":
        diag = int(line_len * 0.75)
        x1   = _safe_ri(margin, max(margin, size - margin - diag))
        y1   = _safe_ri(min(cy, size - margin), size - margin)
        x2   = min(x1 + diag, size - margin)
        y2   = max(y1 - diag, margin)
        cv2.line(canvas, (x1, y1), (x2, y2), gray_ink, thickness)

    elif stroke_name == "línea_oblicua_izquierda":
        diag = int(line_len * 0.75)
        x1   = _safe_ri(margin, max(margin, size - margin - diag))
        y1   = _safe_ri(margin, max(margin, cy))
        x2   = min(x1 + diag, size - margin)
        y2   = min(y1 + diag, size - margin)
        cv2.line(canvas, (x1, y1), (x2, y2), gray_ink, thickness)

    elif stroke_name == "curva":
        half = (size - 2 * margin) // 2
        ax   = _safe_ri(int(size * 0.25), max(int(size * 0.25) + 1, min(int(size * 0.42), half)))
        ay   = _safe_ri(int(size * 0.18), max(int(size * 0.18) + 1, min(int(size * 0.35), half)))
        ang  = random.randint(0, 360)
        a0   = random.randint(0, 90)
        a1   = a0 + random.randint(90, 270)
        ox   = _safe_ri(margin + ax, max(margin + ax, size - margin - ax))
        oy   = _safe_ri(margin + ay, max(margin + ay, size - margin - ay))
        cv2.ellipse(canvas, (ox, oy), (ax, ay), ang, a0, a1, gray_ink, thickness)

    elif stroke_name == "círculo":
        max_r = max(5, (size // 2) - margin - thickness)
        min_r = max(5, min(int(size * 0.18), max_r - 1))
        r     = _safe_ri(min_r, max_r)
        lo    = margin + r
        hi    = max(lo, size - margin - r)
        ox    = _safe_ri(lo, hi)
        oy    = _safe_ri(lo, hi)
        cv2.circle(canvas, (ox, oy), r, gray_ink, thickness)

    else:
        cv2.line(canvas, (margin, size - margin), (size - margin, margin), gray_ink, thickness)

    k = random.choice([3, 3, 5])
    return cv2.GaussianBlur(canvas, (k, k), random.uniform(0.5, 1.2))


# =============================================================================
# Augmentaciones
# =============================================================================

def _safe_odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def _add_shadow(img: np.ndarray) -> np.ndarray:
    h, w   = img.shape[:2]
    result = img.copy().astype(np.float32)
    alpha  = random.uniform(0.25, 0.55)
    stype  = random.choice(["lateral", "corner", "band"])
    if stype == "lateral":
        side   = random.choice(["left", "right", "top", "bottom"])
        extent = random.randint(w // 5, w // 2)
        pts_map = {
            "left":   np.array([[0,0],[extent,0],[extent-30,h],[0,h]]),
            "right":  np.array([[w-extent,0],[w,0],[w,h],[w-extent+30,h]]),
            "top":    np.array([[0,0],[w,0],[w,extent-30],[0,extent]]),
            "bottom": np.array([[0,h-extent],[w,h-extent+30],[w,h],[0,h]]),
        }
        pts = pts_map[side]
    elif stype == "corner":
        corner  = random.choice(["tl","tr","bl","br"])
        ext     = random.randint(w // 4, w * 2 // 3)
        pts_map = {
            "tl": np.array([[0,0],[ext,0],[0,ext]]),
            "tr": np.array([[w-ext,0],[w,0],[w,ext]]),
            "bl": np.array([[0,h-ext],[ext,h],[0,h]]),
            "br": np.array([[w,h-ext],[w-ext,h],[w,h]]),
        }
        pts = pts_map[corner]
    else:
        y0 = random.randint(0, h//2); y1 = y0 + random.randint(h//5, h//2)
        sl = random.randint(-50, 50)
        pts = np.array([[0,y0],[w,y0+sl],[w,y1+sl],[0,y1]])
    mask   = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [pts.reshape(-1,1,2)], 1.0)
    mask   = cv2.GaussianBlur(mask, (_safe_odd(random.randint(31,71)),)*2, 0)
    return (result * (1.0 - alpha * mask[:,:,np.newaxis])).clip(0,255).astype(np.uint8)


def _add_gradient_light(img: np.ndarray) -> np.ndarray:
    h, w  = img.shape[:2]
    gtype = random.choice(["linear","radial","vignette"])
    if gtype == "linear":
        d = random.choice(["h","v","diag"])
        if d == "h":   g = np.tile(np.linspace(random.uniform(0.7,1.0), random.uniform(0.85,1.0), w), (h,1))
        elif d == "v": g = np.tile(np.linspace(random.uniform(0.75,1.0), random.uniform(0.85,1.0), h)[:,None], (1,w))
        else:          g = np.outer(np.linspace(0.85,1.0,h), np.linspace(0.8,1.0,w))
        gradient = g
    elif gtype == "radial":
        cx,cy   = random.randint(w//4,3*w//4), random.randint(h//4,3*h//4)
        Y,X     = np.ogrid[:h,:w]
        dist    = np.sqrt((X-cx)**2+(Y-cy)**2)
        max_d   = np.sqrt(max(cx,w-cx)**2+max(cy,h-cy)**2)
        gradient = 1.0-(dist/max_d)*random.uniform(0.15,0.30)
    else:
        Y,X     = np.ogrid[:h,:w]
        dist    = np.sqrt(((X-w//2)/(w/2))**2+((Y-h//2)/(h/2))**2)
        gradient = 1.0-np.clip(dist-0.4,0,1)*random.uniform(0.2,0.45)
    return (img.astype(np.float32)*gradient[:,:,np.newaxis]).clip(0,255).astype(np.uint8)


def _simulate_pencil(letter: np.ndarray) -> np.ndarray:
    result = letter.copy().astype(np.float32)
    mask   = letter < 128
    noise  = np.random.normal(random.randint(0,40), 15, letter.shape)
    result[mask] = np.clip(noise[mask], 0, 80)
    pts = np.argwhere(mask)
    n   = int(len(pts) * 0.03)
    if n > 0 and len(pts) > n:
        chosen = pts[np.random.choice(len(pts), n, replace=False)]
        result[chosen[:,0], chosen[:,1]] = np.random.randint(60, 100, n)
    return cv2.GaussianBlur(result.astype(np.uint8), (3,3), 0.5)


def _simulate_ink_variation(letter: np.ndarray) -> np.ndarray:
    h, w    = letter.shape
    opacity = cv2.resize(np.random.uniform(0.55,1.0,(8,8)).astype(np.float32),(w,h),interpolation=cv2.INTER_CUBIC)
    result  = letter.astype(np.float32)
    mask    = letter < 128
    result[mask] = (result[mask]*opacity[mask]).clip(0,255)
    return result.astype(np.uint8)


def _apply_global_augmentations(img: np.ndarray) -> np.ndarray:
    if random.random() < PROB_SHADOW:         img = _add_shadow(img)
    if random.random() < PROB_GRADIENT_LIGHT: img = _add_gradient_light(img)
    if random.random() < PROB_BLUR:
        k = _safe_odd(random.choice([3,3,3,5]))
        img = cv2.GaussianBlur(img, (k,k), 0)
    if random.random() < PROB_NOISE:
        noise = np.random.normal(0, random.uniform(3,10), img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16)+noise, 0, 255).astype(np.uint8)
    return img


# =============================================================================
# Composición
# =============================================================================

def _compose_letter_on_bg(
    bg_base: np.ndarray,
    letter:  np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    """Pega la letra sobre el fondo con blending realista. Retorna (img, xc, yc, w, h) normalizados."""
    bg       = cv2.resize(bg_base, (IMG_SIZE, IMG_SIZE))
    h_l, w_l = letter.shape[:2]
    margin   = 30
    x        = _safe_ri(margin, max(margin+1, IMG_SIZE - w_l - margin))
    y        = _safe_ri(margin, max(margin+1, IMG_SIZE - h_l - margin))

    roi        = bg[y:y+h_l, x:x+w_l].copy()
    letter_bgr = cv2.cvtColor(letter, cv2.COLOR_GRAY2BGR) if letter.ndim == 2 else letter
    letter_f   = letter_bgr.astype(np.float32) / 255.0
    roi_f      = roi.astype(np.float32) / 255.0
    alpha      = 1.0 - letter_f
    bg[y:y+h_l, x:x+w_l] = ((roi_f*(1-alpha)+letter_f*alpha)*255).clip(0,255).astype(np.uint8)

    return bg, (x+w_l/2)/IMG_SIZE, (y+h_l/2)/IMG_SIZE, w_l/IMG_SIZE, h_l/IMG_SIZE


# =============================================================================
# Pipeline de preparación de una sola imagen de letra
# =============================================================================

def _prepare_letter(
    char:        str,
    source:      ImageSource | None,
    size:        int,
) -> np.ndarray:
    """
    Obtiene la imagen de la letra lista para composición:
      · Si hay source → carga y normaliza desde el índice unificado
      · Si no hay source → renderiza con Pillow (_render_char_fallback)
    Siempre devuelve imagen en escala de grises, trazo oscuro / fondo claro.
    """
    letter: np.ndarray | None = None

    if source is not None:
        letter = _load_source_image(source)

    if letter is None:
        # Fallback: renderizar con fuente del sistema
        letter = _render_char_fallback(char, size)

    # Redimensionar al tamaño de composición
    letter = cv2.resize(letter, (size, size), interpolation=cv2.INTER_CUBIC)
    # Re-binarizar para eliminar grises del resize
    _, letter = cv2.threshold(letter, 127, 255, cv2.THRESH_BINARY)

    # Augmentaciones del trazo
    if random.random() < PROB_PENCIL_TEXTURE:
        letter = _simulate_pencil(letter)
    if random.random() < PROB_INK_VARIATION:
        letter = _simulate_ink_variation(letter)
    if random.random() < PROB_ROTATION:
        angle  = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
        M      = cv2.getRotationMatrix2D((size//2, size//2), angle, 1.0)
        letter = cv2.warpAffine(letter, M, (size, size),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return letter


# =============================================================================
# Generador principal
# =============================================================================

def generate_synthetic_data(
    char_map_path: str = CHAR_MAP_PATH,
    report_path:   str = REPORT_PATH,
    output_path:   str = OUTPUT_PATH,
    bg_path:       str = BG_PATH,
    emnist_root:   str = EMNIST_ROOT,
    hwc_root:      str = HWC_ROOT,
    spanish_root:  str = SPANISH_ROOT,
) -> dict[str, int]:
    """
    Genera el dataset sintético YOLO completo para TODAS las clases del char_map.

    Flujo:
      1. Lee char_map.json  → lista completa de 107 clases objetivo.
      2. Lee dataset_classes_report.json → clases faltantes vs existentes.
      3. Construye índice unificado de imágenes reales (todos los datasets).
      4. Para cada clase:
           · Trazo primitivo   → _draw_primitive_stroke()  × 150
           · Clase existente   → fuentes del índice         × 50
           · Clase faltante    → _render_char_fallback()    × 100
      5. Compone cada letra sobre un fondo y guarda .jpg + .txt YOLO.

    Returns
    -------
    dict[str, int]   { clase: n_generadas }
    """
    print("═" * 60)
    print("  GENERADOR DE DATOS SINTÉTICOS YOLO")
    print("═" * 60)

    # 1. Char map
    print(f"\n1. Cargando char_map desde '{char_map_path}' ...")
    char_map     = _load_char_map(char_map_path)
    all_classes  = list(char_map["idx2char"].values())
    print(f"   {len(all_classes)} clases objetivo.")

    # 2. Reporte de cobertura
    print("2. Leyendo reporte de cobertura ...")
    missing_classes = load_coverage_report(report_path)

    # 3. Carpetas de salida
    (Path(output_path) / "images" / "train").mkdir(parents=True, exist_ok=True)
    (Path(output_path) / "labels" / "train").mkdir(parents=True, exist_ok=True)
    img_dir = Path(output_path) / "images" / "train"
    lbl_dir = Path(output_path) / "labels" / "train"

    # 4. Fondos
    print("3. Cargando fondos ...")
    bgs = load_backgrounds(bg_path)
    print(
        f"   ✅ {len(bgs)} fondos cargados." if bgs
        else "   ⚠  Sin fondos reales → fondos sintéticos."
    )

    # 5. Índice unificado de imágenes (todos los datasets)
    print("4. Indexando datasets ...")
    image_index = _build_class_image_index(emnist_root, hwc_root, spanish_root)

    # 6. Estadísticas previas
    n_primitives = len(PRIMITIVE_STROKES)
    n_missing    = sum(1 for c in all_classes if c in missing_classes and c not in PRIMITIVE_STROKES)
    n_existing   = len(all_classes) - n_primitives - n_missing
    total_est    = (n_primitives * PRIMITIVE_CLASS_COUNT
                    + n_missing   * MISSING_CLASS_COUNT
                    + n_existing  * EXISTING_CLASS_COUNT)
    print(f"\n   Primitivos : {n_primitives} × {PRIMITIVE_CLASS_COUNT} = {n_primitives*PRIMITIVE_CLASS_COUNT}")
    print(f"   Faltantes  : {n_missing} × {MISSING_CLASS_COUNT} = {n_missing*MISSING_CLASS_COUNT}")
    print(f"   Existentes : {n_existing} × {EXISTING_CLASS_COUNT} = {n_existing*EXISTING_CLASS_COUNT}")
    print(f"   TOTAL EST. : ≈{total_est:,} imágenes\n")

    results: dict[str, int] = {}

    # 7. Generación clase por clase
    for class_idx, char in enumerate(all_classes):

        is_primitive = char in PRIMITIVE_STROKES
        is_missing   = (not is_primitive) and (char in missing_classes or not missing_classes)
        # Si no hay reporte, tratar todo como fallback
        if not missing_classes and not is_primitive:
            is_missing = char not in image_index

        n_images     = (
            PRIMITIVE_CLASS_COUNT if is_primitive
            else get_images_per_char(char, missing_classes)
        )
        tag = "PRIMITIVO" if is_primitive else ("FALTANTE" if char in missing_classes else "existente")

        # Fuentes disponibles para esta clase
        sources: list[ImageSource] = image_index.get(char, [])

        slug = (char
                .replace("\\", "bs").replace("/", "sl").replace(":", "co")
                .replace("*","as").replace("?","qm").replace('"',"dq")
                .replace("<","lt").replace(">","gt").replace("|","pi"))
        # Para chars de 1 carácter imprimible usar ordinal, más limpio en filesystem
        file_prefix = f"cls{class_idx:03d}"

        for i in tqdm(
            range(n_images),
            desc=f"  '{char}' [{tag}] {n_images}",
            leave=False,
            ncols=72,
        ):
            # Fondo
            use_synth = INCLUDE_SYNTHETIC_BG and random.random() < SYNTHETIC_BG_FRACTION
            bg_base   = make_synthetic_bg() if (use_synth or not bgs) else random.choice(bgs).copy()

            # Tamaño de la letra
            size = random.randint(LETTER_SIZE_MIN, LETTER_SIZE_MAX)

            # Imagen del carácter
            if is_primitive:
                letter = _draw_primitive_stroke(char, size)
                # Augmentaciones de trazo (igual que para letras)
                if random.random() < PROB_PENCIL_TEXTURE:  letter = _simulate_pencil(letter)
                if random.random() < PROB_INK_VARIATION:   letter = _simulate_ink_variation(letter)
                if random.random() < PROB_ROTATION:
                    angle  = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
                    M      = cv2.getRotationMatrix2D((size//2, size//2), angle, 1.0)
                    letter = cv2.warpAffine(letter, M, (size, size),
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            else:
                # Elegir fuente aleatoria del índice unificado (o None si no hay)
                source = random.choice(sources) if sources else None
                letter = _prepare_letter(char, source, size)

            # Composición
            composed, xc, yc, nw, nh = _compose_letter_on_bg(bg_base, letter)
            composed = _apply_global_augmentations(composed)

            # Guardar
            stem = f"{file_prefix}_{i:04d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), composed, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(lbl_dir / f"{stem}.txt", "w") as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        print(f"  ✅ '{char}' [{tag}] — {n_images} imgs  "
              f"({'OpenCV' if is_primitive else f'{len(sources):,} fuentes reales' if sources else 'Pillow fallback'})")
        results[char] = n_images

    total_gen = sum(results.values())
    print(f"\n{'═'*60}")
    print(f"  ✨ Dataset generado: {total_gen:,} imágenes")
    print(f"  Ubicación: {Path(output_path).resolve()}")
    print(f"  Siguiente paso: python app/scripts/generate_negatives.py\n")
    return results


if __name__ == "__main__":
    generate_synthetic_data()