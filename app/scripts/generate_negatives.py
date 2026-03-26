"""
app/scripts/generate_negatives.py
===================================
Genera imágenes negativas (fondos sin caracteres) para el dataset YOLO.

INTEGRACIÓN CON verify_dataset_classes.py
------------------------------------------
Lee data/dataset_classes_report.json para calcular el total de negativos:

  total = (n_missing  × NEG_PER_MISSING_CLASS)   [default 100]
        + (n_existing × NEG_PER_EXISTING_CLASS)   [default  50]

Si el JSON no existe se usa NUM_NEGATIVES_FALLBACK como total fijo (200).

COMPATIBILIDAD CON generate_synthetic_yolo.py
----------------------------------------------
Importa load_backgrounds() y make_synthetic_bg() desde generate_synthetic_yolo.
Esto garantiza que los negativos usen los mismos fondos y tipos de papel.
Si el import falla (distinto directorio de trabajo) se activa un fallback local
equivalente para no interrumpir la ejecución.

FONDOS SOPORTADOS
-----------------
  · .webp  — Pillow nativo (>= 9.x)
  · .avif  — requiere pillow_avif
  · .jpg / .jpeg / .png

NOTA YOLO: las imágenes negativas NO tienen archivo .txt.
Esto indica a YOLO que son fondos puros (sin objetos que detectar).

Estructura de salida
--------------------
data/processed/yolo_dataset/images/train/neg_*.jpg
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Import compartido desde generate_synthetic_yolo ─────────────────────────
try:
    from generate_synthetic_yolo import load_backgrounds, make_synthetic_bg
    _SHARED_IMPORT = True
except ImportError:
    _SHARED_IMPORT = False

    try:
        import pillow_avif  # noqa: F401
        _AVIF_OK = True
    except ImportError:
        _AVIF_OK = False

    from PIL import Image

    def load_backgrounds(bg_path: str = "./data/backgrounds") -> list[np.ndarray]:  # type: ignore[misc]
        """Fallback local — idéntico al de generate_synthetic_yolo."""
        pil_exts = {".webp", ".avif"} if _AVIF_OK else {".webp"}
        cv2_exts = {".jpg", ".jpeg", ".png"}
        all_exts = cv2_exts | pil_exts
        bgs: list[np.ndarray] = []
        for f in sorted(Path(bg_path).rglob("*")):
            if f.suffix.lower() not in all_exts:
                continue
            try:
                img = (
                    cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
                    if f.suffix.lower() in pil_exts
                    else cv2.imread(str(f))
                )
                if img is not None and img.size > 0:
                    bgs.append(img)
            except Exception:
                pass
        return bgs

    def make_synthetic_bg(size: int = 640) -> np.ndarray:  # type: ignore[misc]
        """Fallback local — idéntico al de generate_synthetic_yolo."""
        bg_type = random.choice(["white", "grid", "lined", "aged"])
        if bg_type == "white":
            bg = np.full((size, size, 3), 245, dtype=np.uint8)
            return np.clip(bg.astype(np.int16) + np.random.normal(0, 4, bg.shape).astype(np.int16), 200, 255).astype(np.uint8)
        elif bg_type == "grid":
            bg = np.full((size, size, 3), 250, dtype=np.uint8)
            sp = random.randint(20, 35)
            for x in range(0, size, sp): cv2.line(bg, (x, 0), (x, size), (195, 200, 225), 1)
            for y in range(0, size, sp): cv2.line(bg, (0, y), (size, y), (195, 200, 225), 1)
            return bg
        elif bg_type == "lined":
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
# Configuración
# =============================================================================

REPORT_PATH             = "./data/dataset_classes_report.json"
OUTPUT_IMAGES_PATH      = "./data/processed/yolo_dataset/images/train"
BG_PATH                 = "./data/backgrounds"
IMG_SIZE                = 640

NEG_PER_MISSING_CLASS   = 50   # Negativos por cada clase faltante
NEG_PER_EXISTING_CLASS  = 20    # Negativos por cada clase existente
NUM_NEGATIVES_FALLBACK  = 100   # Total si no hay reporte JSON


# =============================================================================
# Cálculo del número de negativos
# =============================================================================

def _compute_num_negatives(report_path: str = REPORT_PATH) -> tuple[int, str]:
    """
    Calcula el total de negativos en base al reporte de cobertura.

    Returns
    -------
    (num_negatives, descripción_log)
    """
    path = Path(report_path)
    if not path.exists():
        return (
            NUM_NEGATIVES_FALLBACK,
            f"Sin reporte JSON → fallback ({NUM_NEGATIVES_FALLBACK} negativos)",
        )

    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    n_missing  = len(report.get("global_missing", []))
    n_total    = report.get("char_map_classes", 0)
    n_existing = max(0, n_total - n_missing)
    num_neg    = (n_missing * NEG_PER_MISSING_CLASS) + (n_existing * NEG_PER_EXISTING_CLASS)

    return (
        num_neg,
        (f"Reporte: {n_total} clases, "
         f"{n_missing} faltantes × {NEG_PER_MISSING_CLASS} + "
         f"{n_existing} existentes × {NEG_PER_EXISTING_CLASS} = {num_neg}"),
    )


# =============================================================================
# Efectos de distractor
# =============================================================================

def _add_scribbles(img: np.ndarray) -> np.ndarray:
    """Tachaduras y rayones erráticos (anotaciones, correcciones, papel reutilizado)."""
    result = img.copy()
    for _ in range(random.randint(1, 4)):
        pts = np.array(
            [[random.randint(20, 610), random.randint(20, 610)]
             for _ in range(random.randint(3, 8))],
            np.int32,
        ).reshape(-1, 1, 2)
        gray = random.randint(40, 110)
        cv2.polylines(result, [pts], isClosed=False,
                      color=(gray, gray, gray), thickness=random.randint(1, 5))
    return result


def _add_smudges(img: np.ndarray) -> np.ndarray:
    """Manchas difusas: borrones de goma, suciedad, humedad."""
    result = img.copy()
    for _ in range(random.randint(1, 3)):
        overlay = result.copy()
        center  = (random.randint(80, 560), random.randint(80, 560))
        axes    = (random.randint(15, 70), random.randint(10, 45))
        gray    = random.randint(140, 210)
        cv2.ellipse(overlay, center, axes, random.randint(0, 360), 0, 360, (gray, gray, gray), -1)
        result = cv2.addWeighted(overlay, random.uniform(0.25, 0.50), result, 1 - random.uniform(0.25, 0.50), 0)
    return cv2.GaussianBlur(result, (random.choice([3, 5]), random.choice([3, 5])), 0)


def _add_graphite_noise(img: np.ndarray) -> np.ndarray:
    """Motas de polvo y grafito sobre la hoja (ruido gaussiano leve)."""
    noise = np.random.normal(0, random.uniform(4, 12), img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _add_ink_bleed(img: np.ndarray) -> np.ndarray:
    """
    Sangrado de tinta desde el reverso del papel.
    Texto gris muy claro, ligeramente borroso, simulando escritura transparentada.
    """
    result = img.copy()
    for _ in range(random.randint(1, 3)):
        y    = random.randint(50, IMG_SIZE - 50)
        font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX])
        text = "".join(random.choices("abcdefghijklmnopqrstuvwxyz ", k=random.randint(8, 18)))
        gray = random.randint(190, 225)
        overlay = result.copy()
        cv2.putText(overlay, text, (random.randint(10, 80), y),
                    font, random.uniform(0.4, 0.9), (gray, gray, gray), 1, cv2.LINE_AA)
        alpha  = random.uniform(0.15, 0.35)
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    return cv2.GaussianBlur(result, (random.choice([3, 5, 5]), random.choice([3, 5, 5])), 0)


# Distractores con su probabilidad de aplicación
_DISTRACTORS = [
    (_add_scribbles,      0.45),
    (_add_smudges,        0.40),
    (_add_graphite_noise, 0.50),
    (_add_ink_bleed,      0.30),
]


# =============================================================================
# Generador de negativos
# =============================================================================

def generate_negatives(
    report_path:  str = REPORT_PATH,
    output_path:  str = OUTPUT_IMAGES_PATH,
    bg_path:      str = BG_PATH,
) -> int:
    """
    Genera imágenes negativas para el dataset YOLO.

    El número de negativos se calcula desde report_path:
      total = n_missing × NEG_PER_MISSING_CLASS + n_existing × NEG_PER_EXISTING_CLASS

    Cada imagen negativa:
      · Fondo real (bg_path) o sintético si no hay fondos
      · 0–N efectos de distractor aplicados aleatoriamente
      · Sin archivo .txt → YOLO lo trata como fondo puro

    Parameters
    ----------
    report_path : str   Ruta a data/dataset_classes_report.json.
    output_path : str   Carpeta de salida de imágenes.
    bg_path     : str   Carpeta de fondos (.webp/.avif/.jpg/.png).

    Returns
    -------
    int   Número de imágenes generadas.
    """
    print("═" * 60)
    print("  GENERADOR DE IMÁGENES NEGATIVAS YOLO")
    print("═" * 60)
    print(f"  Helpers: {'generate_synthetic_yolo (compartido)' if _SHARED_IMPORT else 'fallback local'}")

    # 1. Número de negativos
    print("\n1. Calculando número de negativos ...")
    num_negatives, reason = _compute_num_negatives(report_path)
    print(f"   {reason}")

    if num_negatives == 0:
        print("   Sin clases en el reporte → nada que generar.")
        return 0

    # 2. Carpeta de salida
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Fondos
    print("2. Cargando fondos ...")
    bgs = load_backgrounds(bg_path)
    if bgs:
        print(f"   ✅ {len(bgs)} fondos cargados desde '{bg_path}'")
    else:
        print("   ⚠  Sin fondos reales → fondos sintéticos")

    # 4. Generar
    print(f"\n3. Generando {num_negatives:,} imágenes negativas ...\n")

    generated = 0
    for i in tqdm(range(num_negatives), desc="  Negativos", ncols=72):
        img = (
            cv2.resize(random.choice(bgs).copy(), (IMG_SIZE, IMG_SIZE))
            if bgs
            else make_synthetic_bg(IMG_SIZE)
        )

        for fn, prob in _DISTRACTORS:
            if random.random() < prob:
                img = fn(img)

        # Sin .txt → negativo YOLO
        cv2.imwrite(str(out_dir / f"neg_bg_{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        generated += 1

    print(f"\n✅ {generated:,} imágenes negativas guardadas en:")
    print(f"   {out_dir.resolve()}")
    print("   Recordatorio: YOLO no necesita archivos .txt para estas imágenes.\n")
    return generated


if __name__ == "__main__":
    generate_negatives()