"""
app/utils/visualizer.py  (v4.2)
================================

Genera imágenes de visualización para la UI:
  - "Tu trazo": imagen limpia del carácter escrito por el alumno
  - "Comparación": overlay de plantilla vs alumno con colores

CAMBIOS v4.2 vs v4.1:
  - build_raw_crop_image() corregido: NO usa la máscara del normalizer
    (que puede ser puro ruido en fotos reales). En su lugar:
    1. Prefiere el display_crop (de image_cleaner, limpio)
    2. Fallback al raw_crop_bgr (original de YOLO)
    3. Solo usa la máscara como último recurso

  - Nuevo parámetro display_crop en build_raw_crop_image()

  - _ensure_visible_on_white: convierte cualquier imagen para que el
    trazo sea visible sobre fondo blanco (invierte si es necesario)
"""

import base64
import io

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from app.core.config import TARGET_SHAPE, TARGET_SIZE


# =============================================================================
# Parámetros de visualización
# =============================================================================

VIZ_OUTPUT_PX  = 512
VIZ_DPI        = 100
VIZ_BLUR_K     = (3, 3)
VIZ_BLUR_SIGMA = 0.6

COLOR_GUIDE     = (0,  200,  80)   # Verde BGR — plantilla
COLOR_STUDENT   = (50,  60, 244)   # Rojo  BGR — alumno
COLOR_MATCH     = (0,  214, 255)   # Amarillo — coincidencia

COLOR_GUIDE_MPL    = (0.0,  0.78, 0.32)
COLOR_STUDENT_MPL  = (0.96, 0.16, 0.16)
COLOR_MATCH_MPL    = (1.0,  0.84, 0.0)


# =============================================================================
# Utilidades internas — Limpieza de esqueleto
# =============================================================================

def _clean_small_fragments(skel: np.ndarray, min_frac: float = 0.30) -> np.ndarray:
    """
    Elimina fragmentos del esqueleto cuya área sea menor a min_frac del
    componente más grande.

    min_frac=0.30: preserva bucles estructurales de B/R/P, elimina ruido.
    """
    bin_img = (skel > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )
    if n_labels <= 1:
        return skel

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = float(areas.max())
    threshold = max_area * min_frac

    clean = np.zeros_like(skel)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= threshold:
            clean[labels == i] = skel[labels == i]
    return clean


def _centroid(binary: np.ndarray) -> tuple[float, float] | None:
    """Devuelve el centroide (cx, cy) en píxeles de los puntos activos."""
    pts = np.argwhere(binary > 0)
    if not len(pts):
        return None
    cy = float(pts[:, 0].mean())
    cx = float(pts[:, 1].mean())
    return cx, cy


def _align_by_centroid(
    skel_a: np.ndarray,
    skel_p: np.ndarray,
) -> np.ndarray:
    """
    Alinea skel_a con skel_p usando SOLO traslación de centroide.
    NO escala. NO rotación.
    """
    h, w = skel_p.shape

    c_p = _centroid(skel_p)
    c_a = _centroid(skel_a)

    if c_p is None or c_a is None:
        return skel_a.astype(np.uint8)

    dx = c_p[0] - c_a[0]
    dy = c_p[1] - c_a[1]

    if abs(dx) < 3 and abs(dy) < 3:
        return skel_a.astype(np.uint8)

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        skel_a.astype(np.uint8), M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted


def _adaptive_dilate_k(skel: np.ndarray) -> tuple[int, int]:
    """Kernel de dilatación adaptado al grosor del trazo."""
    pts    = np.sum(skel > 0)
    total  = skel.size
    density = pts / max(total, 1)

    if density < 0.02:
        return (5, 5)
    elif density < 0.06:
        return (4, 4)
    elif density < 0.12:
        return (3, 3)
    else:
        return (2, 2)


def _build_overlay_np(
    skel_p:  np.ndarray,
    skel_a:  np.ndarray,
    img_a:   np.ndarray | None = None,
) -> np.ndarray:
    """
    Genera el overlay BGR de plantilla vs alumno.

    Si img_a (masa binaria del alumno) está disponible, se usa para el
    overlay del alumno en lugar del esqueleto (más continuo en fotos reales).
    """
    h, w = skel_p.shape

    dk_p = _adaptive_dilate_k(skel_p)
    k_p  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dk_p)
    p_thick = cv2.dilate(skel_p.astype(np.uint8), k_p)

    if img_a is not None and np.sum(img_a > 0) > 10:
        dk_a   = _adaptive_dilate_k(img_a)
        k_a    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dk_a)
        a_thick = cv2.dilate(img_a.astype(np.uint8), k_a)
    else:
        dk_a   = _adaptive_dilate_k(skel_a)
        k_a    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dk_a)
        a_thick = cv2.dilate(skel_a.astype(np.uint8), k_a)

    if p_thick.shape != a_thick.shape:
        a_thick = cv2.resize(a_thick, (p_thick.shape[1], p_thick.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    only_p = (p_thick > 0) & ~(a_thick > 0)
    only_a = (a_thick > 0) & ~(p_thick > 0)
    both   = (p_thick > 0) &  (a_thick > 0)

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[only_p] = COLOR_GUIDE
    overlay[only_a] = COLOR_STUDENT
    overlay[both]   = COLOR_MATCH
    return overlay


# =============================================================================
# Utilidades internas — "Tu trazo"
# =============================================================================

def _ensure_visible_on_white(
    img: np.ndarray,
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """
    Toma cualquier imagen (BGR, grayscale, cualquier fondo) y la convierte
    a una imagen BGR con trazo OSCURO sobre fondo BLANCO, centrada en un
    canvas cuadrado.

    Esto normaliza la apariencia para la UI independientemente de si la
    fuente es un crop de foto, un display_crop limpio, o un grayscale.

    Args:
        img: imagen de entrada (BGR o grayscale)
        target_size: tamaño del canvas cuadrado de salida

    Returns:
        BGR uint8 (target_size, target_size, 3) fondo blanco, trazo oscuro
    """
    ts = target_size

    if img is None or img.size == 0:
        return np.full((ts, ts, 3), 255, dtype=np.uint8)

    # Convertir a grayscale para análisis
    if len(img.shape) == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        src_bgr = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) == 2:
        gray = img
        src_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
        src_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        return np.full((ts, ts, 3), 255, dtype=np.uint8)

    # Verificar que hay contraste suficiente
    img_std = float(gray.std())
    if img_std < 8:
        # Sin contraste → probablemente imagen vacía o uniforme
        return np.full((ts, ts, 3), 255, dtype=np.uint8)

    # Determinar si necesitamos invertir (queremos fondo claro, trazo oscuro)
    mean_val = float(gray.mean())
    border_vals = np.concatenate([
        gray[0, :].ravel(), gray[-1, :].ravel(),
        gray[:, 0].ravel(), gray[:, -1].ravel()
    ])
    border_mean = float(border_vals.mean())

    # Si el fondo (bordes) es oscuro → invertir
    if border_mean < 100 and mean_val < 128:
        src_bgr = cv2.bitwise_not(src_bgr)

    # Resize preservando aspect ratio
    h, w = src_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.full((ts, ts, 3), 255, dtype=np.uint8)

    scale   = ts / max(h, w)
    new_w   = max(1, int(w * scale))
    new_h   = max(1, int(h * scale))
    interp  = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=interp)

    # Canvas blanco, centrado
    canvas = np.full((ts, ts, 3), 255, dtype=np.uint8)
    ox = (ts - new_w) // 2
    oy = (ts - new_h) // 2
    canvas[oy:oy+new_h, ox:ox+new_w] = resized

    return canvas


# =============================================================================
# API pública
# =============================================================================

def generate_comparison_plot(
    skel_p:  np.ndarray,
    skel_a:  np.ndarray,
    score:   float,
    level:   str         = "intermedio",
    char:    str         = "",
    img_a:   np.ndarray | None = None,
) -> str:
    """
    Genera el overlay de comparación como base64 PNG.

    Parameters
    ----------
    skel_p  : esqueleto de la plantilla (128×128, uint8 {0,255} o {0,1})
    skel_a  : esqueleto del alumno      (128×128, uint8 {0,255} o {0,1})
    score   : puntuación final [0-100]
    level   : nivel de dificultad (para título)
    char    : carácter evaluado (para título)
    img_a   : masa binaria del alumno (128×128) — mejora fotos reales
    """
    # 1. Normalizar a {0,1}
    skel_p_bin = (skel_p > 0).astype(np.uint8)
    skel_a_bin = (skel_a > 0).astype(np.uint8)

    # 2. Limpiar fragmentos pequeños del alumno
    skel_a_clean = _clean_small_fragments(skel_a_bin, min_frac=0.30)

    # 3. Alineamiento fino por centroide (SIN escala)
    skel_a_aligned = _align_by_centroid(skel_a_clean, skel_p_bin)

    # 4. Alinear img_a con el mismo desplazamiento
    img_a_aligned = None
    if img_a is not None and np.sum(img_a > 0) > 10:
        img_a_bin = (img_a > 0).astype(np.uint8)
        c_p = _centroid(skel_p_bin)
        c_a_mass = _centroid(img_a_bin)
        if c_p is not None and c_a_mass is not None:
            dx = c_p[0] - c_a_mass[0]
            dy = c_p[1] - c_a_mass[1]
            if abs(dx) >= 3 or abs(dy) >= 3:
                h, w = skel_p_bin.shape
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                img_a_aligned = cv2.warpAffine(
                    img_a_bin, M, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            else:
                img_a_aligned = img_a_bin
        else:
            img_a_aligned = img_a_bin

    # 5. Overlay 128×128
    overlay_128 = _build_overlay_np(skel_p_bin, skel_a_aligned, img_a=img_a_aligned)

    # 6. Upscale 128→512 + suavizado leve
    overlay_up = cv2.resize(
        overlay_128,
        (VIZ_OUTPUT_PX, VIZ_OUTPUT_PX),
        interpolation=cv2.INTER_CUBIC,
    )
    overlay_up  = cv2.GaussianBlur(overlay_up, VIZ_BLUR_K, VIZ_BLUR_SIGMA)
    overlay_rgb = cv2.cvtColor(overlay_up, cv2.COLOR_BGR2RGB)

    # 7. Plot matplotlib
    fig, ax = plt.subplots(
        figsize=(VIZ_OUTPUT_PX / VIZ_DPI, VIZ_OUTPUT_PX / VIZ_DPI),
        dpi=VIZ_DPI,
    )
    ax.imshow(overlay_rgb, interpolation="bilinear")

    title = f"Evaluación: {score:.2f}%"
    if char:
        title = f"'{char}'  —  {title}"
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)

    patches = [
        mpatches.Patch(color=COLOR_GUIDE_MPL,   label="Guía"),
        mpatches.Patch(color=COLOR_STUDENT_MPL, label="Alumno"),
        mpatches.Patch(color=COLOR_MATCH_MPL,   label="Acierto"),
    ]
    ax.legend(
        handles=patches, loc="lower center", ncol=3, fontsize=8,
        framealpha=0.65, facecolor="#111111", edgecolor="none",
        labelcolor="white", handlelength=1.2, handleheight=0.8,
        borderpad=0.5, columnspacing=1.0,
    )
    ax.axis("off")
    fig.patch.set_facecolor("#080808")
    ax.set_facecolor("#080808")
    plt.tight_layout(pad=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_raw_crop_image(
    raw_crop_bgr: np.ndarray | None,
    mask: np.ndarray | None,
    display_crop: np.ndarray | None = None,
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """
    Genera la imagen "Tu trazo" para la UI.

    CAMBIO v4.2: Prioriza display_crop (del image_cleaner, limpio y con
    el carácter visible) sobre raw_crop_bgr (foto cruda del cuaderno)
    y sobre la máscara del normalizer (que puede ser ruido en fotos reales).

    Orden de prioridad:
      1. display_crop — limpio por image_cleaner, sin líneas azules,
         carácter visible, fondo blanco. MEJOR opción para UI.
      2. raw_crop_bgr — crop original de YOLO. Puede tener líneas azules
         y fondo gris, pero al menos muestra la foto real.
      3. mask → convertida a imagen — SOLO como último recurso.
         La máscara del normalizer puede ser basura en fotos reales.
      4. Canvas blanco — si nada funciona.

    Args:
        raw_crop_bgr: crop BGR original de YOLO
        mask: máscara binaria del normalizer (puede ser mala en fotos)
        display_crop: crop limpiado por image_cleaner (PREFERIDO)
        target_size: tamaño del canvas cuadrado

    Returns:
        BGR uint8 (target_size, target_size, 3) — para la UI
    """
    ts = target_size

    # ── 1. display_crop: limpio por image_cleaner ──
    if display_crop is not None and isinstance(display_crop, np.ndarray):
        if display_crop.size > 0 and float(display_crop.std()) > 5:
            return _ensure_visible_on_white(display_crop, ts)

    # ── 2. raw_crop_bgr: foto original ──
    if raw_crop_bgr is not None and isinstance(raw_crop_bgr, np.ndarray):
        if raw_crop_bgr.size > 0 and float(
            cv2.cvtColor(raw_crop_bgr, cv2.COLOR_BGR2GRAY).std()
            if len(raw_crop_bgr.shape) == 3
            else raw_crop_bgr.std()
        ) > 10:
            return _ensure_visible_on_white(raw_crop_bgr, ts)

    # ── 3. Máscara como último recurso ──
    if mask is not None and isinstance(mask, np.ndarray):
        if mask.size > 0 and np.sum(mask > 0) > 50:
            # Solo usar si tiene suficientes píxeles activos (>50)
            # para evitar los "puntos dispersos"
            mask_bin = (mask > 0).astype(np.uint8)

            # Verificar que no es solo ruido disperso:
            # calcular ratio entre área del convex hull y píxeles activos
            contours, _ = cv2.findContours(
                mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                # Si el componente más grande tiene suficiente densidad
                largest = max(contours, key=cv2.contourArea)
                hull_area = cv2.contourArea(cv2.convexHull(largest))
                pixel_count = np.sum(mask_bin > 0)

                if hull_area > 100 and pixel_count / max(hull_area, 1) > 0.05:
                    # Parece un trazo real, no solo puntos dispersos
                    gray = np.full(mask_bin.shape, 255, dtype=np.uint8)
                    gray[mask_bin > 0] = 0
                    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    return _ensure_visible_on_white(bgr, ts)

    # ── 4. Último recurso: cualquier imagen disponible ──
    for candidate in [raw_crop_bgr, display_crop]:
        if candidate is not None and isinstance(candidate, np.ndarray):
            if candidate.size > 0:
                return _ensure_visible_on_white(candidate, ts)

    # ── 5. Nada funciona → canvas blanco ──
    return np.full((ts, ts, 3), 255, dtype=np.uint8)