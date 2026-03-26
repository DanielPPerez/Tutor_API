"""
app/utils/visualizer.py  (v3)
==============================

DIAGNÓSTICO DE LOS 2 BUGS ACTUALES
------------------------------------

BUG A — Visualizer rompe fotos reales (B fragmentada, C encogida):
  CAUSA RAÍZ: _clean_small_fragments() con min_frac=0.05 elimina fragmentos
  < 5% del componente mayor. Para una "B" escrita a mano, el trazo se
  esqueletiza en 2-3 arcos separados (bucle superior, bucle inferior, palo
  vertical). El palo vertical puede ser el componente más grande, y los dos
  bucles quedan debajo del 5% → se eliminan → queda solo el palo.
  Luego _scale_skel_to_fit_guide escala ese palo para que llene el bbox de
  la plantilla "B" completa → resultado: una línea vertical estirada.

  Para la "C": el trazo real es una sola curva continua (1 componente), pero
  si hay 2-3 puntos de ruido en esquinas, _largest_component_bbox devuelve
  el bbox de esos puntos si son más grandes. Resultado: escala × 0.109.

  SOLUCIÓN:
  1. _clean_small_fragments: subir min_frac a 0.30 (eliminar solo lo que sea
     < 30% del mayor). Esto preserva los bucles de B/R/P que son estructurales.
  2. _scale_skel_to_fit_guide: NO hacer escala. El normalizer ya centra
     y escala el trazo al 128×128. Si el normalizer hizo bien su trabajo,
     el bbox del alumno ya está en el mismo espacio que la plantilla.
     El visualizer solo debe SUPERPONER, no reescalar de nuevo.
     Si se reescala dos veces (normalizer + visualizer), se distorsiona todo.
  3. Alineamiento por centroide de masa total (no bbox del mayor componente)
     como ajuste fino de posición, sin cambio de escala.

BUG B — Clasificador predice siempre confianza 100% pero letra incorrecta:
  CAUSA RAÍZ: El modelo fue entrenado con EMNIST byclass que tiene letras
  de 28×28 blanco sobre negro. El clasificador en processor.py recibe
  imágenes de 128×128 negro sobre blanco (THRESH_BINARY_INV).
  RandomInvert(p=1.0) en el entrenamiento debería haber corregido esto, pero
  el modelo aprendió características de borde en 28×28 upscaleado a 128×128
  (mucho padding de ceros = borde negro = feature dominante).
  Con confianza 100% siempre = el modelo aprendió a predecir una sola clase
  independientemente de la entrada (modo colapso).

  SOLUCIÓN EN ESTE ARCHIVO: No aplica — el visualizer no puede corregir el
  clasificador. Ver processor.py para la corrección del clasificador.
  AQUÍ solo se documenta el diagnóstico para no confundir los dos bugs.

CAMBIOS EN ESTA VERSIÓN:
  - _clean_small_fragments: min_frac por defecto sube a 0.30
  - _scale_skel_to_fit_guide: ELIMINADO el escalado. Solo traslación por
    centroide. Esto es correcto porque normalize_character() ya garantiza
    que el trazo ocupa el mismo espacio 128×128.
  - Nueva función _align_by_centroid: alineamiento fino sin escala.
  - generate_comparison_plot: ahora acepta img_a (masa del alumno) además
    del esqueleto, para poder hacer overlay de masa vs esqueleto de plantilla
    cuando el esqueleto del alumno está fragmentado (fotos reales).
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
# Utilidades internas
# =============================================================================

def _clean_small_fragments(skel: np.ndarray, min_frac: float = 0.30) -> np.ndarray:
    """
    Elimina fragmentos del esqueleto cuya área sea menor a min_frac del
    componente más grande.

    CORRECCIÓN v3: min_frac sube de 0.05 → 0.30.

    Por qué 0.30:
      - Una "B" tiene palo + 2 bucles. El palo puede tener ~40% del área,
        cada bucle ~30%. Con 0.05, los bucles (30% > 5%) se conservaban, pero
        en fotos reales los bucles pueden fragmentarse en 2-3 arcos cada uno,
        quedando al 10-15% → se eliminaban.
      - Con 0.30: se eliminan solo fragmentos muy pequeños (ruido de papel,
        manchas de tinta) pero se conservan todos los trazos estructurales.
      - El único caso donde 0.30 elimina algo válido: letras con un trazo
        accesorio muy pequeño como la tilde de la "i" o la "j" (punto).
        Para esos casos el punto es ~5-15% del trazo → se pierde, pero es
        aceptable porque el trazo principal (palo) se evalúa correctamente.
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
    # pts[:,0] = filas (y), pts[:,1] = columnas (x)
    cy = float(pts[:, 0].mean())
    cx = float(pts[:, 1].mean())
    return cx, cy


def _align_by_centroid(
    skel_a: np.ndarray,
    skel_p: np.ndarray,
) -> np.ndarray:
    """
    Alinea skel_a con skel_p usando SOLO traslación de centroide a centroide.
    NO escala. NO rotación.

    Por qué NO escalar aquí:
      normalize_character() ya centra y escala el trazo del alumno al canvas
      128×128 con el mismo padding que la plantilla. Si el normalizer funcionó
      correctamente, los bounding boxes de ambos trazos ya son comparables.
      Escalar de nuevo en el visualizer distorsiona la comparación porque
      amplifica cualquier imperfección del normalizer (fragmentos, puntos de
      ruido que ensanchan el bbox, etc.).

      La traslación de centroide es necesaria porque el normalizer centra por
      bounding box, pero si el trazo tiene ruido disperso, el centroide del
      bbox ≠ centroide visual → pequeño desplazamiento que vale la pena corregir.

    Returns
    -------
    np.ndarray uint8 del mismo tamaño que skel_p, con skel_a trasladado.
    """
    h, w = skel_p.shape

    c_p = _centroid(skel_p)
    c_a = _centroid(skel_a)

    if c_p is None or c_a is None:
        return skel_a.astype(np.uint8)

    # Desplazamiento necesario para llevar el centroide de A al de P
    dx = c_p[0] - c_a[0]
    dy = c_p[1] - c_a[1]

    # Solo aplicar traslación si el desplazamiento es significativo (> 3px)
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
    """
    Calcula el tamaño de kernel de dilatación adaptado al grosor del trazo.
    Trazos muy finos (esqueleto puro de 1px) necesitan más dilatación.
    """
    pts    = np.sum(skel > 0)
    total  = skel.size
    density = pts / max(total, 1)

    if density < 0.02:
        return (5, 5)   # Esqueleto puro muy fino
    elif density < 0.06:
        return (4, 4)   # Esqueleto medio
    elif density < 0.12:
        return (3, 3)   # Trazo medio
    else:
        return (2, 2)   # Masa gruesa (imagen sin esqueletizar)


def _build_overlay_np(
    skel_p:  np.ndarray,
    skel_a:  np.ndarray,
    img_a:   np.ndarray | None = None,
) -> np.ndarray:
    """
    Genera el overlay BGR.

    Si img_a (masa binaria del alumno) está disponible, se usa para el
    overlay del alumno en lugar del esqueleto. Esto da mejor resultado
    con fotos reales donde el esqueleto está fragmentado, porque la masa
    normalizada suele ser más continua.

    skel_p siempre usa el esqueleto de la plantilla (trazo fino = guía).
    """
    h, w = skel_p.shape

    # Plantilla: siempre el esqueleto fino
    dk_p = _adaptive_dilate_k(skel_p)
    k_p  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dk_p)
    p_thick = cv2.dilate(skel_p.astype(np.uint8), k_p)

    # Alumno: usar masa si está disponible (más continua que el esqueleto en fotos)
    if img_a is not None and np.sum(img_a > 0) > 10:
        # La masa ya viene en 128×128, solo dilatar levemente para el overlay
        dk_a   = _adaptive_dilate_k(img_a)
        k_a    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dk_a)
        a_thick = cv2.dilate(img_a.astype(np.uint8), k_a)
    else:
        dk_a   = _adaptive_dilate_k(skel_a)
        k_a    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dk_a)
        a_thick = cv2.dilate(skel_a.astype(np.uint8), k_a)

    # Redimensionar p_thick al tamaño de a_thick si difieren (no debería)
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

    CAMBIOS v3:
      - Acepta img_a (masa binaria del alumno 128×128) como parámetro opcional.
        Si se pasa, el overlay del alumno usa la masa en lugar del esqueleto.
        Esto da comparaciones más continuas con fotos reales fragmentadas.
      - _scale_skel_to_fit_guide ELIMINADO. Reemplazado por _align_by_centroid
        que solo traslada (no escala). El normalizer ya garantiza la escala.
      - min_frac sube a 0.30 para preservar bucles estructurales de B/R/P/etc.

    Parameters
    ----------
    skel_p  : esqueleto de la plantilla (128×128, uint8 {0,255} o {0,1})
    skel_a  : esqueleto del alumno      (128×128, uint8 {0,255} o {0,1})
    score   : puntuación final [0-100]
    level   : nivel de dificultad (para título)
    char    : carácter evaluado (para título)
    img_a   : masa binaria del alumno (128×128) — NUEVO, mejora fotos reales
    """
    # 1. Normalizar a {0,1}
    skel_p_bin = (skel_p > 0).astype(np.uint8)
    skel_a_bin = (skel_a > 0).astype(np.uint8)

    # 2. Limpiar fragmentos muy pequeños del alumno (ruido de papel)
    #    min_frac=0.30: preserva bucles estructurales, elimina puntos de tinta
    skel_a_clean = _clean_small_fragments(skel_a_bin, min_frac=0.30)

    # 3. Alineamiento fino por centroide (SIN escala)
    skel_a_aligned = _align_by_centroid(skel_a_clean, skel_p_bin)

    # 4. Normalizar img_a si se pasó
    img_a_bin = None
    if img_a is not None:
        img_a_bin = (img_a > 0).astype(np.uint8)

    # 5. Overlay 128×128
    overlay_128 = _build_overlay_np(skel_p_bin, skel_a_aligned, img_a=img_a_bin)

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
    img_a:        np.ndarray,
) -> np.ndarray:
    """
    Prepara la imagen que se muestra como extracción YOLO.
    Si raw_crop_bgr es None, usa img_a normalizado como fallback.
    """
    ts  = TARGET_SIZE
    src = raw_crop_bgr if raw_crop_bgr is not None else (
        cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR) if img_a.ndim == 2 else img_a
    )
    h, w    = src.shape[:2]
    scale   = ts / max(h, w)
    new_w   = max(1, int(w * scale))
    new_h   = max(1, int(h * scale))
    resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.zeros((ts, ts, 3), dtype=np.uint8)
    ox = (ts - new_w) // 2
    oy = (ts - new_h) // 2
    canvas[oy:oy+new_h, ox:ox+new_w] = (
        resized if resized.ndim == 3
        else cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    )
    return canvas