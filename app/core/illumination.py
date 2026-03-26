"""
app/core/illumination.py
========================
Responsabilidad única (SRP): corregir la iluminación de una imagen en
escala de grises ANTES de binarizarla.

Por qué va en app/core/:
  - Es una etapa del pipeline de preprocesamiento, no una métrica.
  - Es independiente del tipo de carácter o del nivel de dificultad.
  - normalizer.py la importa como un paso más del pipeline.

Por qué un archivo separado y no una función en normalizer.py:
  - OCP: se pueden agregar nuevos métodos (retinex, white balance) sin
    tocar normalizer.py.
  - Testeable de forma aislada con imágenes sintéticas.

Exports públicos
----------------
  correct_background(gray, blur_k)       -> np.ndarray  (división por fondo)
  to_lab_lightness(bgr)                  -> np.ndarray  (canal L de LAB)
  normalize_illumination(gray, params)   -> np.ndarray  (función principal)
"""

from __future__ import annotations

import cv2
import numpy as np


# =============================================================================
# Método 1: Corrección por división de fondo (Background Division)
# =============================================================================

def correct_background(gray: np.ndarray, blur_k: int = 101) -> np.ndarray:
    """
    Elimina variaciones lentas de iluminación (sombras de mano, iluminación
    lateral, gradientes de luz) dividiendo la imagen entre una estimación
    del fondo (papel en blanco).

    Algoritmo:
      1. Estimar el "fondo" con un blur muy grande → elimina el trazo,
         conserva solo la tendencia de iluminación del papel.
      2. Dividir la imagen original entre ese fondo y reescalar a [0,255].
         El resultado tiene iluminación homogénea independientemente de
         dónde esté la sombra.

    Por qué funciona mejor que CLAHE solo:
      CLAHE mejora el contraste local, pero si una zona entera está oscura
      por una sombra, CLAHE no puede "saber" que esa zona debería ser blanca.
      La división de fondo sí lo sabe porque estima la iluminación real
      del papel en cada punto.

    Parameters
    ----------
    gray   : np.ndarray uint8 — imagen en escala de grises
    blur_k : int impar        — kernel del blur para estimar el fondo.
             Debe ser mayor que el trazo más grande (~1/3 del lado menor).

    Returns
    -------
    np.ndarray uint8 — imagen con iluminación homogeneizada [0,255]
    """
    # Garantizar kernel impar y >= 3
    k = blur_k if blur_k % 2 == 1 else blur_k + 1
    k = max(3, k)

    # Estimar fondo: blur suficientemente grande para borrar los trazos
    background = cv2.GaussianBlur(gray, (k, k), 0).astype(np.float32)

    # División: fondo / imagen → zonas claras (papel) quedan ≈1.0
    # Multiplicar por 128 para centrar el rango en gris medio
    f      = gray.astype(np.float32)
    result = (f / (background + 1e-6)) * 128.0

    # Recortar y reescalar a uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# =============================================================================
# Método 2: Canal L del espacio LAB
# =============================================================================

def to_lab_lightness(bgr: np.ndarray) -> np.ndarray:
    """
    Extrae el canal L* (luminosidad) del espacio de color CIE LAB.

    Por qué es mejor que la simple conversión a gris en casos difíciles:
      - La conversión BGR→GRAY pondera R*0.114 + G*0.587 + B*0.299.
        Con un lápiz grafito (gris neutro) y papel azul o amarillo, los
        coeficientes distorsionan el contraste percibido.
      - El canal L* de LAB es perceptualmente uniforme: lo que parece más
        oscuro a la vista tiene valor L* menor, independientemente del tono.
      - Para grafito (gris) sobre papel de colores es especialmente robusto.

    Parameters
    ----------
    bgr : np.ndarray uint8 BGR — imagen en color (recorte de la libreta)

    Returns
    -------
    np.ndarray uint8 — canal L* reescalado a [0,255]
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0]   # L está en [0,255] en OpenCV (mapeado desde [0,100])
    return L


# =============================================================================
# Función principal: decide qué corrección aplicar según los parámetros
# =============================================================================

def normalize_illumination(
    gray:           np.ndarray,
    use_bg_division: bool  = False,
    bg_blur_k:      int   = 101,
    clahe_clip:     float = 3.0,
    clahe_tile:     int   = 8,
) -> np.ndarray:
    """
    Normaliza la iluminación de una imagen en escala de grises.

    Pipeline interno:
      1. Si use_bg_division: corregir_fondo() → elimina sombras y gradientes.
      2. CLAHE adaptativo → mejora contraste local restante.

    Los parámetros vienen de PipelineParams (image_quality.py), NO de
    config.py directamente. Esto permite que cada imagen use los parámetros
    óptimos para su condición particular.

    Parameters
    ----------
    gray            : np.ndarray uint8 — imagen en escala de grises
    use_bg_division : bool — aplicar corrección de fondo (True si hay sombras)
    bg_blur_k       : int  — kernel del blur de fondo (de PipelineParams)
    clahe_clip      : float — clip limit de CLAHE (de PipelineParams)
    clahe_tile      : int   — tamaño de tile de CLAHE (de PipelineParams)

    Returns
    -------
    np.ndarray uint8 — imagen normalizada lista para binarizar
    """
    result = gray.copy()

    # Paso 1: corrección de fondo si hay sombras detectadas
    if use_bg_division:
        result = correct_background(result, blur_k=bg_blur_k)

    # Paso 2: CLAHE para mejorar contraste local residual
    tile  = max(2, min(16, clahe_tile))
    clahe = cv2.createCLAHE(
        clipLimit    = float(clahe_clip),
        tileGridSize = (tile, tile),
    )
    result = clahe.apply(result)

    return result