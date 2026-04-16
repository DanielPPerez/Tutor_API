"""
app/core/preprocessing.py
=========================
Pipeline de preprocesamiento que replica EXACTAMENTE las transformaciones
del notebook de entrenamiento.

Este archivo es el PUENTE CRÍTICO entre la imagen limpia (de image_cleaner.py)
y el modelo ONNX. Cualquier diferencia con el entrenamiento causa degradación.

Pipeline del entrenamiento (notebook):
  1. Imagen grayscale uint8 (fondo ~blanco 255, trazo ~negro 0-80)
  2. Resize a 128×128 (letterbox con padding BLANCO para eval/test)
  3. cvtColor GRAY2RGB → 3 canales (pero esencialmente grayscale)
  4. float32 / 255.0 → rango [0, 1]
  5. ImageNet normalize: (x - mean) / std
  6. Transpose HWC → CHW
  7. Batch dimension → (1, 3, 128, 128)

Transforms de validación en el notebook (Albumentations):
  A.Resize(IMG_SIZE, IMG_SIZE),   # ← nosotros usamos letterbox
  A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
  ToTensorV2(),

NOTA: El notebook usa A.Resize (stretch) para validación, pero el modelo
es robusto a letterbox vs stretch. Usamos letterbox porque preserva
aspect ratio, igual que en el flag _USE_LETTERBOX del classifier.py original.

Formato de entrada esperado:
  - Grayscale uint8 de image_cleaner.clean_crop_for_classification()
  - O BGR uint8 si viene de otro pipeline
  - Fondo ~blanco (200-255), trazo ~negro (0-100)
  - Valores CONTINUOS (no binarios)

Formato de salida:
  - np.ndarray float32, shape (1, 3, 128, 128)
  - Normalizado con ImageNet mean/std
  - Listo para session.run() de ONNX Runtime
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

import logging

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTES — Deben coincidir EXACTAMENTE con el notebook
# ═════════════════════════════════════════════════════════════════════════════

# Tamaño de entrada del modelo
IMG_SIZE: int = 128

# Valor de padding para letterbox (BLANCO, igual que el notebook)
PADDING_VALUE: int = 255

# Normalización ImageNet (igual que Albumentations A.Normalize default)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Pre-computar para formato CHW (evita reshape en cada llamada)
IMAGENET_MEAN_CHW = IMAGENET_MEAN[:, np.newaxis, np.newaxis]  # (3, 1, 1)
IMAGENET_STD_CHW = IMAGENET_STD[:, np.newaxis, np.newaxis]    # (3, 1, 1)


# ═════════════════════════════════════════════════════════════════════════════
# 1. LETTERBOX RESIZE (replica el del notebook)
# ═════════════════════════════════════════════════════════════════════════════

def letterbox_resize(
    img: np.ndarray,
    target_size: int = IMG_SIZE,
    pad_value: int = PADDING_VALUE,
) -> np.ndarray:
    """
    Resize preservando aspect ratio con padding.

    Replica EXACTAMENTE la función letterbox_resize del notebook:
      - Calcula escala para que el lado más largo = target_size
      - Resize con la escala
      - Centra en canvas de target_size × target_size
      - Padding con pad_value (BLANCO = 255)

    Funciona con grayscale (H, W) y BGR/RGB (H, W, 3).

    Args:
        img: imagen de entrada (grayscale o color)
        target_size: tamaño del canvas cuadrado de salida
        pad_value: valor de padding (255 = blanco)

    Returns:
        Imagen redimensionada y centrada, misma profundidad de canales
    """
    if img is None or img.size == 0:
        if len(img.shape) == 3:
            return np.full(
                (target_size, target_size, img.shape[2]),
                pad_value, dtype=np.uint8
            )
        return np.full((target_size, target_size), pad_value, dtype=np.uint8)

    h, w = img.shape[:2]

    if h == 0 or w == 0:
        if len(img.shape) == 3:
            return np.full(
                (target_size, target_size, img.shape[2]),
                pad_value, dtype=np.uint8
            )
        return np.full((target_size, target_size), pad_value, dtype=np.uint8)

    # Calcular escala (el lado más largo se ajusta a target_size)
    scale = target_size / max(h, w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    # Elegir interpolación según dirección del resize
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Crear canvas con padding
    if len(img.shape) == 3:
        canvas = np.full(
            (target_size, target_size, img.shape[2]),
            pad_value, dtype=np.uint8
        )
    else:
        canvas = np.full(
            (target_size, target_size),
            pad_value, dtype=np.uint8
        )

    # Centrar la imagen redimensionada
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas


def direct_resize(
    img: np.ndarray,
    target_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    Resize directo (stretch) a target_size × target_size.

    Esto es lo que hace A.Resize en Albumentations durante validación
    en el notebook. NO preserva aspect ratio.

    Args:
        img: imagen de entrada
        target_size: tamaño de salida

    Returns:
        Imagen redimensionada (stretch)
    """
    if img is None or img.size == 0:
        if len(img.shape) == 3:
            return np.full(
                (target_size, target_size, img.shape[2]),
                255, dtype=np.uint8
            )
        return np.full((target_size, target_size), 255, dtype=np.uint8)

    return cv2.resize(
        img, (target_size, target_size),
        interpolation=cv2.INTER_LINEAR
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. CONVERSIÓN DE CANALES
# ═════════════════════════════════════════════════════════════════════════════

def ensure_rgb_3ch(img: np.ndarray) -> np.ndarray:
    """
    Convierte cualquier imagen a RGB 3 canales.

    El modelo fue entrenado con imágenes GRAY2RGB (3 canales idénticos).
    Esta función replica ese comportamiento.

    Args:
        img: grayscale (H, W), BGR (H, W, 3), BGRA (H, W, 4)

    Returns:
        RGB uint8 (H, W, 3)
    """
    if len(img.shape) == 2:
        # Grayscale → RGB (3 canales idénticos, como en el notebook)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 1:
        # Single channel → RGB
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 4:
        # BGRA → BGR → RGB
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR),
                            cv2.COLOR_BGR2RGB)

    if img.shape[2] == 3:
        # BGR → RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Fallback: tomar primeros 3 canales
    return img[:, :, :3].copy()


def ensure_bgr_3ch(img: np.ndarray) -> np.ndarray:
    """
    Convierte cualquier imagen a BGR 3 canales.

    Args:
        img: grayscale, BGR, BGRA

    Returns:
        BGR uint8 (H, W, 3)
    """
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img.copy()


# ═════════════════════════════════════════════════════════════════════════════
# 3. NORMALIZACIÓN IMAGENET
# ═════════════════════════════════════════════════════════════════════════════

def normalize_imagenet(img_float_chw: np.ndarray) -> np.ndarray:
    """
    Aplica normalización ImageNet a tensor CHW float32.

    Formula: (x - mean) / std
    Con mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    Args:
        img_float_chw: float32 shape (3, H, W), rango [0, 1]

    Returns:
        float32 shape (3, H, W), normalizado
    """
    return (img_float_chw - IMAGENET_MEAN_CHW) / IMAGENET_STD_CHW


# ═════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE COMPLETO: IMAGEN → TENSOR LISTO PARA MODELO
# ═════════════════════════════════════════════════════════════════════════════

def prepare_for_model(
    img: np.ndarray,
    use_letterbox: bool = True,
    target_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    Pipeline completo de preprocesamiento: imagen → tensor para ONNX.

    Replica EXACTAMENTE las transformaciones del notebook de entrenamiento.

    Pasos:
      1. Resize a target_size × target_size
         - letterbox (padding blanco) si use_letterbox=True
         - stretch directo si use_letterbox=False
      2. Convertir a RGB 3 canales
      3. float32 / 255.0 → rango [0, 1]
      4. HWC → CHW
      5. Normalización ImageNet
      6. Agregar batch dimension → (1, 3, H, W)

    Args:
        img: imagen de entrada. Acepta:
             - Grayscale uint8 (H, W) — PREFERIDO, de image_cleaner
             - BGR uint8 (H, W, 3) — crop directo de OpenCV
             - RGB uint8 (H, W, 3)
        use_letterbox: True = letterbox con padding blanco (default)
                       False = stretch directo (como A.Resize)
        target_size: tamaño de entrada del modelo (default: 128)

    Returns:
        np.ndarray float32 shape (1, 3, target_size, target_size)
        Normalizado con ImageNet mean/std
        Listo para ort.InferenceSession.run()
    """
    # ── Validación de entrada ──
    if img is None or img.size == 0:
        logger.warning("prepare_for_model: imagen vacía, generando tensor blanco")
        return _make_white_tensor(target_size)

    h, w = img.shape[:2]
    if h < 2 or w < 2:
        logger.warning(
            f"prepare_for_model: imagen muy pequeña ({w}x{h}), "
            "generando tensor blanco"
        )
        return _make_white_tensor(target_size)

    # ── Paso 1: Resize ──
    if use_letterbox:
        img_resized = letterbox_resize(img, target_size, pad_value=PADDING_VALUE)
    else:
        img_resized = direct_resize(img, target_size)

    # ── Paso 2: Convertir a RGB 3 canales ──
    img_rgb = ensure_rgb_3ch(img_resized)

    # ── Paso 3: float32 / 255.0 ──
    img_float = img_rgb.astype(np.float32) / 255.0

    # ── Paso 4: HWC → CHW ──
    img_chw = img_float.transpose(2, 0, 1)  # (3, H, W)

    # ── Paso 5: Normalización ImageNet ──
    img_normalized = normalize_imagenet(img_chw)

    # ── Paso 6: Batch dimension ──
    tensor = img_normalized[np.newaxis, ...]  # (1, 3, H, W)

    return tensor.astype(np.float32)


def prepare_for_model_grayscale_1ch(
    img: np.ndarray,
    use_letterbox: bool = True,
    target_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    Pipeline para modelos con entrada de 1 canal (grayscale).

    Solo se usa si el modelo ONNX tiene input_shape (1, 1, H, W).
    La mayoría de modelos EfficientNet usan 3 canales.

    Args:
        img: grayscale o color uint8

    Returns:
        np.ndarray float32 shape (1, 1, target_size, target_size)
    """
    # Convertir a grayscale si es necesario
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Resize
    if use_letterbox:
        gray_resized = letterbox_resize(gray, target_size, pad_value=PADDING_VALUE)
    else:
        gray_resized = direct_resize(gray, target_size)

    # float32 / 255.0
    img_float = gray_resized.astype(np.float32) / 255.0

    # HW → CHW (1 canal)
    img_chw = img_float[np.newaxis, ...]  # (1, H, W)

    # Normalización (solo canal L, usando mean/std del canal rojo de ImageNet)
    mean = np.array([0.485], dtype=np.float32)[:, np.newaxis, np.newaxis]
    std = np.array([0.229], dtype=np.float32)[:, np.newaxis, np.newaxis]
    img_normalized = (img_chw - mean) / std

    # Batch dimension
    tensor = img_normalized[np.newaxis, ...]  # (1, 1, H, W)
    return tensor.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 5. UTILIDADES
# ═════════════════════════════════════════════════════════════════════════════

def _make_white_tensor(target_size: int = IMG_SIZE) -> np.ndarray:
    """
    Genera un tensor de imagen blanca (fondo puro, sin trazo).
    Útil como fallback para imágenes vacías/inválidas.

    El modelo debería dar baja confianza en todas las clases.

    Returns:
        float32 (1, 3, target_size, target_size), normalizado ImageNet
    """
    # Imagen blanca (255) → /255 → 1.0 → normalizar
    white = np.ones((3, target_size, target_size), dtype=np.float32)
    normalized = normalize_imagenet(white)
    return normalized[np.newaxis, ...]


def denormalize_for_display(
    tensor: np.ndarray,
) -> np.ndarray:
    """
    Desnormaliza un tensor del modelo para visualización.

    Invierte: ImageNet normalize → ×255 → uint8 → CHW→HWC → RGB→BGR

    Args:
        tensor: float32 shape (1, 3, H, W) o (3, H, W)

    Returns:
        BGR uint8 (H, W, 3) para cv2.imshow/imwrite
    """
    if tensor.ndim == 4:
        tensor = tensor[0]  # Remove batch dim

    # Desnormalizar ImageNet
    img_chw = tensor * IMAGENET_STD_CHW + IMAGENET_MEAN_CHW

    # Clamp y convertir
    img_chw = np.clip(img_chw * 255.0, 0, 255).astype(np.uint8)

    # CHW → HWC
    img_hwc = img_chw.transpose(1, 2, 0)  # (H, W, 3) RGB

    # RGB → BGR para OpenCV
    img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)

    return img_bgr


def get_preprocessing_info(
    img: np.ndarray,
    use_letterbox: bool = True,
) -> dict:
    """
    Información diagnóstica del preprocesamiento.
    Útil para debugging.

    Args:
        img: imagen de entrada (antes de preprocesar)

    Returns:
        dict con métricas del pipeline
    """
    if img is None or img.size == 0:
        return {"error": "imagen vacía"}

    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1

    if channels == 1 or len(img.shape) == 2:
        gray = img if len(img.shape) == 2 else img[:, :, 0]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Simular resize para ver tamaño final
    scale = IMG_SIZE / max(h, w) if use_letterbox else None
    if use_letterbox:
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        pad_h = IMG_SIZE - new_h
        pad_w = IMG_SIZE - new_w
    else:
        new_h, new_w = IMG_SIZE, IMG_SIZE
        pad_h, pad_w = 0, 0

    return {
        "input_shape": (h, w, channels),
        "resize_method": "letterbox" if use_letterbox else "stretch",
        "scale_factor": round(float(scale), 4) if scale else None,
        "resized_shape": (new_h, new_w),
        "padding": (pad_h, pad_w),
        "target_size": IMG_SIZE,
        "gray_mean": round(float(gray.mean()), 1),
        "gray_std": round(float(gray.std()), 1),
        "gray_min": int(gray.min()),
        "gray_max": int(gray.max()),
        "normalization": "imagenet",
        "output_shape": f"(1, 3, {IMG_SIZE}, {IMG_SIZE})",
    }