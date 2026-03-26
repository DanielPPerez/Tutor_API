import cv2
import numpy as np
import onnxruntime as ort
from app.core import config

# Cargamos la sesión
session_cls = ort.InferenceSession(config.MOBILENET_MODEL_PATH, providers=['CPUExecutionProvider'])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def classify_character(normalized_mask):
    # 1. INVERTIR: El entrenamiento se hizo con RandomInvert(p=1.0)
    # EMNIST (Blanco sobre Negro) -> Invertido (Negro sobre Blanco)
    img_inv = cv2.bitwise_not(normalized_mask)
    
    # 2. Asegurar tamaño real esperado por el modelo (ej. 64x64 vs 128x128)
    input_meta = session_cls.get_inputs()[0]
    input_shape = list(input_meta.shape) if input_meta.shape is not None else []

    def _dim_to_int(d: object) -> int | None:
        if isinstance(d, int):
            return d
        if isinstance(d, str):
            return int(d) if d.isdigit() else None
        return None

    expected_c = _dim_to_int(input_shape[1]) if len(input_shape) >= 2 else None
    expected_h = _dim_to_int(input_shape[2]) if len(input_shape) >= 3 else None
    expected_w = _dim_to_int(input_shape[3]) if len(input_shape) >= 4 else None

    expected_h = expected_h or int(config.TARGET_SHAPE[0])
    expected_w = expected_w or int(config.TARGET_SHAPE[1])
    expected_c = expected_c or 3

    img_resized = cv2.resize(img_inv, (expected_w, expected_h))
    
    # --- DEBUG: Mira esta imagen, debe ser letra NEGRA sobre fondo BLANCO ---
    # cv2.imshow("Lo que el modelo ve", img_resized)
    # cv2.waitKey(1)

    # 3. Convertir a RGB (MobileNet espera 3 canales)
    if expected_c == 1:
        img_rgb = img_resized[..., None]  # (H, W, 1)
    else:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    # 4. Preprocesamiento idéntico al entrenamiento
    # HWC -> CHW y escala 0-1
    input_data = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    
    # Normalización ImageNet
    if expected_c == 1:
        mean = np.array([0.485]).reshape(1, 1, 1).astype(np.float32)
        std = np.array([0.229]).reshape(1, 1, 1).astype(np.float32)
    else:
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1).astype(np.float32)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1).astype(np.float32)
    input_data = (input_data - mean) / std
    
    input_data = np.expand_dims(input_data, axis=0)

    # 5. Inferencia
    outputs = session_cls.run(None, {session_cls.get_inputs()[0].name: input_data})
    
    logits = outputs[0][0]
    probs = softmax(logits)
    
    idx = np.argmax(probs)
    confidence = probs[idx]
    
    return config.CLASS_NAMES[str(idx)], float(confidence)