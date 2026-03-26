import cv2
import numpy as np
import onnxruntime as ort
from app.core import config

# Cargamos la sesión de ONNX una sola vez para eficiencia
session_det = ort.InferenceSession(config.YOLO_MODEL_PATH, providers=['CPUExecutionProvider'])

def detect_character(image_bgr):
    """
    Detecta el carácter usando YOLO ONNX y retorna el recorte (crop).
    """
    h_orig, w_orig = image_bgr.shape[:2]
    
    # 1. Preprocesamiento (Resize con Letterbox para mantener aspecto)
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    scale = min(config.YOLO_INPUT_SIZE / h_orig, config.YOLO_INPUT_SIZE / w_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Crear lienzo cuadrado y centrar
    canvas = np.full((config.YOLO_INPUT_SIZE, config.YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[(config.YOLO_INPUT_SIZE - new_h) // 2 : (config.YOLO_INPUT_SIZE - new_h) // 2 + new_h,
           (config.YOLO_INPUT_SIZE - new_w) // 2 : (config.YOLO_INPUT_SIZE - new_w) // 2 + new_w] = img_resized
    
    # Normalizar y cambiar a formato NCHW (batch, canales, alto, ancho)
    input_data = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # 2. Inferencia
    outputs = session_det.run(None, {session_det.get_inputs()[0].name: input_data})
    
    # 3. Post-procesamiento simple (Asumiendo formato YOLOv8: [1, 84, 8400])
    predictions = np.squeeze(outputs[0]).T
    scores = np.max(predictions[:, 4:], axis=1)
    
    # Filtrar por confianza
    mask = scores > config.DETECTION_THRESHOLD
    valid_predictions = predictions[mask]
    valid_scores = scores[mask]

    if len(valid_predictions) == 0:
        return None

    # Obtener la mejor caja (puedes añadir NMS aquí si hay múltiples letras)
    best_idx = np.argmax(valid_scores)
    row = valid_predictions[best_idx]
    
    # Convertir coordenadas de YOLO (centro_x, centro_y, w, h) a coordenadas de imagen original
    x_c, y_c, w, h = row[:4]
    
    # Ajustar por el letterbox y escala
    x_c = (x_c - (config.YOLO_INPUT_SIZE - new_w) / 2) / scale
    y_c = (y_c - (config.YOLO_INPUT_SIZE - new_h) / 2) / scale
    w /= scale
    h /= scale

    x1, y1 = int(x_c - w/2), int(y_c - h/2)
    x2, y2 = int(x1 + w), int(y1 + h)

    # Recorte con seguridad de bordes
    crop = image_bgr[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
    
    return crop