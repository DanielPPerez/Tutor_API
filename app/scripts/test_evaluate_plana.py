"""
test_evaluate_plana.py — Script de verificación del endpoint /evaluate_plana
=============================================================================
Ajustado para coincidir con el detector YOLOv8n entrenado:
  - 1 clase ("trazo")
  - Output ONNX: (1, 5, 8400) → transpuesto (8400, 5) = [cx, cy, w, h, conf]
  - Input: 640×640, letterbox preservando aspect ratio
  - conf=0.25, iou=0.45
  - Reading-order sorting por líneas
"""

import argparse
import base64
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — Debe coincidir con el entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

YOLO_IMG_SIZE     = 640          # Mismo que YOLO_IMG_SIZE del entrenamiento
YOLO_CONF_THRESH  = 0.25         # Mismo que conf=0.25 del entrenamiento
YOLO_IOU_THRESH   = 0.45         # Mismo que iou=0.45 del entrenamiento
YOLO_NUM_CLASSES  = 1            # Solo "trazo"
YOLO_MODEL_PATH   = "app/models/classifier_artifacts/best_detector.onnx"

OUTPUT_DIR = Path("test_plana_output")
OUTPUT_DIR.mkdir(exist_ok=True)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  onnxruntime no instalado. Debug local del detector deshabilitado.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERADOR DE PLANAS SINTÉTICAS
# ─────────────────────────────────────────────────────────────────────────────

def draw_handwritten_char(char: str, size: int = 80) -> np.ndarray:
    """
    Dibuja un carácter simulando escritura manuscrita usando trazos OpenCV.
    Retorna imagen binaria (blanco=trazo sobre negro=fondo).
    """
    img = np.zeros((size, size), dtype=np.uint8)

    cx, cy = size // 2, size // 2
    s = size // 3

    angle_var = random.uniform(-0.15, 0.15)
    thickness = random.randint(2, 4)
    char_upper = char.upper()

    if char_upper == 'A':
        cv2.line(img, (cx, cy - s), (cx - s // 2, cy + s), 255, thickness)
        cv2.line(img, (cx, cy - s), (cx + s // 2, cy + s), 255, thickness)
        cv2.line(img, (cx - s // 3, cy + s // 4), (cx + s // 3, cy + s // 4), 255, thickness)
    elif char_upper == 'B':
        cv2.line(img, (cx - s // 2, cy - s), (cx - s // 2, cy + s), 255, thickness)
        cv2.ellipse(img, (cx - s // 2, cy - s // 2), (s // 2, s // 2), 0, -90, 90, 255, thickness)
        cv2.ellipse(img, (cx - s // 2, cy + s // 2), (s // 2 + 2, s // 2 + 2), 0, -90, 90, 255, thickness)
    elif char_upper == 'C':
        cv2.ellipse(img, (cx, cy), (s // 2 + 5, s), 0, 45, 315, 255, thickness)
    elif char_upper == 'D':
        cv2.line(img, (cx - s // 2, cy - s), (cx - s // 2, cy + s), 255, thickness)
        cv2.ellipse(img, (cx - s // 2, cy), (s // 2 + 5, s), 0, -90, 90, 255, thickness)
    elif char_upper == 'E':
        cv2.line(img, (cx - s // 2, cy - s), (cx - s // 2, cy + s), 255, thickness)
        cv2.line(img, (cx - s // 2, cy - s), (cx + s // 3, cy - s), 255, thickness)
        cv2.line(img, (cx - s // 2, cy), (cx + s // 4, cy), 255, thickness)
        cv2.line(img, (cx - s // 2, cy + s), (cx + s // 3, cy + s), 255, thickness)
    elif char_upper == 'M':
        cv2.line(img, (cx - s // 2, cy + s), (cx - s // 2, cy - s), 255, thickness)
        cv2.line(img, (cx - s // 2, cy - s), (cx, cy + s // 2), 255, thickness)
        cv2.line(img, (cx, cy + s // 2), (cx + s // 2, cy - s), 255, thickness)
        cv2.line(img, (cx + s // 2, cy - s), (cx + s // 2, cy + s), 255, thickness)
    elif char_upper == 'O' or char == '0':
        cv2.ellipse(img, (cx, cy), (s // 2, s), 0, 0, 360, 255, thickness)
    elif char == '1':
        cv2.line(img, (cx, cy - s), (cx, cy + s), 255, thickness)
        cv2.line(img, (cx - s // 3, cy - s // 2), (cx, cy - s), 255, thickness)
    elif char == '2':
        cv2.ellipse(img, (cx, cy - s // 2), (s // 2, s // 2), 0, 180, 360, 255, thickness)
        cv2.line(img, (cx + s // 2, cy - s // 2), (cx - s // 2, cy + s), 255, thickness)
        cv2.line(img, (cx - s // 2, cy + s), (cx + s // 2, cy + s), 255, thickness)
    elif char == '3':
        cv2.ellipse(img, (cx, cy - s // 2), (s // 2, s // 2), 0, -90, 90, 255, thickness)
        cv2.ellipse(img, (cx, cy + s // 2), (s // 2, s // 2), 0, -90, 90, 255, thickness)
    else:
        cv2.ellipse(img, (cx, cy), (s // 2, s), 0, 0, 360, 255, thickness)

    if angle_var != 0:
        M = cv2.getRotationMatrix2D((cx, cy), angle_var * 30, 1.0)
        img = cv2.warpAffine(img, M, (size, size))

    return img


def generate_plana_image_handwritten(
    char: str,
    n_repetitions: int = 6,
    img_width: int = 800,
    img_height: int = 200,
    char_size: int = 100,
    add_noise: bool = True,
    add_lines: bool = True,
) -> np.ndarray:
    """Genera una imagen de plana con caracteres manuscritos simulados."""
    img = np.ones((img_height, img_width), dtype=np.uint8) * 255

    spacing = img_width // (n_repetitions + 1)
    y_center = img_height // 2

    for i in range(n_repetitions):
        x = spacing * (i + 1)
        char_img = draw_handwritten_char(char, size=char_size)

        x_offset = random.randint(-8, 8)
        y_offset = random.randint(-10, 10)
        scale = random.uniform(0.85, 1.15)
        new_size = int(char_size * scale)
        char_img = cv2.resize(char_img, (new_size, new_size))

        x_pos = max(0, min(x - new_size // 2 + x_offset, img_width - new_size))
        y_pos = max(0, min(y_center - new_size // 2 + y_offset, img_height - new_size))

        roi = img[y_pos:y_pos + new_size, x_pos:x_pos + new_size]
        if roi.shape[0] == new_size and roi.shape[1] == new_size:
            mask = char_img > 128
            roi[mask] = 0

    if add_lines:
        cv2.line(img, (20, img_height - 35), (img_width - 20, img_height - 35), 180, 1)
        cv2.line(img, (20, 35), (img_width - 20, 35), 200, 1)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if add_noise:
        noise = np.random.normal(0, 5, img_bgr.shape).astype(np.int16)
        img_bgr = np.clip(img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img_bgr


def generate_plana_from_template(
    char: str,
    n_repetitions: int = 6,
    template_dir: str = "app/templates",
) -> Optional[np.ndarray]:
    """Genera plana usando templates .npy existentes."""

    def _safe_name(c: str) -> str:
        if c.isdigit():
            return f"digit_{c}"
        base = "N_tilde" if c.upper() in ("Ñ", "N\u0303") else c
        suffix = "upper" if c.isupper() else "lower"
        return f"{base}_{suffix}"

    base_name = _safe_name(char)
    possible_paths = [
        Path(template_dir) / "skeleton" / f"{base_name}_skeleton.npy",
        Path(template_dir) / "intermedio" / f"{base_name}_intermedio.npy",
        Path(template_dir) / f"{base_name}.npy",
    ]

    template_img = None
    for p in possible_paths:
        if p.exists():
            arr = np.load(str(p))
            template_img = (arr > 0).astype(np.uint8) * 255
            break

    if template_img is None:
        return None

    char_size = 80
    template_img = cv2.resize(template_img, (char_size, char_size))

    img_width, img_height = 800, 200
    img = np.ones((img_height, img_width), dtype=np.uint8) * 255

    spacing = img_width // (n_repetitions + 1)
    y_center = img_height // 2

    for i in range(n_repetitions):
        x = spacing * (i + 1)
        x_offset = random.randint(-8, 8)
        y_offset = random.randint(-10, 10)
        scale = random.uniform(0.9, 1.1)
        angle = random.uniform(-5, 5)

        new_size = int(char_size * scale)
        char_img = cv2.resize(template_img, (new_size, new_size))

        M = cv2.getRotationMatrix2D((new_size // 2, new_size // 2), angle, 1.0)
        char_img = cv2.warpAffine(char_img, M, (new_size, new_size))

        x_pos = max(0, min(x - new_size // 2 + x_offset, img_width - new_size))
        y_pos = max(0, min(y_center - new_size // 2 + y_offset, img_height - new_size))

        roi = img[y_pos:y_pos + new_size, x_pos:x_pos + new_size]
        if roi.shape[0] == new_size and roi.shape[1] == new_size:
            mask = char_img > 128
            roi[mask] = 0

    cv2.line(img, (20, img_height - 35), (img_width - 20, img_height - 35), 180, 1)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    noise = np.random.normal(0, 3, img_bgr.shape).astype(np.int16)
    img_bgr = np.clip(img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img_bgr


# ─────────────────────────────────────────────────────────────────────────────
# 2. DETECTOR YOLO LOCAL — Ajustado al modelo entrenado
# ─────────────────────────────────────────────────────────────────────────────

class YOLODetector:
    """
    Detector ONNX ajustado al modelo YOLOv8n entrenado.

    Formato de salida ONNX del modelo: (1, 5, 8400)
      - Transpuesto: (8400, 5)
      - Cada fila: [cx, cy, w, h, conf]
      - cx, cy, w, h en escala de píxeles del input (640)
      - 1 sola clase ("trazo"), NO hay class scores separados
      - conf es directamente la confianza del objeto
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = YOLO_CONF_THRESH,
        iou_threshold: float = YOLO_IOU_THRESH,
        img_size: int = YOLO_IMG_SIZE,
    ):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime no instalado")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Validar output shape esperado
        out_shape = self.session.get_outputs()[0].shape
        print(f"  ✅ YOLO cargado.")
        print(f"     Input : {self.input_name} {self.input_shape}")
        print(f"     Output: {out_shape}")
        print(f"     Conf  : {self.conf_threshold}, IoU: {self.iou_threshold}")
        print(f"     ImgSz : {self.img_size}")

    def _letterbox(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Letterbox resize preservando aspect ratio (como hace Ultralytics).
        Retorna: (imagen resized, ratio, (pad_w, pad_h))
        """
        h, w = img.shape[:2]
        target = self.img_size

        # Ratio para que el lado más largo quede en target
        ratio = min(target / h, target / w)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Padding para llegar a target×target
        pad_w = (target - new_w) // 2
        pad_h = (target - new_h) // 2

        canvas = np.full((target, target, 3), 114, dtype=np.uint8)  # gris 114 como YOLO
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        return canvas, ratio, (pad_w, pad_h)

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocesamiento idéntico al de Ultralytics."""
        letterboxed, ratio, (pad_w, pad_h) = self._letterbox(img_bgr)

        # BGR → RGB
        img_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

        # Normalizar [0, 1] float32
        blob = img_rgb.astype(np.float32) / 255.0

        # HWC → CHW
        blob = blob.transpose(2, 0, 1)

        # Añadir batch dimension
        blob = np.expand_dims(blob, axis=0)

        return blob, ratio, (pad_w, pad_h)

    def detect(self, img_bgr: np.ndarray) -> List[Dict]:
        """
        Detecta todos los caracteres ("trazo") en la imagen.
        Retorna lista de dicts con bboxes en coordenadas de la imagen original.
        """
        h_orig, w_orig = img_bgr.shape[:2]

        # Preprocesar
        blob, ratio, (pad_w, pad_h) = self._preprocess(img_bgr)

        # Inferencia
        outputs = self.session.run(None, {self.input_name: blob})
        preds = outputs[0]  # Shape: (1, 5, 8400)

        # Parsear
        detections = self._parse_yolov8_output(preds, ratio, pad_w, pad_h, w_orig, h_orig)

        return detections

    def _parse_yolov8_output(
        self,
        preds: np.ndarray,
        ratio: float,
        pad_w: int,
        pad_h: int,
        w_orig: int,
        h_orig: int,
    ) -> List[Dict]:
        """
        Parsea output YOLOv8 de (1, 5, 8400) para modelo de 1 clase.

        El output de YOLOv8 con 1 clase es (1, 5, 8400):
          - Dim 0: batch
          - Dim 1: [cx, cy, w, h, conf_clase_0]  (5 valores)
          - Dim 2: 8400 anchor predictions

        Nota: YOLOv8 NO tiene objectness separado.
        La fila 4 es directamente el score de la clase "trazo".
        """
        # (1, 5, 8400) → (8400, 5)
        if preds.shape[1] == 5 and preds.shape[2] == 8400:
            preds = preds[0].T  # (8400, 5)
        elif preds.shape[1] == 8400 and preds.shape[2] == 5:
            preds = preds[0]    # (8400, 5)
        else:
            print(f"  ⚠️ Output shape inesperado: {preds.shape}")
            return []

        # Columnas: cx, cy, w, h, conf
        cx   = preds[:, 0]
        cy   = preds[:, 1]
        w    = preds[:, 2]
        h    = preds[:, 3]
        conf = preds[:, 4]

        # Filtrar por confianza
        mask = conf >= self.conf_threshold
        cx   = cx[mask]
        cy   = cy[mask]
        w    = w[mask]
        h    = h[mask]
        conf = conf[mask]

        if len(conf) == 0:
            return []

        # Convertir center → corner (en escala 640 con letterbox)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Deshacer letterbox padding
        x1 = (x1 - pad_w) / ratio
        y1 = (y1 - pad_h) / ratio
        x2 = (x2 - pad_w) / ratio
        y2 = (y2 - pad_h) / ratio

        # Clamp a imagen original
        x1 = np.clip(x1, 0, w_orig).astype(int)
        y1 = np.clip(y1, 0, h_orig).astype(int)
        x2 = np.clip(x2, 0, w_orig).astype(int)
        y2 = np.clip(y2, 0, h_orig).astype(int)

        # Construir lista de detecciones
        detections = []
        for i in range(len(conf)):
            if x2[i] - x1[i] < 5 or y2[i] - y1[i] < 5:
                continue
            detections.append({
                "x1": int(x1[i]),
                "y1": int(y1[i]),
                "x2": int(x2[i]),
                "y2": int(y2[i]),
                "confidence": float(conf[i]),
            })

        # NMS
        detections = self._nms(detections, self.iou_threshold)

        # Reading-order sort (igual que detect_characters del entrenamiento)
        detections = self._reading_order_sort(detections)

        return detections

    def _nms(self, dets: List[Dict], iou_thresh: float) -> List[Dict]:
        """Non-Maximum Suppression."""
        if not dets:
            return []

        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        keep = []

        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [d for d in dets if self._iou(best, d) < iou_thresh]

        return keep

    def _iou(self, a: Dict, b: Dict) -> float:
        """Calcula IoU entre dos bboxes."""
        ix1 = max(a["x1"], b["x1"])
        iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"])
        iy2 = min(a["y2"], b["y2"])

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def _reading_order_sort(
        self, detections: List[Dict], line_tolerance: float = 0.5
    ) -> List[Dict]:
        """
        Ordena detecciones en reading order (izq→der, arriba→abajo).
        Idéntico al algoritmo de detect_characters() del entrenamiento.
        """
        if not detections:
            return []

        # Calcular median height
        heights = [d["y2"] - d["y1"] for d in detections]
        median_h = sorted(heights)[len(heights) // 2]
        tol = line_tolerance * median_h

        # Ordenar por centro Y
        detections.sort(key=lambda d: (d["y1"] + d["y2"]) / 2)

        # Agrupar en líneas
        lines = []
        current_line = [detections[0]]
        current_y = (detections[0]["y1"] + detections[0]["y2"]) / 2

        for d in detections[1:]:
            y_center = (d["y1"] + d["y2"]) / 2
            if abs(y_center - current_y) <= tol:
                current_line.append(d)
            else:
                lines.append(current_line)
                current_line = [d]
                current_y = y_center
        lines.append(current_line)

        # Dentro de cada línea, ordenar por X
        ordered = []
        for line_idx, line in enumerate(lines):
            line.sort(key=lambda d: d["x1"])
            for d in line:
                d["line"] = line_idx
                ordered.append(d)

        return ordered


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def visualize_detections(
    img_bgr: np.ndarray, detections: List[Dict], char: str
) -> np.ndarray:
    """Dibuja bounding boxes con info de línea y reading order."""
    vis = img_bgr.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]
        line_idx = det.get("line", 0)

        # Color por línea
        line_colors = [
            (0, 255, 0),    # verde
            (255, 0, 0),    # azul
            (0, 165, 255),  # naranja
            (255, 0, 255),  # magenta
            (0, 255, 255),  # amarillo
        ]
        color = line_colors[line_idx % len(line_colors)]

        # El primero (plantilla) tiene borde más grueso
        thickness = 3 if i == 0 else 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Label con índice de reading order y confianza
        label = f"#{i} L{line_idx} {conf:.0%}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            vis, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )

    # Info general
    n_lines = max((d.get("line", 0) for d in detections), default=0) + 1 if detections else 0
    info = f"'{char}' - Detectados: {len(detections)}, Lineas: {n_lines}"
    cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

    return vis


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLIENTE API
# ─────────────────────────────────────────────────────────────────────────────

def call_evaluate_plana(api_url: str, img_bgr: np.ndarray, level: str = "intermedio") -> dict:
    """Llama al endpoint /evaluate_plana."""
    endpoint = f"{api_url.rstrip('/')}/evaluate_plana"

    _, buffer = cv2.imencode(".png", img_bgr)

    files = {"file": ("plana.png", buffer.tobytes(), "image/png")}
    data = {"level": level}

    try:
        response = requests.post(endpoint, files=files, data=data, timeout=120)

        if response.status_code == 422:
            try:
                detail = response.json().get("detail", "Error 422")
            except Exception:
                detail = response.text
            return {"error": f"422: {detail}"}

        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        return {"error": "Timeout"}
    except requests.exceptions.ConnectionError as e:
        return {"error": f"Conexión: {e}"}
    except Exception as e:
        return {"error": str(e)}


def print_results(result: dict, char: str) -> bool:
    """Imprime resultados formateados."""
    print("\n" + "=" * 60)
    print(f"📊 RESULTADOS - PLANA '{char}'")
    print("=" * 60)

    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
        return False

    print(f"📋 Plantilla: '{result.get('template_char', '?')}' "
          f"({result.get('template_confidence', 0):.0%})")
    print(f"📦 Detectados: {result.get('n_detected', 0)} | "
          f"Evaluados: {result.get('n_evaluated', 0)}")
    print(f"📈 PROMEDIO: {result.get('avg_score', 0):.1f}%")

    for r in result.get("results", []):
        score = r.get("score_final", 0)
        emoji = (
            "🌟" if score >= 85
            else ("✅" if score >= 70
                  else ("⚠️" if score >= 50 else "❌"))
        )
        feedback = r.get("feedback", "")[:50]
        print(f"   #{r.get('index')}: {emoji} {score:.1f}% - {feedback}...")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test /evaluate_plana")
    parser.add_argument("--url", required=True, help="URL de la API")
    parser.add_argument("--chars", default="AB", help="Caracteres a probar")
    parser.add_argument("--n", type=int, default=5, help="Repeticiones por plana")
    parser.add_argument("--level", default="intermedio")
    parser.add_argument("--debug-detector", action="store_true", help="Debug YOLO local")
    parser.add_argument("--template-dir", default="app/templates", help="Dir de templates .npy")
    parser.add_argument("--use-templates", action="store_true", help="Usar templates .npy")
    parser.add_argument("--model-path", default=YOLO_MODEL_PATH, help="Path al .onnx")
    parser.add_argument("--conf", type=float, default=YOLO_CONF_THRESH, help="Umbral de confianza")
    parser.add_argument("--iou", type=float, default=YOLO_IOU_THRESH, help="Umbral IoU para NMS")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🧪 TEST /evaluate_plana")
    print("=" * 60)
    print(f"URL          : {args.url}")
    print(f"Caracteres   : {args.chars}")
    print(f"Repeticiones : {args.n}")
    print(f"Nivel        : {args.level}")
    print(f"Modelo       : {args.model_path}")
    print(f"Conf thresh  : {args.conf}")
    print(f"IoU thresh   : {args.iou}")

    # Cargar detector para debug
    detector = None
    if args.debug_detector and ONNX_AVAILABLE:
        try:
            print("\n🔧 Cargando detector YOLO local...")
            detector = YOLODetector(
                model_path=args.model_path,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
            )
        except Exception as e:
            print(f"⚠️ No se pudo cargar detector: {e}")

    # Procesar cada carácter
    all_ok = True
    for char in args.chars:
        print(f"\n{'─' * 50}")
        print(f"🔤 Generando plana para '{char}'...")

        # Generar imagen
        if args.use_templates:
            img = generate_plana_from_template(char, args.n, args.template_dir)
            if img is None:
                print(f"   ⚠️ No hay template para '{char}', usando manuscrito sintético")
                img = generate_plana_image_handwritten(char, args.n)
        else:
            img = generate_plana_image_handwritten(char, args.n)

        # Guardar imagen generada
        img_path = OUTPUT_DIR / f"{char}_plana.png"
        cv2.imwrite(str(img_path), img)
        print(f"   💾 Guardada: {img_path}")
        print(f"   📐 Tamaño: {img.shape[1]}×{img.shape[0]} px")

        # Debug detector local
        if detector:
            print("   🔍 Probando detector local...")
            t0 = time.time()
            detections = detector.detect(img)
            dt = (time.time() - t0) * 1000
            print(f"   📦 Detectados: {len(detections)} ({dt:.1f} ms)")

            if detections:
                for i, d in enumerate(detections):
                    print(f"      #{i}: bbox=({d['x1']},{d['y1']},{d['x2']},{d['y2']}) "
                          f"conf={d['confidence']:.4f} line={d.get('line', '?')}")

                vis = visualize_detections(img, detections, char)
                vis_path = OUTPUT_DIR / f"{char}_detections.png"
                cv2.imwrite(str(vis_path), vis)
                print(f"   💾 Debug: {vis_path}")
            else:
                print("   ⚠️ El detector NO encontró caracteres en la imagen")
                print("   💡 Posible causa: las planas sintéticas son muy diferentes")
                print("      al estilo de crops usados en el entrenamiento.")
                print("      El modelo fue entrenado con compose_image() usando crops reales.")

        # Llamar endpoint
        print(f"   🌐 Llamando {args.url}/evaluate_plana...")
        t0 = time.time()
        result = call_evaluate_plana(args.url, img, args.level)
        elapsed = time.time() - t0
        print(f"   ⏱️ Tiempo: {elapsed:.2f}s")

        success = print_results(result, char)
        if not success:
            all_ok = False

    print(f"\n📁 Imágenes en: {OUTPUT_DIR.absolute()}")
    if all_ok:
        print("✅ Todos los tests completados exitosamente\n")
    else:
        print("⚠️ Algunos tests fallaron — revisa los errores arriba\n")


if __name__ == "__main__":
    main()