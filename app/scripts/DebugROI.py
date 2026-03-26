"""
debug_and_refine_roi.py
========================
Script de validación y refinamiento de ROI para el detector YOLO de caracteres.

Dos modos de uso:
  1. Modo DEBUG: dibuja las bounding boxes sobre las imágenes originales y las
     guarda en una carpeta 'debug/' para inspección visual.
  2. Modo REFINE: toma la detección de YOLO, le aplica un margen (padding),
     luego dentro del recorte busca el contorno real del carácter con Canny/Sobel
     y ajusta el bbox exactamente a ese contorno.

Uso rápido:
  python debug_and_refine_roi.py --mode debug  --source ruta/a/imagenes --weights ruta/al/modelo.onnx
  python debug_and_refine_roi.py --mode refine --source ruta/a/imagen.jpg --weights ruta/al/modelo.onnx
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────
# Parámetros globales ajustables
# ──────────────────────────────────────────────
CONF_THRESHOLD   = 0.25   # Confianza mínima YOLO (baja si recorta letras)
IOU_THRESHOLD    = 0.45   # NMS IoU
PADDING_PX       = 12     # Margen alrededor de la caja YOLO (en píxeles)
MIN_CONTOUR_FILL = 0.30   # El contorno debe cubrir al menos este % del área del recorte
CANNY_LOW        = 30     # Umbral bajo Canny (baja en imágenes con poco contraste)
CANNY_HIGH       = 120    # Umbral alto Canny
BLUR_KSIZE       = 5      # Tamaño del kernel de suavizado antes de Canny
OUTPUT_SIZE      = (64, 64)  # Tamaño final normalizado del carácter extraído


# ──────────────────────────────────────────────
# Utilidades de imagen
# ──────────────────────────────────────────────

def load_model(weights_path: str) -> YOLO:
    """Carga el modelo YOLO (pt u onnx)."""
    model = YOLO(weights_path)
    return model


def preprocess_for_contour(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento adaptativo pensado para libretas (cuadriculadas, rayadas,
    blancas) y condiciones de iluminación variadas.

    Pipeline:
      1. Escala de grises
      2. CLAHE  → mejora contraste local (útil en iluminación desigual)
      3. GaussianBlur → reduce ruido antes de Canny
      4. Canny adaptativo
      5. Dilatación ligera para cerrar trazos discontinuos
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE para normalizar contraste (tolera fondos de libreta)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Suavizado
    blurred = cv2.GaussianBlur(enhanced, (BLUR_KSIZE, BLUR_KSIZE), 0)

    # Canny
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # Dilatar para conectar trazos fragmentados
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def find_character_contour(edges: np.ndarray, min_fill: float = MIN_CONTOUR_FILL):
    """
    Busca el contorno más grande que cubre al menos `min_fill` del área del recorte.
    Devuelve el bounding rect (x, y, w, h) relativo al recorte, o None si no encuentra.
    """
    roi_area = edges.shape[0] * edges.shape[1]
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Ordenar por área descendente
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area / roi_area >= min_fill:
            x, y, w, h = cv2.boundingRect(cnt)
            return x, y, w, h

    # Si ninguno supera el umbral, devolver el más grande de todos
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h


def refine_bbox(image_bgr: np.ndarray, yolo_box, padding: int = PADDING_PX):
    """
    Toma la caja YOLO, añade margen, extrae el ROI, detecta el contorno
    real del carácter y devuelve:
      - roi_raw:      recorte con padding (sin refinar)
      - roi_refined:  recorte ajustado al contorno exacto
      - char_norm:    carácter binarizado y normalizado a OUTPUT_SIZE
      - refined_abs:  bbox absoluta (x1,y1,x2,y2) en la imagen original
    """
    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in yolo_box]

    # Añadir padding sin salirse de la imagen
    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(W, x2 + padding)
    py2 = min(H, y2 + padding)

    roi_raw = image_bgr[py1:py2, px1:px2].copy()

    if roi_raw.size == 0:
        return None, None, None, None

    # Detectar contorno dentro del ROI
    edges = preprocess_for_contour(roi_raw)
    result = find_character_contour(edges)

    if result is None:
        roi_refined = roi_raw
        refined_abs = (px1, py1, px2, py2)
    else:
        rx, ry, rw, rh = result
        # Convertir a coordenadas absolutas
        ax1 = px1 + rx
        ay1 = py1 + ry
        ax2 = ax1 + rw
        ay2 = ay1 + rh
        # Agregar un margen mínimo al contorno también
        m = 4
        ax1 = max(0, ax1 - m)
        ay1 = max(0, ay1 - m)
        ax2 = min(W, ax2 + m)
        ay2 = min(H, ay2 + m)
        roi_refined = image_bgr[ay1:ay2, ax1:ax2].copy()
        refined_abs = (ax1, ay1, ax2, ay2)

    # Normalizar: binarización + resize
    gray = cv2.cvtColor(roi_refined, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    char_norm = cv2.resize(binary, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

    return roi_raw, roi_refined, char_norm, refined_abs


# ──────────────────────────────────────────────
# Modo DEBUG
# ──────────────────────────────────────────────

def run_debug(source: str, weights: str, output_dir: str = "debug"):
    """
    Procesa todas las imágenes en `source` (carpeta o archivo único),
    dibuja las cajas YOLO y las guarda en `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(weights)

    source_path = Path(source)
    if source_path.is_file():
        image_paths = [source_path]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [p for p in source_path.rglob("*") if p.suffix.lower() in exts]

    print(f"🔍 Procesando {len(image_paths)} imagen(es) en modo DEBUG...")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠ No se pudo leer: {img_path}")
            continue

        results = model.predict(
            img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        debug_img = img.copy()
        detections = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                label = f"{r.names[cls]} {conf:.2f}"

                # Caja YOLO en rojo
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(debug_img, label, (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

                # Refinar y dibujar contorno real en verde
                _, _, _, refined_abs = refine_bbox(img, (x1, y1, x2, y2))
                if refined_abs:
                    rx1, ry1, rx2, ry2 = refined_abs
                    cv2.rectangle(debug_img, (rx1, ry1), (rx2, ry2), (0, 220, 0), 2)

                detections += 1

        out_name = Path(output_dir) / f"debug_{img_path.name}"
        cv2.imwrite(str(out_name), debug_img)
        print(f"  ✅ {img_path.name} → {detections} detección(es) → {out_name}")

    print(f"\n📁 Imágenes de debug guardadas en: {os.path.abspath(output_dir)}")
    print("💡 Rojo = caja YOLO | Verde = contorno refinado del carácter")


# ──────────────────────────────────────────────
# Modo REFINE (extracción limpia del carácter)
# ──────────────────────────────────────────────

def run_refine(source: str, weights: str, output_dir: str = "refined"):
    """
    Para cada imagen en `source`, extrae cada carácter detectado,
    lo refina con el contorno real y guarda:
      - El carácter binarizado normalizado (listo para comparar con plantilla)
      - Una visualización con ambas cajas superpuestas
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "chars"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "viz"), exist_ok=True)

    model = load_model(weights)

    source_path = Path(source)
    if source_path.is_file():
        image_paths = [source_path]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [p for p in source_path.rglob("*") if p.suffix.lower() in exts]

    print(f"🔬 Extrayendo caracteres de {len(image_paths)} imagen(es)...")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(
            img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        viz_img = img.copy()
        char_idx = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])

                roi_raw, roi_refined, char_norm, refined_abs = refine_bbox(
                    img, (x1, y1, x2, y2)
                )

                if char_norm is None:
                    continue

                stem = img_path.stem
                # Guardar carácter normalizado (para comparar con plantilla)
                char_out = Path(output_dir) / "chars" / f"{stem}_char{char_idx:02d}.png"
                cv2.imwrite(str(char_out), char_norm)

                # Visualización
                if refined_abs:
                    rx1, ry1, rx2, ry2 = refined_abs
                    cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)     # YOLO
                    cv2.rectangle(viz_img, (rx1, ry1), (rx2, ry2), (0, 220, 0), 2) # Refinado
                    cv2.putText(viz_img, f"conf:{conf:.2f}", (x1, max(y1-6,0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

                print(f"  ✅ Carácter guardado: {char_out}")
                char_idx += 1

        viz_out = Path(output_dir) / "viz" / f"viz_{img_path.name}"
        cv2.imwrite(str(viz_out), viz_img)

    print(f"\n📁 Resultados en: {os.path.abspath(output_dir)}")
    print("   chars/ → caracteres binarizados 64×64 (para comparar con plantilla)")
    print("   viz/   → visualizaciones con cajas YOLO (rojo) y contorno real (verde)")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validación de BBoxes YOLO y refinamiento de ROI para caracteres"
    )
    parser.add_argument(
        "--mode", choices=["debug", "refine"], default="debug",
        help="debug: solo dibuja cajas | refine: extrae y normaliza el carácter"
    )
    parser.add_argument(
        "--source", required=True,
        help="Ruta a imagen o carpeta de imágenes"
    )
    parser.add_argument(
        "--weights", required=True,
        help="Ruta al modelo YOLO (.pt o .onnx)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Carpeta de salida (default: 'debug' o 'refined' según el modo)"
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help=f"Umbral de confianza YOLO (default: {CONF_THRESHOLD})"
    )
    parser.add_argument(
        "--padding", type=int, default=PADDING_PX,
        help=f"Margen alrededor de la caja YOLO en px (default: {PADDING_PX})"
    )

    args = parser.parse_args()

    # Sobreescribir parámetros globales si se pasaron por CLI
    CONF_THRESHOLD = args.conf
    PADDING_PX     = args.padding

    if args.mode == "debug":
        out = args.output or "debug"
        run_debug(args.source, args.weights, out)
    else:
        out = args.output or "refined"
        run_refine(args.source, args.weights, out)