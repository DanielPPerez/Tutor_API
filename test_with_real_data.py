#!/usr/bin/env python3
"""
test_with_real_data.py — Test suite v5 compatible
Cambios vs versión anterior:
  1. TTA con 5 transforms (matching training)
  2. Elastic warp en TTA
  3. Mejor reporte por tipo de error
"""
import os, sys, json, time, math, random
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
MODEL_PATH     = "app/models/classifier_artifacts/best_classifier.onnx"
CLASS_MAP_PATH = "app/models/classifier_artifacts/char_map.json"
IMG_SIZE       = 128

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

def letterbox_resize(img_bgr, size=IMG_SIZE):
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def preprocess(img_bgr):
    if img_bgr is None:
        raise ValueError("Imagen es None")
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    img = letterbox_resize(img_bgr, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0).astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# =============================================================================
# CHAR MAP
# =============================================================================

def load_class_map(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    idx2char = {int(k): v for k, v in data["idx2char"].items()}
    num_classes = data.get("num_classes", len(idx2char))
    for i in range(num_classes):
        if i not in idx2char:
            idx2char[i] = f"UNK_{i}"
    return idx2char, num_classes


def build_char2idx(idx2char):
    return {v: k for k, v in idx2char.items()}


# =============================================================================
# GENERACIÓN DE IMÁGENES — Idéntica al Cell 8 del entrenamiento v5
# =============================================================================

def find_system_fonts():
    search_paths = [
        '/usr/share/fonts', '/usr/local/share/fonts',
        'C:/Windows/Fonts',
        '/System/Library/Fonts', '/Library/Fonts',
        os.path.expanduser('~/Library/Fonts'),
    ]
    import glob
    found = []
    for p in search_paths:
        if os.path.exists(p):
            found += glob.glob(f'{p}/**/*.ttf', recursive=True)
            found += glob.glob(f'{p}/**/*.otf', recursive=True)
    usable = []
    for fp in found[:50]:
        try:
            ImageFont.truetype(fp, 60)
            usable.append(fp)
        except Exception:
            pass
    return usable


USABLE_FONTS = find_system_fonts()


def make_synthetic_image(char, img_size=IMG_SIZE):
    """Genera imagen sintética — MATCHING Cell 8 v5 del entrenamiento."""
    img = Image.new('L', (img_size, img_size), color=255)
    draw = ImageDraw.Draw(img)

    font_size = random.randint(50, 85)
    font = None
    if USABLE_FONTS:
        try:
            font = ImageFont.truetype(random.choice(USABLE_FONTS), font_size)
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img_size - tw) // 2 - bbox[0] + random.randint(-8, 8)
    y = (img_size - th) // 2 - bbox[1] + random.randint(-8, 8)
    draw.text((x, y), char, fill=random.randint(0, 80), font=font)

    arr = np.array(img)

    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((img_size // 2, img_size // 2), angle, 1.0)
    arr = cv2.warpAffine(arr, M, (img_size, img_size),
                          borderValue=255, flags=cv2.INTER_LINEAR)

    if random.random() < 0.5:
        k = random.choice([2, 3])
        kernel = np.ones((k, k), np.uint8)
        arr = cv2.erode(arr, kernel, 1) if random.random() < 0.5 \
              else cv2.dilate(arr, kernel, 1)

    if random.random() < 0.4:
        noise = np.random.randint(-25, 25, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-20, 20)
        arr = np.clip(arr.astype(float) * alpha + beta, 0, 255).astype(np.uint8)

    # NUEVO v5: Elastic warp (matching Cell 8 del entrenamiento)
    if random.random() < 0.3:
        rows, cols = arr.shape
        dx = (np.random.rand(rows, cols).astype(np.float32) - 0.5) * 6
        dy = (np.random.rand(rows, cols).astype(np.float32) - 0.5) * 6
        x_map, y_map = np.meshgrid(np.arange(cols), np.arange(rows))
        map_x = (x_map + dx).astype(np.float32)
        map_y = (y_map + dy).astype(np.float32)
        arr = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)

    return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)


def make_stroke_image(stroke_type, img_size=IMG_SIZE):
    """Genera trazos — idéntico al Cell 8."""
    canvas = np.full((img_size, img_size), 255, dtype=np.uint8)
    thickness = random.randint(2, 5)
    color = random.randint(0, 60)
    margin = random.randint(15, 30)
    center = img_size // 2
    jx = random.randint(-10, 10)
    jy = random.randint(-10, 10)

    if stroke_type == 'línea_vertical':
        x1 = center + jx + random.randint(-3, 3)
        x2 = center + jx + random.randint(-3, 3)
        y1 = margin + random.randint(-5, 5)
        y2 = img_size - margin + random.randint(-5, 5)
        pts = []
        n = random.randint(5, 10)
        for i in range(n + 1):
            t = i / n
            px = int(x1 + (x2 - x1) * t + random.randint(-2, 2))
            py = int(y1 + (y2 - y1) * t)
            pts.append([px, py])
        cv2.polylines(canvas, [np.array(pts, np.int32)], False,
                       color, thickness, cv2.LINE_AA)

    elif stroke_type == 'línea_horizontal':
        y1 = center + jy + random.randint(-3, 3)
        y2 = center + jy + random.randint(-3, 3)
        x1 = margin + random.randint(-5, 5)
        x2 = img_size - margin + random.randint(-5, 5)
        pts = []
        n = random.randint(5, 10)
        for i in range(n + 1):
            t = i / n
            px = int(x1 + (x2 - x1) * t)
            py = int(y1 + (y2 - y1) * t + random.randint(-2, 2))
            pts.append([px, py])
        cv2.polylines(canvas, [np.array(pts, np.int32)], False,
                       color, thickness, cv2.LINE_AA)

    elif stroke_type == 'línea_oblicua_derecha':
        x1, y1 = margin + jx, margin + jy
        x2, y2 = img_size - margin + jx, img_size - margin + jy
        pts = []
        n = random.randint(5, 10)
        for i in range(n + 1):
            t = i / n
            px = int(x1 + (x2 - x1) * t + random.randint(-2, 2))
            py = int(y1 + (y2 - y1) * t + random.randint(-2, 2))
            pts.append([px, py])
        cv2.polylines(canvas, [np.array(pts, np.int32)], False,
                       color, thickness, cv2.LINE_AA)

    elif stroke_type == 'línea_oblicua_izquierda':
        x1, y1 = img_size - margin + jx, margin + jy
        x2, y2 = margin + jx, img_size - margin + jy
        pts = []
        n = random.randint(5, 10)
        for i in range(n + 1):
            t = i / n
            px = int(x1 + (x2 - x1) * t + random.randint(-2, 2))
            py = int(y1 + (y2 - y1) * t + random.randint(-2, 2))
            pts.append([px, py])
        cv2.polylines(canvas, [np.array(pts, np.int32)], False,
                       color, thickness, cv2.LINE_AA)

    elif stroke_type == 'curva':
        curve_type = random.choice(['S', 'C', 'U', 'wave'])
        pts = []
        n = 20
        if curve_type == 'S':
            for i in range(n + 1):
                t = i / n
                px = int(center + 35 * math.sin(t * math.pi * 2) + jx)
                py = int(margin + (img_size - 2 * margin) * t + jy)
                pts.append([px, py])
        elif curve_type == 'C':
            for i in range(n + 1):
                t = i / n
                a = -math.pi / 2 + math.pi * t
                px = int(center + 35 * math.cos(a) + jx)
                py = int(center + 35 * math.sin(a) + jy)
                pts.append([px, py])
        elif curve_type == 'U':
            for i in range(n + 1):
                t = i / n
                a = math.pi * t
                px = int(center + 35 * math.cos(a) + jx)
                py = int(center + 30 * math.sin(a) + jy)
                pts.append([px, py])
        else:
            for i in range(n + 1):
                t = i / n
                px = int(margin + (img_size - 2 * margin) * t)
                py = int(center + 25 * math.sin(t * math.pi * 3) + jy)
                pts.append([px, py])
        for p in pts:
            p[0] += random.randint(-2, 2)
            p[1] += random.randint(-2, 2)
        cv2.polylines(canvas, [np.array(pts, np.int32)], False,
                       color, thickness, cv2.LINE_AA)

    elif stroke_type == 'círculo':
        rx = random.randint(25, 45)
        ry = random.randint(25, 45)
        cx, cy = center + jx, center + jy
        angle = random.uniform(-15, 15)
        pts = []
        n = 30
        gap = random.uniform(0, 0.3)
        for i in range(n + 1):
            t = i / n * (2 * math.pi - gap)
            px = int(cx + rx * math.cos(t + math.radians(angle))
                     + random.randint(-2, 2))
            py = int(cy + ry * math.sin(t + math.radians(angle))
                     + random.randint(-2, 2))
            pts.append([px, py])
        cv2.polylines(canvas, [np.array(pts, np.int32)], False,
                       color, thickness, cv2.LINE_AA)

    # Post-noise
    angle = random.uniform(-12, 12)
    h, w = canvas.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    canvas = cv2.warpAffine(canvas, M, (w, h), borderValue=255)
    if random.random() < 0.5:
        k = random.choice([2, 3])
        kernel = np.ones((k, k), np.uint8)
        canvas = cv2.erode(canvas, kernel, 1) if random.random() < 0.5 \
                 else cv2.dilate(canvas, kernel, 1)
    if random.random() < 0.5:
        noise = np.random.randint(-20, 20, canvas.shape, dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def generate_test_image(char, idx2char, img_size=IMG_SIZE):
    stroke_names = {
        'línea_vertical', 'línea_horizontal',
        'línea_oblicua_derecha', 'línea_oblicua_izquierda',
        'curva', 'círculo',
    }
    if char in stroke_names:
        return make_stroke_image(char, img_size)
    else:
        return make_synthetic_image(char, img_size)


# =============================================================================
# TTA — ACTUALIZADO para v5 (5 transforms)
# =============================================================================

def _augment_rotate(img_bgr):
    h, w = img_bgr.shape[:2]
    angle = random.uniform(-7, 7)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), borderValue=(255, 255, 255))


def _augment_brightness(img_bgr):
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-15, 15)
    return np.clip(img_bgr.astype(float) * alpha + beta, 0, 255).astype(np.uint8)


def _augment_blur(img_bgr):
    return cv2.GaussianBlur(img_bgr, (3, 3), 0)


def _augment_elastic(img_bgr):
    """NUEVO v5: Elastic warp leve — matching TTA transform #5."""
    h, w = img_bgr.shape[:2]
    # Trabajar en cada canal
    result = np.zeros_like(img_bgr)
    dx = (np.random.rand(h, w).astype(np.float32) - 0.5) * 4
    dy = (np.random.rand(h, w).astype(np.float32) - 0.5) * 4
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_map + dx).astype(np.float32)
    map_y = (y_map + dy).astype(np.float32)
    for c in range(3):
        result[:, :, c] = cv2.remap(
            img_bgr[:, :, c], map_x, map_y,
            cv2.INTER_LINEAR, borderValue=255
        )
    return result


def predict_with_tta(sess, img_bgr, input_name, output_name, num_classes):
    """TTA v5: 5 transforms (matching training)."""
    aug_fns = [
        lambda img: img,                    # TTA 0: clean
        lambda img: _augment_rotate(img),   # TTA 1: rotation
        lambda img: _augment_blur(img),     # TTA 2: blur
        lambda img: _augment_brightness(img), # TTA 3: brightness
        lambda img: _augment_elastic(img),  # TTA 4: elastic (NUEVO v5)
    ]

    avg_probs = np.zeros(num_classes, dtype=np.float32)

    for fn in aug_fns:
        augmented = fn(img_bgr.copy())
        tensor = preprocess(augmented)
        logits = sess.run([output_name], {input_name: tensor})[0][0]
        probs = softmax(logits)
        avg_probs += probs

    avg_probs /= len(aug_fns)
    return avg_probs


# =============================================================================
# PREDICCIÓN
# =============================================================================

def predict_single(sess, img_bgr, input_name, output_name,
                   idx2char, num_classes, use_tta=False):
    if use_tta:
        probs = predict_with_tta(sess, img_bgr, input_name,
                                  output_name, num_classes)
    else:
        tensor = preprocess(img_bgr)
        logits = sess.run([output_name], {input_name: tensor})[0][0]
        probs = softmax(logits)

    top5_idx = np.argsort(probs)[-5:][::-1]
    pred_idx = top5_idx[0]

    return {
        'pred_char': idx2char[pred_idx],
        'pred_idx': int(pred_idx),
        'confidence': float(probs[pred_idx]),
        'top5': [(idx2char[i], float(probs[i])) for i in top5_idx],
        'probs': probs,
    }


# =============================================================================
# TESTS
# =============================================================================

def classify_error_type(gt_char, pred_char):
    """NUEVO v5: Clasifica el tipo de error."""
    accent_bases = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ü': 'u', 'ñ': 'n',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'Ü': 'U', 'Ñ': 'N',
    }
    # base → accent
    if pred_char in accent_bases and gt_char == accent_bases[pred_char]:
        return 'base→accent'
    # accent → base
    if gt_char in accent_bases and pred_char == accent_bases[gt_char]:
        return 'accent→base'
    # case
    if gt_char.isalpha() and pred_char.isalpha():
        if gt_char.lower() == pred_char.lower() and gt_char != pred_char:
            return 'case'
    # shape
    shape_groups = [{'1', 'l', 'I'}, {'0', 'O', 'o'}, {'5', 'S', 's'}]
    for group in shape_groups:
        if gt_char in group and pred_char in group:
            return 'shape'
    return 'other'


def run_test_group(sess, input_name, output_name, idx2char, char2idx,
                   num_classes, group_name, chars, trials_per_char=5,
                   use_tta=False):
    print(f"\n{'─' * 65}")
    print(f"  {group_name}")
    print(f"{'─' * 65}")

    correct, total = 0, 0
    failures = []

    for char in chars:
        if char not in char2idx:
            continue

        votes = Counter()
        confidences = []

        for _ in range(trials_per_char):
            img = generate_test_image(char, idx2char)
            result = predict_single(sess, img, input_name, output_name,
                                     idx2char, num_classes, use_tta=use_tta)
            votes[result['pred_char']] += 1
            confidences.append(result['confidence'])

        best_pred, best_count = votes.most_common(1)[0]
        avg_conf = np.mean(confidences)
        hit_rate = votes.get(char, 0) / trials_per_char

        is_correct = (best_pred == char)
        if is_correct:
            correct += 1
        else:
            err_type = classify_error_type(char, best_pred)
            failures.append((char, best_pred, hit_rate, avg_conf, err_type))
        total += 1

        mark = "✅" if is_correct else "❌"
        top3_votes = votes.most_common(3)
        votes_str = ", ".join(f"'{c}'×{n}" for c, n in top3_votes)
        print(f"  {mark} '{char}' → best='{best_pred}' "
              f"({best_count}/{trials_per_char}) "
              f"avg_conf={avg_conf:.3f} | votes: {votes_str}")

    acc = correct / max(total, 1)
    status = "✅" if acc >= 0.7 else "⚠️ " if acc >= 0.5 else "❌"
    print(f"\n  {status} {group_name}: {correct}/{total} = {acc:.1%}")

    if failures:
        print(f"\n  Errores ({len(failures)}):")
        for gt, pred, hr, conf, etype in sorted(failures, key=lambda x: x[2]):
            print(f"    '{gt}' → '{pred}' "
                  f"(hit={hr:.0%}, conf={conf:.3f}, type={etype})")

    return correct, total, failures


def run_all_tests(sess, input_name, output_name, idx2char, char2idx,
                  num_classes, trials_per_char=5, use_tta=False):
    test_groups = {
        'Minúsculas (a-z)': list('abcdefghijklmnopqrstuvwxyz'),
        'Mayúsculas (A-Z)': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        'Dígitos (0-9)':    list('0123456789'),
        'Acentuadas':       list('áéíóúüñÁÉÍÓÚÜÑ'),
        'Puntuación':       list('.,;:¿?¡!()-_\'"/@#$%&*+=<>'),
        'Trazos básicos':   [
            'línea_vertical', 'línea_horizontal',
            'línea_oblicua_derecha', 'línea_oblicua_izquierda',
            'curva', 'círculo',
        ],
    }

    total_correct, total_all = 0, 0
    all_failures = []
    group_results = {}

    for group_name, chars in test_groups.items():
        c, t, fails = run_test_group(
            sess, input_name, output_name, idx2char, char2idx,
            num_classes, group_name, chars,
            trials_per_char=trials_per_char, use_tta=use_tta,
        )
        total_correct += c
        total_all += t
        all_failures += fails
        group_results[group_name] = (c, t)

    return total_correct, total_all, all_failures, group_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  OCR CHARACTER CLASSIFIER — TEST SUITE v5")
    print("=" * 65)

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Modelo no encontrado: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(CLASS_MAP_PATH):
        print(f"\n❌ char_map no encontrado: {CLASS_MAP_PATH}")
        sys.exit(1)

    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    input_shape = sess.get_inputs()[0].shape
    print(f"\n  Modelo:     {MODEL_PATH}")
    print(f"  Input:      {input_name} {input_shape}")
    print(f"  Output:     {output_name} {sess.get_outputs()[0].shape}")
    print(f"  Provider:   {sess.get_providers()[0]}")

    idx2char, num_classes = load_class_map(CLASS_MAP_PATH)
    char2idx = build_char2idx(idx2char)
    print(f"  Clases:     {num_classes}")
    print(f"  Fuentes:    {len(USABLE_FONTS)} disponibles")

    # Verificar escala
    print(f"\n  Verificando escala del modelo...")
    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    dummy_out = sess.run([output_name], {input_name: dummy})[0][0]
    max_logit = np.abs(dummy_out).max()
    print(f"  Max |logit| con input aleatorio: {max_logit:.1f}")
    if max_logit > 5.0:
        print(f"  ✅ Escala ArcFace incluida")
    else:
        print(f"  ⚠️ Logits bajos")

    # ═══ TEST 1: Sin TTA ═══
    print(f"\n{'═' * 65}")
    print(f"  TEST 1: SIN TTA (7 muestras/carácter, majority vote)")
    print(f"{'═' * 65}")

    t0 = time.time()
    c1, t1, f1, g1 = run_all_tests(
        sess, input_name, output_name, idx2char, char2idx,
        num_classes, trials_per_char=7, use_tta=False,
    )
    elapsed1 = time.time() - t0

    print(f"\n{'═' * 65}")
    print(f"  RESULTADO SIN TTA: {c1}/{t1} = {c1/max(t1,1):.1%} "
          f"({elapsed1:.1f}s)")
    print(f"{'═' * 65}")

    # ═══ TEST 2: Con TTA ═══
    print(f"\n{'═' * 65}")
    print(f"  TEST 2: CON TTA ×5 (5 muestras/carácter, majority vote)")
    print(f"{'═' * 65}")

    t0 = time.time()
    c2, t2, f2, g2 = run_all_tests(
        sess, input_name, output_name, idx2char, char2idx,
        num_classes, trials_per_char=5, use_tta=True,
    )
    elapsed2 = time.time() - t0

    print(f"\n{'═' * 65}")
    print(f"  RESULTADO CON TTA: {c2}/{t2} = {c2/max(t2,1):.1%} "
          f"({elapsed2:.1f}s)")
    print(f"{'═' * 65}")

    # ═══ RESUMEN ═══
    print(f"\n{'═' * 65}")
    print(f"  RESUMEN FINAL")
    print(f"{'═' * 65}")
    print(f"  Sin TTA: {c1}/{t1} = {c1/max(t1,1):.1%}")
    print(f"  Con TTA: {c2}/{t2} = {c2/max(t2,1):.1%}")
    print(f"  Tiempo:  {elapsed1:.0f}s sin TTA | {elapsed2:.0f}s con TTA")

    # Resultados por grupo
    print(f"\n  Por grupo (con TTA):")
    for gname, (gc, gt) in g2.items():
        pct = gc / max(gt, 1)
        tag = "✅" if pct >= 0.8 else "⚠️" if pct >= 0.5 else "❌"
        print(f"    {tag} {gname:30s}: {gc}/{gt} = {pct:.0%}")

    # Errores por tipo
    if f2:
        error_types = Counter(e[4] for e in f2)
        print(f"\n  Errores por tipo:")
        for etype, cnt in error_types.most_common():
            print(f"    {etype:>15s}: {cnt}")

        print(f"\n  Top-10 errores:")
        for gt, pred, hr, conf, etype in sorted(f2, key=lambda x: x[2])[:10]:
            print(f"    '{gt}' → '{pred}' "
                  f"(hit={hr:.0%}, type={etype})")

    # ═══ TEST CARPETA ═══
    test_dirs = [Path("data/Prueba"), Path("test_images"), Path("prueba")]
    for test_dir in test_dirs:
        if not test_dir.exists():
            continue

        print(f"\n{'═' * 65}")
        print(f"  TEST EXTRA: Imágenes de {test_dir}")
        print(f"{'═' * 65}")

        correct_t, total_t = 0, 0
        for img_path in sorted(test_dir.glob("*.png")):
            stem = img_path.stem
            parts = stem.split('_')
            if not parts:
                continue
            gt = parts[0]
            if gt not in char2idx:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            result = predict_single(sess, img, input_name, output_name,
                                     idx2char, num_classes, use_tta=True)
            total_t += 1
            if result['pred_char'] == gt:
                correct_t += 1

            mark = "✅" if result['pred_char'] == gt else "❌"
            etype = classify_error_type(gt, result['pred_char']) \
                    if result['pred_char'] != gt else ''
            print(f"  {mark} '{gt}' → '{result['pred_char']}' "
                  f"({result['confidence']:.3f}) {etype}")

        if total_t > 0:
            print(f"\n  Carpeta accuracy: {correct_t}/{total_t} = "
                  f"{correct_t/total_t:.1%}")

    print(f"\n✅ Tests completados")


if __name__ == '__main__':
    main()