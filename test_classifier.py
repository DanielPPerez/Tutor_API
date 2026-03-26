"""
test_classifier.py
==================
Script de evaluación del clasificador ONNX sobre las imágenes de la carpeta Prueba.

Uso:
    python test_classifier.py
    python test_classifier.py --prueba-dir app/Prueba --model app/models/weights/mobilenet_classifier.onnx
    python test_classifier.py --top-k 5 --show-failures --save-report

Las imágenes deben tener fondo BLANCO y letra NEGRA (mismo formato que el entrenamiento).

El ground truth se extrae del nombre del archivo con estas estrategias (en orden):
  1. Primer carácter del stem:  "a_001.png"   → "a"
  2. cls{idx}_*:                "cls000_0001.png" → EMNIST_CHARS[0]
  3. prim_{slug}_*:             "prim_circulo_0001.png" → "círculo"
  4. char_{idx}_*:              "char_00_0001.png" → EMNIST_CHARS[0]
  5. Nombre completo de 1 char: "A.png"        → "A"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime no instalado.\n  pip install onnxruntime")
    sys.exit(1)

try:
    from PIL import Image as PilImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")   # no necesita display
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    print("[INFO] matplotlib no disponible — se omitirán las gráficas visuales.")

# =============================================================================
# Rutas por defecto (relativas a la raíz del proyecto E:\Estadia)
# =============================================================================

DEFAULT_PRUEBA_DIR  = "data/Prueba"
DEFAULT_MODEL_PATH  = "app/models/final_models/mobilenet_classifier.onnx"
DEFAULT_CHARMAP     = "app/models/final_models/char_map.json"
IMG_EXTS            = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

EMNIST_CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
PRIM_SLUG_MAP = {
    "linea_vertical":         "línea_vertical",
    "linea_horizontal":       "línea_horizontal",
    "linea_oblicua_derecha":  "línea_oblicua_derecha",
    "linea_oblicua_izquierda":"línea_oblicua_izquierda",
    "curva":                  "curva",
    "circulo":                "círculo",
}

# Normalización ImageNet — misma que en el entrenamiento
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


# =============================================================================
# Carga de modelos
# =============================================================================

def load_classifier(model_path: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(model_path, providers=providers)
        inp  = sess.get_inputs()[0]
        print(f"  Modelo cargado: {Path(model_path).name}")
        print(f"  Input shape   : {inp.shape}  dtype={inp.type}")
        return sess
    except Exception as e:
        print(f"ERROR cargando modelo: {e}")
        sys.exit(1)


def load_charmap(charmap_path: str) -> dict[int, str]:
    path = Path(charmap_path)
    if not path.exists():
        print(f"[WARN] char_map.json no encontrado en '{charmap_path}'. "
              "Se usará EMNIST_CLASS_ORDER como fallback.")
        return {i: c for i, c in enumerate(EMNIST_CHARS)}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "idx2char" in raw:
        return {int(k): str(v) for k, v in raw["idx2char"].items()}
    return {int(k): str(v) for k, v in raw.items() if str(k).isdigit()}


def get_model_input_shape(sess: ort.InferenceSession) -> tuple[int, int, int]:
    """Devuelve (channels, height, width) del modelo."""
    shape = list(sess.get_inputs()[0].shape)
    def _to_int(d):
        if isinstance(d, int): return d
        if isinstance(d, str) and d.isdigit(): return int(d)
        return None
    c = _to_int(shape[1]) if len(shape) >= 2 else None
    h = _to_int(shape[2]) if len(shape) >= 3 else None
    w = _to_int(shape[3]) if len(shape) >= 4 else None
    return (c or 3, h or 64, w or 64)


# =============================================================================
# Ground truth desde nombre de archivo
# =============================================================================

def decode_gt_from_stem(stem: str) -> str | None:
    """
    Extrae el carácter ground truth del nombre del archivo (sin extensión).
    Devuelve None si no puede determinarse.
    """
    # 1. cls{idx}_*
    m = re.match(r'^cls(\d{3})_', stem)
    if m:
        idx = int(m.group(1))
        return EMNIST_CHARS[idx] if idx < len(EMNIST_CHARS) else None

    # 2. prim_{slug}_*
    m = re.match(r'^prim_([a-z_]+)_', stem)
    if m and m.group(1) in PRIM_SLUG_MAP:
        return PRIM_SLUG_MAP[m.group(1)]

    # 3. char_{idx}_*
    m = re.match(r'^char_(\d{2})_', stem)
    if m:
        idx = int(m.group(1))
        return EMNIST_CHARS[idx] if idx < len(EMNIST_CHARS) else None

    # 4. Primer carácter + guión/underscore: "a_001", "A-002"
    m = re.match(r'^(.)[\-_]', stem)
    if m and m.group(1).strip():
        return m.group(1)

    # 5. Stem de 1 carácter: "A.png", "ñ.png"
    if len(stem) == 1 and stem.strip():
        return stem

    return None


# =============================================================================
# Preprocesado — IGUAL QUE EN EL NOTEBOOK DE ENTRENAMIENTO
# =============================================================================

def _find_label_file(img_path: Path) -> Path | None:
    """
    Busca el archivo .txt de etiquetas YOLO correspondiente a la imagen.
    Estrategias en orden:
      1. Mismo stem en ../labels/train/ (estructura YOLO estándar)
      2. Mismo stem en ../labels/val/
      3. Mismo directorio que la imagen
    """
    stem = img_path.stem
    candidates = [
        img_path.parent.parent.parent / "labels" / img_path.parent.parent.name / img_path.parent.name / (stem + ".txt"),
        img_path.parent.parent / "labels" / img_path.parent.name / (stem + ".txt"),
        img_path.parent.parent / "labels" / "train" / (stem + ".txt"),
        img_path.parent.parent / "labels" / "val"   / (stem + ".txt"),
        img_path.with_suffix(".txt"),
    ]
    return next((p for p in candidates if p.exists()), None)


def _extract_best_crop(img_bgr: np.ndarray, label_path: Path) -> np.ndarray | None:
    """
    Extrae el crop más grande del label YOLO (mayor área = letra principal).
    Devuelve None si no hay labels válidos.
    """
    lines = label_path.read_text().strip().splitlines()
    H, W  = img_bgr.shape[:2]
    best_crop, best_area = None, 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = max(0, int((xc - bw/2)*W));  y1 = max(0, int((yc - bh/2)*H))
        x2 = min(W, int((xc + bw/2)*W));  y2 = min(H, int((yc + bh/2)*H))
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2-x1)*(y2-y1)
        if area > best_area:
            best_area = area
            best_crop = img_bgr[y1:y2, x1:x2]

    return best_crop


def preprocess_image(img_path: str, expected_c: int,
                     expected_h: int, expected_w: int) -> np.ndarray | None:
    """
    Pipeline IDÉNTICO a VAL_TRANSFORM del notebook:
    Grayscale → PadToSquare → Resize → (x/255 - 0.5) / 0.5
    """
    img_path = Path(img_path)

    # 1. Leer y convertir a grayscale SIEMPRE
    if _PIL_OK:
        try:
            img = PilImage.open(img_path).convert('L')  # fuerza 1 canal
        except Exception:
            return None
    else:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        img = PilImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    # 2. Buscar label YOLO y extraer crop si existe
    label_path = _find_label_file(img_path)
    if label_path is not None:
        img_bgr_for_crop = cv2.imread(str(img_path))
        if img_bgr_for_crop is not None:
            crop = _extract_best_crop(img_bgr_for_crop, label_path)
            if crop is not None and crop.size > 0:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                img = PilImage.fromarray(crop_gray)

    # 3. PadToSquare (crítico — preserva aspect ratio)
    w, h = img.size
    if w != h:
        side = max(w, h)
        padded = PilImage.new('L', (side, side), 255)  # fondo blanco
        padded.paste(img, ((side - w) // 2, (side - h) // 2))
        img = padded

    # 4. Resize
    img = img.resize((expected_w, expected_h), PilImage.BILINEAR)

    # 5. Normalizar: EXACTAMENTE igual que el notebook
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]
    arr = (arr - 0.5) / 0.5                          # [-1, 1]

    # 6. NCHW: mismo mapa grayscale repetido en el eje de canales si el ONNX espera 3 (p. ej. mobilenet).
    out = arr[np.newaxis, np.newaxis, :, :].astype(np.float32)
    if expected_c != 1:
        out = np.repeat(out, expected_c, axis=1)
    return out


# =============================================================================
# Inferencia
# =============================================================================

def predict(
    sess:       ort.InferenceSession,
    tensor:     np.ndarray,
    idx2char:   dict[int, str],
    top_k:      int = 5,
) -> list[tuple[str, float]]:
    """
    Ejecuta inferencia y devuelve lista de (char, prob) ordenada por prob desc.
    """
    input_name = sess.get_inputs()[0].name
    logits     = sess.run(None, {input_name: tensor})[0][0]   # (num_classes,)

    # Softmax numéricamente estable
    exp_l = np.exp(logits - logits.max())
    probs  = exp_l / exp_l.sum()

    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(idx2char.get(int(idx), f"idx_{int(idx)}"), float(probs[int(idx)])) for idx in top_idx]


# =============================================================================
# Evaluación completa
# =============================================================================

def evaluate(
    prueba_dir:   str,
    model_path:   str,
    charmap_path: str,
    top_k:        int  = 5,
    show_failures: bool = True,
    save_report:  bool = False,
    verbose:      bool = True,
) -> dict:
    """
    Evalúa el clasificador sobre todas las imágenes de prueba_dir.

    Returns
    -------
    dict con métricas completas.
    """
    print("\n" + "=" * 65)
    print("  EVALUACIÓN DEL CLASIFICADOR ONNX")
    print("=" * 65)

    # ── Cargar modelos ────────────────────────────────────────────────────────
    print("\n📦 Cargando modelos...")
    sess    = load_classifier(model_path)
    idx2char = load_charmap(charmap_path)
    char2idx = {v: k for k, v in idx2char.items()}
    ec, eh, ew = get_model_input_shape(sess)
    print(f"  Clases en char_map : {len(idx2char)}")
    print(f"  Input esperado     : {ec}×{eh}×{ew}")

    # ── Recopilar imágenes ────────────────────────────────────────────────────
    prueba_path = Path(prueba_dir)
    if not prueba_path.exists():
        print(f"\nERROR: Carpeta no encontrada: '{prueba_path.resolve()}'")
        sys.exit(1)

    img_files = sorted(
        f for f in prueba_path.rglob("*")
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )
    if not img_files:
        print(f"ERROR: No se encontraron imágenes en '{prueba_path}'")
        sys.exit(1)

    print(f"\n🖼  Imágenes encontradas: {len(img_files)}")

    # ── Contadores ────────────────────────────────────────────────────────────
    top1_correct  = 0
    top_k_correct = 0
    no_gt         = 0
    no_load       = 0
    total_with_gt = 0

    confidence_correct   = []   # confianzas cuando acertó Top-1
    confidence_incorrect = []   # confianzas cuando falló Top-1
    per_class_results    = defaultdict(lambda: {"correct": 0, "total": 0, "wrong_preds": []})
    failures             = []   # (img_path, gt, pred1, conf1, top_k_preds)

    print(f"\n⚙  Procesando imágenes...\n")
    t0 = time.time()

    for img_path in img_files:
        gt = decode_gt_from_stem(img_path.stem)

        tensor = preprocess_image(str(img_path), ec, eh, ew)
        if tensor is None:
            no_load += 1
            if verbose:
                print(f"  [SKIP] No se pudo cargar: {img_path.name}")
            continue

        preds = predict(sess, tensor, idx2char, top_k=top_k)
        pred1, conf1 = preds[0]
        top_k_chars  = [p[0] for p in preds]

        if gt is None:
            no_gt += 1
            # Mostrar igualmente la predicción
            if verbose:
                print(f"  [?GT] {img_path.name:<40s} → pred={pred1!r:>4s}  conf={conf1:.3f}")
            continue

        total_with_gt += 1
        is_top1    = (pred1 == gt)
        is_top_k   = (gt in top_k_chars)

        per_class_results[gt]["total"] += 1

        if is_top1:
            top1_correct  += 1
            top_k_correct += 1
            confidence_correct.append(conf1)
            per_class_results[gt]["correct"] += 1
            if verbose:
                print(f"  ✅ {img_path.name:<40s}  GT={gt!r:>4s}  pred={pred1!r:>4s}  conf={conf1:.3f}")
        elif is_top_k:
            top_k_correct += 1
            confidence_incorrect.append(conf1)
            per_class_results[gt]["wrong_preds"].append(pred1)
            failures.append((img_path, gt, pred1, conf1, preds))
            if verbose:
                top_str = "  ".join(f"{c!r}:{p:.2f}" for c, p in preds[:3])
                print(f"  🔶 {img_path.name:<40s}  GT={gt!r:>4s}  pred={pred1!r:>4s}  conf={conf1:.3f}  "
                      f"[en top{top_k}]  → {top_str}")
        else:
            confidence_incorrect.append(conf1)
            per_class_results[gt]["wrong_preds"].append(pred1)
            failures.append((img_path, gt, pred1, conf1, preds))
            if verbose:
                top_str = "  ".join(f"{c!r}:{p:.2f}" for c, p in preds[:3])
                print(f"  ❌ {img_path.name:<40s}  GT={gt!r:>4s}  pred={pred1!r:>4s}  conf={conf1:.3f}  "
                      f"→ {top_str}")

    elapsed = time.time() - t0

    # ── Métricas ──────────────────────────────────────────────────────────────
    top1_acc = top1_correct  / max(total_with_gt, 1)
    topk_acc = top_k_correct / max(total_with_gt, 1)

    avg_conf_ok  = float(np.mean(confidence_correct))   if confidence_correct   else 0.0
    avg_conf_bad = float(np.mean(confidence_incorrect)) if confidence_incorrect else 0.0

    # ── Reporte consola ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTADOS")
    print("=" * 65)
    print(f"  Total imágenes      : {len(img_files)}")
    print(f"  Con GT deducible    : {total_with_gt}")
    print(f"  Sin GT en nombre    : {no_gt}")
    print(f"  No se pudo cargar   : {no_load}")
    print(f"  Tiempo total        : {elapsed:.2f}s  ({elapsed/max(len(img_files),1)*1000:.1f}ms/img)")
    print()
    print(f"  Accuracy Top-1      : {top1_correct}/{total_with_gt} = {top1_acc*100:.2f}%")
    print(f"  Accuracy Top-{top_k:<2d}    : {top_k_correct}/{total_with_gt} = {topk_acc*100:.2f}%")
    print()
    print(f"  Confianza promedio (aciertos): {avg_conf_ok*100:.1f}%")
    print(f"  Confianza promedio (errores) : {avg_conf_bad*100:.1f}%")

    # Distribución de errores
    if failures:
        wrong_preds_flat = [f[2] for f in failures]
        print(f"\n  Top errores de predicción:")
        for wrong_char, cnt in Counter(wrong_preds_flat).most_common(10):
            print(f"    pred={wrong_char!r:>5s}  {cnt:>3d} veces")

    # Por clase
    print(f"\n  Rendimiento por clase (solo clases con fallos):")
    for cls in sorted(per_class_results):
        info = per_class_results[cls]
        if info["total"] == 0: continue
        acc_cls = info["correct"] / info["total"]
        if acc_cls < 1.0 or info["total"] > 0:
            wrong_freq = Counter(info["wrong_preds"]).most_common(3)
            wrong_str  = "  ".join(f"{c!r}×{n}" for c, n in wrong_freq)
            mark = "✅" if acc_cls == 1.0 else ("🔶" if acc_cls >= 0.5 else "❌")
            print(f"    {mark} GT={cls!r:>5s}  {info['correct']:>3d}/{info['total']:<3d} "
                  f"= {acc_cls*100:5.1f}%"
                  + (f"  errores→ {wrong_str}" if wrong_str else ""))

    # ── Mostrar fallos detallados ─────────────────────────────────────────────
    if show_failures and failures:
        print(f"\n  ─── Fallos detallados (Top-1 incorrecto) ───")
        for img_path, gt, pred1, conf1, preds in failures[:30]:
            top_str = "  ".join(f"{c!r}:{p:.3f}" for c, p in preds[:top_k])
            print(f"    {img_path.name:<45s}  GT={gt!r}  → {top_str}")

    # ── Gráficas ──────────────────────────────────────────────────────────────
    if _MPL_OK:
        _generate_plots(
            failures         = failures,
            confidence_correct   = confidence_correct,
            confidence_incorrect = confidence_incorrect,
            per_class_results = per_class_results,
            top_k             = top_k,
            top1_acc          = top1_acc,
            topk_acc          = topk_acc,
            save_report       = save_report,
        )

    # ── Reporte JSON ──────────────────────────────────────────────────────────
    results = {
        "total_images"         : len(img_files),
        "total_with_gt"        : total_with_gt,
        "no_gt"                : no_gt,
        "top1_correct"         : top1_correct,
        "top_k_correct"        : top_k_correct,
        "top_k"                : top_k,
        "top1_accuracy"        : round(top1_acc, 4),
        "topk_accuracy"        : round(topk_acc, 4),
        "avg_conf_correct"     : round(avg_conf_ok, 4),
        "avg_conf_incorrect"   : round(avg_conf_bad, 4),
        "elapsed_sec"          : round(elapsed, 3),
        "ms_per_image"         : round(elapsed / max(len(img_files), 1) * 1000, 2),
        "per_class"            : {
            cls: {
                "accuracy": round(v["correct"] / max(v["total"], 1), 4),
                "correct":  v["correct"],
                "total":    v["total"],
                "top_wrong": Counter(v["wrong_preds"]).most_common(3),
            }
            for cls, v in per_class_results.items()
        },
    }

    if save_report:
        report_path = Path("test_classifier_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 Reporte JSON guardado: {report_path.resolve()}")

    print("\n" + "=" * 65)
    return results


# =============================================================================
# Gráficas
# =============================================================================

def _generate_plots(
    failures,
    confidence_correct,
    confidence_incorrect,
    per_class_results,
    top_k,
    top1_acc,
    topk_acc,
    save_report,
):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Evaluación del Clasificador OCR Español", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── 1. Distribución de confianza ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 21)
    if confidence_correct:
        ax1.hist(confidence_correct,   bins=bins, alpha=0.7, color="#2ecc71", label=f"Acierto (n={len(confidence_correct)})")
    if confidence_incorrect:
        ax1.hist(confidence_incorrect, bins=bins, alpha=0.7, color="#e74c3c", label=f"Error   (n={len(confidence_incorrect)})")
    ax1.set_xlabel("Confianza (softmax)"); ax1.set_ylabel("Frecuencia")
    ax1.set_title("Distribución de confianza"); ax1.legend(fontsize=8)
    ax1.axvline(0.5, color="gray", ls="--", lw=0.8)

    # ── 2. Accuracy por clase (barras) ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    classes_sorted = sorted(
        [(cls, v["correct"] / max(v["total"], 1), v["total"])
         for cls, v in per_class_results.items() if v["total"] > 0],
        key=lambda x: x[1],
    )
    if classes_sorted:
        labels = [c[0] for c in classes_sorted]
        accs   = [c[1] for c in classes_sorted]
        colors = ["#2ecc71" if a == 1.0 else ("#f39c12" if a >= 0.5 else "#e74c3c") for a in accs]
        bars   = ax2.barh(range(len(labels)), accs, color=colors, height=0.6)
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel("Accuracy"); ax2.set_xlim(0, 1.05)
        ax2.set_title("Accuracy por clase")
        ax2.axvline(1.0, color="gray", ls="--", lw=0.8)
        # Añadir valor
        for i, (bar, acc) in enumerate(zip(bars, accs)):
            ax2.text(min(acc + 0.02, 1.02), i, f"{acc*100:.0f}%", va="center", fontsize=7)

    # ── 3. Errores más frecuentes ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if failures:
        wrong_counter = Counter(f[2] for f in failures)
        top_wrongs = wrong_counter.most_common(12)
        if top_wrongs:
            labels_w = [c for c, _ in top_wrongs]
            counts_w = [n for _, n in top_wrongs]
            ax3.barh(range(len(labels_w)), counts_w, color="#9b59b6", height=0.6)
            ax3.set_yticks(range(len(labels_w)))
            ax3.set_yticklabels(labels_w, fontsize=9)
            ax3.set_xlabel("Veces predicho incorrectamente")
            ax3.set_title("Predicciones erróneas más frecuentes")
    else:
        ax3.text(0.5, 0.5, "¡Sin errores!", ha="center", va="center",
                 fontsize=16, color="#2ecc71", fontweight="bold")
        ax3.set_title("Errores")

    # ── 4. Scatter confianza vs correcto/incorrecto ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if confidence_correct:
        ax4.scatter(range(len(confidence_correct)), sorted(confidence_correct),
                    c="#2ecc71", s=12, alpha=0.6, label="Aciertos")
    if confidence_incorrect:
        ax4.scatter(range(len(confidence_incorrect)), sorted(confidence_incorrect),
                    c="#e74c3c", s=12, alpha=0.6, label="Errores")
    ax4.set_xlabel("Muestra (ordenada por conf)"); ax4.set_ylabel("Confianza")
    ax4.set_title("Confianza ordenada"); ax4.legend(fontsize=8)
    ax4.axhline(0.5, color="gray", ls="--", lw=0.8)

    # ── 5. Resumen textual ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    summary = (
        f"RESUMEN\n"
        f"{'─'*30}\n"
        f"Top-1 Accuracy : {top1_acc*100:.2f}%\n"
        f"Top-{top_k} Accuracy : {topk_acc*100:.2f}%\n\n"
        f"Confianza aciertos : {np.mean(confidence_correct)*100:.1f}%\n" if confidence_correct else
        f"Confianza aciertos : —\n"
    )
    if confidence_incorrect:
        summary += f"Confianza errores  : {np.mean(confidence_incorrect)*100:.1f}%\n"
    summary += f"\nTotal evaluadas : {len(confidence_correct)+len(confidence_incorrect)}"

    ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
             fontsize=11, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="#95a5a6"))

    out_path = "test_classifier_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  📊 Gráficas guardadas: {Path(out_path).resolve()}")
    if not save_report:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


# =============================================================================
# Modo demo: mostrar imágenes de fallo con predicciones
# =============================================================================

def show_failure_grid(failures, idx2char, sess, ec, eh, ew, max_show=16):
    """Muestra una cuadrícula de las imágenes donde el modelo se equivocó."""
    if not _MPL_OK or not failures:
        return

    n   = min(len(failures), max_show)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle(f"Fallos del clasificador (primeros {n})", fontsize=13, fontweight="bold")

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (img_path, gt, pred1, conf1, preds) in enumerate(failures[:n]):
        ax = axes_flat[i]
        img = cv2.imread(str(img_path))
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.text(0.5, 0.5, "?", ha="center", va="center", fontsize=24)

        top3 = "  ".join(f"{c!r}:{p:.2f}" for c, p in preds[:3])
        ax.set_title(f"GT={gt!r}  pred={pred1!r}({conf1:.2f})\n{top3}", fontsize=7.5)
        ax.axis("off")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    out_path = "test_classifier_failures.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  🖼  Cuadrícula de fallos: {Path(out_path).resolve()}")
    plt.close()


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Evalúa el clasificador ONNX sobre las imágenes de la carpeta Prueba.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prueba-dir",  default=DEFAULT_PRUEBA_DIR,
                   help="Carpeta con imágenes de prueba.")
    p.add_argument("--model",       default=DEFAULT_MODEL_PATH,
                   help="Ruta al modelo ONNX.")
    p.add_argument("--charmap",     default=DEFAULT_CHARMAP,
                   help="Ruta a char_map.json.")
    p.add_argument("--top-k",       type=int, default=5,
                   help="Considerar correcto si el GT está en Top-K.")
    p.add_argument("--show-failures", action="store_true", default=True,
                   help="Mostrar detalle de cada fallo.")
    p.add_argument("--save-report", action="store_true",
                   help="Guardar reporte JSON y gráficas.")
    p.add_argument("--quiet",       action="store_true",
                   help="No mostrar resultado imagen por imagen.")
    p.add_argument("--failure-grid", action="store_true",
                   help="Generar cuadrícula visual de imágenes fallidas.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    results = evaluate(
        prueba_dir    = args.prueba_dir,
        model_path    = args.model,
        charmap_path  = args.charmap,
        top_k         = args.top_k,
        show_failures = args.show_failures,
        save_report   = args.save_report,
        verbose       = not args.quiet,
    )

    if args.failure_grid:
        # Re-cargar para la cuadrícula (simplificado)
        print("\n  Generando cuadrícula de fallos...")