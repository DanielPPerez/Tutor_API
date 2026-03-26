"""
training/prepare_yolo_dataset.py
=================================
Prepara el dataset YOLO final fusionando TODAS las fuentes disponibles y
creando la partición train/val + dataset.yaml listo para Ultralytics.

MANEJO DE CADA DATASET
-----------------------

1. SINTÉTICO (data/processed/yolo_dataset/)
   Estructura: images/train/*.jpg  +  labels/train/*.txt
   → Copia directa; labels YOLO ya existen.

2. handwritting_characters_database (data/raw/handwritting_characters_database/)
   Estructura: split/curated.tar.gz.01 + curated.tar.gz.02  (partes de un tar)
   → Concatena las partes y extrae el .tar.gz en un directorio temporal.
   → Dentro del tar busca imágenes de carácter individual.
   → Label generado: bbox = imagen completa (1 carácter recortado).

3. IAM Handwriting (data/raw/iam_handwriting/)
   Estructura: iam_words/words/a01/a01-000u/a01-000u-00-00.png
   → Profundidad variable; busca todos los .png recursivamente.
   → Son imágenes de PALABRAS completas → útiles para el detector.
   → Label generado: bbox = imagen completa (1 palabra = N caracteres).

4. spanish_handwritten_characters_words
   Estructura: carpetas de carácter → imágenes (ya detectadas correctamente).
   → Label generado: bbox = imagen completa.

5. EMNIST (data/raw/emnist_byclass/)
   Formato: dataset binario de torchvision (no archivos sueltos).
   → Se exportan imágenes a data/processed/emnist_images/ (una vez).
   → Label generado: bbox = imagen completa (28×28 → 640×640 al exportar).

6. data/augmented/
   Estructura flexible: busca imágenes en cualquier subcarpeta.
   Si hay .txt del mismo nombre → usa ese label.
   Si no → label de imagen completa.

Estructura de salida
--------------------
data/processed/yolo_dataset_final/
  images/train/  images/val/
  labels/train/  labels/val/
  dataset.yaml
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import random
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}
IMG_SIZE  = 640
NC        = 1
NAMES     = ["trazo"]

_FULL_BOX_LABEL = "0 0.500000 0.500000 1.000000 1.000000\n"


# =============================================================================
# Helpers generales
# =============================================================================

def _ensure_dirs(base: Path) -> tuple[Path, Path, Path, Path]:
    ti = base / "images" / "train"; ti.mkdir(parents=True, exist_ok=True)
    vi = base / "images" / "val";   vi.mkdir(parents=True, exist_ok=True)
    tl = base / "labels" / "train"; tl.mkdir(parents=True, exist_ok=True)
    vl = base / "labels" / "val";   vl.mkdir(parents=True, exist_ok=True)
    return ti, vi, tl, vl


def _imread_unicode(path: Path) -> "np.ndarray | None":
    """
    Lee una imagen ignorando caracteres no-ASCII en la ruta (fix Windows).

    cv2.imread() en Windows falla silenciosamente cuando la ruta contiene
    ñ, tildes u otros caracteres fuera de ASCII.  La solución es leer los
    bytes del archivo con Python (que sí maneja Unicode) y pasarlos a
    cv2.imdecode(), que trabaja sobre el buffer en memoria.

    Orden de intentos:
      1. np.fromfile + cv2.imdecode   — rápido, soporta jpg/png/bmp
      2. Pillow                        — fallback para ppm/pgm/tif/tiff
    """
    # Intento 1: leer bytes y decodificar en memoria
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    # Intento 2: Pillow (cubre ppm, pgm, tif, tiff, webp, avif…)
    try:
        from PIL import Image
        pil_img = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    return None


def _resize_save(src: Path, dst: Path) -> bool:
    """
    Lee src (tolerante a rutas Unicode en Windows), redimensiona a
    IMG_SIZE×IMG_SIZE y guarda en dst como JPEG. Retorna True si ok.
    """
    img = _imread_unicode(src)
    if img is None:
        return False
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def _write_label(dst: Path, label_src: Optional[Path], full_box_fallback: bool) -> None:
    """Escribe el .txt de etiquetas en dst."""
    if label_src is not None and label_src.exists():
        content = label_src.read_text().strip()
        if content:
            dst.write_text(content + "\n")
            return
    if full_box_fallback:
        dst.write_text(_FULL_BOX_LABEL)
    else:
        dst.write_text("")   # negativo YOLO


# Par (img_path, label_path | None, use_full_box_if_no_label)
Sample = tuple[Path, Optional[Path], bool]


# =============================================================================
# 1. Sintético
# =============================================================================

def collect_synthetic(synth_dir: Path) -> list[Sample]:
    """
    Lee data/processed/yolo_dataset/images/train/ + labels/train/.
    Labels YOLO ya existen — copia directa.
    """
    pairs: list[Sample] = []
    img_dir = synth_dir / "images" / "train"
    lbl_dir = synth_dir / "labels" / "train"

    if not img_dir.exists():
        print(f"  [SKIP] Sintético: '{img_dir}' no encontrado.")
        return pairs

    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        pairs.append((img, lbl if lbl.exists() else None, True))

    print(f"  Sintético                            : {len(pairs):>8,} imágenes")
    return pairs


# =============================================================================
# 2. handwritting_characters_database — extracción de .tar.gz particionado
# =============================================================================

def _extract_handwritting(hwc_root: Path) -> Path:
    """
    Concatena curated.tar.gz.01 + curated.tar.gz.02 y extrae el contenido.

    Retorna la ruta a la carpeta extraída. Si ya fue extraída, la devuelve
    directamente sin repetir el proceso.
    """
    extracted_dir = hwc_root / "_extracted"
    done_flag     = extracted_dir / ".done"

    if done_flag.exists():
        print(f"  [OK] handwritting ya extraído en '{extracted_dir}'")
        return extracted_dir

    split_dir = hwc_root 
    part1     = split_dir / "curated.tar.gz.01"
    part2     = split_dir / "curated.tar.gz.02"

    if not part1.exists():
        print(f"  [SKIP] handwritting: no se encontró '{part1}'")
        return extracted_dir   # vacío

    print(f"  Extrayendo handwritting_characters_database ...")
    print(f"    Parte 1: {part1}  ({part1.stat().st_size / 1e6:.0f} MB)")

    combined_gz = hwc_root / "_curated_combined.tar.gz"

    # Concatenar partes
    with open(combined_gz, "wb") as out_f:
        for part in [part1, part2]:
            if part.exists():
                print(f"    Concatenando {part.name} ...")
                with open(part, "rb") as pf:
                    shutil.copyfileobj(pf, out_f)

    # Extraer
    extracted_dir.mkdir(parents=True, exist_ok=True)
    print(f"    Extrayendo en '{extracted_dir}' ...")
    try:
        with tarfile.open(combined_gz, "r:gz") as tar:
            tar.extractall(extracted_dir)
        done_flag.touch()
        print(f"    ✅ Extracción completada.")
    except Exception as e:
        print(f"    [ERROR] Extracción fallida: {e}")
        print("    Intenta extraer manualmente con: "
              "cat curated.tar.gz.01 curated.tar.gz.02 | tar -xz")
    finally:
        combined_gz.unlink(missing_ok=True)

    return extracted_dir


def collect_handwritting(hwc_root: Path, max_images: int = 50_000) -> list[Sample]:
    """
    Extrae y recoge imágenes de handwritting_characters_database.
    Cada imagen es un carácter individual → label = imagen completa.
    """
    if not hwc_root.exists():
        print(f"  [SKIP] handwritting_characters_database: '{hwc_root}' no existe.")
        return []

    # Verificar si hay imágenes directamente (sin extraer)
    direct_imgs = [
        f for f in hwc_root.rglob("*")
        if f.is_file() and f.suffix.lower() in IMG_EXTS
        and "_extracted" not in str(f)
    ]

    if not direct_imgs:
        # Necesita extracción
        extracted_dir = _extract_handwritting(hwc_root)
        search_root   = extracted_dir
    else:
        search_root   = hwc_root

    # Recoger imágenes
    all_imgs = sorted(
        f for f in search_root.rglob("*")
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )

    # Limitar para no desbalancear el dataset
    if len(all_imgs) > max_images:
        random.seed(42)
        all_imgs = random.sample(all_imgs, max_images)

    pairs: list[Sample] = []
    for img in all_imgs:
        # Buscar label en la misma carpeta o en labels/ hermana
        lbl_same    = img.with_suffix(".txt")
        lbl_sibling = img.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")
        lbl = lbl_same if lbl_same.exists() else (lbl_sibling if lbl_sibling.exists() else None)
        pairs.append((img, lbl, True))   # True = generar bbox completo si no hay label

    n_with = sum(1 for _, l, _ in pairs if l is not None)
    print(
        f"  handwritting_characters_database     : {len(pairs):>8,} imágenes  "
        f"({n_with:,} con label)"
    )
    return pairs


# =============================================================================
# 3. IAM Handwriting — estructura nested iam_words/words/aXX/.../img.png
# =============================================================================

def collect_iam(iam_root: Path, max_images: int = 30_000) -> list[Sample]:
    """
    Recoge imágenes de IAM Handwriting.

    Estructura real:
        iam_handwriting/iam_words/words/a01/a01-000u/a01-000u-00-00.png

    Cada imagen es una palabra (múltiples caracteres recortados).
    Son muy útiles para el detector: imagen real con texto manuscrito.
    Label: bbox = imagen completa (1 word box).
    """
    if not iam_root.exists():
        print(f"  [SKIP] IAM Handwriting: '{iam_root}' no existe.")
        return []

    # Buscar recursivamente desde iam_root
    # La profundidad es: iam_root / iam_words / words / aXX / aXX-NNN / img.png
    all_imgs = sorted(
        f for f in iam_root.rglob("*")
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )

    if not all_imgs:
        print(f"  [SKIP] IAM: no se encontraron imágenes bajo '{iam_root}'")
        return []

    # Muestrear si hay demasiadas
    if len(all_imgs) > max_images:
        random.seed(42)
        all_imgs = random.sample(all_imgs, max_images)

    # IAM no tiene labels YOLO → label = imagen completa (word box)
    pairs: list[Sample] = [(img, None, True) for img in all_imgs]

    print(
        f"  iam_handwriting                      : {len(pairs):>8,} imágenes  "
        f"(label=word bbox completo)"
    )
    return pairs


# =============================================================================
# 4. Spanish handwritten — ya detectado correctamente (carpetas por clase)
# =============================================================================

def collect_spanish(spanish_root: Path, max_images: int = 80_000) -> list[Sample]:
    """
    Recoge imágenes de spanish_handwritten_characters_words.
    Cada imagen es un carácter → label = imagen completa.
    """
    if not spanish_root.exists():
        print(f"  [SKIP] Spanish: '{spanish_root}' no existe.")
        return []

    all_imgs = sorted(
        f for f in spanish_root.rglob("*")
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )

    if len(all_imgs) > max_images:
        random.seed(42)
        all_imgs = random.sample(all_imgs, max_images)

    pairs: list[Sample] = []
    for img in all_imgs:
        lbl_same = img.with_suffix(".txt")
        lbl      = lbl_same if lbl_same.exists() else None
        pairs.append((img, lbl, True))

    n_with = sum(1 for _, l, _ in pairs if l is not None)
    print(
        f"  spanish_handwritten_characters_words : {len(pairs):>8,} imágenes  "
        f"({n_with:,} con label)"
    )
    return pairs


# =============================================================================
# 5. EMNIST — exportar imágenes desde torchvision
# =============================================================================

def _export_emnist_images(
    emnist_root:  Path,
    export_dir:   Path,
    max_per_class: int = 800,
) -> int:
    """
    Exporta imágenes de EMNIST byclass a archivos .png en export_dir.

    Solo exporta hasta max_per_class imágenes por clase para no saturar
    el dataset con 800K imágenes repetidas.

    Retorna el número total de imágenes exportadas.
    """
    done_flag = export_dir / ".done"
    if done_flag.exists():
        n = sum(1 for f in export_dir.rglob("*.png"))
        print(f"  [OK] EMNIST ya exportado: {n:,} imágenes en '{export_dir}'")
        return n

    print(f"  Exportando EMNIST byclass → '{export_dir}' ...")
    print(f"    (máx {max_per_class} imágenes/clase × 62 clases)")

    try:
        import torchvision
        import numpy as np

        ds = torchvision.datasets.EMNIST(
            root=str(emnist_root), split="byclass", train=True, download=False
        )
    except Exception as e:
        print(f"  [SKIP] EMNIST: no se pudo cargar el dataset ({e})")
        return 0

    export_dir.mkdir(parents=True, exist_ok=True)

    # Indexar por clase
    from collections import defaultdict
    class_indices: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(ds.targets):
        class_indices[int(label)].append(i)

    emnist_chars = (
        [str(d) for d in range(10)]
        + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    )

    total = 0
    for cls_idx, char in enumerate(emnist_chars):
        indices = class_indices.get(cls_idx, [])
        if not indices:
            continue

        # Muestrear
        if len(indices) > max_per_class:
            random.seed(cls_idx)
            indices = random.sample(indices, max_per_class)

        char_slug = f"cls{cls_idx:02d}"
        char_dir  = export_dir / char_slug
        char_dir.mkdir(exist_ok=True)

        for j, ds_idx in enumerate(indices):
            img_pil, _ = ds[ds_idx]
            arr = np.array(img_pil)
            # Corrección orientación EMNIST byclass
            arr = cv2.flip(cv2.transpose(arr), flipCode=1)
            # Invertir: trazo oscuro / fondo claro
            arr = cv2.bitwise_not(arr)
            # Guardar como PNG grayscale
            out = char_dir / f"{char_slug}_{j:04d}.png"
            cv2.imwrite(str(out), arr)
            total += 1

    done_flag.touch()
    print(f"  ✅ EMNIST exportado: {total:,} imágenes")
    return total


def collect_emnist(emnist_root: Path, processed_root: Path) -> list[Sample]:
    """
    Exporta (si no existe) y recoge imágenes de EMNIST byclass.
    Cada imagen es un carácter 28×28 → label = imagen completa.
    """
    export_dir = processed_root / "emnist_images"
    n = _export_emnist_images(emnist_root, export_dir, max_per_class=500)

    if n == 0:
        return []

    all_imgs   = sorted(f for f in export_dir.rglob("*.png"))
    pairs: list[Sample] = [(img, None, True) for img in all_imgs]

    print(
        f"  emnist_byclass (exportado)           : {len(pairs):>8,} imágenes"
    )
    return pairs


# =============================================================================
# 6. Augmented
# =============================================================================

def collect_augmented(aug_root: Path) -> list[Sample]:
    """
    Recoge imágenes de data/augmented/.
    Estructura flexible: busca imágenes en cualquier subcarpeta.
    """
    if not aug_root.exists():
        print(f"  [SKIP] augmented: '{aug_root}' no existe.")
        return []

    all_imgs = sorted(
        f for f in aug_root.rglob("*")
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )

    if not all_imgs:
        print(f"  [SKIP] augmented: carpeta existe pero está vacía.")
        return []

    pairs: list[Sample] = []
    for img in all_imgs:
        lbl_same    = img.with_suffix(".txt")
        lbl_sibling = img.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")
        lbl = lbl_same if lbl_same.exists() else (lbl_sibling if lbl_sibling.exists() else None)
        pairs.append((img, lbl, True))

    n_with = sum(1 for _, l, _ in pairs if l is not None)
    print(
        f"  augmented                            : {len(pairs):>8,} imágenes  "
        f"({n_with:,} con label)"
    )
    return pairs


# =============================================================================
# Copia paralela de muestras
# =============================================================================

def _copy_one(args: tuple) -> bool:
    """Worker para ProcessPoolExecutor: copia una imagen + escribe su label."""
    img_path, lbl_path, use_full_box, img_dst, lbl_dst = args
    ok = _resize_save(Path(img_path), Path(img_dst))
    if ok:
        _write_label(Path(lbl_dst), Path(lbl_path) if lbl_path else None, use_full_box)
    return ok


def _split_and_copy_parallel(
    all_samples: list[Sample],
    train_img:   Path,
    val_img:     Path,
    train_lbl:   Path,
    val_lbl:     Path,
    val_split:   float = 0.15,
    seed:        int   = 42,
    n_workers:   int   = 0,
) -> tuple[int, int]:
    """
    Divide en train/val y copia en paralelo con ProcessPoolExecutor.

    n_workers=0 → modo secuencial (para debug o Windows sin __main__ guard).
    n_workers>0 → paralelo.
    """
    random.seed(seed)
    indices = list(range(len(all_samples)))
    random.shuffle(indices)
    n_val   = max(1, int(len(indices) * val_split))
    val_set = set(indices[:n_val])

    tasks: list[tuple] = []
    for i, (img_path, lbl_path, use_full_box) in enumerate(all_samples):
        is_val      = (i in val_set)
        img_dst_dir = val_img   if is_val else train_img
        lbl_dst_dir = val_lbl   if is_val else train_lbl

        stem    = f"{img_path.stem}_{i:07d}"
        img_dst = str(img_dst_dir / f"{stem}.jpg")
        lbl_dst = str(lbl_dst_dir / f"{stem}.txt")
        lbl_str = str(lbl_path) if lbl_path is not None else None

        tasks.append((str(img_path), lbl_str, use_full_box, img_dst, lbl_dst))

    n_train_ok = 0
    n_val_ok   = 0
    total      = len(tasks)
    val_count  = len(val_set)
    train_count = total - val_count

    print(f"\n  Copiando {total:,} imágenes "
          f"({'paralelo ×' + str(n_workers) if n_workers > 0 else 'secuencial'}) ...")

    if n_workers > 0:
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_copy_one, t): i for i, t in enumerate(tasks)}
            for fut in as_completed(futures):
                ok = fut.result()
                i  = futures[fut]
                if ok:
                    if i in val_set:
                        n_val_ok += 1
                    else:
                        n_train_ok += 1
                done += 1
                if done % 5000 == 0:
                    print(f"    {done:,}/{total:,} copiadas ...")
    else:
        for i, task in enumerate(tasks):
            ok = _copy_one(task)
            if ok:
                if i in val_set:
                    n_val_ok += 1
                else:
                    n_train_ok += 1
            if (i + 1) % 5000 == 0:
                print(f"    {i+1:,}/{total:,} copiadas ...")

    return n_train_ok, n_val_ok


# =============================================================================
# dataset.yaml
# =============================================================================

def _write_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "dataset.yaml"
    content   = {
        "path":  str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NC,
        "names": NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
    print(f"\n  dataset.yaml → {yaml_path}")
    return yaml_path


# =============================================================================
# Función principal
# =============================================================================

def prepare_yolo_dataset(
    data_root:          str   = "./data",
    output_dir:         str | None = None,
    use_raw:            bool  = True,
    use_augmented:      bool  = True,
    use_synthetic_yolo: bool  = True,
    use_emnist:         bool  = True,
    use_iam:            bool  = True,
    val_split:          float = 0.15,
    seed:               int   = 42,
    n_workers:          int   = 0,
) -> str:
    """
    Prepara el dataset YOLO final fusionando todas las fuentes.

    Parameters
    ----------
    data_root : str
        Raíz del proyecto (contiene raw/, processed/, augmented/).
    output_dir : str | None
        Carpeta de salida. Default: data_root/processed/yolo_dataset_final.
    use_raw : bool
        Incluir handwritting_characters_database y spanish_handwritten.
    use_augmented : bool
        Incluir data/augmented/.
    use_synthetic_yolo : bool
        Incluir data/processed/yolo_dataset/ (imágenes sintéticas).
    use_emnist : bool
        Exportar y incluir EMNIST byclass como imágenes.
    use_iam : bool
        Incluir IAM Handwriting word images.
    val_split : float
        Fracción de validación.
    seed : int
        Semilla aleatoria.
    n_workers : int
        Procesos paralelos para copia. 0 = secuencial.
        Recomendado: 0 en Windows, os.cpu_count()//2 en Linux/Kaggle.

    Returns
    -------
    str  — ruta al dataset.yaml generado.
    """
    root      = Path(data_root)
    out_dir   = Path(output_dir) if output_dir else root / "processed" / "yolo_dataset_final"
    processed = root / "processed"

    print("=" * 65)
    print("  PREPARACIÓN DATASET YOLO FINAL")
    print("=" * 65)
    print(f"  data_root  : {root.resolve()}")
    print(f"  output_dir : {out_dir.resolve()}")
    print(f"  val_split  : {val_split}    workers: {n_workers}")
    print(f"\n  Recolectando fuentes ...\n")

    all_samples: list[Sample] = []

    # ── 1. Sintético ──────────────────────────────────────────────────────────
    if use_synthetic_yolo:
        all_samples += collect_synthetic(root / "processed" / "yolo_dataset")

    # ── 2. handwritting_characters_database ───────────────────────────────────
    if use_raw:
        all_samples += collect_handwritting(root / "raw" / "handwritting_characters_database")

    # ── 3. IAM Handwriting ────────────────────────────────────────────────────
    if use_iam:
        all_samples += collect_iam(root / "raw" / "iam_handwriting")

    # ── 4. Spanish Handwritten ────────────────────────────────────────────────
    if use_raw:
        all_samples += collect_spanish(root / "raw" / "spanish_handwritten_characters_words")

    # ── 5. EMNIST (exportar a archivos) ───────────────────────────────────────
    if use_emnist:
        all_samples += collect_emnist(root / "raw" / "emnist_byclass", processed)

    # ── 6. Augmented ──────────────────────────────────────────────────────────
    if use_augmented:
        all_samples += collect_augmented(root / "augmented")

    # ── Resumen ───────────────────────────────────────────────────────────────
    if not all_samples:
        raise RuntimeError(
            "No se encontraron imágenes en ninguna fuente.\n"
            "Verifica que hayas ejecutado:\n"
            "  1. dataset_downloads.py\n"
            "  2. generate_synthetic_yolo.py"
        )

    n_with_label = sum(1 for _, l, _ in all_samples if l is not None)
    n_negatives  = sum(1 for _, l, _ in all_samples if l is None)
    print(f"\n  ─────────────────────────────────────────────────────────")
    print(f"  TOTAL recolectado : {len(all_samples):>10,} imágenes")
    print(f"  Con label YOLO    : {n_with_label:>10,}")
    print(f"  Label generado    : {n_negatives:>10,}  (bbox imagen completa)")

    # ── Crear dirs de salida ──────────────────────────────────────────────────
    train_img, val_img, train_lbl, val_lbl = _ensure_dirs(out_dir)

    # ── Split y copia ─────────────────────────────────────────────────────────
    n_train, n_val = _split_and_copy_parallel(
        all_samples, train_img, val_img, train_lbl, val_lbl,
        val_split=val_split, seed=seed, n_workers=n_workers,
    )

    print(f"\n  Train : {n_train:,} imágenes")
    print(f"  Val   : {n_val:,}   imágenes")
    print(f"  Total : {n_train + n_val:,} imágenes")

    # ── dataset.yaml ──────────────────────────────────────────────────────────
    yaml_path = _write_yaml(out_dir)

    print("\n✅ Dataset preparado.")
    print(f"   Usa: python training/train_detector.py --data-yaml {yaml_path}\n")
    return str(yaml_path)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fusiona datasets y genera dataset.yaml para YOLOv8.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root",        default="./data")
    parser.add_argument("--output-dir",       default=None)
    parser.add_argument("--val-split",        type=float, default=0.15)
    parser.add_argument("--no-raw",           action="store_true",
                        help="Omitir handwritting y spanish.")
    parser.add_argument("--no-augmented",     action="store_true")
    parser.add_argument("--no-synthetic",     action="store_true")
    parser.add_argument("--no-emnist",        action="store_true",
                        help="Omitir exportación de EMNIST.")
    parser.add_argument("--no-iam",           action="store_true")
    parser.add_argument("--workers",          type=int, default=0,
                        help="Procesos paralelos (0=secuencial). "
                             "En Linux/Kaggle: os.cpu_count()//2")
    args = parser.parse_args()

    prepare_yolo_dataset(
        data_root          = args.data_root,
        output_dir         = args.output_dir,
        use_raw            = not args.no_raw,
        use_augmented      = not args.no_augmented,
        use_synthetic_yolo = not args.no_synthetic,
        use_emnist         = not args.no_emnist,
        use_iam            = not args.no_iam,
        val_split          = args.val_split,
        n_workers          = args.workers,
    )