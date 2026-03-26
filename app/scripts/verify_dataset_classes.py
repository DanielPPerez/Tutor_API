"""
app/scripts/verify_dataset_classes.py
=======================================
Cruza las clases de cada dataset en ``data/raw/`` contra las 101 clases
definidas en ``app/models/char_map.json`` y escribe un reporte JSON.

Clases soportadas (char_map.json — 101 clases):
  - 26 minúsculas (a–z)
  - 27 mayúsculas (A–Z + Ñ)
  - 10 dígitos (0–9)
  - Vocales con tilde (á, é, í, ó, ú — minúsculas y mayúsculas: 10)
  - ñ minúscula
  - Trazos primitivos: línea_vertical, línea_horizontal,
    línea_oblicua_derecha, línea_oblicua_izquierda, curva, círculo
  - Símbolos / puntuación (hasta completar 101)

Uso:
    python verify_dataset_classes.py
    python verify_dataset_classes.py \\
        --data-root data \\
        --char-map app/models/char_map.json \\
        --out data/dataset_classes_report.json

Expected Interface (ver PLAN_IMPLEMENTACIONES_ESTADIA.md § 3.2):
  - Módulo:   verify_dataset_classes
  - Función:  run_verification(data_root, char_map_path, output_path) -> dict
              Retorna {
                "char_map_classes": int,
                "by_dataset": {
                  "<nombre>": {
                    "classes": list[str],
                    "covered": list[str],
                    "missing": list[str]
                  }
                },
                "global_missing": list[str]
              }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import gzip
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Clases de trazos primitivos que deben existir en char_map.json
# (si no están, se agregan como "faltantes" en el reporte)
# ---------------------------------------------------------------------------
PRIMITIVE_STROKE_CLASSES: list[str] = [
    "línea_vertical",
    "línea_horizontal",
    "línea_oblicua_derecha",
    "línea_oblicua_izquierda",
    "curva",
    "círculo",
]


# ---------------------------------------------------------------------------
# Helpers — lectura de char_map.json
# ---------------------------------------------------------------------------

def _load_char_map(char_map_path: str) -> dict[str, Any]:
    """
    Carga ``char_map.json``.

    Formatos aceptados:
      A) { "idx2char": {"0": "a", ...}, "char2idx": {"a": 0, ...}, "num_classes": 101 }
      B) { "0": "a", "1": "b", ... }   (solo índice → carácter)
      C) ["a", "b", ...]               (lista ordenada)

    Devuelve siempre un dict con claves "idx2char", "char2idx", "num_classes".
    """
    path = Path(char_map_path)
    if not path.exists():
        # Genera un char_map por defecto y lo escribe para que el resto funcione
        print(f"  [WARN] char_map.json no encontrado en '{char_map_path}'.")
        print("         Generando char_map por defecto con 101 clases ...")
        return _build_default_char_map(char_map_path)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        idx2char = {str(i): c for i, c in enumerate(raw)}
    elif isinstance(raw, dict):
        if "idx2char" in raw:
            idx2char = {str(k): v for k, v in raw["idx2char"].items()}
        else:
            # Asume formato B
            idx2char = {str(k): v for k, v in raw.items() if str(k).isdigit()}
    else:
        raise ValueError(f"Formato de char_map.json no reconocido: {type(raw)}")

    char2idx = {v: int(k) for k, v in idx2char.items()}
    return {
        "idx2char": idx2char,
        "char2idx": char2idx,
        "num_classes": len(idx2char),
    }


def _build_default_char_map(output_path: str | None = None) -> dict[str, Any]:
    """
    Construye un char_map con las 101 clases esperadas del proyecto y,
    si se indica output_path, lo escribe en disco.
    """
    chars: list[str] = []
    # Minúsculas a–z
    chars += [chr(c) for c in range(ord("a"), ord("z") + 1)]
    # Mayúsculas A–Z + Ñ
    chars += [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    chars.append("Ñ")
    # ñ minúscula
    chars.append("ñ")
    # Vocales con tilde
    chars += ["á", "é", "í", "ó", "ú", "Á", "É", "Í", "Ó", "Ú"]
    # Dígitos 0–9
    chars += [str(d) for d in range(10)]
    # Trazos primitivos
    chars += PRIMITIVE_STROKE_CLASSES
    # Símbolos hasta llegar a 101
    extra_symbols = [".", ",", ";", ":", "!", "?", "-", "_", "(", ")", "'"]
    remaining = 101 - len(chars)
    chars += extra_symbols[:remaining]

    idx2char = {str(i): c for i, c in enumerate(chars)}
    char2idx = {c: i for i, c in enumerate(chars)}
    result = {"idx2char": idx2char, "char2idx": char2idx, "num_classes": len(chars)}

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  char_map por defecto escrito en: {path}")

    return result


# ---------------------------------------------------------------------------
# Inspectores por dataset
# ---------------------------------------------------------------------------

def _classes_from_folder_names(root: Path) -> list[str]:
    """
    Estrategia genérica: cada subcarpeta directa de root cuyo nombre
    tenga 1 carácter (o sea un nombre conocido de trazo primitivo)
    se considera una clase.
    """
    classes: list[str] = []
    if not root.exists():
        return classes
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            name = entry.name
            # carpeta de 1 carácter → clase directa
            if len(name) == 1:
                classes.append(name)
            # carpeta que coincide con trazos primitivos
            elif name in PRIMITIVE_STROKE_CLASSES:
                classes.append(name)
            # Formato común: "class_A", "char_a", "label_0" → extraer el char
            else:
                m = re.match(r"^(?:class|char|label|sample)[_\-]?(.+)$", name, re.IGNORECASE)
                if m and len(m.group(1)) == 1:
                    classes.append(m.group(1))
    return classes


def _classes_from_emnist(dataset_root: Path) -> list[str]:
    """
    EMNIST By Class: 62 clases (0–9, A–Z, a–z).
    Verifica la existencia de los archivos (comprimidos o descomprimidos)
    y devuelve las 62 clases estándar si están presentes.
    """
    # Ruta donde torchvision/tu script organiza los archivos
    emnist_raw = dataset_root / "EMNIST" / "raw"
    
    # Buscamos archivos que empiecen con 'emnist-byclass-' 
    # Quitamos el '.gz' del glob para que reconozca los archivos descomprimidos
    found_files = []
    if emnist_raw.exists():
        found_files = list(emnist_raw.glob("emnist-byclass-*"))
    
    if not found_files:
        # Búsqueda recursiva por si están en otra subcarpeta
        found_files = list(dataset_root.rglob("emnist-byclass-*"))

    if found_files:
        # Si encontró archivos del split 'byclass', retornamos el mapeo estándar
        digits   = [str(d) for d in range(10)]
        uppers   = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        lowers   = [chr(c) for c in range(ord("a"), ord("z") + 1)]
        
        print(f"  [EMNIST] Archivos detectados: {len(found_files)}. Mapeando 62 clases.")
        return digits + uppers + lowers
    
    print("  [EMNIST] No se encontraron archivos que empiecen con 'emnist-byclass-'.")
    return []


def _classes_from_handwritting_characters(dataset_root: Path) -> list[str]:
    """
    handwritting_characters_database (sueiras/GitHub).
    Estructura esperada: carpetas con nombre del carácter o índice.
    Inspecciona README.md para extraer lista de clases si existe.
    """
    classes: list[str] = []

    # Intentar extraer del README
    for readme in dataset_root.rglob("README*"):
        try:
            text = readme.read_text(encoding="utf-8", errors="ignore")
            # Buscar patrones tipo "Classes: a, b, c" o tabla markdown
            found = re.findall(r"\b([A-Za-záéíóúÁÉÍÓÚñÑ0-9])\b", text)
            if found:
                classes = list(dict.fromkeys(found))  # preservar orden, deduplicar
                break
        except Exception:
            continue

    # Fallback: carpetas
    if not classes:
        classes = _classes_from_folder_names(dataset_root)

    # Segunda pasada: buscar en subcarpetas típicas (data/, images/, chars/)
    if not classes:
        for sub in ("data", "images", "chars", "characters", "samples"):
            sub_path = dataset_root / sub
            if sub_path.exists():
                classes = _classes_from_folder_names(sub_path)
                if classes:
                    break

    return classes


def _classes_from_iam_handwriting(dataset_root: Path) -> list[str]:
    """
    IAM Handwriting Word Database.
    Contiene palabras completas (no chars individuales); sin embargo
    los labels pueden ser texto ASCII. Reportamos las clases inferidas
    de los archivos de anotación (.txt) o de la estructura de carpetas.
    Clases esperadas: a–z, A–Z (inglés, sin tilde ni ñ).
    """
    classes: list[str] = []

    # Buscar archivos de etiquetas IAM estándar (words.txt, lines.txt)
    for label_file in dataset_root.rglob("*.txt"):
        if label_file.name in ("words.txt", "lines.txt", "sentences.txt"):
            try:
                text = label_file.read_text(encoding="utf-8", errors="ignore")
                # Extraer caracteres únicos de los campos de texto
                # Formato IAM: ... ok <transcripción>
                transcriptions = re.findall(r"\bok\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(.+)", text)
                chars_found: set[str] = set()
                for t in transcriptions:
                    chars_found.update(c for c in t.strip() if c.strip())
                if chars_found:
                    classes = sorted(chars_found)
                    break
            except Exception:
                continue

    # Fallback: clases conocidas del IAM (inglés estándar)
    if not classes:
        classes = (
            [chr(c) for c in range(ord("a"), ord("z") + 1)]
            + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
            + [str(d) for d in range(10)]
        )

    return classes


def _classes_from_spanish_handwritten(dataset_root: Path) -> list[str]:
    """
    Spanish Handwritten: Extrae caracteres únicos desde el archivo 0annotation.json
    y los nombres de las carpetas.
    """
    chars_found: set[str] = set()

    # 1. Intentar leer desde el archivo de anotaciones (lo más preciso)
    # Buscamos 0annotation.json en cualquier subcarpeta
    annotation_files = list(dataset_root.rglob("0annotation.json"))
    
    if annotation_files:
        print(f"  [Spanish] Procesando {len(annotation_files)} archivos de anotación...")
        for ann_path in annotation_files:
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # data es un dict: {"archivo.jpg": "palabra"}
                    for palabra in data.values():
                        # Añadimos cada letra de la palabra al set
                        for char in palabra:
                            if char.strip(): # Evitar espacios
                                chars_found.add(char)
            except Exception as e:
                print(f"  [Spanish] Error leyendo {ann_path.name}: {e}")

    # 2. Fallback: Si no hay JSON o queremos complementar con nombres de carpetas
    # Esto ayuda a detectar clases si las carpetas se llaman "A", "B", "Enie", etc.
    folder_classes = _classes_from_folder_names(dataset_root)
    for c in folder_classes:
        if len(c) == 1:
            chars_found.add(c)

    # 3. Limpieza de caracteres
    # Filtramos para quedarnos solo con lo que nos interesa (letras y números)
    # y evitamos símbolos extraños si los hubiera
    final_classes = sorted([c for c in chars_found if re.match(r'[a-zA-Z0-9ñÑáéíóúÁÉÍÓÚüÜ]', c)])

    if final_classes:
        print(f"  [Spanish] Caracteres detectados: {''.join(final_classes)}")
    else:
        print(f"  [Spanish] No se detectaron caracteres en {dataset_root}")

    return final_classes


# ---------------------------------------------------------------------------
# Lógica central de verificación
# ---------------------------------------------------------------------------

_DATASET_INSPECTORS = {
    "emnist_byclass":                       _classes_from_emnist,
    "handwritting_characters_database":     _classes_from_handwritting_characters,
    "iam_handwriting":                      _classes_from_iam_handwriting,
    "spanish_handwritten_characters_words": _classes_from_spanish_handwritten,
}


def _inspect_dataset(name: str, dataset_root: Path) -> list[str]:
    """Despacha al inspector correcto según el nombre del dataset."""
    inspector = _DATASET_INSPECTORS.get(name)
    if inspector:
        return inspector(dataset_root)
    # Dataset desconocido: estrategia genérica por carpetas
    return _classes_from_folder_names(dataset_root)


def run_verification(
    data_root: str,
    char_map_path: str,
    output_path: str,
) -> dict[str, Any]:
    """
    Ejecuta la verificación de clases y escribe el reporte JSON.

    Parameters
    ----------
    data_root : str
        Ruta base del proyecto (contiene ``raw/`` con los datasets).
    char_map_path : str
        Ruta a ``app/models/char_map.json``.
    output_path : str
        Ruta de salida para el reporte JSON.

    Returns
    -------
    dict con:
      - ``"char_map_classes"`` (int)   — total de clases en char_map.json.
      - ``"by_dataset"``      (dict)   — por dataset: classes, covered, missing.
      - ``"global_missing"``  (list)   — clases de char_map sin cobertura en ningún dataset.
    """
    raw_path = Path(data_root) / "raw"

    # 1. Cargar char_map
    print(f"\nCargando char_map desde: {char_map_path}")
    char_map = _load_char_map(char_map_path)
    all_target_classes: list[str] = list(char_map["idx2char"].values())

    # Asegurar que los trazos primitivos estén en la lista objetivo
    for prim in PRIMITIVE_STROKE_CLASSES:
        if prim not in all_target_classes:
            all_target_classes.append(prim)
            print(f"  [INFO] Clase de trazo primitivo agregada al objetivo: '{prim}'")

    print(f"  Total de clases objetivo: {len(all_target_classes)}")

    # 2. Inspeccionar cada dataset
    by_dataset: dict[str, dict] = {}
    global_covered: set[str] = set()

    # Datasets conocidos + cualquier carpeta desconocida bajo raw/
    known_datasets = set(_DATASET_INSPECTORS.keys())
    found_datasets: list[tuple[str, Path]] = []

    if raw_path.exists():
        for entry in sorted(raw_path.iterdir()):
            if entry.is_dir():
                found_datasets.append((entry.name, entry))
    else:
        print(f"  [WARN] Carpeta raw/ no encontrada en '{raw_path}'.")
        print("         Ejecuta dataset_downloads.py primero.")

    if not found_datasets:
        # Agregar los esperados como vacíos para que el reporte sea completo
        for ds_name in _DATASET_INSPECTORS:
            found_datasets.append((ds_name, raw_path / ds_name))

    for ds_name, ds_path in found_datasets:
        print(f"\n  Inspeccionando dataset: {ds_name} ({ds_path}) ...")

        if not ds_path.exists():
            by_dataset[ds_name] = {
                "path": str(ds_path),
                "status": "not_found",
                "classes": [],
                "covered": [],
                "missing": all_target_classes[:],
            }
            print(f"    [WARN] Carpeta no encontrada; dataset no descargado.")
            continue

        raw_classes = _inspect_dataset(ds_name, ds_path)
        # Normalizar: quitar duplicados, mantener orden
        classes_found = list(dict.fromkeys(raw_classes))

        covered = sorted(set(classes_found) & set(all_target_classes))
        missing = sorted(set(all_target_classes) - set(classes_found))
        global_covered.update(covered)

        by_dataset[ds_name] = {
            "path": str(ds_path),
            "status": "ok",
            "classes": classes_found,
            "covered": covered,
            "missing": missing,
        }
        print(f"    Clases encontradas : {len(classes_found)}")
        print(f"    Cubiertas (vs map) : {len(covered)}")
        print(f"    Faltantes (vs map) : {len(missing)}")

    # 3. Clases sin cobertura en ningún dataset
    global_missing = sorted(set(all_target_classes) - global_covered)

    # 4. Construir reporte
    report: dict[str, Any] = {
        "char_map_path": str(Path(char_map_path).resolve()),
        "char_map_classes": len(all_target_classes),
        "target_classes": all_target_classes,
        "primitive_strokes": PRIMITIVE_STROKE_CLASSES,
        "datasets_inspected": len(by_dataset),
        "by_dataset": by_dataset,
        "global_covered": sorted(global_covered),
        "global_missing": global_missing,
        "coverage_summary": {
            "total_target": len(all_target_classes),
            "total_covered": len(global_covered),
            "total_missing": len(global_missing),
            "coverage_pct": round(len(global_covered) / max(len(all_target_classes), 1) * 100, 2),
        },
    }

    # 5. Escribir reporte
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 6. Imprimir resumen
    _print_summary(report)
    print(f"\nReporte escrito en: {out_path.resolve()}\n")

    return {
        "char_map_classes": report["char_map_classes"],
        "by_dataset": {
            k: {
                "classes": v["classes"],
                "covered": v["covered"],
                "missing": v["missing"],
            }
            for k, v in by_dataset.items()
        },
        "global_missing": global_missing,
    }


def _print_summary(report: dict[str, Any]) -> None:
    """Imprime un resumen legible en consola."""
    cs = report["coverage_summary"]
    print("\n" + "=" * 60)
    print("RESUMEN DE VERIFICACIÓN DE CLASES")
    print("=" * 60)
    print(f"  Clases objetivo (char_map) : {cs['total_target']}")
    print(f"  Clases cubiertas           : {cs['total_covered']}")
    print(f"  Clases faltantes           : {cs['total_missing']}")
    print(f"  Cobertura global           : {cs['coverage_pct']}%")

    print("\n  Por dataset:")
    for ds_name, info in report["by_dataset"].items():
        status_tag = info.get("status", "ok")
        if status_tag == "not_found":
            print(f"    ✗ {ds_name:45s}  [NO DESCARGADO]")
        else:
            cov = len(info["covered"])
            tot = len(report["target_classes"])
            print(f"    ✓ {ds_name:45s}  {cov:3d}/{tot} cubiertas")

    if report["global_missing"]:
        print("\n  Clases SIN cobertura en ningún dataset:")
        for cls in report["global_missing"]:
            print(f"    · {cls}")
        print(
            "\n  ACCIÓN RECOMENDADA: Generar datos sintéticos o buscar fuentes\n"
            "  adicionales para las clases faltantes (especialmente trazos\n"
            "  primitivos y caracteres especiales del español)."
        )
    else:
        print("\n  ✓ Todas las clases están cubiertas por al menos un dataset.")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Punto de entrada CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verifica la cobertura de clases de cada dataset respecto a "
            "char_map.json y genera un reporte JSON."
        )
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Ruta base del proyecto (contiene raw/). Default: ./data",
    )
    parser.add_argument(
        "--char-map",
        default="app/models/char_map.json",
        help="Ruta a char_map.json. Default: app/models/char_map.json",
    )
    parser.add_argument(
        "--out",
        default="data/dataset_classes_report.json",
        help="Ruta de salida del reporte JSON. Default: data/dataset_classes_report.json",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_verification(
        data_root=args.data_root,
        char_map_path=args.char_map,
        output_path=args.out,
    )