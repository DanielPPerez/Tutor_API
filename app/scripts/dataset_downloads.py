"""
app/scripts/dataset_downloads.py
=================================
Descarga los datasets necesarios para el Tutor Inteligente de Caligrafía.

Datasets descargados (versión disponible a 15/03/2026):
  1. EMNIST By Class      — torchvision (split='byclass', train + test)
  2. handwritting_characters_database — GitHub: sueiras/handwritting_characters_database
  3. iam-handwriting-word-database    — Kaggle: nibinv23/iam-handwriting-word-database
  4. spanish-handwritten-characterswords — Kaggle: verack/spanish-handwritten-characterswords

Uso:
    python dataset_downloads.py                  # descarga en ./data
    python dataset_downloads.py --data-root /ruta/personalizada

Variables de entorno para Kaggle API (alternativa a kagglehub interactivo):
    KAGGLE_USERNAME, KAGGLE_KEY

Expected Interface (ver PLAN_IMPLEMENTACIONES_ESTADIA.md § 3.1):
  - Módulo:  dataset_downloads
  - Función: download_all(data_root: str = "data") -> dict[str, dict]
             Retorna { dataset_name: {"path": str, "ok": bool, "message": str} }
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
import torchvision
from tqdm import tqdm
import kagglehub

# ---------------------------------------------------------------------------
# Constantes de URLs / identificadores (fijadas a 15/03/2026)
# ---------------------------------------------------------------------------

GITHUB_HWC_URL = (
    "https://github.com/sueiras/handwritting_characters_database/archive/refs/heads/master.zip"
)
GITHUB_HWC_DIR_IN_ZIP = "handwritting_characters_database-master"

KAGGLE_IAM_DATASET      = "nibinv23/iam-handwriting-word-database"
KAGGLE_SPANISH_DATASET  = "verack/spanish-handwritten-characterswords"


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, desc: str = "") -> None:
    """Descarga un archivo con barra de progreso."""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=desc or dest.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            f.write(chunk)
            bar.update(len(chunk))


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extrae un ZIP en dest_dir."""
    print(f"  Extrayendo {zip_path.name} → {dest_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)


def _kaggle_download(dataset: str, dest_dir: Path) -> None:
    """
    Descarga un dataset de Kaggle usando kagglehub (preferido) o
    la CLI de Kaggle como fallback.

    Requiere que las credenciales estén configuradas:
      - kagglehub: ~/.config/kaggle/kaggle.json  o  KAGGLE_USERNAME + KAGGLE_KEY
      - CLI:       ~/.kaggle/kaggle.json
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # --- Intento 1: kagglehub ---
    try:
        import kagglehub  # pip install kagglehub

        print(f"  Usando kagglehub para '{dataset}' ...")
        path = kagglehub.dataset_download(dataset)
        # kagglehub descarga en caché; copiamos al destino del proyecto
        src = Path(path)
        if src.resolve() != dest_dir.resolve():
            shutil.copytree(src, dest_dir, dirs_exist_ok=True)
        print(f"  Copiado desde caché kagglehub → {dest_dir}")
        return
    except ImportError:
        print("  kagglehub no instalado; intentando CLI de Kaggle ...")
    except Exception as e:
        print(f"  kagglehub falló ({e}); intentando CLI de Kaggle ...")

    # --- Intento 2: CLI de Kaggle ---
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle",
                "datasets", "download",
                "-d", dataset,
                "-p", str(dest_dir),
                "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"No se pudo descargar '{dataset}' con kaggle CLI.\n"
            f"Asegúrate de tener KAGGLE_USERNAME y KAGGLE_KEY configurados.\n"
            f"Stderr: {e.stderr}"
        ) from e


# ---------------------------------------------------------------------------
# Descargadores individuales
# ---------------------------------------------------------------------------

def _download_emnist(raw_path: Path) -> dict:
    """
    Dataset 1 — EMNIST By Class (train + test).
    Modificado para descargar desde Kaggle (crawford/emnist) 
    debido a la inestabilidad de los servidores de NIST.
    """
    dataset_key = "emnist_byclass"
    # Ruta donde torchvision espera encontrar los archivos
    # torchvision busca en: root/EMNIST/raw/
    dest_root = raw_path / dataset_key
    target_raw_dir = dest_root / "EMNIST" / "raw"

    if (target_raw_dir / "emnist-byclass-train-images-idx3-ubyte.gz").exists():
        return {"path": str(dest_root), "ok": True, "message": "EMNIST ya existe; omitiendo descarga."}

    try:
        print("\n[1/4] EMNIST By Class — descargando desde Kaggle (crawford/emnist) ...")
        
        # 1. Descargar usando kagglehub
        import kagglehub
        # El dataset 'crawford/emnist' contiene tanto CSVs como archivos GZIP originales
        cache_path = kagglehub.dataset_download("crawford/emnist")
        src_path = Path(cache_path)

        # 2. Crear carpetas de destino
        target_raw_dir.mkdir(parents=True, exist_ok=True)

        # 3. Localizar los archivos .gz (están dentro de una carpeta llamada 'gzip' en ese dataset)
        gzip_src_folder = src_path / "gzip"
        if not gzip_src_folder.exists():
            # Si no existe la carpeta gzip, buscamos en la raíz de la descarga
            gzip_src_folder = src_path

        print(f"      Organizando archivos para torchvision en {target_raw_dir}...")
        for gz_file in gzip_src_folder.glob("*.gz"):
            shutil.copy(gz_file, target_raw_dir)

        # 4. Verificación final mediante torchvision (no descargará nada porque ya están ahí)
        print("      Verificando integridad con torchvision...")
        torchvision.datasets.EMNIST(root=str(dest_root), split="byclass", train=True, download=True)
        torchvision.datasets.EMNIST(root=str(dest_root), split="byclass", train=False, download=True)

        return {
            "path": str(dest_root),
            "ok": True,
            "message": "EMNIST descargado desde Kaggle y organizado correctamente.",
        }
    except Exception as e:
        return {"path": str(dest_root), "ok": False, "message": f"ERROR en EMNIST: {e}"}

def _download_handwritting_characters(raw_path: Path) -> dict:
    """
    Dataset 2 — handwritting_characters_database (GitHub sueiras).
    URL fijada a 15/03/2026.
    """
    dataset_key = "handwritting_characters_database"
    dest = raw_path / dataset_key

    if dest.exists() and any(dest.iterdir()):
        return {"path": str(dest), "ok": True, "message": "Ya existe; omitiendo descarga."}

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = raw_path / f"{dataset_key}.zip"

    try:
        print(f"\n[2/4] handwritting_characters_database — descargando desde GitHub ...")
        _download_file(GITHUB_HWC_URL, zip_path, desc=dataset_key)
        _extract_zip(zip_path, raw_path)

        # GitHub crea subcarpeta con el nombre del branch
        extracted = raw_path / GITHUB_HWC_DIR_IN_ZIP
        if extracted.exists():
            if dest.exists():
                shutil.rmtree(dest)
            extracted.rename(dest)

        zip_path.unlink(missing_ok=True)
        return {
            "path": str(dest),
            "ok": True,
            "message": "handwritting_characters_database descargado correctamente.",
        }
    except Exception as e:
        return {"path": str(dest), "ok": False, "message": f"ERROR: {e}"}


def _download_iam_handwriting(raw_path: Path) -> dict:
    """
    Dataset 3 — IAM Handwriting Word Database.
    Fuente: Kaggle nibinv23/iam-handwriting-word-database.
    """
    dataset_key = "iam_handwriting"
    dest = raw_path / dataset_key

    if dest.exists() and any(dest.iterdir()):
        return {"path": str(dest), "ok": True, "message": "Ya existe; omitiendo descarga."}

    try:
        print(f"\n[3/4] IAM Handwriting Word Database — descargando desde Kaggle ...")
        _kaggle_download(KAGGLE_IAM_DATASET, dest)
        return {
            "path": str(dest),
            "ok": True,
            "message": "iam-handwriting-word-database descargado correctamente.",
        }
    except Exception as e:
        return {"path": str(dest), "ok": False, "message": f"ERROR: {e}"}


def _download_spanish_handwritten(raw_path: Path) -> dict:
    """
    Dataset 4 — Spanish Handwritten Characters/Words.
    Fuente: Kaggle verack/spanish-handwritten-characterswords.
    """
    dataset_key = "spanish_handwritten_characters_words"
    dest = raw_path / dataset_key

    if dest.exists() and any(dest.iterdir()):
        return {"path": str(dest), "ok": True, "message": "Ya existe; omitiendo descarga."}

    try:
        print(f"\n[4/4] Spanish Handwritten Characters/Words — descargando desde Kaggle ...")
        _kaggle_download(KAGGLE_SPANISH_DATASET, dest)
        return {
            "path": str(dest),
            "ok": True,
            "message": "spanish-handwritten-characterswords descargado correctamente.",
        }
    except Exception as e:
        return {"path": str(dest), "ok": False, "message": f"ERROR: {e}"}


# ---------------------------------------------------------------------------
# Función principal — Expected Interface § 3.1
# ---------------------------------------------------------------------------

def download_all(data_root: str = "data") -> dict[str, dict]:
    """
    Orquesta la descarga de los cuatro datasets hacia ``data_root``.

    Parameters
    ----------
    data_root : str
        Ruta base donde se creará la subcarpeta ``raw/``.
        Por defecto: ``"data"`` (relativa al directorio de trabajo).

    Returns
    -------
    dict[str, dict]
        Clave = nombre del dataset; valor = dict con:
          - ``"path"``    (str)  — ruta final en disco.
          - ``"ok"``      (bool) — True si la descarga fue exitosa.
          - ``"message"`` (str)  — descripción del resultado o del error.

    Notes
    -----
    Crea ``data_root/raw`` si no existe.
    Los datasets de Kaggle requieren credenciales configuradas
    (KAGGLE_USERNAME + KAGGLE_KEY o ~/.kaggle/kaggle.json).
    """
    raw_path = Path(data_root) / "raw"
    raw_path.mkdir(parents=True, exist_ok=True)

    # Carpetas auxiliares que el pipeline necesita
    Path(data_root, "custom_n").mkdir(parents=True, exist_ok=True)
    Path(data_root, "backgrounds", "avif").mkdir(parents=True, exist_ok=True)
    Path(data_root, "processed").mkdir(parents=True, exist_ok=True)
    Path(data_root, "augmented").mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    results["emnist_byclass"]                      = _download_emnist(raw_path)
    results["handwritting_characters_database"]    = _download_handwritting_characters(raw_path)
    results["iam_handwriting"]                     = _download_iam_handwriting(raw_path)
    results["spanish_handwritten_characters_words"]= _download_spanish_handwritten(raw_path)

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE DESCARGA")
    print("=" * 60)
    for name, info in results.items():
        status = "✓ OK" if info["ok"] else "✗ ERROR"
        print(f"  {status}  {name}")
        print(f"          {info['message']}")
        if info["ok"]:
            print(f"          Ruta: {info['path']}")
    print("=" * 60)

    failed = [k for k, v in results.items() if not v["ok"]]
    if failed:
        print(f"\nATENCIÓN: {len(failed)} dataset(s) no se descargaron correctamente.")
        print("  Revisa credenciales de Kaggle o conectividad y vuelve a ejecutar.\n")
    else:
        print("\nTodos los datasets descargados correctamente.")
        print(
            "SIGUIENTE PASO: Ejecuta 'app/scripts/verify_dataset_classes.py' "
            "para cruzar clases con char_map.json\n"
        )

    return results


# ---------------------------------------------------------------------------
# Punto de entrada CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Descarga todos los datasets del Tutor Inteligente de Caligrafía."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Ruta base para almacenar los datasets (default: ./data).",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    download_all(data_root=args.data_root)