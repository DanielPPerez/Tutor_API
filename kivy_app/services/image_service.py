"""
kivy_app/services/image_service.py
====================================
SRP: único responsable de operaciones sobre imágenes:
  - Decodificar base64 → archivo temporal
  - Redimensionar para pantalla
  - Limpiar temporales
"""
from __future__ import annotations

import base64
import io
import os
import uuid
from pathlib import Path

from config import TEMP_DIR


def b64_to_texture_path(b64_string: str, prefix: str = "img") -> str | None:
    """
    Convierte una cadena base64 (PNG/JPEG) a un archivo temporal en disco.

    Kivy no carga imágenes desde bytes directamente en Image.source,
    así que las escribimos en un temp file y devolvemos la ruta.

    Returns
    -------
    str | None  — ruta al archivo temporal, o None si falla.
    """
    if not b64_string:
        return None
    try:
        img_bytes = base64.b64decode(b64_string)
        tmp_path  = TEMP_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        tmp_path.write_bytes(img_bytes)
        return str(tmp_path)
    except Exception as e:
        print(f"[image_service] Error decodificando base64: {e}")
        return None


def cleanup_temp_images() -> None:
    """Elimina todos los archivos temporales generados por esta sesión."""
    try:
        for f in TEMP_DIR.glob("img_*.png"):
            f.unlink(missing_ok=True)
    except Exception as e:
        print(f"[image_service] Error limpiando temporales: {e}")


def resize_image_for_upload(
    image_path: str,
    max_side: int = 1024,
    quality: int  = 85,
) -> bytes:
    """
    Redimensiona una imagen al tamaño máximo indicado y devuelve bytes JPEG.
    Útil para reducir el tamaño antes de enviar a la API.

    Parameters
    ----------
    image_path : str   Ruta a la imagen original.
    max_side   : int   Lado máximo en píxeles.
    quality    : int   Calidad JPEG 1-100.
    """
    try:
        # Intentar con PIL si está disponible
        from PIL import Image as PilImage   # noqa: PLC0415

        img = PilImage.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            img   = img.resize((int(w * ratio), int(h * ratio)), PilImage.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()

    except ImportError:
        # Sin PIL: devolver bytes crudos
        return Path(image_path).read_bytes()
    except Exception as e:
        print(f"[image_service] Error redimensionando: {e}")
        return Path(image_path).read_bytes()