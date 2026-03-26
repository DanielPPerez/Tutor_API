"""
kivy_app/api_client.py
=======================
Cliente HTTP para la API del Tutor Inteligente.

SRP: única responsabilidad — comunicación HTTP con el backend FastAPI.
No conoce nada de Kivy, UI ni lógica de negocio.

Expected Interface (PLAN_IMPLEMENTACIONES_ESTADIA.md § 3.7)
-----------------------------------------------------------
evaluate_image(image_path, target_char, level, base_url) -> dict
evaluate_plana (image_path, level, base_url)             -> dict
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import requests

from config import API_TIMEOUT_SECONDS, DEFAULT_API_URL, MAX_IMAGE_SIZE_MB


# ── Excepciones específicas del cliente ──────────────────────────────────────

class APIError(Exception):
    """Error de comunicación con la API."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class ImageTooLargeError(APIError):
    """Imagen supera el límite de tamaño."""


class ConnectionError(APIError):   # noqa: A001
    """No se pudo conectar con la API."""


# ── Error response estándar ───────────────────────────────────────────────────

def _error_response(message: str, status_code: int = 0) -> dict:
    return {
        "error"           : message,
        "status_code"     : status_code,
        "score_final"     : 0.0,
        "feedback"        : message,
        "detected_char"   : None,
        "confidence"      : 0.0,
        "image_student_b64": "",
        "template_b64"    : "",
        "comparison_b64"  : "",
        "scores_breakdown": {},
        "weights_used"    : {},
        "metadata"        : {},
        "metrics_extra"   : {},
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_image_bytes(image_path: Union[str, bytes, Path]) -> bytes:
    """
    Lee los bytes de la imagen desde ruta o bytes directos.
    Valida tamaño máximo.
    """
    if isinstance(image_path, bytes):
        data = image_path
    else:
        path = Path(image_path)
        if not path.exists():
            raise APIError(f"Imagen no encontrada: {path}")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            raise ImageTooLargeError(
                f"Imagen demasiado grande ({size_mb:.1f} MB). "
                f"Máximo: {MAX_IMAGE_SIZE_MB} MB"
            )
        data = path.read_bytes()

    return data


def _get_filename(image_path: Union[str, bytes, Path]) -> str:
    """Extrae nombre de archivo o devuelve default."""
    if isinstance(image_path, bytes):
        return "capture.jpg"
    return Path(image_path).name or "image.jpg"


def _build_base_url(base_url: str) -> str:
    """Normaliza la URL base (elimina trailing slash)."""
    return base_url.rstrip("/")


# ── Función principal — POST /evaluate ───────────────────────────────────────

def evaluate_image(
    image_path: Union[str, bytes, Path],
    target_char: str,
    level:       str  = "intermedio",
    base_url:    str  = DEFAULT_API_URL,
) -> dict:
    """
    Envía una imagen al endpoint POST /evaluate y devuelve el JSON de respuesta.

    Parameters
    ----------
    image_path  : str | bytes | Path
        Ruta a la imagen, o bytes directos (ej. captura de cámara).
    target_char : str
        Carácter objetivo que el alumno intentó escribir ("A", "b", "3"…).
    level       : str
        Nivel de dificultad: "principiante" | "intermedio" | "avanzado".
    base_url    : str
        URL base de la API FastAPI (por defecto la de Render en ``config.DEFAULT_API_URL``).

    Returns
    -------
    dict
        JSON de la respuesta del endpoint, incluyendo:
        score_final, feedback, detected_char, confidence,
        image_student_b64, template_b64, comparison_b64,
        scores_breakdown, weights_used, metadata, metrics_extra.
        En caso de error, devuelve un dict con clave "error".
    """
    url = f"{_build_base_url(base_url)}/evaluate"

    try:
        img_bytes = _read_image_bytes(image_path)
        filename  = _get_filename(image_path)

        files = {
            "file": (filename, io.BytesIO(img_bytes), "image/jpeg"),
        }
        data = {
            "target_char": str(target_char).strip(),
            "level"      : str(level).strip(),
        }

        response = requests.post(
            url,
            files   = files,
            data    = data,
            timeout = API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    except ImageTooLargeError as e:
        return _error_response(str(e))
    except requests.exceptions.ConnectionError:
        return _error_response(
            f"No se pudo conectar con la API en {base_url}. "
            "Verifica la URL y que el servidor esté activo."
        )
    except requests.exceptions.Timeout:
        return _error_response(
            f"La API tardó más de {API_TIMEOUT_SECONDS}s en responder. "
            "Intenta con una imagen más pequeña o verifica la conexión."
        )
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else 0
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return _error_response(f"Error del servidor ({status}): {detail}", status)
    except Exception as e:
        return _error_response(f"Error inesperado: {e}")


# ── POST /evaluate_plana ──────────────────────────────────────────────────────

def evaluate_plana(
    image_path: Union[str, bytes, Path],
    level:      str = "intermedio",
    base_url:   str = DEFAULT_API_URL,
) -> dict:
    """
    Envía una imagen al endpoint POST /evaluate_plana.

    El primer carácter detectado en la imagen se usa como plantilla;
    los demás se califican contra esa referencia.

    Returns
    -------
    dict con:
        template_char, template_b64, n_evaluated, avg_score, level,
        results: list[dict]  (uno por carácter calificado)
    """
    url = f"{_build_base_url(base_url)}/evaluate_plana"

    try:
        img_bytes = _read_image_bytes(image_path)
        filename  = _get_filename(image_path)

        files = {"file": (filename, io.BytesIO(img_bytes), "image/jpeg")}
        data  = {"level": str(level).strip()}

        response = requests.post(
            url, files=files, data=data, timeout=API_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()

    except ImageTooLargeError as e:
        return {"error": str(e), "results": [], "n_evaluated": 0}
    except requests.exceptions.ConnectionError:
        return {
            "error": f"Sin conexión con {base_url}",
            "results": [], "n_evaluated": 0,
        }
    except requests.exceptions.Timeout:
        return {
            "error": f"Tiempo de espera agotado ({API_TIMEOUT_SECONDS}s)",
            "results": [], "n_evaluated": 0,
        }
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else 0
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"error": f"Error {status}: {detail}", "results": [], "n_evaluated": 0}
    except Exception as e:
        return {"error": f"Error: {e}", "results": [], "n_evaluated": 0}


# ── Health check ──────────────────────────────────────────────────────────────

def check_api_health(base_url: str) -> tuple[bool, str]:
    """
    Verifica que la API esté accesible.

    Returns
    -------
    (is_ok: bool, message: str)
    """
    try:
        response = requests.get(
            f"{_build_base_url(base_url)}/docs",
            timeout=5,
        )
        if response.status_code < 500:
            return True, "API accesible ✓"
        return False, f"API respondió con error {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"No se pudo conectar con {base_url}"
    except requests.exceptions.Timeout:
        return False, "Tiempo de espera agotado"
    except Exception as e:
        return False, str(e)

