"""
kivy_app/config.py
==================
Configuración centralizada de la aplicación.
Sigue SRP: único responsable de todos los parámetros configurables.
Se persiste en un JSON local para que la URL de la API sobreviva reinicios.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ── Rutas del sistema ────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).parent
ASSETS_DIR   = APP_DIR / "assets"
FONTS_DIR    = ASSETS_DIR / "fonts"
IMAGES_DIR   = ASSETS_DIR / "images"
KV_DIR       = ASSETS_DIR / "kv"

# En Android, los datos persistentes van en app_storage_path()
# En escritorio usamos el directorio del proyecto
_DATA_DIR    = Path(os.environ.get("ANDROID_APP_PATH", str(APP_DIR)))
CONFIG_FILE  = _DATA_DIR / "user_config.json"
TEMP_DIR     = _DATA_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Paleta de colores de la app ───────────────────────────────────────────────
COLORS = {
    "primary"        : (0.294, 0.000, 0.510, 1),   # Deep Purple #4B0082
    "primary_light"  : (0.565, 0.196, 0.773, 1),   # Violet #9032C5
    "accent"         : (1.000, 0.600, 0.000, 1),   # Amber #FF9900
    "accent_light"   : (1.000, 0.800, 0.400, 1),   # Light Amber
    "surface"        : (0.980, 0.976, 1.000, 1),   # Near-white with purple tint
    "surface_card"   : (1.000, 1.000, 1.000, 1),   # White cards
    "background"     : (0.961, 0.949, 0.996, 1),   # Soft lavender background
    "on_primary"     : (1.000, 1.000, 1.000, 1),   # White text on primary
    "on_surface"     : (0.129, 0.129, 0.192, 1),   # Dark text
    "text_secondary" : (0.435, 0.435, 0.537, 1),   # Gray text
    "divider"        : (0.878, 0.859, 0.953, 1),   # Soft divider
    # Score colors
    "score_excellent": (0.188, 0.722, 0.369, 1),   # Green  #30B85E
    "score_good"     : (0.086, 0.627, 0.522, 1),   # Teal   #16A085
    "score_fair"     : (1.000, 0.757, 0.027, 1),   # Yellow #FFC107
    "score_poor"     : (0.937, 0.325, 0.314, 1),   # Red    #EF5350
    "error"          : (0.796, 0.094, 0.094, 1),   # Dark Red
}

# ── Niveles de dificultad ────────────────────────────────────────────────────
LEVELS = ["principiante", "intermedio", "avanzado"]

LEVEL_LABELS = {
    "principiante": "Principiante",
    "intermedio"  : "Intermedio",
    "avanzado"    : "Avanzado",
}

LEVEL_COLORS = {
    "principiante": (0.188, 0.722, 0.369, 1),   # Green
    "intermedio"  : (1.000, 0.600, 0.000, 1),   # Amber
    "avanzado"    : (0.796, 0.094, 0.094, 1),   # Red
}

LEVEL_ICONS = {
    "principiante": "⭐",
    "intermedio"  : "⭐⭐",
    "avanzado"    : "⭐⭐⭐",
}

# ── Rangos de puntuación ─────────────────────────────────────────────────────
SCORE_RANGES = [
    (0.85, "excellent", "¡Excelente! Tu trazo es muy preciso."),
    (0.70, "good",      "¡Bien hecho! Sigue practicando."),
    (0.50, "fair",      "Vas mejorando. ¡No te rindas!"),
    (0.00, "poor",      "Necesitas más práctica. ¡Tú puedes!"),
]

def get_score_category(score: float) -> tuple[str, str, tuple]:
    """Devuelve (category, message, color) para un score [0-1]."""
    for threshold, cat, msg in SCORE_RANGES:
        if score >= threshold:
            color_key = f"score_{cat}"
            return cat, msg, COLORS.get(color_key, COLORS["score_poor"])
    return "poor", SCORE_RANGES[-1][2], COLORS["score_poor"]

# ── Configuración de red ─────────────────────────────────────────────────────
# API desplegada en Render (HTTPS)
DEFAULT_API_URL     = "http://127.0.0.1:8000"
API_TIMEOUT_SECONDS = 60
MAX_IMAGE_SIZE_MB   = 5

# ── Tipografía ───────────────────────────────────────────────────────────────
FONT_REGULAR  = str(FONTS_DIR / "Roboto-Regular.ttf")
FONT_BOLD     = str(FONTS_DIR / "Roboto-Bold.ttf")
FONT_LIGHT    = str(FONTS_DIR / "Roboto-Light.ttf")

# Tamaños (dp)
FONT_SIZE = {
    "h1"      : "32sp",
    "h2"      : "24sp",
    "h3"      : "20sp",
    "body"    : "15sp",
    "body_sm" : "13sp",
    "caption" : "11sp",
    "button"  : "14sp",
}

# ── Persistencia de configuración de usuario ─────────────────────────────────
_DEFAULTS = {
    "api_url"       : DEFAULT_API_URL,
    "last_level"    : "intermedio",
    "last_char"     : "A",
    "haptic_enabled": True,
    "sound_enabled" : True,
}


def load_user_config() -> dict:
    """Carga la configuración del usuario desde disco. Devuelve defaults si no existe."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            return {**_DEFAULTS, **saved}
        except Exception:
            pass
    return dict(_DEFAULTS)


def save_user_config(cfg: dict) -> None:
    """Persiste la configuración del usuario en disco."""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[config] Error guardando configuración: {e}")

