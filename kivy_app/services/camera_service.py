"""
kivy_app/services/camera_service.py
=====================================
SRP: abstracción de la cámara y la galería de fotos.
Funciona tanto en Android (via plyer) como en escritorio (filechooser).
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

# Kivy utils para detección de plataforma confiable
from kivy.utils import platform
from config import TEMP_DIR

# Detectar plataforma de forma estándar en Kivy
_IS_ANDROID = (platform == 'android')


def get_temp_image_path() -> str:
    """Genera una ruta temporal única para guardar una foto capturada."""
    return str(TEMP_DIR / f"capture_{uuid.uuid4().hex[:8]}.jpg")


def open_gallery(callback) -> None:
    """
    Abre el selector de galería/archivos.
    callback(path: str | None) — llamado con la ruta de la imagen seleccionada
    o None si el usuario canceló.
    """
    try:
        from plyer import filechooser   # noqa: PLC0415
        filechooser.open_file(
            on_selection = lambda sel: callback(sel[0] if sel else None),
            filters      = [("Imágenes", "*.jpg", "*.jpeg", "*.png", "*.webp")],
            title        = "Selecciona una imagen",
            multiple     = False,
        )
    except Exception as e:
        print(f"[camera_service] Error abriendo galería: {e}")
        # Fallback Kivy nativo
        _kivy_filechooser(callback)


def _kivy_filechooser(callback) -> None:
    """Filechooser de Kivy como fallback (escritorio)."""
    try:
        from kivy.uix.popup import Popup                       # noqa: PLC0415
        from kivy.uix.boxlayout import BoxLayout               # noqa: PLC0415
        from kivy.uix.filechooser import FileChooserIconView   # noqa: PLC0415
        from kivy.uix.button import Button                     # noqa: PLC0415

        content  = BoxLayout(orientation="vertical", spacing=8, padding=8)
        chooser  = FileChooserIconView(
            filters  = ["*.jpg", "*.jpeg", "*.png", "*.webp"],
            path     = str(Path.home()),
        )
        btn_row  = BoxLayout(size_hint_y=None, height="48dp", spacing=8)
        btn_ok   = Button(text="Seleccionar", background_color=(0.294, 0, 0.51, 1))
        btn_cancel = Button(text="Cancelar",  background_color=(0.5, 0.5, 0.5, 1))

        btn_row.add_widget(btn_cancel)
        btn_row.add_widget(btn_ok)
        content.add_widget(chooser)
        content.add_widget(btn_row)

        popup = Popup(
            title         = "Seleccionar imagen",
            content       = content,
            size_hint     = (0.95, 0.85),
            auto_dismiss  = False,
        )

        def _on_ok(_):
            sel = chooser.selection
            popup.dismiss()
            callback(sel[0] if sel else None)

        def _on_cancel(_):
            popup.dismiss()
            callback(None)

        btn_ok.bind(on_press=_on_ok)
        btn_cancel.bind(on_press=_on_cancel)
        popup.open()

    except Exception as e:
        print(f"[camera_service] Fallback filechooser falló: {e}")
        callback(None)


def request_camera_permission() -> bool:
    """
    Solicita permiso de cámara en Android.
    En escritorio siempre devuelve True.
    """
    if not _IS_ANDROID:
        return True

    try:
        # Los imports de 'android' solo funcionan en el dispositivo. 
        # Usamos # type: ignore para que VS Code no marque error en Windows.
        from android.permissions import request_permissions, Permission   # type: ignore
        from android.permissions import check_permission                   # type: ignore

        permissions = [Permission.CAMERA, Permission.READ_EXTERNAL_STORAGE]

        # Verificar si ya tenemos los permisos
        if all(check_permission(p) for p in permissions):
            return True

        # Nota: request_permissions es ASÍNCRONO. 
        # Esta función devolverá False la primera vez mientras el diálogo está abierto.
        def _cb(permissions, results):
            if all(results):
                print("[camera_service] Permisos concedidos por el usuario")
            else:
                print("[camera_service] Permisos denegados por el usuario")

        request_permissions(permissions, _cb)
        
        # Devolvemos False inicialmente porque el usuario aún está decidiendo en el diálogo
        return False

    except ImportError:
        # Si estamos en Android pero no encuentra el módulo (raro), permitimos intentar
        return True
    except Exception as e:
        print(f"[camera_service] Error solicitando permisos: {e}")
        return False