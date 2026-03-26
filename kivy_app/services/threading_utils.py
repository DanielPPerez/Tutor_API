"""
kivy_app/services/threading_utils.py
=====================================
SRP: ejecutar tareas pesadas (llamadas HTTP) sin bloquear el hilo de UI.
Kivy corre en un solo hilo; cualquier operación de red debe ir en background.
"""
from __future__ import annotations

import threading
from typing import Any, Callable


def run_in_background(
    task:       Callable[[], Any],
    on_success: Callable[[Any], None],
    on_error:   Callable[[Exception], None] | None = None,
) -> threading.Thread:
    """
    Ejecuta `task()` en un hilo secundario.

    Cuando termina, llama a `on_success(result)` o `on_error(exception)`
    usando Clock.schedule_once para que la callback corra en el hilo de UI
    (requisito de Kivy para modificar widgets).

    Parameters
    ----------
    task       : callable que no recibe argumentos y devuelve un resultado.
    on_success : callable(result) — se llama en el hilo de UI con el resultado.
    on_error   : callable(exception) — se llama en el hilo de UI si hay error.

    Returns
    -------
    Thread iniciado (daemon=True).
    """
    from kivy.clock import Clock   # noqa: PLC0415 — import diferido

    def _worker():
        try:
            result = task()
            Clock.schedule_once(lambda dt: on_success(result), 0)
        except Exception as exc:
            if on_error:
                Clock.schedule_once(lambda dt: on_error(exc), 0)
            else:
                Clock.schedule_once(
                    lambda dt: on_success({"error": str(exc)}), 0
                )

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t