"""
kivy_app/main.py
=================
Punto de entrada de la aplicación Kivy.

Responsabilidades (SRP):
  - Inicializar la app Kivy con la configuración visual correcta
  - Registrar todas las pantallas en el ScreenManager
  - Gestionar el ciclo de vida de la app (on_start, on_stop)

Principio de inversión de dependencias:
  - main.py conoce los nombres de las pantallas pero no su lógica interna
  - Cada pantalla es autocontenida y se importa aquí solo para registrarla
"""

import os
import sys

# ── Configuración ANTES de importar Kivy ─────────────────────────────────────
# Debe hacerse antes de cualquier import de Kivy para tener efecto

os.environ.setdefault("KIVY_NO_ENV_CONFIG", "1")

from kivy.config import Config                   # noqa: E402
Config.set("graphics", "resizable", "1")
Config.set("kivy", "log_level",  "warning")
Config.set("input", "mouse", "mouse,disable_multitouch")  # escritorio: no multicursor

# Añadir el directorio raíz de la app al path (necesario en Android)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# ── Imports de Kivy ───────────────────────────────────────────────────────────
from kivy.app import App                          # noqa: E402
from kivy.core.window import Window               # noqa: E402
from kivy.uix.screenmanager import ScreenManager, SlideTransition  # noqa: E402

# ── Pantallas ─────────────────────────────────────────────────────────────────
from screens.home_screen     import HomeScreen     # noqa: E402
from screens.capture_screen  import CaptureScreen  # noqa: E402
from screens.plana_screen    import PlanaScreen    # noqa: E402
from screens.evaluate_screen import EvaluateScreen # noqa: E402
from screens.result_screen   import ResultScreen   # noqa: E402
from screens.config_screen   import ConfigScreen   # noqa: E402


class TutorCaligrafiaApp(App):
    """
    Aplicación principal del Tutor de Caligrafía.

    El ScreenManager gestiona la transición entre pantallas.
    El flujo principal es:
        home → capture → evaluate → result → (capture | home)
        home → plana   → evaluate → result → (plana   | home)
        home → config  → home
    """

    title = "Tutor de Caligrafía"

    def build(self):
        # ── Ventana ───────────────────────────────────────────────────────────
        # Simular tamaño de tablet/móvil en escritorio
        if os.environ.get("ANDROID_ARGUMENT") is None:
            Window.size = (400, 800)

        # ── ScreenManager ─────────────────────────────────────────────────────
        sm = ScreenManager(
            transition=SlideTransition(duration=0.25)
        )

        # Registrar todas las pantallas
        # El nombre de cada screen es la clave de navegación usada en go_to()
        sm.add_widget(HomeScreen    (name="home"))
        sm.add_widget(CaptureScreen (name="capture"))
        sm.add_widget(PlanaScreen   (name="plana"))
        sm.add_widget(EvaluateScreen(name="evaluate"))
        sm.add_widget(ResultScreen  (name="result"))
        sm.add_widget(ConfigScreen  (name="config"))

        sm.current = "home"
        return sm

    def on_start(self):
        """Llamado una vez que la app está completamente iniciada."""
        from services.image_service import cleanup_temp_images   # noqa: PLC0415
        cleanup_temp_images()
        print("[TutorCaligrafiaApp] App iniciada.")

    def on_stop(self):
        """Llamado cuando el usuario cierra la app."""
        from services.image_service import cleanup_temp_images   # noqa: PLC0415
        cleanup_temp_images()
        print("[TutorCaligrafiaApp] App cerrada.")

    def on_pause(self):
        """Android: app va a segundo plano. Devolver True para no destruirla."""
        return True

    def on_resume(self):
        """Android: app vuelve a primer plano."""
        pass


if __name__ == "__main__":
    TutorCaligrafiaApp().run()