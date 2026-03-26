"""
kivy_app/screens/base_screen.py
================================
Pantalla base con fondo y helpers compartidos.

PATRÓN CANVAS CORRECTO (sin canvas.before.clear()):
  Screen ya tiene su propio canvas.before con instrucciones internas.
  Llamar canvas.before.clear() en un Screen dentro de ScreenManager
  desbalancea el stack OpenGL → RenderContext.pop_states crash.

  Solución: usar canvas (no canvas.before) para el fondo del Screen,
  o colocar el fondo en un widget hijo en lugar del Screen directamente.
"""
from __future__ import annotations

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen

from components.styled_box import ColorBox, RoundedBox
from components.icon import Icon
from config import COLORS, FONT_SIZE


def _safe_update_text_size(inst, size):
    """
    Evita que `Label` propague valores inválidos a `text_size`.
    """
    try:
        if not size or len(size) != 2:
            return
        w, h = size
        if w is None or h is None:
            return
        w_i = int(float(w))
        h_i = int(float(h))
        if w_i <= 0 or h_i <= 0:
            return
        inst.text_size = (w_i, h_i)
    except Exception:
        return


class BaseScreen(Screen):
    """
    Pantalla base.

    El fondo se dibuja en un widget ColorBox hijo que cubre toda la pantalla,
    NO directamente en canvas.before del Screen (evita el crash).
    """

    BG_COLOR = COLORS["background"]

    def on_enter(self):
        pass

    def go_to(self, screen_name: str, direction: str = "left") -> None:
        if self.manager and screen_name in self.manager.screen_names:
            self.manager.transition.direction = direction
            self.manager.current = screen_name

    def show_toast(self, message: str, duration: float = 2.5) -> None:
        """Muestra un mensaje flotante temporal."""
        length = min(len(message) * 9 + 40, 380)
        toast  = Label(
            text      = message,
            size_hint = (None, None),
            size      = (length, dp(48)),
            font_size = FONT_SIZE["body_sm"],
            color     = (1, 1, 1, 1),
            halign    = "center",
        )
        toast.x = (Window.width  - toast.width)  / 2
        toast.y = Window.height * 0.12

        # Fondo del toast — fondo en canvas (no canvas.before)
        with toast.canvas.before:
            _c = Color(*COLORS["on_surface"])
            _r = RoundedRectangle(pos=toast.pos, size=toast.size, radius=[dp(24)])
        toast.bind(
            pos  = lambda w, v: setattr(_r, 'pos',  v),
            size = lambda w, v: setattr(_r, 'size', v),
        )

        Window.add_widget(toast)
        Clock.schedule_once(lambda dt: Window.remove_widget(toast), duration)


# =============================================================================
# Fábrica de widgets de UI compartidos
# =============================================================================

def make_topbar(title: str, on_back=None) -> ColorBox:
    """
    Barra superior con título y botón de retroceso opcional.
    Usa ColorBox (patrón correcto) en lugar de manipular canvas.before.
    """
    bar = ColorBox(
        bg_color    = COLORS["primary"],
        orientation = "horizontal",
        size_hint_y = None,
        height      = "56dp",
        padding     = ["12dp", "8dp"],
        spacing     = "8dp",
    )
    if on_back:
        btn = Button(size_hint=(None, None), size=("40dp", "40dp"),
                     background_color=(0, 0, 0, 0))
        btn.add_widget(Icon(name="back", color=list(COLORS["on_primary"]),
                            size_hint=(None, None), size=("22dp", "22dp"),
                            pos_hint={"center_x": 0.5, "center_y": 0.5}))
        btn.bind(on_press=lambda _: on_back())
        bar.add_widget(btn)

    lbl = Label(
        text      = title,
        font_size = FONT_SIZE["h3"],
        color     = COLORS["on_primary"],
        bold      = True,
        halign    = "left",
        valign    = "middle",
    )
    lbl.bind(size=_safe_update_text_size)
    bar.add_widget(lbl)
    return bar


def make_primary_button(text: str, on_press: callable) -> Button:
    """Botón primario estándar."""
    btn = Button(
        text             = text,
        size_hint_y      = None,
        height           = "52dp",
        font_size        = FONT_SIZE["button"],
        background_color = COLORS["primary"],
        color            = COLORS["on_primary"],
        bold             = True,
    )
    btn.bind(on_press=lambda _: on_press())
    return btn


def make_accent_button(text: str, on_press: callable) -> Button:
    """Botón de acento (amber)."""
    btn = Button(
        text             = text,
        size_hint_y      = None,
        height           = "52dp",
        font_size        = FONT_SIZE["button"],
        background_color = COLORS["accent"],
        color            = COLORS["on_primary"],
        bold             = True,
    )
    btn.bind(on_press=lambda _: on_press())
    return btn


def make_ghost_button(text: str, on_press: callable) -> Button:
    """Botón secundario (sin fondo)."""
    btn = Button(
        text             = text,
        size_hint_y      = None,
        height           = "52dp",
        font_size        = FONT_SIZE["button"],
        background_color = COLORS["surface"],
        color            = COLORS["primary"],
    )
    btn.bind(on_press=lambda _: on_press())
    return btn


def make_section_label(text: str) -> Label:
    """Etiqueta de sección (solo texto)."""
    lbl = Label(
        text=text,
        font_size=FONT_SIZE["body"],
        color=COLORS["on_surface"],
        bold=True,
        size_hint_y=None,
        height="30dp",
        halign="left",
    )
    lbl.bind(size=_safe_update_text_size)
    return lbl


def make_section_header(text: str, icon_name: str | None = None) -> BoxLayout:
    """Encabezado de sección con icono dibujado (sin emojis)."""
    row = BoxLayout(orientation="horizontal", size_hint_y=None, height="30dp", spacing="8dp")
    if icon_name:
        row.add_widget(Icon(
            name=icon_name,
            color=list(COLORS["primary"]),
            size_hint=(None, None),
            size=("18dp", "18dp"),
            pos_hint={"center_y": 0.5},
        ))
    row.add_widget(make_section_label(text))
    return row


def make_spacer(height: str = "12dp") -> BoxLayout:
    return BoxLayout(size_hint_y=None, height=height)