"""
kivy_app/components/styled_box.py
====================================
Widgets Kivy reutilizables con fondo pintado mediante el patrón correcto:

  CORRECTO  (Kivy docs "best practice"):
    - Crear Color + Shape UNA SOLA VEZ en __init__
    - Guardar referencia a la instrucción
    - Actualizar .pos / .size en bind callbacks (sin llamar clear())

  INCORRECTO (causa RenderContext.pop_states crash):
    - Llamar canvas.before.clear() + recrear instrucciones en callbacks

Referencia: https://kivy.org/doc/stable/guide/graphics.html
Issue conocido: github.com/kivy/kivy/issues/2205
"""
from __future__ import annotations

from kivy.graphics import Color, RoundedRectangle, Rectangle, Ellipse
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.properties import ListProperty


# =============================================================================
# Mixin base — aplica el patrón "guardar referencia + actualizar props"
# =============================================================================

class _BgMixin:
    """
    Mixin que añade un fondo dibujado con el patrón correcto de Kivy.
    Subclases deben llamar _init_bg() al final de __init__.
    """
    # Exponemos `bg_color` como propiedad animable de Kivy.
    # Las animaciones del app usan `Animation(..., bg_color=...)`.
    bg_color = ListProperty([1, 1, 1, 1])

    def _update_bg_color(self, instance, value) -> None:
        """Sincroniza la propiedad animada con el Color de canvas."""
        if getattr(self, "_bg_color_ref", None):
            self._bg_color_ref.rgba = value

    def _init_bg(
        self,
        bg_color: tuple,
        radius: float = 0.0,
        shape: str = "rect",   # "rect" | "rounded" | "ellipse"
    ) -> None:
        """
        Crea las instrucciones de canvas UNA sola vez y guarda referencias.

        Parameters
        ----------
        bg_color : tuple RGBA
        radius   : radio de las esquinas en dp (solo para "rounded")
        shape    : tipo de forma
        """
        self._bg_color_ref: Color | None = None
        self._bg_shape_ref = None
        self._bg_radius    = radius
        self.bg_color = list(bg_color)

        with self.canvas.before:
            self._bg_color_ref = Color(*self.bg_color)
            if shape == "rounded":
                self._bg_shape_ref = RoundedRectangle(
                    pos    = self.pos,
                    size   = self.size,
                    radius = [dp(radius)] if radius else [0],
                )
            elif shape == "ellipse":
                self._bg_shape_ref = Ellipse(pos=self.pos, size=self.size)
            else:
                self._bg_shape_ref = Rectangle(pos=self.pos, size=self.size)

        # Actualizar propiedades de la instrucción existente — NUNCA clear()
        self.bind(
            pos  = self._update_bg_pos,
            size = self._update_bg_size,
            bg_color = self._update_bg_color,
        )

    def _update_bg_pos(self, instance, value):
        if self._bg_shape_ref:
            self._bg_shape_ref.pos = value

    def _update_bg_size(self, instance, value):
        if self._bg_shape_ref:
            self._bg_shape_ref.size = value

    def set_bg_color(self, color: tuple) -> None:
        """Cambia el color del fondo en tiempo de ejecución."""
        self.bg_color = list(color)


# =============================================================================
# Widgets styled listos para usar
# =============================================================================

class ColorBox(BoxLayout, _BgMixin):
    """
    BoxLayout con fondo de color sólido.

    Uso:
        box = ColorBox(bg_color=(0.2, 0.2, 0.8, 1), orientation="vertical")
    """
    def __init__(self, bg_color=(1, 1, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self._init_bg(bg_color, shape="rect")


class RoundedBox(BoxLayout, _BgMixin):
    """
    BoxLayout con fondo redondeado.

    Uso:
        card = RoundedBox(bg_color=(1, 1, 1, 1), radius=12)
    """
    def __init__(self, bg_color=(1, 1, 1, 1), radius: float = 12.0, **kwargs):
        super().__init__(**kwargs)
        self._init_bg(bg_color, radius=radius, shape="rounded")


class CircleBox(BoxLayout, _BgMixin):
    """BoxLayout con fondo elíptico/circular."""
    def __init__(self, bg_color=(1, 1, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self._init_bg(bg_color, shape="ellipse")


class ColorFloat(FloatLayout, _BgMixin):
    """FloatLayout con fondo de color."""
    def __init__(self, bg_color=(1, 1, 1, 1), radius: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        shape = "rounded" if radius > 0 else "rect"
        self._init_bg(bg_color, radius=radius, shape=shape)


# =============================================================================
# Barra de progreso simple
# =============================================================================

class StyledProgressBar(Widget):
    """
    Barra de progreso horizontal personalizada.
    Se cambió el nombre de ProgressBar -> StyledProgressBar para evitar 
    conflictos con el estilo interno de Kivy que busca la propiedad '.max'.
    """

    def __init__(
        self,
        value:      float = 0.5,       # 0.0 – 1.0
        bg_color:   tuple = (0.9, 0.9, 0.9, 1),
        fill_color: tuple = (0.3, 0.7, 0.4, 1),
        radius:     float = 6.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._value      = max(0.0, min(1.0, value))
        self._fill_color = fill_color
        self._bg_color   = bg_color
        self._radius     = radius

        with self.canvas:
            # Fondo
            self._bg_c   = Color(*bg_color)
            self._bg_r   = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(radius)]
            )
            # Relleno (width proporcional al valor)
            self._fill_c = Color(*fill_color)
            self._fill_r = RoundedRectangle(
                pos    = self.pos,
                size   = (self.width * self._value, self.height),
                radius = [dp(radius)],
            )

        self.bind(pos=self._update, size=self._update)

    def _update(self, *args):
        self._bg_r.pos   = self.pos
        self._bg_r.size  = self.size
        self._fill_r.pos = self.pos
        self._fill_r.size = (self.width * self._value, self.height)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self._fill_r.size = (self.width * self._value, self.height)

    def set_fill_color(self, color: tuple) -> None:
        self._fill_color = color
        self._fill_c.rgba = color


# =============================================================================
# Divisor horizontal
# =============================================================================

class HDivider(Widget):
    """Línea divisoria horizontal de 1dp."""

    def __init__(self, color=(0.88, 0.86, 0.95, 1), **kwargs):
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", "1dp")
        super().__init__(**kwargs)
        with self.canvas:
            self._c = Color(*color)
            self._r = Rectangle(pos=self.pos, size=self.size)
        self.bind(
            pos  = lambda w, v: setattr(self._r, 'pos', v),
            size = lambda w, v: setattr(self._r, 'size', v),
        )
