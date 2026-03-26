"""
kivy_app/components/score_gauge.py
====================================
Widget personalizado: medidor circular de puntuación.

NOTA SOBRE canvas.clear():
  ScoreGauge extiende Widget (no Screen), y usa self.canvas (no canvas.before).
  En un Widget simple, canvas.clear() + redibujado es seguro porque el widget
  no es gestionado por ScreenManager (no hay stack OpenGL que desbalancear).
  El crash RenderContext.pop_states ocurre SOLO con canvas.before en Screen.
"""
from __future__ import annotations

from kivy.animation import Animation
from kivy.graphics import Color, Ellipse, Line, RoundedRectangle
from kivy.metrics import dp
from kivy.properties import ColorProperty, NumericProperty, StringProperty
from kivy.uix.widget import Widget

from config import COLORS, get_score_category


class ScoreGauge(Widget):
    """
    Medidor circular animado de puntuación.

    Dibuja un arco de 240° que se rellena proporcionalmente al score.
    Uso:
        gauge = ScoreGauge(size_hint=(1, 1))
        gauge.set_score(0.87, animate=True)
    """

    score       = NumericProperty(0.0)
    arc_color   = ColorProperty(COLORS["score_excellent"])
    track_color = ColorProperty(COLORS["divider"])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # canvas.clear() es seguro en Widget simple (no Screen) ✓
        self.bind(
            score       = self._redraw,
            arc_color   = self._redraw,
            track_color = self._redraw,
            pos         = self._redraw,
            size        = self._redraw,
        )

    def set_score(self, value: float, animate: bool = True) -> None:
        """Establece el score con animación de arco."""
        value = max(0.0, min(1.0, float(value)))
        _, _, color = get_score_category(value)
        self.arc_color = color

        if animate:
            anim = Animation(score=value, duration=1.0, t="out_cubic")
            anim.start(self)
        else:
            self.score = value

    def _redraw(self, *args):
        # canvas.clear() en Widget (no Screen): seguro ✓
        self.canvas.clear()
        if self.width < 10 or self.height < 10:
            return

        cx, cy   = self.center
        radius   = min(self.width, self.height) / 2 - dp(10)
        if radius < dp(5):
            return

        with self.canvas:
            # Sombra suave
            Color(0, 0, 0, 0.07)
            Ellipse(
                pos  = (cx - radius - dp(6), cy - radius - dp(6)),
                size = (radius * 2 + dp(12), radius * 2 + dp(12)),
            )

            # Pista de fondo del arco (arco completo 240°)
            Color(*self.track_color)
            Line(circle=(cx, cy, radius, -210, 30), width=dp(9), cap="round")

            # Arco de progreso
            if self.score > 0.005:
                arc_end = -210 + (-240 * self.score)
                Color(*self.arc_color)
                Line(
                    circle = (cx, cy, radius, -210, arc_end),
                    width  = dp(9),
                    cap    = "round",
                )

            # Círculo interior blanco (efecto "donut")
            inner_r = max(radius - dp(16), dp(4))
            Color(1, 1, 1, 1)
            Ellipse(
                pos  = (cx - inner_r, cy - inner_r),
                size = (inner_r * 2, inner_r * 2),
            )


class FeedbackCard(Widget):
    """
    Tarjeta coloreada para mostrar el mensaje de retroalimentación.
    Usa el patrón correcto: referencias guardadas, sin clear().
    """

    text       = StringProperty("")
    card_color = ColorProperty(COLORS["score_excellent"])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Crear instrucciones UNA vez y guardar referencias ✓
        with self.canvas.before:
            self._bg_c = Color(*self.card_color)
            self._bg_r = RoundedRectangle(
                pos    = self.pos,
                size   = self.size,
                radius = [dp(12)],
            )

        # Actualizar propiedades de la instrucción existente — nunca clear() ✓
        self.bind(
            pos        = lambda w, v: setattr(self._bg_r, 'pos',  v),
            size       = lambda w, v: setattr(self._bg_r, 'size', v),
            card_color = lambda w, v: setattr(self._bg_c, 'rgba', v),
        )