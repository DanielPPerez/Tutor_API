"""
components/icon.py
==================
Widget de Iconos Vectoriales — versión corregida.
  • Añadidos: chevron_right, list, star, book, arrow_left
  • Corregido: todos los iconos usan _pad_box consistentemente
  • El color de acento sólo se aplica si la subrutina lo pide explícitamente
"""
from __future__ import annotations
import math

from kivy.animation import Animation
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, StringProperty
from kivy.uix.widget import Widget


class Icon(Widget):
    name         = StringProperty("info")
    color        = ListProperty([0.2, 0.2, 0.2, 1])
    accent_color = ListProperty([0.1, 0.6, 0.9, 1])
    stroke       = NumericProperty(2.5)
    active       = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            pos=self._redraw, size=self._redraw,
            name=self._redraw, color=self._redraw,
            stroke=self._redraw, active=self._redraw,
        )

    def on_name(self, *_):
        self.opacity = 0
        Animation(opacity=1, duration=0.25, t="out_quad").start(self)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _pad_box(self, pad=0.18):
        x, y = self.pos
        w, h = self.size
        p = min(w, h) * pad
        return x + p, y + p, max(1.0, w - 2 * p), max(1.0, h - 2 * p)

    def _set_main(self):
        Color(*(self.accent_color if self.active else self.color))

    def _set_accent(self):
        Color(*self.accent_color)

    # ── redraw ────────────────────────────────────────────────────────────────

    def _redraw(self, *_):
        self.canvas.clear()
        with self.canvas:
            if self.active:
                Color(*self.accent_color[:3], 0.15)
                Ellipse(pos=self.pos, size=self.size)

            self._set_main()

            dispatch = {
                "pencil":        self._draw_pencil,
                "camera":        self._draw_camera,
                "home":          self._draw_home,
                "check":         self._draw_check,
                "gear":          self._draw_gear,
                "trophy":        self._draw_trophy,
                "info":          self._draw_info,
                "image":         self._draw_image,
                "chart":         self._draw_chart,
                "close":         self._draw_close,
                "back":          self._draw_back,
                "arrow_left":    self._draw_back,          # alias
                "chevron_right": self._draw_chevron_right,
                "chevron_left":  self._draw_chevron_left,
                "list":          self._draw_list,
                "star":          self._draw_star,
                "book":          self._draw_book,
            }
            dispatch.get((self.name or "info").lower().strip(), self._draw_info)()

    # ── iconos individuales ───────────────────────────────────────────────────

    def _draw_pencil(self):
        x, y, w, h = self._pad_box(0.12)
        s = self.stroke
        # Cuerpo diagonal del lápiz
        Line(points=[x + w * 0.15, y + h * 0.1,
                     x + w * 0.75, y + h * 0.70], width=s, cap="round")
        Line(points=[x + w * 0.28, y + h * 0.0,
                     x + w * 0.88, y + h * 0.60], width=s, cap="round")
        # Costados
        Line(points=[x + w * 0.15, y + h * 0.1,
                     x + w * 0.28, y + h * 0.00], width=s, cap="round")
        Line(points=[x + w * 0.75, y + h * 0.70,
                     x + w * 0.88, y + h * 0.60], width=s, cap="round")
        # Punta (triángulo inferior-izq)
        Line(points=[x + w * 0.15, y + h * 0.10,
                     x,            y,
                     x + w * 0.28, y + h * 0.00],
             width=s, joint="round", cap="round")
        # Borrador (acento)
        self._set_accent()
        Line(points=[x + w * 0.75, y + h * 0.70,
                     x + w * 0.88, y + h * 0.88,
                     x + w * 1.00, y + h * 0.76,
                     x + w * 0.88, y + h * 0.60],
             width=s, joint="round", cap="round")

    def _draw_home(self):
        x, y, w, h = self._pad_box(0.15)
        s = self.stroke
        Line(points=[x, y + h * 0.50,
                     x + w * 0.50, y + h,
                     x + w, y + h * 0.50],
             width=s, cap="round", joint="round")
        Line(points=[x + w * 0.15, y + h * 0.50,
                     x + w * 0.15, y,
                     x + w * 0.85, y,
                     x + w * 0.85, y + h * 0.50],
             width=s, joint="round")
        self._set_accent()
        Line(points=[x + w * 0.40, y,
                     x + w * 0.40, y + h * 0.28,
                     x + w * 0.60, y + h * 0.28,
                     x + w * 0.60, y],
             width=s)

    def _draw_trophy(self):
        x, y, w, h = self._pad_box(0.16)
        s = self.stroke
        Line(circle=(x + w * 0.5, y + h * 0.58, w * 0.32, 180, 360), width=s)
        Line(points=[x + w * 0.18, y + h * 0.58,
                     x + w * 0.82, y + h * 0.58], width=s, cap="round")
        Line(points=[x + w * 0.50, y + h * 0.26,
                     x + w * 0.50, y + h * 0.12], width=s)
        Line(points=[x + w * 0.28, y + h * 0.08,
                     x + w * 0.72, y + h * 0.08], width=s, cap="round")
        # Asas
        Line(circle=(x + w * 0.13, y + h * 0.63, w * 0.10, 0, 180), width=s - 0.5)
        Line(circle=(x + w * 0.87, y + h * 0.63, w * 0.10, 0, 180), width=s - 0.5)

    def _draw_check(self):
        x, y, w, h = self._pad_box(0.15)
        self._set_accent()
        Line(points=[x, y + h * 0.45,
                     x + w * 0.38, y + h * 0.12,
                     x + w, y + h * 0.88],
             width=self.stroke + 1, cap="round", joint="round")

    def _draw_back(self):
        x, y, w, h = self._pad_box(0.20)
        s = self.stroke
        Line(points=[x + w, y + h * 0.5, x, y + h * 0.5],
             width=s, cap="round")
        Line(points=[x + w * 0.45, y + h * 0.88,
                     x, y + h * 0.5,
                     x + w * 0.45, y + h * 0.12],
             width=s, cap="round", joint="round")

    def _draw_chevron_right(self):
        """Flecha > limpia y centrada"""
        x, y, w, h = self._pad_box(0.28)
        s = self.stroke + 0.5
        Line(points=[x, y + h,
                     x + w, y + h * 0.5,
                     x, y],
             width=s, cap="round", joint="round")

    def _draw_chevron_left(self):
        x, y, w, h = self._pad_box(0.28)
        s = self.stroke + 0.5
        Line(points=[x + w, y + h,
                     x, y + h * 0.5,
                     x + w, y],
             width=s, cap="round", joint="round")

    def _draw_list(self):
        """Tres líneas con punto (modo plana / lista)"""
        x, y, w, h = self._pad_box(0.15)
        s = self.stroke
        fracs = [0.75, 0.50, 0.25]
        dot_r = max(2.0, min(w, h) * 0.07)
        for f in fracs:
            # Línea de texto
            self._set_main()
            Line(points=[x + w * 0.28, y + h * f,
                         x + w, y + h * f],
                 width=s, cap="round")
            # Punto de lista (acento)
            self._set_accent()
            Ellipse(pos=(x + w * 0.02 - dot_r * 0.5,
                         y + h * f - dot_r),
                    size=(dot_r * 2, dot_r * 2))

    def _draw_star(self):
        """Estrella de 5 puntas"""
        x, y, w, h = self._pad_box(0.12)
        cx, cy = x + w / 2, y + h / 2
        outer = min(w, h) / 2
        inner = outer * 0.42
        pts = []
        for i in range(10):
            angle = math.radians(-90 + i * 36)
            r = outer if i % 2 == 0 else inner
            pts += [cx + r * math.cos(angle), cy + r * math.sin(angle)]
        pts += pts[:2]
        Line(points=pts, width=self.stroke, joint="round", cap="round")

    def _draw_book(self):
        """Libro abierto (educación)"""
        x, y, w, h = self._pad_box(0.15)
        s = self.stroke
        cx = x + w * 0.5
        # Tapas izq/der
        Line(points=[x, y, x, y + h, cx, y + h * 0.9, cx, y],
             width=s, joint="round", cap="round")
        Line(points=[x + w, y, x + w, y + h, cx, y + h * 0.9, cx, y],
             width=s, joint="round", cap="round")
        # Lomo
        self._set_accent()
        Line(points=[cx, y, cx, y + h * 0.9], width=s, cap="round")
        # Líneas de texto (pág izquierda)
        self._set_main()
        for frac in [0.65, 0.48, 0.31]:
            Line(points=[x + w * 0.12, y + h * frac,
                         cx - w * 0.08, y + h * frac],
                 width=s * 0.7, cap="round")

    def _draw_camera(self):
        x, y, w, h = self._pad_box(0.15)
        s = self.stroke
        Line(rectangle=(x, y, w, h * 0.75), width=s, joint="round")
        Line(circle=(x + w * 0.5, y + h * 0.37, w * 0.22), width=s)
        self._set_accent()
        Line(points=[x + w * 0.15, y + h * 0.75,
                     x + w * 0.15, y + h * 0.88,
                     x + w * 0.38, y + h * 0.88,
                     x + w * 0.38, y + h * 0.75],
             width=s)

    def _draw_info(self):
        x, y, w, h = self._pad_box(0.15)
        cx = x + w * 0.5
        Line(circle=(cx, y + h * 0.5, w * 0.45), width=self.stroke)
        Line(points=[cx, y + h * 0.72, cx, y + h * 0.38],
             width=self.stroke, cap="round")
        Ellipse(pos=(cx - 2, y + h * 0.20 - 2), size=(4, 4))

    def _draw_gear(self):
        x, y, w, h = self._pad_box(0.20)
        cx, cy = x + w / 2, y + h / 2
        r = w * 0.34
        s = self.stroke
        Line(circle=(cx, cy, r), width=s)
        Line(circle=(cx, cy, r * 0.42), width=s)
        for i in range(8):
            angle = math.radians(i * 45)
            x1 = cx + r * math.cos(angle)
            y1 = cy + r * math.sin(angle)
            x2 = cx + (r + w * 0.16) * math.cos(angle)
            y2 = cy + (r + w * 0.16) * math.sin(angle)
            Line(points=[x1, y1, x2, y2], width=s, cap="round")

    def _draw_close(self):
        x, y, w, h = self._pad_box(0.20)
        s = self.stroke + 0.5
        Line(points=[x, y, x + w, y + h], width=s, cap="round")
        Line(points=[x, y + h, x + w, y], width=s, cap="round")

    def _draw_chart(self):
        x, y, w, h = self._pad_box(0.15)
        s = self.stroke
        Line(points=[x, y + h, x, y, x + w, y], width=s, cap="round")
        self._set_accent()
        Rectangle(pos=(x + w * 0.15, y + s), size=(w * 0.14, h * 0.40))
        self._set_main()
        Rectangle(pos=(x + w * 0.45, y + s), size=(w * 0.14, h * 0.68))
        self._set_accent()
        Rectangle(pos=(x + w * 0.75, y + s), size=(w * 0.14, h * 0.30))

    def _draw_image(self):
        x, y, w, h = self._pad_box(0.15)
        s = self.stroke
        Line(rectangle=(x, y, w, h), width=s, joint="round")
        Line(points=[x, y + h * 0.22,
                     x + w * 0.38, y + h * 0.58,
                     x + w * 0.62, y + h * 0.36,
                     x + w * 0.82, y + h * 0.50,
                     x + w, y + h * 0.22],
             width=s, cap="round", joint="round")
        self._set_accent()
        Line(circle=(x + w * 0.76, y + h * 0.76,
                     min(w, h) * 0.10), width=s)