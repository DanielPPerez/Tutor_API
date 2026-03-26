"""
kivy_app/screens/evaluate_screen.py
=====================================
Pantalla de carga/evaluación — versión corregida.
  • Spinner centrado con AnchorLayout (no pos_hint en BoxLayout)
  • Animación de pulso más suave
  • Info card con carácter objetivo legible
"""
from __future__ import annotations

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

from api_client import evaluate_image, evaluate_plana
from components.icon import Icon
from components.styled_box import ColorBox, RoundedBox
from config import COLORS, DEFAULT_API_URL, FONT_SIZE, load_user_config
from screens.base_screen import BaseScreen, make_spacer
from services.threading_utils import run_in_background


class EvaluateScreen(BaseScreen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._image_path  = ""
        self._target_char = "A"
        self._level       = "intermedio"
        self._mode        = "single"
        self._anim        = None
        self._dots_event  = None

    def set_params(self, image_path, target_char="A", level="intermedio", mode="single"):
        self._image_path  = image_path
        self._target_char = target_char
        self._level       = level
        self._mode        = mode

    def on_enter(self):
        super().on_enter()
        self.clear_widgets()
        self._build_ui()
        Clock.schedule_once(lambda dt: self._start_evaluation(), 0.2)

    def on_leave(self):
        if self._anim:       self._anim.cancel(self._spinner)
        if self._dots_event: self._dots_event.cancel()

    def _build_ui(self):
        root = ColorBox(bg_color=COLORS["background"], orientation="vertical")

        # ── Top bar ───────────────────────────────────────────────────────────
        topbar = ColorBox(
            bg_color    = COLORS["primary"],
            orientation = "horizontal",
            size_hint_y = None,
            height      = dp(56),
            padding     = [dp(16), dp(10)],
        )
        topbar.add_widget(Label(
            text      = "Analizando tu trazo…",
            font_size = FONT_SIZE["h3"],
            color     = COLORS["on_primary"],
            bold      = True,
            halign    = "left",
        ))
        root.add_widget(topbar)

        # ── Centro ────────────────────────────────────────────────────────────
        center = BoxLayout(
            orientation = "vertical",
            padding     = dp(32),
            spacing     = dp(20),
        )

        # Spinner — AnchorLayout para centrado correcto en BoxLayout vertical
        spinner_wrap = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (1, None),
            height    = dp(88),
        )
        self._spinner = Icon(
            name      = "pencil",
            color     = list(COLORS["primary"]),
            size_hint = (None, None),
            size      = (dp(72), dp(72)),
        )
        spinner_wrap.add_widget(self._spinner)
        center.add_widget(spinner_wrap)

        # Mensaje principal
        self._main_msg = Label(
            text      = "Evaluando tu caligrafía…",
            font_size = FONT_SIZE["h3"],
            color     = COLORS["on_surface"],
            bold      = True,
            halign    = "center",
            size_hint = (1, None),
            height    = dp(36),
        )
        center.add_widget(self._main_msg)

        # Sub-mensaje
        self._sub_msg = Label(
            text      = "Esto puede tardar unos segundos.",
            font_size = FONT_SIZE["body_sm"],
            color     = COLORS["text_secondary"],
            halign    = "center",
            size_hint = (1, None),
            height    = dp(24),
        )
        center.add_widget(self._sub_msg)

        # Dots animados — centrados con AnchorLayout
        dots_wrap = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (1, None),
            height    = dp(32),
        )
        self._dots_lbl = Label(
            text      = "●  ○  ○",
            font_size = "20sp",
            color     = COLORS["primary"],
            halign    = "center",
            size_hint = (None, None),
            size      = (dp(100), dp(28)),
        )
        dots_wrap.add_widget(self._dots_lbl)
        center.add_widget(dots_wrap)

        center.add_widget(make_spacer("12dp"))

        # ── Info card ─────────────────────────────────────────────────────────
        info = RoundedBox(
            bg_color    = (*COLORS["primary"][:3], 0.08),
            radius      = 14,
            orientation = "vertical",
            size_hint_y = None,
            height      = dp(100),
            padding     = dp(20),
            spacing     = dp(6),
        )

        # Fila carácter objetivo con icono
        char_row = BoxLayout(
            orientation = "horizontal",
            spacing     = dp(8),
            size_hint_y = None,
            height      = dp(36),
        )
        char_icon_wrap = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (None, None),
            size      = (dp(24), dp(24)),
        )
        char_icon_wrap.add_widget(Icon(
            name      = "pencil",
            color     = list(COLORS["primary"]),
            size_hint = (None, None),
            size      = (dp(18), dp(18)),
        ))
        char_row.add_widget(char_icon_wrap)
        char_row.add_widget(Label(
            text      = f"Carácter objetivo: {self._target_char}",
            font_size = FONT_SIZE["body"],
            color     = COLORS["on_surface"],
            bold      = True,
            halign    = "left",
        ))
        info.add_widget(char_row)

        # Fila nivel/modo
        meta_row = BoxLayout(
            orientation = "horizontal",
            spacing     = dp(8),
            size_hint_y = None,
            height      = dp(28),
        )
        meta_icon_wrap = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (None, None),
            size      = (dp(24), dp(24)),
        )
        meta_icon_wrap.add_widget(Icon(
            name      = "chart",
            color     = list(COLORS["text_secondary"]),
            size_hint = (None, None),
            size      = (dp(16), dp(16)),
        ))
        meta_row.add_widget(meta_icon_wrap)
        meta_row.add_widget(Label(
            text      = (
                f"Nivel: {self._level.capitalize()}  |  "
                f"{'Individual' if self._mode == 'single' else 'Plana completa'}"
            ),
            font_size = FONT_SIZE["body_sm"],
            color     = COLORS["text_secondary"],
            halign    = "left",
        ))
        info.add_widget(meta_row)
        center.add_widget(info)

        root.add_widget(center)
        self.add_widget(root)

        # ── Animaciones ───────────────────────────────────────────────────────
        self._anim = (
            Animation(opacity=0.35, duration=0.5, t="in_out_sine")
            + Animation(opacity=1.0, duration=0.5, t="in_out_sine")
        )
        self._anim.repeat = True
        self._anim.start(self._spinner)

        dots_seq = ["●  ○  ○", "○  ●  ○", "○  ○  ●"]
        self._dot_i = 0

        def _cycle(dt):
            self._dot_i = (self._dot_i + 1) % 3
            if hasattr(self, "_dots_lbl"):
                self._dots_lbl.text = dots_seq[self._dot_i]

        self._dots_event = Clock.schedule_interval(_cycle, 0.45)

    # ── Evaluación ────────────────────────────────────────────────────────────

    def _start_evaluation(self):
        cfg      = load_user_config()
        base_url = cfg.get("api_url", DEFAULT_API_URL)
        self._sub_msg.text = f"Enviando a {base_url}…"

        if self._mode == "plana":
            task = lambda: evaluate_plana(self._image_path, self._level, base_url)
        else:
            task = lambda: evaluate_image(
                self._image_path, self._target_char, self._level, base_url
            )
        run_in_background(task, self._on_result, self._on_error)

    def _on_result(self, result):
        if self._dots_event: self._dots_event.cancel()
        if self._anim:       self._anim.cancel(self._spinner)
        rs = self.manager.get_screen("result")
        rs.set_result(result, mode=self._mode, target_char=self._target_char)
        self.go_to("result")

    def _on_error(self, exc):
        self._on_result({
            "error"       : str(exc),
            "score_final" : 0.0,
            "feedback"    : str(exc),
        })