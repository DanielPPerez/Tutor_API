"""
kivy_app/components/detected_char_card.py
==========================================
Tarjeta de OCR — muestra lo que el modelo detectó vs lo que se intentó trazar.

Uso en result_screen.py:
    from components.detected_char_card import DetectedCharCard
    content.add_widget(DetectedCharCard(
        target_char   = result.get("target_char", "?"),
        detected_char = result.get("detected_char", "?"),
        confidence    = result.get("confidence", 0.0),   # 0.0–1.0
    ))
"""
from __future__ import annotations

from kivy.metrics import dp
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

from components.icon import Icon
from components.styled_box import RoundedBox
from config import COLORS, FONT_SIZE


def _conf_color(conf: float):
    """Verde si >70 %, naranja si >40 %, rojo si ≤40 %."""
    if conf >= 0.70:
        return (0.18, 0.72, 0.37, 1)   # verde
    if conf >= 0.40:
        return (1.0,  0.60, 0.0,  1)   # naranja
    return (0.80, 0.09, 0.09, 1)       # rojo


class DetectedCharCard(RoundedBox):
    """
    Sección visual del OCR en la pantalla de resultado.

    Layout:
    ┌──────────────────────────────────────────────┐
    │  🔍  ¿Qué letra reconoció la IA?             │
    │                                              │
    │   [ B ]          →           [ A ]          │
    │  Lo que          →        Lo que             │
    │  detectó                  intentaste         │
    │                                              │
    │  Confianza del modelo:  ████░░░░  16.7 %     │
    └──────────────────────────────────────────────┘
    """

    def __init__(self, target_char: str, detected_char: str, confidence: float, **kw):
        match = (detected_char.strip().upper() == target_char.strip().upper())
        accent = _conf_color(confidence)

        super().__init__(
            bg_color    = (1, 1, 1, 1),
            radius      = 16,
            orientation = "vertical",
            size_hint_y = None,
            height      = dp(192),
            padding     = [dp(16), dp(14)],
            spacing     = dp(10),
            **kw,
        )

        # ── Encabezado ────────────────────────────────────────────────────────
        header = BoxLayout(
            orientation = "horizontal",
            size_hint_y = None,
            height      = dp(28),
            spacing     = dp(8),
        )
        icon_wrap = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (None, None),
            size      = (dp(22), dp(22)),
        )
        icon_wrap.add_widget(Icon(
            name      = "info",
            color     = list(COLORS["primary"]),
            size_hint = (None, None),
            size      = (dp(18), dp(18)),
        ))
        header.add_widget(icon_wrap)
        header.add_widget(Label(
            text      = "¿Qué letra reconoció la IA?",
            font_size = FONT_SIZE["h3"],
            color     = COLORS["on_surface"],
            bold      = True,
            halign    = "left",
            valign    = "middle",
        ))
        self.add_widget(header)

        # ── Comparación de caracteres ─────────────────────────────────────────
        compare_row = BoxLayout(
            orientation = "horizontal",
            size_hint_y = None,
            height      = dp(80),
            spacing     = dp(8),
        )

        def _char_bubble(char: str, label_text: str, color) -> BoxLayout:
            col = BoxLayout(orientation="vertical", spacing=dp(4))
            bubble = RoundedBox(
                bg_color  = (*color[:3], 0.12),
                radius    = 12,
                size_hint = (1, None),
                height    = dp(52),
            )
            bubble_anchor = AnchorLayout(anchor_x="center", anchor_y="center")
            bubble_anchor.add_widget(Label(
                text      = char or "?",
                font_size = "32sp",
                color     = color,
                bold      = True,
                halign    = "center",
            ))
            bubble.add_widget(bubble_anchor)
            col.add_widget(bubble)
            col.add_widget(Label(
                text      = label_text,
                font_size = FONT_SIZE["caption"],
                color     = COLORS["text_secondary"],
                halign    = "center",
                size_hint_y = None,
                height    = dp(18),
            ))
            return col

        # Carácter detectado (OCR)
        compare_row.add_widget(
            _char_bubble(detected_char or "?", "Detectado", accent)
        )

        # Flecha central con ícono de match
        arrow_col = BoxLayout(orientation="vertical")
        arrow_anchor = AnchorLayout(anchor_x="center", anchor_y="center")
        match_icon = "check" if match else "close"
        match_color = (0.18, 0.72, 0.37, 1) if match else (0.80, 0.09, 0.09, 1)
        arrow_anchor.add_widget(Icon(
            name      = match_icon,
            color     = list(match_color),
            size_hint = (None, None),
            size      = (dp(26), dp(26)),
        ))
        arrow_col.add_widget(arrow_anchor)
        compare_row.add_widget(BoxLayout(size_hint=(None, 1), width=dp(40)))
        arrow_col_wrap = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (None, 1),
            width     = dp(40),
        )
        arrow_col_wrap.add_widget(Icon(
            name      = match_icon,
            color     = list(match_color),
            size_hint = (None, None),
            size      = (dp(26), dp(26)),
        ))
        compare_row.add_widget(arrow_col_wrap)

        # Carácter objetivo (lo que intentó escribir)
        compare_row.add_widget(
            _char_bubble(target_char or "?", "Objetivo", list(COLORS["primary"]))
        )
        self.add_widget(compare_row)

        # ── Barra de confianza ────────────────────────────────────────────────
        conf_pct = round(confidence * 100, 1)
        conf_row = BoxLayout(
            orientation = "horizontal",
            size_hint_y = None,
            height      = dp(20),
            spacing     = dp(8),
        )
        conf_row.add_widget(Label(
            text      = "Confianza del modelo:",
            font_size = FONT_SIZE["caption"],
            color     = COLORS["text_secondary"],
            size_hint = (None, 1),
            width     = dp(148),
            halign    = "left",
        ))
        # Barra de progreso manual
        bar_bg = RoundedBox(
            bg_color  = (*COLORS["primary"][:3], 0.12),
            radius    = 6,
            size_hint = (1, None),
            height    = dp(10),
        )
        bar_fill = RoundedBox(
            bg_color  = accent,
            radius    = 6,
            size_hint = (min(confidence, 1.0), 1),
        )
        bar_bg.add_widget(bar_fill)
        conf_row.add_widget(bar_bg)
        conf_row.add_widget(Label(
            text      = f"{conf_pct}%",
            font_size = FONT_SIZE["caption"],
            color     = accent,
            bold      = True,
            size_hint = (None, 1),
            width     = dp(44),
            halign    = "right",
        ))
        self.add_widget(conf_row)


# ─── Snippet para result_screen.py ───────────────────────────────────────────
# Agrega estas líneas en _build_results_ui() de result_screen.py,
# después de la sección "Desglose de métricas" y antes de make_spacer final:
#
#   from components.detected_char_card import DetectedCharCard
#
#   content.add_widget(DetectedCharCard(
#       target_char   = self._result.get("target_char",   self._target_char),
#       detected_char = self._result.get("detected_char", "?"),
#       confidence    = self._result.get("confidence",    0.0),
#   ))