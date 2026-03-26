"""
kivy_app/screens/home_screen.py
================================
Pantalla de inicio — versión corregida.
  • Iconos centrados con AnchorLayout en lugar de pos_hint
  • Tarjetas interactivas con feedback visual en tap
  • Estética moderna para público joven
"""
from __future__ import annotations

from kivy.animation import Animation
from kivy.metrics import dp
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView

from components.icon import Icon
from components.styled_box import CircleBox, ColorBox, RoundedBox
from config import COLORS, FONT_SIZE
from screens.base_screen import BaseScreen, make_spacer


# ── Botón de icono (AnchorLayout + ButtonBehavior) ───────────────────────────

class IconButton(ButtonBehavior, AnchorLayout):
    pass


# ── Tarjeta de modo interactiva ───────────────────────────────────────────────

class ModeCard(RoundedBox):
    """
    Tarjeta con retroalimentación táctil:
      • Escala 0.97× al presionar
      • Tinte suave mientras está pulsada
      • Efecto rebote al soltar
    """

    def __init__(self, icon_name, title, desc, color, screen_name, nav_fn, **kwargs):
        super().__init__(
            bg_color    = (1, 1, 1, 1),
            radius      = 16,
            orientation = "horizontal",
            size_hint_y = None,
            height      = dp(116),
            padding     = [dp(14), dp(12)],
            spacing     = dp(12),
            **kwargs,
        )
        self._nav_fn = nav_fn
        self._screen = screen_name
        self._base_color = (1, 1, 1, 1)

        # ── Franja lateral
        stripe = RoundedBox(
            bg_color  = color,
            radius    = 4,
            size_hint = (None, 1),
            width     = dp(5),
        )
        self.add_widget(stripe)

        # ── Círculo con icono (centrado via AnchorLayout interno)
        circle_bg = CircleBox(
            bg_color  = (*color[:3], 0.14),
            size_hint = (None, None),
            size      = (dp(58), dp(58)),
        )
        inner_anchor = AnchorLayout(anchor_x="center", anchor_y="center")
        inner_anchor.add_widget(Icon(
            name      = icon_name,
            color     = [*color[:3], 1],
            size_hint = (None, None),
            size      = (dp(28), dp(28)),
        ))
        circle_bg.add_widget(inner_anchor)

        # Alineación vertical del círculo
        circle_wrap = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (None, 1),
            width     = dp(58),
        )
        circle_wrap.add_widget(circle_bg)
        self.add_widget(circle_wrap)

        # ── Texto
        text_col = BoxLayout(orientation="vertical", spacing=dp(3))
        t = Label(
            text      = title,
            font_size = FONT_SIZE["h3"],
            color     = COLORS["on_surface"],
            bold      = True,
            halign    = "left",
            valign    = "middle",
        )
        t.bind(size=t.setter("text_size"))
        d = Label(
            text      = desc,
            font_size = FONT_SIZE["body_sm"],
            color     = COLORS["text_secondary"],
            halign    = "left",
            valign    = "top",
        )
        d.bind(size=d.setter("text_size"))
        text_col.add_widget(t)
        text_col.add_widget(d)
        self.add_widget(text_col)

        # ── Chevron centrado verticalmente
        chev_wrap = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (None, 1),
            width     = dp(24),
        )
        chev_wrap.add_widget(Icon(
            name      = "chevron_right",
            color     = [*color[:3], 0.85],
            size_hint = (None, None),
            size      = (dp(18), dp(18)),
        ))
        self.add_widget(chev_wrap)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # opacity + bg_color: ambas son propiedades nativas de Widget/RoundedBox
            anim = Animation(opacity=0.75, duration=0.07, t="out_quad")
            anim &= Animation(bg_color=(0.96, 0.96, 1.0, 1), duration=0.07)
            anim.start(self)
            touch.grab(self)
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            anim = Animation(opacity=1.0, duration=0.18, t="out_quad")
            anim &= Animation(bg_color=self._base_color, duration=0.12)
            anim.start(self)
            if self.collide_point(*touch.pos):
                self._nav_fn(self._screen)
            return True
        return super().on_touch_up(touch)


# ── HomeScreen ────────────────────────────────────────────────────────────────

class HomeScreen(BaseScreen):

    def on_enter(self):
        super().on_enter()
        self.clear_widgets()
        self._build_ui()

    def _build_ui(self):
        root = ColorBox(bg_color=COLORS["background"], orientation="vertical")

        # ── Top bar ───────────────────────────────────────────────────────────
        topbar = ColorBox(
            bg_color    = COLORS["primary"],
            orientation = "horizontal",
            size_hint_y = None,
            height      = dp(56),
            padding     = [dp(12), dp(8)],
            spacing     = dp(8),
        )

        # Logo/ícono izquierdo
        logo_btn = IconButton(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (None, None),
            size      = (dp(40), dp(40)),
        )
        logo_btn.add_widget(Icon(
            name      = "pencil",
            color     = list(COLORS["on_primary"]),
            size_hint = (None, None),
            size      = (dp(22), dp(22)),
        ))
        topbar.add_widget(logo_btn)

        topbar.add_widget(Label(
            text      = "Tutor de Caligrafía",
            font_size = FONT_SIZE["h3"],
            color     = COLORS["on_primary"],
            bold      = True,
            halign    = "left",
        ))

        # Botón de configuración
        btn_cfg = IconButton(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (None, None),
            size      = (dp(44), dp(44)),
        )
        btn_cfg.add_widget(Icon(
            name      = "gear",
            color     = list(COLORS["on_primary"]),
            size_hint = (None, None),
            size      = (dp(22), dp(22)),
        ))
        btn_cfg.bind(on_press=lambda _: self.go_to("config"))
        topbar.add_widget(btn_cfg)
        root.add_widget(topbar)

        # ── Scrollable content ────────────────────────────────────────────────
        scroll = ScrollView()
        content = BoxLayout(
            orientation = "vertical",
            padding     = dp(20),
            spacing     = dp(16),
            size_hint_y = None,
        )
        content.bind(minimum_height=content.setter("height"))

        # ── Hero card ─────────────────────────────────────────────────────────
        hero = RoundedBox(
            bg_color    = COLORS["primary"],
            radius      = 20,
            orientation = "vertical",
            size_hint_y = None,
            height      = dp(200),
            spacing     = dp(8),
            padding     = dp(20),
        )
        # Icono hero centrado
        hero_icon_wrap = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (1, None),
            height    = dp(64),
        )
        hero_icon_wrap.add_widget(Icon(
            name      = "pencil",
            color     = list(COLORS["on_primary"]),
            size_hint = (None, None),
            size      = (dp(52), dp(52)),
        ))
        hero.add_widget(hero_icon_wrap)

        hero.add_widget(Label(
            text      = "¡Mejora tu caligrafía!",
            font_size = FONT_SIZE["h2"],
            color     = COLORS["on_primary"],
            bold      = True,
            halign    = "center",
        ))
        desc_lbl = Label(
            text      = "Sube una foto de tu libreta y recibe\nretroalimentación inmediata.",
            font_size = FONT_SIZE["body_sm"],
            color     = (1, 1, 1, 0.85),
            halign    = "center",
        )
        desc_lbl.bind(size=desc_lbl.setter("text_size"))
        hero.add_widget(desc_lbl)
        content.add_widget(hero)

        # ── Sección modos ─────────────────────────────────────────────────────
        content.add_widget(Label(
            text        = "¿Qué quieres practicar?",
            font_size   = FONT_SIZE["h3"],
            color       = COLORS["on_surface"],
            bold        = True,
            size_hint_y = None,
            height      = dp(36),
            halign      = "left",
        ))

        cards_data = [
            ("pencil", "Evaluar\nCarácter",
             "Fotografía un carácter y recibe\ntu puntuación al instante.",
             COLORS["primary"], "capture"),
            ("list", "Modo Plana",
             "Fotografía una plana completa\ny evalúa cada repetición.",
             (0.086, 0.627, 0.522, 1), "plana"),
        ]
        for icon_n, title, desc_t, color, screen in cards_data:
            content.add_widget(ModeCard(
                icon_name   = icon_n,
                title       = title,
                desc        = desc_t,
                color       = color,
                screen_name = screen,
                nav_fn      = self.go_to,
            ))

        # ── Niveles ───────────────────────────────────────────────────────────
        content.add_widget(Label(
            text        = "Niveles disponibles",
            font_size   = FONT_SIZE["h3"],
            color       = COLORS["on_surface"],
            bold        = True,
            size_hint_y = None,
            height      = dp(36),
            halign      = "left",
        ))

        levels = [
            ("Principiante", (0.188, 0.722, 0.369, 1), "check"),
            ("Intermedio",   (1.0,   0.60,  0.0,   1), "check"),
            ("Avanzado",     (0.796, 0.094, 0.094, 1), "trophy"),
        ]
        grid = GridLayout(
            cols        = 3,
            size_hint_y = None,
            height      = dp(96),
            spacing     = dp(8),
        )
        for label, lcolor, icon_n in levels:
            chip = RoundedBox(
                bg_color    = (*lcolor[:3], 0.13),
                radius      = 12,
                orientation = "vertical",
                padding     = dp(8),
                spacing     = dp(4),
            )
            chip_icon_anchor = AnchorLayout(
                anchor_x  = "center",
                anchor_y  = "center",
                size_hint = (1, None),
                height    = dp(28),
            )
            chip_icon_anchor.add_widget(Icon(
                name      = icon_n,
                color     = [*lcolor[:3], 1],
                size_hint = (None, None),
                size      = (dp(22), dp(22)),
            ))
            chip.add_widget(chip_icon_anchor)
            chip.add_widget(Label(
                text        = label,
                font_size   = FONT_SIZE["caption"],
                color       = (*lcolor[:3], 1),
                bold        = True,
                halign      = "center",
                size_hint_y = None,
                height      = dp(24),
            ))
            grid.add_widget(chip)
        content.add_widget(grid)

        content.add_widget(make_spacer("24dp"))
        scroll.add_widget(content)
        root.add_widget(scroll)
        self.add_widget(root)