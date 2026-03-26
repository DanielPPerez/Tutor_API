"""
kivy_app/screens/capture_screen.py
====================================
Pantalla de captura — versión corregida.
  • Iconos de sección centrados con AnchorLayout
  • Level-chips estilizados (RoundedBox) en lugar de ToggleButton plano
  • Botón "Seleccionar imagen" con icono centrado
  • Botón principal con feedback táctil
"""
from __future__ import annotations

from kivy.animation import Animation
from kivy.metrics import dp
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput

from components.icon import Icon
from components.styled_box import ColorBox, RoundedBox
from config import (
    COLORS, FONT_SIZE, LEVELS, LEVEL_LABELS, LEVEL_COLORS,
    load_user_config, save_user_config,
)
from screens.base_screen import BaseScreen, make_spacer
from services.camera_service import open_gallery


# ─── Helpers locales ──────────────────────────────────────────────────────────

class _IconBtn(ButtonBehavior, AnchorLayout):
    """AnchorLayout con comportamiento de botón (sin fondo nativo de Button)."""
    pass


def _section_header(text: str, icon_name: str) -> BoxLayout:
    """
    Encabezado de sección con icono centrado usando AnchorLayout.
    Reemplaza make_section_header para garantizar centrado correcto.
    """
    row = BoxLayout(
        orientation = "horizontal",
        size_hint_y = None,
        height      = dp(32),
        spacing     = dp(8),
    )
    icon_wrap = AnchorLayout(
        anchor_x  = "center",
        anchor_y  = "center",
        size_hint = (None, None),
        size      = (dp(24), dp(24)),
    )
    icon_wrap.add_widget(Icon(
        name      = icon_name,
        color     = list(COLORS["primary"]),
        size_hint = (None, None),
        size      = (dp(18), dp(18)),
    ))
    row.add_widget(icon_wrap)
    row.add_widget(Label(
        text      = text,
        font_size = FONT_SIZE["h3"],
        color     = COLORS["on_surface"],
        bold      = True,
        halign    = "left",
        valign    = "middle",
    ))
    return row


# ─── Level chip (reemplaza ToggleButton) ─────────────────────────────────────

class LevelChip(RoundedBox):
    """
    Chip de nivel seleccionable con animación de color.
    """
    def __init__(self, label: str, level_key: str, color, on_select, **kw):
        super().__init__(
            bg_color    = (*color[:3], 0.12),
            radius      = 10,
            orientation = "vertical",
            padding     = dp(8),
            spacing     = dp(4),
            **kw,
        )
        self._color      = color
        self._level_key  = level_key
        self._on_select  = on_select
        self._selected   = False

        self._icon_wrap = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (1, None),
            height    = dp(22),
        )
        self._icon = Icon(
            name      = "check",
            color     = [*color[:3], 0.0],   # invisible hasta seleccionar
            size_hint = (None, None),
            size      = (dp(16), dp(16)),
        )
        self._icon_wrap.add_widget(self._icon)
        self.add_widget(self._icon_wrap)

        self._lbl = Label(
            text      = label,
            font_size = FONT_SIZE["body_sm"],
            color     = (*color[:3], 0.7),
            bold      = False,
            halign    = "center",
        )
        self.add_widget(self._lbl)

    def set_selected(self, selected: bool):
        self._selected = selected
        if selected:
            Animation(bg_color=(*self._color[:3], 1.0), duration=0.15).start(self)
            self._lbl.color     = (1, 1, 1, 1)
            self._lbl.bold      = True
            self._icon.color    = [1, 1, 1, 1]
        else:
            Animation(bg_color=(*self._color[:3], 0.12), duration=0.15).start(self)
            self._lbl.color     = (*self._color[:3], 0.7)
            self._lbl.bold      = False
            self._icon.color    = [*self._color[:3], 0.0]

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            Animation(opacity=0.70, duration=0.07).start(self)
            touch.grab(self)
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            Animation(opacity=1.0, duration=0.15).start(self)
            if self.collide_point(*touch.pos):
                self._on_select(self._level_key)
            return True
        return super().on_touch_up(touch)


# ─── CaptureScreen ────────────────────────────────────────────────────────────

class CaptureScreen(BaseScreen):

    def on_enter(self):
        super().on_enter()
        self._cfg        = load_user_config()
        self._image_path = None
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
            padding     = [dp(8), dp(8)],
            spacing     = dp(8),
        )
        back_btn = _IconBtn(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (None, None),
            size      = (dp(44), dp(44)),
        )
        back_btn.add_widget(Icon(
            name      = "back",
            color     = list(COLORS["on_primary"]),
            size_hint = (None, None),
            size      = (dp(22), dp(22)),
        ))
        back_btn.bind(on_press=lambda _: self.go_to("home", "right"))
        topbar.add_widget(back_btn)

        topbar.add_widget(Label(
            text      = "Evaluar Carácter",
            font_size = FONT_SIZE["h3"],
            color     = COLORS["on_primary"],
            bold      = True,
            halign    = "left",
        ))
        root.add_widget(topbar)

        # ── Scroll content ────────────────────────────────────────────────────
        scroll  = ScrollView()
        content = BoxLayout(
            orientation = "vertical",
            padding     = dp(16),
            spacing     = dp(14),
            size_hint_y = None,
        )
        content.bind(minimum_height=content.setter("height"))

        content.add_widget(_section_header("Imagen del trazo", "camera"))
        content.add_widget(self._image_area())
        content.add_widget(_section_header("Carácter objetivo", "pencil"))
        content.add_widget(self._char_input_row())
        content.add_widget(_section_header("Nivel de dificultad", "chart"))
        content.add_widget(self._level_selector())
        content.add_widget(make_spacer("8dp"))
        content.add_widget(self._eval_button())
        content.add_widget(make_spacer("24dp"))

        scroll.add_widget(content)
        root.add_widget(scroll)
        self.add_widget(root)

    # ── Secciones ─────────────────────────────────────────────────────────────

    def _image_area(self):
        col = BoxLayout(
            orientation = "vertical",
            size_hint_y = None,
            height      = dp(236),
            spacing     = dp(8),
        )

        # Card de preview
        self._preview_card = RoundedBox(
            bg_color    = (1, 1, 1, 1),
            radius      = 14,
            size_hint_y = None,
            height      = dp(182),
            padding     = dp(4),
        )
        self._placeholder = Label(
            text      = "Toca 'Seleccionar imagen'\npara agregar una foto",
            font_size = FONT_SIZE["body_sm"],
            color     = COLORS["text_secondary"],
            halign    = "center",
        )
        self._preview_img = KivyImage(source="", allow_stretch=True, keep_ratio=True)
        self._preview_card.add_widget(self._placeholder)
        self._preview_card.add_widget(self._preview_img)
        col.add_widget(self._preview_card)

        # Botón seleccionar — RoundedBox táctil con icono centrado
        sel_btn = _IconBtn(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (1, None),
            height    = dp(46),
        )
        sel_bg = RoundedBox(
            bg_color = COLORS["primary"],
            radius   = 12,
        )
        sel_row = BoxLayout(
            orientation = "horizontal",
            spacing     = dp(8),
            size_hint   = (None, None),
            size        = (dp(210), dp(30)),
        )
        icon_a = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (None, None),
            size      = (dp(22), dp(22)),
        )
        icon_a.add_widget(Icon(
            name      = "image",
            color     = list(COLORS["on_primary"]),
            size_hint = (None, None),
            size      = (dp(18), dp(18)),
        ))
        sel_row.add_widget(icon_a)
        sel_row.add_widget(Label(
            text      = "Seleccionar imagen",
            font_size = FONT_SIZE["button"],
            color     = COLORS["on_primary"],
            bold      = True,
            halign    = "left",
            valign    = "middle",
        ))
        sel_bg.add_widget(AnchorLayout(anchor_x="center", anchor_y="center"))
        sel_btn.add_widget(sel_bg)
        # Overlay del texto+icon centrado sobre el bg
        overlay = AnchorLayout(anchor_x="center", anchor_y="center")
        overlay.add_widget(sel_row)
        sel_btn.add_widget(overlay)
        sel_btn.bind(on_press=lambda _: open_gallery(self._on_img_selected))

        # Efecto táctil
        def _press(inst, *_):
            Animation(opacity=0.75, duration=0.07).start(sel_bg)
        def _release(inst, *_):
            Animation(opacity=1.0, duration=0.15).start(sel_bg)
        sel_btn.bind(on_press=_press, on_release=_release)
        col.add_widget(sel_btn)
        return col

    def _char_input_row(self):
        row = BoxLayout(
            orientation = "horizontal",
            size_hint_y = None,
            height      = dp(68),
            spacing     = dp(12),
            padding     = dp(4),
        )
        hint = Label(
            text      = "Escribe el carácter\nque intentaste trazar:",
            font_size = FONT_SIZE["body_sm"],
            color     = COLORS["text_secondary"],
            size_hint = (0.6, 1),
            halign    = "left",
        )
        hint.bind(size=hint.setter("text_size"))
        self._char_input = TextInput(
            text             = self._cfg.get("last_char", "A"),
            font_size        = "28sp",
            size_hint        = (None, None),
            size             = (dp(74), dp(58)),
            multiline        = False,
            halign           = "center",
            background_color = COLORS["surface"],
            foreground_color = COLORS["on_surface"],
            cursor_color     = COLORS["primary"],
            padding          = [dp(8), dp(8)],
        )
        self._char_input.bind(
            text=lambda _, v: setattr(self._char_input, "text", v[-1]) if len(v) > 1 else None
        )
        row.add_widget(hint)
        row.add_widget(self._char_input)
        return row

    def _level_selector(self):
        grid = GridLayout(
            cols        = 3,
            size_hint_y = None,
            height      = dp(76),
            spacing     = dp(8),
        )
        self._level_chips: dict[str, LevelChip] = {}
        cur = self._cfg.get("last_level", "intermedio")

        for lvl in LEVELS:
            chip = LevelChip(
                label     = LEVEL_LABELS[lvl],
                level_key = lvl,
                color     = LEVEL_COLORS[lvl],
                on_select = self._select_level,
            )
            chip.set_selected(lvl == cur)
            self._level_chips[lvl] = chip
            grid.add_widget(chip)
        return grid

    def _eval_button(self):
        """Botón principal de acción con feedback táctil."""
        wrap = _IconBtn(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (1, None),
            height    = dp(52),
        )
        bg = RoundedBox(bg_color=COLORS["primary"], radius=14)
        row = BoxLayout(
            orientation = "horizontal",
            spacing     = dp(10),
            size_hint   = (None, None),
            size        = (dp(220), dp(32)),
        )
        icon_a = AnchorLayout(
            anchor_x  = "center",
            anchor_y  = "center",
            size_hint = (None, None),
            size      = (dp(24), dp(24)),
        )
        icon_a.add_widget(Icon(
            name      = "pencil",
            color     = list(COLORS["on_primary"]),
            size_hint = (None, None),
            size      = (dp(20), dp(20)),
        ))
        row.add_widget(icon_a)
        row.add_widget(Label(
            text      = "Evaluar mi trazo",
            font_size = FONT_SIZE["button"],
            color     = COLORS["on_primary"],
            bold      = True,
            halign    = "left",
            valign    = "middle",
        ))
        wrap.add_widget(bg)
        overlay = AnchorLayout(anchor_x="center", anchor_y="center")
        overlay.add_widget(row)
        wrap.add_widget(overlay)

        def _press(*_):
            anim = Animation(opacity=0.75, duration=0.07)
            anim &= Animation(bg_color=(*COLORS["primary"][:3], 0.82), duration=0.07)
            anim.start(bg)
        def _release(*_):
            anim = Animation(opacity=1.0, duration=0.18)
            anim &= Animation(bg_color=COLORS["primary"], duration=0.12)
            anim.start(bg)
        wrap.bind(on_press=_press, on_release=_release)
        wrap.bind(on_press=lambda _: self._on_evaluate())
        self._eval_btn = wrap
        return wrap

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _select_level(self, level: str):
        for key, chip in self._level_chips.items():
            chip.set_selected(key == level)

    def _get_level(self) -> str:
        for key, chip in self._level_chips.items():
            if chip._selected:
                return key
        return "intermedio"

    def _on_img_selected(self, path):
        if not path:
            return
        self._image_path = path
        if self._placeholder.parent:
            self._preview_card.remove_widget(self._placeholder)
        self._preview_img.source = path
        self._preview_img.reload()
        self.show_toast("Imagen cargada ✓")

    def _on_evaluate(self):
        if not self._image_path:
            self.show_toast("⚠ Selecciona una imagen primero")
            return
        char = self._char_input.text.strip()
        if not char:
            self.show_toast("⚠ Ingresa el carácter objetivo")
            return
        level = self._get_level()
        cfg = load_user_config()
        cfg["last_char"]  = char
        cfg["last_level"] = level
        save_user_config(cfg)
        ev = self.manager.get_screen("evaluate")
        ev.set_params(image_path=self._image_path, target_char=char, level=level, mode="single")
        self.go_to("evaluate")