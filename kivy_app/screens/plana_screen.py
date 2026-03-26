"""kivy_app/screens/plana_screen.py — sin canvas.before.clear()"""
from __future__ import annotations

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.togglebutton import ToggleButton

from components.styled_box import ColorBox, RoundedBox
from config import COLORS, FONT_SIZE, LEVELS, LEVEL_LABELS, LEVEL_COLORS, load_user_config, save_user_config
from components.icon import Icon
from screens.base_screen import BaseScreen, make_topbar, make_primary_button, make_section_header, make_spacer
from services.camera_service import open_gallery


class PlanaScreen(BaseScreen):

    def on_enter(self):
        super().on_enter()
        self._cfg = load_user_config()
        self._image_path = None
        self.clear_widgets()
        self._build_ui()

    def _build_ui(self):
        root = ColorBox(bg_color=COLORS["background"], orientation="vertical")
        root.add_widget(make_topbar("Modo Plana", on_back=lambda: self.go_to("home","right")))

        scroll = ScrollView()
        content = BoxLayout(orientation="vertical", padding="16dp", spacing="14dp", size_hint_y=None)
        content.bind(minimum_height=content.setter("height"))

        # Descripción
        desc_card = RoundedBox(
            bg_color=(*COLORS["primary"][:3], 0.08), radius=12,
            orientation="vertical", size_hint_y=None, height="88dp",
            padding="16dp",
        )
        desc_lbl = Label(
            text="Fotografía una plana completa.\n"
                 "El primer carácter detectado será la plantilla;\n"
                 "los demás se calificarán contra él.",
            font_size=FONT_SIZE["body_sm"], color=COLORS["on_surface"],
            halign="left", valign="middle",
        )
        desc_lbl.bind(size=desc_lbl.setter("text_size"))
        desc_card.add_widget(desc_lbl)
        content.add_widget(desc_card)

        content.add_widget(make_section_header("Imagen de la plana", icon_name="image"))
        content.add_widget(self._image_area())
        content.add_widget(make_section_header("Nivel", icon_name="chart"))
        content.add_widget(self._level_selector())
        content.add_widget(make_spacer("8dp"))
        content.add_widget(make_primary_button("Evaluar plana  ▶", self._on_evaluate))
        content.add_widget(make_spacer("24dp"))

        scroll.add_widget(content)
        root.add_widget(scroll)
        self.add_widget(root)

    def _image_area(self):
        col = BoxLayout(orientation="vertical", size_hint_y=None, height="216dp", spacing="8dp")
        card = RoundedBox(
            bg_color=(1,1,1,1), radius=12,
            size_hint_y=None, height="164dp", padding="4dp",
        )
        self._ph = Label(
            text="Selecciona la foto de la plana",
            font_size=FONT_SIZE["body_sm"], color=COLORS["text_secondary"], halign="center",
        )
        self._prev = KivyImage(source="", allow_stretch=True, keep_ratio=True)
        card.add_widget(self._ph); card.add_widget(self._prev)
        self._card = card
        col.add_widget(card)
        btn = Button(text="  Seleccionar imagen",
                     font_size=FONT_SIZE["button"],
                     background_color=COLORS["primary_light"],
                     color=COLORS["on_primary"], bold=True,
                     size_hint_y=None, height="44dp")
        btn.add_widget(Icon(
            name="image",
            color=list(COLORS["on_primary"]),
            size_hint=(None, None),
            size=("18dp", "18dp"),
            pos_hint={"x": 0.08, "center_y": 0.5},
        ))
        btn.bind(on_press=lambda _: open_gallery(self._on_img))
        col.add_widget(btn)
        return col

    def _level_selector(self):
        grid = GridLayout(cols=3, size_hint_y=None, height="68dp", spacing="8dp")
        self._lvl_btns = {}
        cur = self._cfg.get("last_level","intermedio")
        for lvl in LEVELS:
            c = LEVEL_COLORS[lvl]
            btn = ToggleButton(
                text=LEVEL_LABELS[lvl], group="plana_level",
                state="down" if lvl==cur else "normal",
                font_size=FONT_SIZE["body_sm"],
                background_color=c if lvl==cur else COLORS["surface"],
                color=COLORS["on_primary"] if lvl==cur else COLORS["on_surface"],
            )
            btn.bind(on_press=lambda b, l=lvl: self._sel(l))
            self._lvl_btns[lvl] = btn; grid.add_widget(btn)
        return grid

    def _sel(self, level):
        for l, b in self._lvl_btns.items():
            b.background_color = LEVEL_COLORS[l] if l==level else COLORS["surface"]
            b.color = COLORS["on_primary"] if l==level else COLORS["on_surface"]

    def _get_level(self):
        for l, b in self._lvl_btns.items():
            if b.state == "down": return l
        return "intermedio"

    def _on_img(self, path):
        if not path: return
        self._image_path = path
        if self._ph.parent: self._card.remove_widget(self._ph)
        self._prev.source = path; self._prev.reload()

    def _on_evaluate(self):
        if not self._image_path:
            self.show_toast("⚠ Selecciona una imagen"); return
        level = self._get_level()
        cfg = load_user_config(); cfg["last_level"] = level; save_user_config(cfg)
        ev = self.manager.get_screen("evaluate")
        ev.set_params(image_path=self._image_path, level=level, mode="plana")
        self.go_to("evaluate")