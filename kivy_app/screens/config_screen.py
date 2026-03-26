"""kivy_app/screens/config_screen.py — sin canvas.before.clear()"""
from __future__ import annotations

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.switch import Switch
from kivy.uix.textinput import TextInput

from api_client import check_api_health
from components.styled_box import ColorBox, RoundedBox
from components.icon import Icon
from config import COLORS, FONT_SIZE, DEFAULT_API_URL, load_user_config, save_user_config
from screens.base_screen import BaseScreen, make_topbar, make_section_header, make_spacer
from services.threading_utils import run_in_background


class ConfigScreen(BaseScreen):

    def on_enter(self):
        super().on_enter()
        self._cfg = load_user_config()
        self.clear_widgets()
        self._build_ui()

    def _build_ui(self):
        root = ColorBox(bg_color=COLORS["background"], orientation="vertical")
        root.add_widget(make_topbar("Configuración", on_back=lambda: self.go_to("home","right")))

        scroll = ScrollView()
        content = BoxLayout(orientation="vertical", padding="16dp", spacing="16dp", size_hint_y=None)
        content.bind(minimum_height=content.setter("height"))

        content.add_widget(make_section_header("Conexión a la API", icon_name="globe"))
        content.add_widget(self._url_section())
        content.add_widget(make_section_header("Preferencias", icon_name="gear"))
        content.add_widget(self._preferences())
        content.add_widget(make_section_header("Acerca de", icon_name="info"))
        content.add_widget(self._about())
        content.add_widget(make_spacer("24dp"))

        scroll.add_widget(content)
        root.add_widget(scroll)
        self.add_widget(root)

    def _url_section(self):
        card = RoundedBox(
            bg_color=(1,1,1,1), radius=12,
            orientation="vertical", size_hint_y=None, height="208dp",
            padding="16dp", spacing="10dp",
        )
        hint = Label(
            text="URL base del servidor FastAPI\nEjemplo: http://192.168.1.100:8000",
            font_size=FONT_SIZE["body_sm"], color=COLORS["text_secondary"],
            size_hint_y=None, height="40dp", halign="left",
        )
        hint.bind(size=hint.setter("text_size"))
        card.add_widget(hint)

        self._url_input = TextInput(
            text=self._cfg.get("api_url", DEFAULT_API_URL),
            font_size=FONT_SIZE["body"],
            size_hint_y=None, height="48dp", multiline=False,
            background_color=COLORS["surface"],
            foreground_color=COLORS["on_surface"],
            cursor_color=COLORS["primary"],
            padding=["12dp","10dp"],
        )
        card.add_widget(self._url_input)

        self._status_lbl = Label(
            text="○  Sin verificar",
            font_size=FONT_SIZE["body_sm"], color=COLORS["text_secondary"],
            size_hint_y=None, height="24dp", halign="left",
        )
        self._status_lbl.bind(size=self._status_lbl.setter("text_size"))
        card.add_widget(self._status_lbl)

        btn_row = BoxLayout(size_hint_y=None, height="44dp", spacing="8dp")
        btn_test = Button(
            text="  Probar conexión",
            font_size=FONT_SIZE["button"],
            background_color=COLORS["primary_light"],
            color=COLORS["on_primary"],
        )
        btn_test.add_widget(Icon(
            name="plug",
            color=list(COLORS["on_primary"]),
            size_hint=(None, None),
            size=("18dp", "18dp"),
            pos_hint={"x": 0.06, "center_y": 0.5},
        ))
        btn_test.bind(on_press=lambda _: self._test())
        btn_save = Button(
            text="  Guardar",
            font_size=FONT_SIZE["button"],
            background_color=COLORS["accent"],
            color=COLORS["on_primary"], bold=True,
        )
        btn_save.add_widget(Icon(
            name="save",
            color=list(COLORS["on_primary"]),
            size_hint=(None, None),
            size=("18dp", "18dp"),
            pos_hint={"x": 0.10, "center_y": 0.5},
        ))
        btn_save.bind(on_press=lambda _: self._save())
        btn_row.add_widget(btn_test); btn_row.add_widget(btn_save)
        card.add_widget(btn_row)
        return card

    def _preferences(self):
        card = RoundedBox(
            bg_color=(1,1,1,1), radius=12,
            orientation="vertical", size_hint_y=None, height="96dp",
            padding="16dp", spacing="8dp",
        )
        for label, key in [("Vibración al recibir resultado", "haptic_enabled"),
                            ("Sonido de resultado", "sound_enabled")]:
            row = BoxLayout(size_hint_y=None, height="36dp")
            row.add_widget(Label(
                text=label, font_size=FONT_SIZE["body_sm"],
                color=COLORS["on_surface"], halign="left",
            ))
            sw = Switch(active=self._cfg.get(key, True),
                        size_hint=(None,None), size=("80dp","36dp"))
            sw.bind(active=lambda _, v, k=key: self._set_pref(k, v))
            row.add_widget(sw)
            card.add_widget(row)
        return card

    def _about(self):
        card = RoundedBox(
            bg_color=(*COLORS["primary"][:3], 0.06), radius=12,
            orientation="vertical", size_hint_y=None, height="112dp",
            padding="16dp", spacing="4dp",
        )
        for text in [
            "Tutor de Caligrafía — v1.0",
            "Backend: FastAPI + YOLOv8 + EfficientNet-B2",
            "Frontend: Kivy (Android APK)",
            "Evaluación de trazos con IA",
        ]:
            card.add_widget(Label(
                text=text, font_size=FONT_SIZE["body_sm"],
                color=COLORS["on_surface"], halign="left",
            ))
        return card

    def _test(self):
        url = self._url_input.text.strip()
        self._status_lbl.text = "Verificando…"
        self._status_lbl.color = COLORS["text_secondary"]
        run_in_background(
            task=lambda: check_api_health(url),
            on_success=self._on_health,
        )

    def _on_health(self, result):
        ok, msg = result
        self._status_lbl.text  = f"OK — {msg}" if ok else f"Error — {msg}"
        self._status_lbl.color = COLORS["score_excellent"] if ok else COLORS["error"]

    def _save(self):
        cfg = load_user_config()
        cfg["api_url"] = self._url_input.text.strip()
        save_user_config(cfg)
        self.show_toast("Configuración guardada ✓")

    def _set_pref(self, key, value):
        cfg = load_user_config(); cfg[key] = value; save_user_config(cfg)