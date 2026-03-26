"""kivy_app/screens/result_screen.py — corregido"""
from __future__ import annotations

from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView

from components.styled_box import ColorBox, RoundedBox, StyledProgressBar
from components.icon import Icon
from config import COLORS, FONT_SIZE, get_score_category
from screens.base_screen import BaseScreen, make_topbar, make_spacer
from services.image_service import b64_to_texture_path, cleanup_temp_images


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


class ResultScreen(BaseScreen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._result      = {}
        self._mode        = "single"
        self._target_char = "?"

    def set_result(self, result, mode="single", target_char="?"):
        self._result      = result
        self._mode        = mode
        self._target_char = target_char

    def on_enter(self):
        super().on_enter()
        cleanup_temp_images()
        self.clear_widgets()
        if self._mode == "plana":
            self._build_plana()
        else:
            self._build_single()

    # ── Modo individual ───────────────────────────────────────────────────────

    def _build_single(self):
        r = self._result
        has_error = bool(r.get("error"))

        root = ColorBox(bg_color=COLORS["background"], orientation="vertical")
        root.add_widget(make_topbar("Resultado", on_back=lambda: self.go_to("capture", "right")))

        scroll = ScrollView()
        content = BoxLayout(orientation="vertical", padding="16dp", spacing="14dp", size_hint_y=None)
        content.bind(minimum_height=content.setter("height"))

        if has_error:
            content.add_widget(self._error_card(r["error"]))
        else:
            score_final = float(r.get("score_final", 0) or 0)
            score01, score100, cat, color = self._score_to_scale_and_color(score_final)

            # Score hero
            content.add_widget(self._score_hero(score01, score100, cat, color, r))

            # Feedback
            fb = r.get("feedback", "")
            if fb:
                content.add_widget(self._feedback_card(fb, color))

            # Imágenes
            content.add_widget(self._section("Tu trazo vs la guía", icon_name="image"))
            content.add_widget(self._images_row(r))

            # Métricas
            content.add_widget(self._section("Desglose de métricas", icon_name="chart"))
            content.add_widget(self._metrics_grid(r))

        content.add_widget(self._action_btns("capture"))
        content.add_widget(make_spacer("24dp"))
        scroll.add_widget(content)
        root.add_widget(scroll)
        self.add_widget(root)

    def _score_hero(self, score01, score100, cat, color, r):
        """Card principal con el score - SIN gauge circular"""
        hero = RoundedBox(
            bg_color=(*color[:3], 0.12),
            radius=16,
            orientation="vertical",
            size_hint_y=None,
            height="180dp",
            padding="16dp",
            spacing="10dp",
        )

        # ─── Porcentaje grande centrado ───
        score_lbl = Label(
            text=f"{score100:.1f}%",
            font_size="36sp",
            color=color,
            bold=True,
            halign="center",
            valign="middle",
            size_hint_y=None,
            height="50dp",
        )
        hero.add_widget(score_lbl)

        # ─── Barra de progreso lineal ───
        bar = StyledProgressBar(
            value=score01,
            fill_color=color,
            bg_color=(*COLORS["divider"][:3], 0.5),
            size_hint_y=None,
            height="14dp",
        )
        hero.add_widget(bar)

        # ─── Info: Detectado | Confianza | Nivel ───
        info_row = BoxLayout(spacing="8dp", size_hint_y=None, height="36dp")
        detected = r.get("detected_char") or self._target_char
        conf = float(r.get("confidence", 0) or 0)

        for txt in [
            f"Detectado: {detected}",
            f"Confianza: {int(conf * 100)}%",
            f"Nivel: {r.get('level', 'intermedio').capitalize()}",
        ]:
            lbl = Label(
                text=txt,
                font_size=FONT_SIZE["body_sm"],
                color=COLORS["text_secondary"],
                halign="center",
                valign="middle",
            )
            info_row.add_widget(lbl)
        hero.add_widget(info_row)

        # ─── Calificación (icono + texto) ───
        grades = {
            "excellent": "Excelente",
            "good": "Bien hecho",
            "fair": "Aceptable",
            "poor": "Sigue practicando",
        }
        grade_row = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height="32dp",
            spacing="8dp",
        )
        # Spacer izquierdo para centrar
        grade_row.add_widget(BoxLayout(size_hint_x=1))

        icon_name = "trophy" if cat == "excellent" else ("check" if cat in ("good", "fair") else "info")
        grade_row.add_widget(Icon(
            name=icon_name,
            color=list(color),
            size_hint=(None, None),
            size=("20dp", "20dp"),
            pos_hint={"center_y": 0.5},
        ))
        grade_row.add_widget(Label(
            text=grades.get(cat, ""),
            font_size=FONT_SIZE["h3"],
            color=color,
            bold=True,
            size_hint=(None, 1),
            width="150dp",
            halign="left",
            valign="middle",
        ))
        # Spacer derecho para centrar
        grade_row.add_widget(BoxLayout(size_hint_x=1))
        hero.add_widget(grade_row)

        return hero

    def _feedback_card(self, feedback, color):
        card = RoundedBox(
            bg_color=color,
            radius=12,
            orientation="horizontal",
            size_hint_y=None,
            height="80dp",
            padding="16dp",
            spacing="12dp",
        )
        card.add_widget(Icon(
            name="info",
            color=list(COLORS["on_primary"]),
            size_hint=(None, None),
            size=("22dp", "22dp"),
            pos_hint={"center_y": 0.5},
        ))
        lbl = Label(
            text=feedback,
            font_size=FONT_SIZE["body"],
            color=COLORS["on_primary"],
            halign="left",
            valign="middle",
        )
        lbl.bind(size=_safe_update_text_size)
        card.add_widget(lbl)
        return card

    def _images_row(self, r):
        grid = GridLayout(cols=3, size_hint_y=None, height="180dp", spacing="8dp")
        metrics = r.get("metrics") or {}
        student_b64 = (
            r.get("image_student_b64")
            or r.get("image_b64")
            or (metrics.get("image_b64") if isinstance(metrics, dict) else "")
        )
        template_b64 = r.get("template_b64") or (
            metrics.get("template_b64") if isinstance(metrics, dict) else ""
        )
        comparison_b64 = (
            r.get("comparison_b64")
            or (metrics.get("comparison_b64") if isinstance(metrics, dict) else "")
            or student_b64
        )

        imgs = [
            (student_b64 or "", "Tu trazo", "stu"),
            (template_b64 or "", "Guía", "tpl"),
            (comparison_b64 or "", "Comparación", "cmp"),
        ]
        for b64, label, pfx in imgs:
            cell = BoxLayout(orientation="vertical", spacing="4dp")
            path = b64_to_texture_path(b64, prefix=pfx)
            img_card = RoundedBox(
                bg_color=(1, 1, 1, 1),
                radius=8,
                size_hint_y=None,
                height="144dp",
                padding="4dp",
            )
            if path:
                img_card.add_widget(KivyImage(source=path, allow_stretch=True, keep_ratio=True))
            else:
                img_card.add_widget(Label(
                    text="—",
                    font_size="24sp",
                    color=COLORS["text_secondary"],
                ))
            cell.add_widget(img_card)
            cell.add_widget(Label(
                text=label,
                font_size=FONT_SIZE["caption"],
                color=COLORS["text_secondary"],
                size_hint_y=None,
                height="22dp",
                halign="center",
            ))
            grid.add_widget(cell)
        return grid

    def _metrics_grid(self, r):
        breakdown = r.get("scores_breakdown") or {}
        weights = r.get("weights_used") or {}
        if not isinstance(breakdown, dict):
            breakdown = {}
        if not isinstance(weights, dict):
            weights = {}

        if not breakdown:
            metrics = r.get("metrics") or {}
            if isinstance(metrics, dict):
                breakdown = self._breakdown_from_metrics(metrics)
                weights = {
                    "dt_precision": 0.30,
                    "dt_coverage": 0.20,
                    "topology": 0.20,
                    "ssim": 0.12,
                    "procrustes": 0.10,
                    "hausdorff": 0.04,
                    "trajectory": 0.02,
                    "cosine": 0.02,
                }
        if not breakdown:
            return Label(
                text="—",
                font_size=FONT_SIZE["body_sm"],
                color=COLORS["text_secondary"],
                size_hint_y=None,
                height="32dp",
            )

        rows_cfg = [
            ("dt_precision", "DT Precisión"),
            ("dt_coverage", "DT Cobertura"),
            ("topology", "Topología"),
            ("ssim", "SSIM"),
            ("procrustes", "Procrustes"),
            ("hausdorff", "Hausdorff"),
            ("trajectory", "Trayectoria"),
            ("cosine", "Coseno"),
        ]

        col = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            height=dp(44 * len(rows_cfg)),
            spacing="8dp",
        )

        for key, label in rows_cfg:
            try:
                val_pts = float(breakdown.get(key, 0) or 0)
            except Exception:
                val_pts = 0.0
            val01 = (val_pts / 100.0) if val_pts > 1.0 else val_pts
            _, _, mcolor = get_score_category(val01)

            row = BoxLayout(size_hint_y=None, height="36dp", spacing="8dp")

            # Label de la métrica
            lbl_metric = Label(
                text=label,
                font_size=FONT_SIZE["body_sm"],
                color=COLORS["on_surface"],
                size_hint=(0.32, 1),
                halign="left",
                valign="middle",
            )
            lbl_metric.bind(size=_safe_update_text_size)
            row.add_widget(lbl_metric)

            # Barra de progreso
            bar = StyledProgressBar(
                value=val01,
                fill_color=mcolor,
                bg_color=(*COLORS["divider"][:3], 0.5),
                size_hint=(0.50, None),
                height="12dp",
                pos_hint={"center_y": 0.5},
            )
            row.add_widget(bar)

            # Valor numérico
            row.add_widget(Label(
                text=f"{val_pts:.1f}",
                font_size=FONT_SIZE["caption"],
                color=COLORS["primary"],
                size_hint=(0.18, 1),
                bold=True,
                halign="right",
                valign="middle",
            ))
            col.add_widget(row)

        return col

    # ── Modo plana ────────────────────────────────────────────────────────────

    def _build_plana(self):
        r = self._result
        root = ColorBox(bg_color=COLORS["background"], orientation="vertical")
        root.add_widget(make_topbar("Resultado — Plana", on_back=lambda: self.go_to("plana", "right")))

        scroll = ScrollView()
        content = BoxLayout(
            orientation="vertical",
            padding="16dp",
            spacing="14dp",
            size_hint_y=None,
        )
        content.bind(minimum_height=content.setter("height"))

        if r.get("error"):
            content.add_widget(self._error_card(r["error"]))
        else:
            content.add_widget(self._plana_summary(r))
            content.add_widget(self._section("Detalle por carácter", icon_name="list"))
            for cr in r.get("results", []):
                content.add_widget(self._plana_char_row(cr))

        content.add_widget(self._action_btns("plana"))
        content.add_widget(make_spacer("24dp"))
        scroll.add_widget(content)
        root.add_widget(scroll)
        self.add_widget(root)

    def _plana_summary(self, r):
        avg_final = float(r.get("avg_score", 0) or 0)
        score01, score100, cat, color = self._score_to_scale_and_color(avg_final)
        card = RoundedBox(
            bg_color=(*color[:3], 0.12),
            radius=16,
            orientation="vertical",
            size_hint_y=None,
            height="130dp",
            padding="16dp",
            spacing="10dp",
        )
        card.add_widget(Label(
            text=f"Promedio: {score100:.1f}%",
            font_size=FONT_SIZE["h2"],
            color=color,
            bold=True,
            size_hint_y=None,
            height="36dp",
        ))
        card.add_widget(Label(
            text=f"Plantilla: «{r.get('template_char', '?')}»  |  {r.get('n_evaluated', 0)} calificados",
            font_size=FONT_SIZE["body_sm"],
            color=COLORS["text_secondary"],
            size_hint_y=None,
            height="24dp",
        ))
        card.add_widget(StyledProgressBar(
            value=score01,
            fill_color=color,
            bg_color=(*COLORS["divider"][:3], 0.5),
            size_hint_y=None,
            height="14dp",
        ))
        return card

    def _plana_char_row(self, cr):
        score_final = float(cr.get("score_final", 0) or 0)
        score01, score100, cat, color = self._score_to_scale_and_color(score_final)
        idx = cr.get("index", "?")

        row = RoundedBox(
            bg_color=(1, 1, 1, 1),
            radius=10,
            orientation="horizontal",
            size_hint_y=None,
            height="80dp",
            spacing="12dp",
            padding="12dp",
        )
        row.add_widget(Label(
            text=f"#{idx}\n{score100:.1f}%",
            font_size=FONT_SIZE["body_sm"],
            color=color,
            bold=True,
            size_hint=(None, 1),
            width="56dp",
            halign="center",
        ))

        info = BoxLayout(orientation="vertical", spacing="4dp")
        info.add_widget(Label(
            text=f"Detectado: {cr.get('detected_char', '?')}",
            font_size=FONT_SIZE["body"],
            color=COLORS["on_surface"],
            bold=True,
            halign="left",
        ))
        fb = cr.get("feedback", "")
        fb_lbl = Label(
            text=(fb[:80] + "…") if len(fb) > 80 else fb,
            font_size=FONT_SIZE["body_sm"],
            color=COLORS["text_secondary"],
            halign="left",
        )
        fb_lbl.bind(size=_safe_update_text_size)
        info.add_widget(fb_lbl)
        row.add_widget(info)

        comp = b64_to_texture_path(cr.get("comparison_b64", ""), prefix=f"pc{idx}")
        if comp:
            row.add_widget(KivyImage(
                source=comp,
                allow_stretch=True,
                keep_ratio=True,
                size_hint=(None, 1),
                width="64dp",
            ))
        return row

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _error_card(self, msg):
        card = RoundedBox(
            bg_color=(*COLORS["error"][:3], 0.1),
            radius=12,
            orientation="vertical",
            size_hint_y=None,
            height="100dp",
            padding="16dp",
            spacing="8dp",
        )
        title = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height="28dp",
            spacing="8dp",
        )
        title.add_widget(Icon(
            name="close",
            color=list(COLORS["error"]),
            size_hint=(None, None),
            size=("18dp", "18dp"),
        ))
        title.add_widget(Label(
            text="Error",
            font_size=FONT_SIZE["h3"],
            color=COLORS["error"],
            bold=True,
        ))
        card.add_widget(title)
        lbl = Label(
            text=msg,
            font_size=FONT_SIZE["body_sm"],
            color=COLORS["text_secondary"],
            halign="left",
        )
        lbl.bind(size=_safe_update_text_size)
        card.add_widget(lbl)
        return card

    def _action_btns(self, retry_screen):
        row = BoxLayout(size_hint_y=None, height="52dp", spacing="12dp")
        b1 = Button(
            text="  Intentar de nuevo",
            font_size=FONT_SIZE["button"],
            background_color=COLORS["primary"],
            color=COLORS["on_primary"],
            bold=True,
        )
        b1.add_widget(Icon(
            name="reload",
            color=list(COLORS["on_primary"]),
            size_hint=(None, None),
            size=("18dp", "18dp"),
            pos_hint={"x": 0.08, "center_y": 0.5},
        ))
        b1.bind(on_press=lambda _: self.go_to(retry_screen, "right"))

        b2 = Button(
            text="  Inicio",
            font_size=FONT_SIZE["button"],
            background_color=COLORS["surface"],
            color=COLORS["primary"],
        )
        b2.add_widget(Icon(
            name="home",
            color=list(COLORS["primary"]),
            size_hint=(None, None),
            size=("18dp", "18dp"),
            pos_hint={"x": 0.12, "center_y": 0.5},
        ))
        b2.bind(on_press=lambda _: self.go_to("home", "right"))
        row.add_widget(b1)
        row.add_widget(b2)
        return row

    def _section(self, text, icon_name: str | None = None):
        row = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height="32dp",
            spacing="8dp",
        )
        if icon_name:
            row.add_widget(Icon(
                name=icon_name,
                color=list(COLORS["primary"]),
                size_hint=(None, None),
                size=("20dp", "20dp"),
                pos_hint={"center_y": 0.5},
            ))
        lbl = Label(
            text=text,
            font_size=FONT_SIZE["body"],
            color=COLORS["on_surface"],
            bold=True,
            size_hint_y=None,
            height="32dp",
            halign="left",
            valign="middle",
        )
        lbl.bind(size=_safe_update_text_size)
        row.add_widget(lbl)
        return row

    def _breakdown_from_metrics(self, metrics: dict) -> dict:
        """Fallback para el esquema viejo que llega desde Render."""
        metrics = metrics or {}
        geo = metrics.get("geometric") or {}
        topo = metrics.get("topology") or {}
        seg_cos = metrics.get("segment_cosine") or {}

        topo_match = bool(topo.get("match", False))
        topology = 100.0 if topo_match else 30.0

        ssim = float(geo.get("ssim_score", 0) or 0)
        procrustes = float(geo.get("procrustes_score", 0) or 0)

        h_dist = float(geo.get("hausdorff", 999.0) or 999.0)
        HAUSDORFF_TOLERANCE = 5.0
        HAUSDORFF_FACTOR = 2.0
        adjusted = max(0.0, h_dist - HAUSDORFF_TOLERANCE)
        hausdorff = max(0.0, 100.0 - adjusted * HAUSDORFF_FACTOR)

        traj_err = float(metrics.get("trajectory_error", 0) or 0)
        SCORING_TRAJ_FACTOR = 3.0
        trajectory = max(0.0, 100.0 - traj_err * SCORING_TRAJ_FACTOR)

        cosine_score = seg_cos.get("score")
        if cosine_score is None:
            cosine_val = float(seg_cos.get("cosine", 0) or 0)
            cosine = max(0.0, min(100.0, cosine_val * 100.0))
        else:
            cosine = float(cosine_score or 0)

        dt_precision_raw = metrics.get("dt_precision_score") or metrics.get("dt_precision")
        dt_precision = float(dt_precision_raw or 0) if dt_precision_raw is not None else 0.0
        if dt_precision <= 1.0:
            dt_precision *= 100.0

        dt_coverage_raw = metrics.get("dt_coverage") or metrics.get("dt_coverage_ratio")
        dt_coverage = float(dt_coverage_raw or 0) if dt_coverage_raw is not None else 0.0
        if dt_coverage <= 1.0:
            dt_coverage *= 100.0

        def _clamp100(x: float) -> float:
            return max(0.0, min(100.0, float(x)))

        return {
            "dt_precision": _clamp100(dt_precision),
            "dt_coverage": _clamp100(dt_coverage),
            "topology": _clamp100(topology),
            "ssim": _clamp100(ssim),
            "procrustes": _clamp100(procrustes),
            "hausdorff": _clamp100(hausdorff),
            "trajectory": _clamp100(trajectory),
            "cosine": _clamp100(cosine),
        }

    def _score_to_scale_and_color(self, score_final: float) -> tuple[float, float, str, tuple]:
        """
        Backend devuelve `score_final` en 0..100.
        Kivy (get_score_category) asume 0..1.
        """
        score_final = float(score_final or 0)
        score100 = score_final if score_final > 1.0 else score_final * 100.0
        score01 = max(0.0, min(1.0, score100 / 100.0))
        cat, _, color = get_score_category(score01)
        return score01, score100, cat, color