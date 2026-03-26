"""
kivy_app/utils/smoke_test.py
=============================
Prueba rápida de la app sin necesitar Kivy ni la API corriendo.
Verifica:
  1. Imports de módulos no-Kivy
  2. Config carga y persiste correctamente
  3. api_client.evaluate_image() devuelve estructura de error correcta
     cuando la API no está disponible (no crash)
  4. image_service.b64_to_texture_path() maneja base64 inválido sin crash
  5. threading_utils.run_in_background() ejecuta task y callback

Ejecución:
    python utils/smoke_test.py
"""

import os, sys, json, base64, time, threading

# Agregar raíz de la app al path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

PASS = "\u2705"
FAIL = "\u274c"
results = []

def test(name, fn):
    try:
        fn()
        results.append((True, name))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((False, name))
        print(f"  {FAIL} {name}: {e}")


print("\n=== Smoke Test — Tutor de Caligrafía App ===\n")

# ── 1. Config ─────────────────────────────────────────────────────────────────
print("1. Módulo config")

def t_config_load():
    from config import load_user_config, COLORS, FONT_SIZE, LEVELS
    cfg = load_user_config()
    assert "api_url"    in cfg
    assert "last_level" in cfg
    assert len(COLORS)  > 5
    assert len(LEVELS)  == 3

def t_config_save():
    from config import load_user_config, save_user_config
    cfg = load_user_config()
    cfg["_smoke_test"] = True
    save_user_config(cfg)
    cfg2 = load_user_config()
    assert cfg2.get("_smoke_test") is True
    # Cleanup
    cfg2.pop("_smoke_test")
    save_user_config(cfg2)

def t_score_category():
    from config import get_score_category
    cat, msg, color = get_score_category(0.90)
    assert cat == "excellent"
    assert len(color) == 4
    cat2, _, _ = get_score_category(0.0)
    assert cat2 == "poor"

test("load_user_config()", t_config_load)
test("save_user_config()", t_config_save)
test("get_score_category()", t_score_category)

# ── 2. api_client ─────────────────────────────────────────────────────────────
print("\n2. Módulo api_client")

def t_api_error_response():
    """Sin servidor corriendo: debe devolver dict con 'error', nunca crashear."""
    from api_client import evaluate_image
    # Imagen inexistente — espera ConnectionError manejado
    result = evaluate_image(
        image_path  = "/tmp/nonexistent_test_img.jpg",
        target_char = "A",
        level       = "intermedio",
        base_url    = "http://127.0.0.1:19999",   # puerto que no existe
    )
    assert isinstance(result, dict), "Debe devolver dict"
    assert "error" in result or "score_final" in result, "Estructura inválida"
    # No debe tener campos faltantes críticos
    assert "score_final" in result
    assert "feedback"    in result

def t_api_plana_error():
    from api_client import evaluate_plana
    result = evaluate_plana(
        image_path = b"not_an_image",
        level      = "intermedio",
        base_url   = "http://127.0.0.1:19999",
    )
    assert isinstance(result, dict)
    assert "error" in result or "results" in result

def t_health_check():
    from api_client import check_api_health
    ok, msg = check_api_health("http://127.0.0.1:19999")
    assert isinstance(ok,  bool)
    assert isinstance(msg, str)
    assert ok is False   # servidor no corre

test("evaluate_image() maneja server-down", t_api_error_response)
test("evaluate_plana() maneja server-down", t_api_plana_error)
test("check_api_health() server-down",      t_health_check)

# ── 3. image_service ──────────────────────────────────────────────────────────
print("\n3. Módulo image_service")

def t_b64_invalid():
    from services.image_service import b64_to_texture_path
    result = b64_to_texture_path("not_valid_base64!!!", prefix="test")
    assert result is None, "Base64 inválido debe devolver None"

def t_b64_valid_png():
    from services.image_service import b64_to_texture_path
    # PNG mínimo válido 1×1 pixel
    png_1x1 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDQADMA"
        "HCBC7N4QAAAABJRU5ErkJggg=="
    )
    path = b64_to_texture_path(png_1x1, prefix="smoke")
    assert path is not None, "Base64 válido debe devolver ruta"
    assert os.path.exists(path), f"Archivo temporal debe existir: {path}"

def t_cleanup():
    from services.image_service import cleanup_temp_images
    cleanup_temp_images()   # No debe crashear aunque no haya archivos

test("b64_to_texture_path() base64 inválido → None", t_b64_invalid)
test("b64_to_texture_path() PNG válido → path",       t_b64_valid_png)
test("cleanup_temp_images() sin crash",                t_cleanup)

# ── 4. threading_utils (sin Kivy Clock) ──────────────────────────────────────
print("\n4. Módulo threading_utils (mock Clock)")

def t_threading_mock():
    """
    threading_utils usa Clock.schedule_once. Lo mockeamos para testear
    sin Kivy activo.
    """
    # Mock kivy.clock antes de importar
    import types
    mock_kivy = types.ModuleType("kivy")
    mock_clock = types.ModuleType("kivy.clock")

    results_cb = []

    class MockClock:
        @staticmethod
        def schedule_once(fn, delay):
            fn(delay)   # Ejecutar sincrónicamente en test

    mock_clock.Clock = MockClock
    sys.modules.setdefault("kivy",       mock_kivy)
    sys.modules.setdefault("kivy.clock", mock_clock)

    # Importar DESPUÉS del mock
    if "services.threading_utils" in sys.modules:
        del sys.modules["services.threading_utils"]

    from services.threading_utils import run_in_background

    def my_task():
        return {"ok": True, "value": 42}

    def my_success(result):
        results_cb.append(result)

    t = run_in_background(my_task, my_success)
    t.join(timeout=3.0)
    time.sleep(0.1)   # Dar tiempo al Clock mock

    assert len(results_cb) == 1, f"Callback no llamado: {results_cb}"
    assert results_cb[0]["value"] == 42

test("run_in_background() task → callback", t_threading_mock)

# ── Resumen ────────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for ok, _ in results if ok)
failed = total - passed

print(f"\n{'='*45}")
print(f"Resultado: {passed}/{total} tests pasaron")
if failed:
    print(f"\nFallidos:")
    for ok, name in results:
        if not ok: print(f"  {FAIL} {name}")
print(f"{'='*45}\n")

sys.exit(0 if failed == 0 else 1)