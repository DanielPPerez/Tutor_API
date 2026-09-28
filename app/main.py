import os
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.api.endpoints import router

app = FastAPI(title="Aprendia Edge Backend")

# Define los orígenes permitidos explícitamente y soporte para CORS total
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Directorios de frontend y recursos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
FRONTEND_INDEX = os.path.join(FRONTEND_DIR, "index.html")
SAMPLES_DIR = os.path.join(FRONTEND_DIR, "samples")
PRUEBA_DIR = os.path.join(BASE_DIR, "data", "Prueba")


# ── Ruta principal: servir la interfaz web ────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Sirve la aplicación de frontend interactiva."""
    if os.path.exists(FRONTEND_INDEX):
        with open(FRONTEND_INDEX, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        "<h2>Frontend no encontrado. Asegúrate de que existe frontend/index.html</h2>",
        status_code=404,
    )


# ── Endpoints de Keep-Alive / Health Check ─────────────────────────────────────
@app.get("/health")
@app.get("/ping")
async def health_check():
    """
    Endpoint ligero para verificar disponibilidad y mantener la instancia
    despierta en Render mediante servicios de monitoreo o cron jobs (ej. cron-job.org / UptimeRobot).
    """
    return {
        "status": "online",
        "service": "Aprendia Edge — Tutor API",
        "version": "4.2",
    }


# ── Servir imágenes de muestra para pruebas y demo ────────────────────────────
@app.get("/samples/{filename}")
async def get_sample_image(filename: str):
    """
    Entrega las imágenes de muestra de caligrafía para demostración rápida en el frontend.
    """
    safe_name = os.path.basename(filename)

    # 1. Buscar en frontend/samples
    file_path = os.path.join(SAMPLES_DIR, safe_name)
    if not os.path.exists(file_path):
        # 2. Buscar en data/Prueba
        file_path_alt = os.path.join(PRUEBA_DIR, safe_name)
        if os.path.exists(file_path_alt):
            file_path = file_path_alt
        else:
            raise HTTPException(status_code=404, detail=f"Muestra '{safe_name}' no encontrada")

    with open(file_path, "rb") as f:
        content = f.read()

    ext = safe_name.lower().split(".")[-1]
    media_type = "image/png" if ext == "png" else "image/jpeg"
    return Response(content=content, media_type=media_type)


# Montar estáticos de frontend si están disponibles
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)