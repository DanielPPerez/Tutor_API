import os
import psutil
import logging
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.api.endpoints import router

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aprendia Edge Backend")

def get_memory_usage():
    """Retorna el uso de memoria RSS en MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

@app.on_event("startup")
async def startup_event():
    """Log de memoria al iniciar la aplicación."""
    mem_usage = get_memory_usage()
    logger.info(f"[STARTUP] Memoria RSS inicial: {mem_usage:.2f} MB")


@app.middleware("http")
async def log_memory_middleware(request, call_next):
    """Middleware para logging de memoria en cada request."""
    mem_before = get_memory_usage()
    response = await call_next(request)
    mem_after = get_memory_usage()
    mem_delta = mem_after - mem_before
    
    logger.info(
        f"[MEMORY] {request.method} {request.url.path} | "
        f"Before: {mem_before:.2f} MB | After: {mem_after:.2f} MB | "
        f"Delta: {mem_delta:+.2f} MB"
    )
    return response


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
    try:
        safe_name = os.path.basename(filename)

        # 1. Buscar en frontend/samples
        file_path = os.path.join(SAMPLES_DIR, safe_name)
        if not os.path.exists(file_path):
            # 2. Buscar en data/Prueba
            file_path_alt = os.path.join(PRUEBA_DIR, safe_name)
            if os.path.exists(file_path_alt):
                file_path = file_path_alt
            else:
                logger.warning(f"[SAMPLES] Archivo no encontrado: {safe_name}")
                raise HTTPException(
                    status_code=404, 
                    detail=f"Muestra '{safe_name}' no encontrada en frontend/samples ni data/Prueba"
                )

        with open(file_path, "rb") as f:
            content = f.read()

        ext = safe_name.lower().split(".")[-1]
        media_type = "image/png" if ext == "png" else "image/jpeg"
        logger.info(f"[SAMPLES] Sirviendo: {safe_name} ({len(content)} bytes)")
        return Response(content=content, media_type=media_type)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SAMPLES] Error al servir {filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al procesar imagen: {str(e)}"
        )


# Montar estáticos de frontend si están disponibles
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)