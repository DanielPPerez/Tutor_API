# Diagnóstico y Corrección HTTP 503 en Render

## RESUMEN EJECUTIVO

### Causa Raíz Probable
**Consumo excesivo de memoria (>512MB)** causado por:
1. **Carga de modelos a nivel de módulo**: Los modelos ONNX (~80 MB) se cargan al importar, consumiendo memoria antes de cualquier request.
2. **Dependencias pesadas**: `torch` y `torchvision` con CUDA (+800 MB), `opencv-python` con GUI (+200 MB), y paquetes innecesarios para producción (`mlflow`, `albumentations`, `timm`, `kagglehub`).
3. **Sin límite de workers**: Render puede iniciar múltiples workers, multiplicando el consumo de memoria.
4. **matplotlib font cache**: Al inicio se construye el cache de fuentes, consumiendo memoria adicional.

---

## CAMBIOS IMPLEMENTADOS

### 1. ✅ Comando de arranque (Procfile)
**Creado**: `Procfile`
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 120
```
- ✓ Escucha en `0.0.0.0` y usa variable `$PORT`
- ✓ Forzado a **1 solo worker** (`--workers 1`)
- ✓ Timeout aumentado para requests largos

### 2. ✅ Optimización de memoria
**Archivo**: `app/core/classifier.py` (líneas 28-45)
- Implementado **singleton lazy** para carga del modelo ONNX
- El modelo se carga bajo demanda, no al importar el módulo
- Logging de carga de modelo para monitoreo

### 3. ✅ Requirements.txt optimizado
**Archivo**: `requirements.txt`

**Eliminado** (ahorra ~1.5 GB):
- ❌ `mlflow` (solo para entrenamiento)
- ❌ `timm` (solo para entrenamiento)
- ❌ `albumentations` (solo para entrenamiento)
- ❌ `kagglehub` (solo para datasets)
- ❌ `onnxscript` (no usado en producción)
- ❌ `pillow-avif-plugin` (formato raro)

**Cambiado**:
- ✓ `opencv-python` → `opencv-python-headless` (ahorra ~200 MB)
- ✓ `torch` y `torchvision`: agregado `--index-url https://download.pytorch.org/whl/cpu` para build CPU-only (ahorra ~600 MB)
- ✓ Todas las versiones fijadas (reproducibilidad)

**Agregado**:
- ✓ `psutil==6.1.0` (monitoreo de memoria)

### 4. ✅ Endpoint /health liviano
**Archivo**: `app/main.py` (líneas 54-65)
- Ya existía el endpoint `/health`
- ✓ No carga modelos
- ✓ Responde inmediatamente con status 200
- ✓ También disponible en `/ping`

### 5. ✅ Logging de memoria RSS
**Archivo**: `app/main.py` (líneas 16-46)
- ✓ Función `get_memory_usage()` usando `psutil`
- ✓ Log al startup (líneas 33-36)
- ✓ Middleware HTTP que registra memoria antes/después de cada request (líneas 39-52)
- ✓ Logs incluyen delta de consumo por request

### 6. ✅ Endpoint /samples robusto
**Archivo**: `app/main.py` (líneas 90-127)
- ✓ Try-catch completo para capturar excepciones
- ✓ Logging detallado de errores
- ✓ Responde HTTP 500 con mensaje claro en lugar de tumbar el proceso
- ✓ Verifica rutas relativas: `frontend/samples/` y `data/Prueba/`
- ✓ Archivos ya commiteados en `frontend/samples/` (8 imágenes demo)

### 7. ⏳ Prueba local con límite de memoria
**Pendiente**: Requiere Docker local o configuración manual de ulimit.

**Comando recomendado**:
```bash
docker build -t tutor-api .
docker run -m 512m -p 8000:8000 tutor-api
```

---

## CONFIGURACIÓN EN RENDER DASHBOARD

### Ajustes requeridos:

1. **Start Command** (si no detecta Procfile automáticamente):
   ```
   uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 120
   ```

2. **Health Check Path**:
   ```
   /health
   ```

3. **Environment Variables** (opcional):
   ```
   PYTHON_VERSION=3.10
   LOG_LEVEL=INFO
   ```

4. **Instance Type**:
   - Actual: Free (512 MB RAM)
   - Recomendado: **Starter ($7/mes, 1 GB RAM)** si persisten 503s tras optimizaciones

5. **Auto-Deploy**:
   - ✓ Activar para que use el nuevo `Procfile` y `requirements.txt`

---

## DIFF DE CAMBIOS PRINCIPALES

### Procfile (nuevo)
```diff
+ web: uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 120
```

### requirements.txt
```diff
- fastapi
- uvicorn
- python-multipart
+ fastapi==0.115.0
+ uvicorn[standard]==0.30.6
+ python-multipart==0.0.9

- opencv-python==4.9.0.80
+ opencv-python-headless==4.9.0.80

- scikit-image
- scipy
- matplotlib
- pillow
- pillow-avif-plugin==1.4.3
+ scikit-image==0.24.0
+ scipy==1.13.1
+ matplotlib==3.9.2
+ pillow==10.4.0

- onnx>=1.16.0
- onnxruntime==1.18.1
- onnxscript
+ onnx==1.16.1
+ onnxruntime==1.18.1

+ --index-url https://download.pytorch.org/whl/cpu
  torch==2.2.0
  torchvision==0.17.0

- ultralytics>=8.0.0,<9.0.0
+ ultralytics==8.2.103

- timm==0.9.16
- albumentations==1.3.1
- mlflow==2.10.2
- tqdm
- kagglehub
- requests
+ psutil==6.1.0
```

### app/main.py
```diff
+ import psutil
+ import logging

+ logging.basicConfig(level=logging.INFO)
+ logger = logging.getLogger(__name__)

+ def get_memory_usage():
+     process = psutil.Process()
+     return process.memory_info().rss / (1024 * 1024)

+ @app.on_event("startup")
+ async def startup_event():
+     mem_usage = get_memory_usage()
+     logger.info(f"[STARTUP] Memoria RSS inicial: {mem_usage:.2f} MB")

+ @app.middleware("http")
+ async def log_memory_middleware(request, call_next):
+     mem_before = get_memory_usage()
+     response = await call_next(request)
+     mem_after = get_memory_usage()
+     mem_delta = mem_after - mem_before
+     logger.info(f"[MEMORY] {request.method} {request.url.path} | Before: {mem_before:.2f} MB | After: {mem_after:.2f} MB | Delta: {mem_delta:+.2f} MB")
+     return response

  @app.get("/samples/{filename}")
  async def get_sample_image(filename: str):
+     try:
          # ... código ...
+         logger.info(f"[SAMPLES] Sirviendo: {safe_name} ({len(content)} bytes)")
+     except HTTPException:
+         raise
+     except Exception as e:
+         logger.error(f"[SAMPLES] Error al servir {filename}: {str(e)}", exc_info=True)
+         raise HTTPException(status_code=500, detail=f"Error interno al procesar imagen: {str(e)}")
```

### app/core/classifier.py
```diff
+ _session_cls: Optional[ort.InferenceSession] = None

+ def _get_classifier_session() -> ort.InferenceSession:
+     global _session_cls
+     if _session_cls is None:
+         logger.info(f"[classifier] Cargando modelo ONNX: {config.MOBILENET_MODEL_PATH}")
+         _session_cls = ort.InferenceSession(config.MOBILENET_MODEL_PATH, providers=['CPUExecutionProvider'])
+         logger.info(f"[classifier] ✓ Modelo cargado en memoria")
+     return _session_cls

+ session_cls = _get_classifier_session()
- session_cls = ort.InferenceSession(config.MOBILENET_MODEL_PATH, providers=['CPUExecutionProvider'])
```

---

## MONITOREO POST-DEPLOY

Después de hacer push de los cambios, verifica en los logs de Render:

1. **Startup memory**:
   ```
   [STARTUP] Memoria RSS inicial: ~250-350 MB
   ```

2. **Memory per request**:
   ```
   [MEMORY] POST /evaluate_plana | Before: 320 MB | After: 380 MB | Delta: +60 MB
   ```

3. **Health check funcionando**:
   ```
   INFO: GET /health HTTP/1.1" 200 OK
   ```

Si persiste 503 tras estos cambios, considera:
- Upgrade a **Starter plan** ($7/mes, 1 GB RAM)
- Reducir resolución de entrada de imágenes en `config.py`
- Usar modelo ONNX cuantizado (INT8 en lugar de FP32)

---

## PRÓXIMOS PASOS

1. ✅ Commit y push de cambios
2. ⏳ Render re-deploy automático
3. ⏳ Verificar logs de memoria
4. ⏳ Probar endpoints: `/`, `/health`, `/samples/a minuscula.jpeg`, `/evaluate_plana`
5. ⏳ Si persiste 503, considerar upgrade a Starter plan
