# Plan de Refactorización, Pruebas y Despliegue - Tutor Inteligente de Caligrafía

## Objetivo
Implementar la instalación de habilidades (`qaskills`, `playwright-skill`, `aesthetic-frontend-skills`), organización de arquitectura, limpieza de notebooks, creación de scripts de pruebas automatizadas de rendimiento (accuracy, letras difíciles, latencias, cuellos de botella de pipelines, vocales con acento, y caracteres especiales como **ñ** y **ch**), reporte en Markdown, unificación de API e interfaz en el mismo repositorio con paleta de colores corporativa (Rojo `#a12336`, Negro `#000000`, Verde `#009887`, Beige `#d3c2b3` y Blanco `#ffffff`), actualización de interfaz (soporte plana vs letra individual, calificaciones enteras) y mitigación de delay en Render (`https://tutor-api-bn0p.onrender.com`).

---

## Fases de Implementación

### Fase 1: Instalación de Skills y Limpieza de Notebooks
1. **Instalación de Skills**:
   - `https://github.com/PramodDutta/qaskills`
   - `playwright-skill`
   - `https://github.com/alexiseverage/aesthetic-frontend-skills`
2. **Depuración de la carpeta `Notebooks/`**:
   - Inspeccionar fechas de modificación y contenidos de los notebooks en `Notebooks/`.
   - Identificar el notebook de entrenamiento más reciente y correcto para el Detector (YOLOv8) y el Clasificador (EfficientNetV2-S + ArcFace v5).
   - Conservar exclusivamente dichos notebooks en la carpeta `Notebooks/` y eliminar los obsoletos.

### Fase 2: Reorganización de la Arquitectura de Carpetas e Integración Frontend-Backend
1. Reestructurar carpetas de acuerdo a estándares limpios:
   - `app/core/`: Lógica de negocio (detector, clasificador, binarizador, normalizador, etc.).
   - `app/api/`: Endpoints de FastAPI.
   - `app/scripts/`: Scripts de pruebas y evaluación.
   - `app/models/`: Modelos y artefactos (`classifier_artifacts`).
   - `frontend/`: Interfaz web unificada en el mismo repositorio junto a la API, aplicando la paleta de colores requerida (Rojo `#a12336`, Negro `#000000`, Verde `#009887`, Beige `#d3c2b3`, Blanco `#ffffff`).
   - `tests/`: Pruebas unitarias e integración centralizadas.

### Fase 3: Pruebas Automatizadas y Reporte de Rendimiento (Vocales, Ñ y Ch)
1. **Crear script de evaluación integral (`app/scripts/evaluate_performance.py`)** para medir automáticamente:
   - Porcentaje de accuracy global y por categoría.
   - Identificación de letras más difíciles (top confused pairs).
   - Tiempo de análisis por letra (latencia).
   - Identificación del proceso de pipeline más lento (profiling).
   - Verificación específica del reconocimiento correcto de caracteres especiales del español: vocales con acento, **ñ** y **ch**.
2. **Generar reporte completo** en un archivo `reporte_rendimiento_modelo.md`.

### Fase 4: Actualización de la Interfaz y Puntuaciones Enteras
1. **Puntuaciones Enteras**: Asegurar que todas las notas y puntajes devueltos por el backend y mostrados en la interfaz sean números enteros (`int`, redondeados de 0 a 100).
2. **Funcionalidades en Interfaz**: Actualizar la interfaz web para soportar explícitamente:
   - Análisis de plana completa (`/evaluate_plana`).
   - Análisis de carácter individual (`/evaluate`).

### Fase 5: Mitigación de Delay en Render (Keep-Alive / Wake-up)
1. Configurar un mecanismo en la interfaz web que realice un ping automático o evento de activación al cargar la página en el dominio `https://tutor-api-bn0p.onrender.com` para despertar la instancia gratuita antes de que el usuario interactúe, evitando el retraso de ~50 segundos.

---

## Criterios de Validación
- Skills instaladas correctamente.
- Pruebas automatizadas ejecutadas exitosamente con `pytest` y Playwright.
- Reporte `reporte_rendimiento_modelo.md` generado con métricas reales incluyendo **ñ** y **ch**.
- API e interfaz integradas en el repositorio con la paleta de colores especificada.
- Endpoints y UI respondiendo con puntajes enteros y soportando plana vs carácter individual.
- Mecanismo de keep-alive implementado para Render.
