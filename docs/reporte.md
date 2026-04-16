# Tutor Inteligente de Caligrafía

Documento técnico de continuidad para entender, ejecutar, operar y extender el proyecto sin transferencia verbal.

---

## 1) Resumen ejecutivo

El proyecto evalúa caligrafía manuscrita en imágenes usando:
- API en `FastAPI` (`app/main.py`, `app/api/endpoints.py`)
- Detector de caracteres YOLO ONNX (`best_detector.onnx`)
- Clasificador OCR ONNX (`best_classifier.onnx`)
- Pipeline de normalización robusta para fotos reales de cuaderno
- Cliente de escritorio en `Kivy` (`kivy_app/`)

Casos de uso principales:
- Evaluación de un carácter: `POST /evaluate`
- Evaluación de plana: `POST /evaluate_plana`
- Reconocimiento libre de texto: `POST /recognize`

---

## 2) Estado real del repositorio

### Stack técnico confirmado
- Python: `3.10.11`
- Backend: `fastapi`, `uvicorn`, `python-multipart`
- Inferencia: `onnxruntime==1.18.1`
- CV: `opencv-python-headless==4.9.0.80`, `scikit-image`, `scipy`, `pillow`
- Cliente escritorio: `kivy==2.3.0`

### Estructura funcional mínima
- `app/main.py`: inicializa FastAPI y CORS
- `app/api/endpoints.py`: define `/evaluate`, `/evaluate_plana`, `/recognize`
- `app/core/processor.py`: detección + clasificación + preprocesado robusto (incluye fallbacks)
- `app/core/normalizer.py`: normalización a máscara binaria final
- `app/core/config.py`: rutas, umbrales, pesos y niveles
- `app/metrics/*.py`: cálculo de métricas de trazo
- `app/utils/visualizer.py`: imagen comparativa base64
- `app/models/classifier_artifacts/`: artefactos ONNX y reportes de entrenamiento
- `kivy_app/`: app de escritorio

---

## 3) Requisitos y setup

### Prerrequisitos
- Python `3.10+`
- `pip`
- Espacio suficiente para modelos y dependencias (~2 GB recomendado)

### Instalación backend
```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\activate
pip install -r requirements.txt
```

### Verificación de artefactos críticos
Deben existir:
- `app/models/classifier_artifacts/best_detector.onnx`
- `app/models/classifier_artifacts/best_classifier.onnx`
- `app/models/classifier_artifacts/best_classifier.onnx.data`

Si falta alguno, la API levantará pero fallará en inferencia.

### Generación de plantillas (primera vez o al cambiar fuente/alfabeto)
```bash
python -m app.scripts.generate_templates
```

Genera plantillas en `app/templates/` por nivel y esqueletos.

---

## 4) Ejecución del sistema

### API (desarrollo)
```bash
uvicorn app.main:app --reload
```

### API (host explícito)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Documentación interactiva:
- `http://localhost:8000/docs`

### Cliente Kivy (opcional)
```bash
cd kivy_app
pip install -r requirements.txt
python main.py
```

---

## 5) Endpoints y contrato de uso

## `POST /evaluate`
Evalúa un único carácter contra una plantilla esperada.

Form-data:
- `file` (requerido)
- `target_char` (requerido)
- `level` (opcional, default `intermedio`)

Niveles válidos reales en código (`config.TEMPLATE_DIFFICULTY_KERNELS`):
- `principiante`
- `intermedio`
- `avanzado`

Retorna (campos clave):
- `score_final`, `scores_breakdown`, `weights_used`
- `feedback`
- `metadata`
- `metrics_extra`
- `image_student_b64`, `template_b64`, `comparison_b64`

---

## `POST /evaluate_plana`
Evalúa una imagen con múltiples caracteres.

Form-data:
- `file` (requerido)
- `target_char` (opcional, default `""`)
- `level` (opcional, default `intermedio`)

Flujo:
- Usa `preprocess_multi` con SmartOCR
- Si falla, aplica fallback a detección YOLO directa
- Primer carácter detectado se usa como plantilla (o `target_char` si se envía)

Retorna (campos clave):
- `template_char`, `template_confidence`
- `n_detected`, `n_evaluated`, `avg_score`
- `smart_ocr` (texto, palabras, líneas, confianza)
- `results` (lista por carácter evaluado)

---

## `POST /recognize`
Reconocimiento libre sin evaluación de calidad.

Form-data:
- `file` (requerido)

Retorna:
- `text`, `n_detected`, `confidence`
- `words`, `lines`, `characters`

También implementa fallback cuando falla `preprocess_multi`.

---

## 6) Pipeline de procesamiento (operativo)

1. Decodificación de imagen y detección de cajas de caracteres.
2. Recorte por detección y normalización robusta (`normalizer.py`).
3. Binarización + limpieza + corrección geométrica.
4. Salida de máscara en `TARGET_SIZE=128`.
5. Esqueletización de plantilla y alumno.
6. Cálculo de métricas de similitud.
7. Cálculo de nota final y feedback.
8. Render de comparación para retorno visual.

Notas importantes de precisión:
- El tamaño objetivo real del pipeline es `128x128`, no `224x224`.
- El nivel de dificultad modifica tolerancias de `Distance Transform`.

---

## 7) Métricas y scoring (alineado a `config.py`)

Componentes usados en score:
- `dt_precision`: 0.30
- `dt_coverage`: 0.20
- `topology`: 0.20
- `ssim`: 0.12
- `procrustes`: 0.10
- `hausdorff`: 0.04
- `trajectory`: 0.02
- `cosine`: 0.02

Tolerancias DT por nivel:
- `principiante`: 8.0
- `intermedio`: 5.0
- `avanzado`: 3.0

Topología:
- Match: `100`
- Mismatch: `30`

---

## 8) Modelo OCR y dataset

Fuente: `app/models/classifier_artifacts/train_config.json` y `metrics_report.json`

Resumen:
- Arquitectura: `tf_efficientnetv2_s + ProjectionHead + ArcFace v5`
- Clases: `107`
- Entrada: `128x128`
- Dataset train: `99,354` imágenes
- Métricas globales:
  - `best_val_acc`: `0.8126`
  - `test_acc`: `0.8097`
  - `weighted_f1`: `0.8093`

Interpretación recomendada:
- Priorizar `real_test_acc` (`0.7934`) para expectativas en producción.
- `synth_test_acc` es útil, pero optimista por naturaleza del set sintético.

---

## 9) Operación y troubleshooting

### Error: "Nivel inválido"
Causa: uso de `basico` en lugar de `principiante`.
Acción: enviar uno de `principiante|intermedio|avanzado`.

### Error: "No existe plantilla..."
Causa: plantillas no generadas.
Acción: ejecutar `python -m app.scripts.generate_templates`.

### API levanta pero no detecta/clasifica
Revisar:
- rutas de artefactos ONNX en `app/core/config.py`
- presencia física de modelos en `app/models/classifier_artifacts/`
- calidad de imagen de entrada (enfoque, contraste, oclusión)

### `evaluate_plana` con detecciones inconsistentes
El endpoint ya tiene fallback a YOLO directo; verificar:
- imágenes con caracteres más separados
- iluminación sin sombras duras
- resolución suficiente

---

## 10) Limitaciones actuales

- Sin Docker (`Dockerfile` y `docker-compose` no presentes).
- Sin persistencia en base de datos.
- Evaluación centrada en similitud geométrica, no en legibilidad semántica completa.
- Sensible a casos extremos: trazos muy tenues, oclusión, superposición fuerte de caracteres.

---

## 11) Continuidad del proyecto (handoff checklist)

Antes de entregar a otra persona, validar:
- [ ] `uvicorn app.main:app --reload` inicia sin errores
- [ ] `/docs` visible y funcional
- [ ] `/evaluate` responde con imagen de prueba
- [ ] `/evaluate_plana` responde con `results` y `smart_ocr`
- [ ] `/recognize` devuelve texto y caracteres
- [ ] Plantillas generadas en `app/templates/`
- [ ] Artefactos ONNX presentes
- [ ] Dependencias instalables desde `requirements.txt`

Siguiente documentación recomendada:
- Manual de pruebas con casos reales y criterios de aceptación.
- Versionado de artefactos de modelo (checksum + fecha + origen).
- Guía de reentrenamiento reproducible (dataset, seeds, comandos, export ONNX).

---

## 12) Enlaces

- Repositorio: [https://github.com/DanielPPerez/Tutor_API](https://github.com/DanielPPerez/Tutor_API)
- Notebook detector: [https://www.kaggle.com/code/danielperegrinoperez/detector-train](https://www.kaggle.com/code/danielperegrinoperez/detector-train)
- Notebook clasificador: [https://www.kaggle.com/code/danielperegrinoperez/clasificador-ocr-spanish](https://www.kaggle.com/code/danielperegrinoperez/clasificador-ocr-spanish)
