# Documentación API (Tutor Inteligente de Caligrafía)

Esta documentación describe los endpoints disponibles, el pipeline de evaluación (preprocesamiento, comparación y cálculo del score) y el formato de salida.

## Endpoints

### `POST /evaluate`
Evalúa una imagen para un carácter objetivo.

#### Request (`multipart/form-data`)
- `file`: imagen (JPG/PNG/WEBP…).
- `target_char`: carácter esperado (por ejemplo: `A`, `b`, `3`).
- `level`: dificultad del carril: `principiante` | `intermedio` | `avanzado` (default: `intermedio`).

#### Response (`application/json`)
Campos principales:
- `target_char`: el carácter pedido.
- `detected_char`: carácter detectado por el clasificador (string, `?` si no aplica).
- `confidence`: confianza del clasificador (`float`).
- `score_final`: nota final combinada `[0-100]` (`float`).
- `level`: nivel usado (`string`).
- `scores_breakdown`: desglose de métricas que forman `score_final` (`dict`).
- `weights_used`: pesos de cada componente usados para el score (`dict`).
- `feedback`: texto pedagógico (`string`).
- `metadata`: metadatos del preprocesamiento (ROI refinada, corrección de ángulo, escala y dimensiones, etc.).
- `metrics_extra`: métricas auxiliares para debug/expansión (geometría, topología, calidad, DT coverage ratio, etc.).

Imágenes devueltas en base64 (PNG):
- `image_student_b64`: crop RAW de la caja YOLO (foto real recortada).
- `template_b64`: carril del nivel seleccionado (plantilla visual).
- `comparison_b64`: overlay de comparación (verde/rojo/amarillo) generado por el visualizador.

Si no se detecta trazo válido, se devuelve:
- `error`: mensaje (`string`)
- `target_char`, `detected_char` (null), `confidence` (0.0)

### `POST /evaluate_plana`
Evalúa una “plana” (imagen con múltiples caracteres), usando como plantilla el primer carácter detectado.

#### Request (`multipart/form-data`)
- `file`: imagen con múltiples caracteres.
- `level`: dificultad del carril: `principiante` | `intermedio` | `avanzado` (default: `intermedio`).

#### Response (`application/json`)
- `template_char`: carácter detectado del primer bbox (referencia).
- `template_confidence`: confianza del clasificador para la plantilla.
- `template_b64`: imagen base64 del crop del primer carácter.
- `n_detected`: número total de caracteres detectados por YOLO.
- `n_evaluated`: número de caracteres calificados (sin contar el template).
- `avg_score`: promedio de `score_final` de los caracteres evaluados con score > 0.
- `level`: nivel usado.
- `results`: lista con un dict por cada carácter evaluado (del segundo en adelante), incluyendo:
- `index` (1-based)
- `detected_char`, `confidence`
- `score_final`, `scores_breakdown`, `weights_used`
- `feedback`
- `metadata`, `metrics_extra`
- `image_student_b64`, `comparison_b64`

## Preprocesamiento de imagen

El preprocesamiento transforma la imagen cruda en una representación normalizada del trazo del alumno.

### 1) Decodificación y detección YOLO
1. Se decodifican los bytes del archivo a `BGR`.
2. Se ejecuta YOLO (`app/core/processor.py`) para detectar bboxes del carácter.
3. Se usa:
- `/evaluate`: la bbox con mayor confianza.
- `/evaluate_plana`: todas las bboxes en orden de lectura (arriba→abajo; dentro de cada línea, izquierda→derecha).
4. Se extrae `raw_crop_bgr` del bbox (para visualización y, cuando está disponible, para clasificación con distribución idéntica a la del entrenamiento).

### 2) Extracción de ROI y limpieza de líneas de cuaderno (normalizer)
La función `normalize_character()` orquesta el flujo de normalización:
1. `extract_roi()`:
- Si se recibió `yolo_box`: usa el bbox YOLO con `ROI_PADDING` para refinar la ROI con bordes/contornos.
- Si no: encuentra bbox por contornos desde Canny.
2. `remove_color_lines()` (HSV):
- Elimina líneas de libreta con rangos HSV configurables.
- Reemplaza por blanco donde la máscara detecta líneas.
3. Decisión por “digital vs foto”:
- Se analiza calidad (blur/contraste/iluminación/sombra) para elegir ruta.
- En imágenes digitales: se evita corrección agresiva de iluminación.
- En fotos: se normaliza iluminación con `normalize_illumination()` (incluye corrección por “background division” si hay sombras y CLAHE adaptativo para contraste local residual).
4. Binarización (`binarizer.py`):
- Jerarquía típica: Otsu si `use_otsu`, o Adaptativo Gaussiano por defecto.
- Sauvola es opcional (si `scikit-image` está disponible).
- La salida es una máscara binaria con convención: `trazo=255`, `fondo=0`.
5. Para fotos: `remove_grid_lines()` (morfología + inpaint) para borrar líneas de cuadriculado.
6. Limpieza:
- `remove_specks()` elimina componentes pequeñas conservando siempre la componente mayor (previene borrar trazos finos).
- `clean_noise()` aplica apertura/cierre morfológico (kernel adaptativo; en “digital” puede ser identidad).
- Si el trazo está muy fragmentado (fotos): `_fill_internal_gaps()` rellena huecos interiores por flood-fill inverso.
7. `deskew()`:
- Estima rotación con momentos (moments) y corrige el ángulo si está dentro de `MAX_DESKEW_ANGLE`.
8. `crop_and_center()`:
- Recorta el bounding box del trazo y lo centra en un canvas `128x128`.
- Re-binariza tras el resize para mantener el carácter binario.

Resultado del preprocesamiento:
- `img_a`: `np.ndarray uint8` de tamaño `128x128` con `trazo=255` y `fondo=0`.
- `metadata`: ángulo corregido, escala, dimensiones del trazo y un conjunto de flags/calidad para diagnóstico.

## Métricas de comparación

Las métricas comparan el trazo normalizado del alumno contra la plantilla del carácter, representadas como:
- `skel_p`: esqueleto 1px de la plantilla (guía).
- `skel_a`: esqueleto 1px del alumno.
- `img_a`: masa binaria del alumno (para DT y overlay).

### Distance Transform (DT) fidelity
Métrica principal de “fidelidad al carril” (`app/metrics/distance_transform.py`):
- Se construye un mapa de distancias desde el esqueleto de la plantilla.
- Para cada píxel activo del alumno se mide cuánto está fuera del radio permitido (tolerancia).
- Produce:
- `score_precision` (0-100): castiga píxeles fuera del carril.
- `coverage` (0-1): fracción del esqueleto cubierta por el trazo del alumno dentro de la tolerancia.
- `score_final_dt = w_prec * score_precision + w_cov * score_coverage`.
- La tolerancia depende del `level` (`config.DT_TOLERANCE_BY_LEVEL`).

### Métricas geométricas entre esqueletos
Calculadas sobre `skel_p` vs `skel_a` (`app/metrics/geometric.py`):
- `SSIM`: similitud estructural (mapeada a `[0-100]`).
- `Procrustes`: ajuste global con disparidad de secuencias remuestreadas (mapeado a `[0-100]`).
- `Hausdorff`: distancia de borde en puntos del esqueleto (penalizada con tolerancia y factor).

### Topología (bucles/agujeros)
Métrica topológica de integridad estructural (`app/metrics/topologic.py`):
- Cuenta `loops` en el esqueleto.
- El score de topología depende de si el número de loops coincide: `topo_match=True` => `100.0`; `topo_match=False` => `30.0`

### Trayectoria (DTW sobre puntos)
Compara la “trayectoria” a lo largo del esqueleto (`app/metrics/trajectory.py`):
- Convierte el esqueleto en secuencia de puntos ordenada por ángulo alrededor del centroide.
- Submuestrea a un máximo (`MAX_POINTS_TRAJECTORY`).
- Calcula DTW con ventana de Sakoe–Chiba (`DTW_BAND_RATIO`) para reducir coste.
- Mapea distancia DTW a `[0-100]` restando un factor por unidad de distancia.

### Coherencia direccional por segmentos (coseno)
Métrica por segmentos direccionales (`app/metrics/segment_cosine.py`):
- Divide el esqueleto (ordenado) en `N_SEGMENTS=12` segmentos.
- Para cada segmento obtiene un vector dirección y compara con coseno.
- Mapea cosenos promedio de `[-1,1]` a `[0,100]`.

### Nota sobre “quality” (calidad intrínseca)
Existe una métrica de calidad intrínseca del trazo del alumno (`app/metrics/quality.py`) para:
- `metrics_extra` y diagnóstico visual.
- En el score actual (`calculate_final_score`) **no** se usa directamente como componente ponderado.

## Cálculo del score

El score final se calcula en `app/metrics/scorer.py` (`calculate_final_score()`):
1. Cada métrica se convierte a `score` en rango `[0-100]`.
2. Se aplica una suma ponderada con pesos configurables en `config.SCORING_WEIGHTS`:
- `dt_precision`: 0.30
- `dt_coverage`: 0.20
- `topology`: 0.20
- `ssim`: 0.12
- `procrustes`: 0.10
- `hausdorff`: 0.04
- `trajectory`: 0.02
- `cosine`: 0.02
3. El score final se redondea a 2 decimales y se devuelve como `score_final`.

Retroalimentación:
- `get_feedback()` usa el valor global y umbrales del desglose para generar un texto pedagógico.

## Salida y formato

### Encoding de imágenes
Las claves `*_b64` devuelven imágenes en base64:
- Son PNG generados en backend (no requieren conversión adicional).
- `comparison_b64` es el overlay generado por `app/utils/visualizer.py`.

### Convención de campos numéricos
- `score_final`, `scores_breakdown.*`: `float` en `[0-100]` (con redondeo a 2 decimales).
- `confidence`: `float` en `[0-1]` (redondeado a 4 decimales).
- `coverage` en `metrics_extra`: `float` en `[0-1]`.

### Estructura esperada (compatibilidad frontend)
Los endpoints devuelven exactamente las llaves indicadas arriba (por ejemplo `scores_breakdown`, `weights_used`, `metadata` y las 3 imágenes base64 esperadas).

## Requerimientos mínimos

### Dependencias Python (producción / API)
Según `requirements.txt`, el API requiere:
- Web: `fastapi`, `uvicorn`, `python-multipart`
- Imágenes: `numpy`, `opencv-python-headless`
- Métricas/algoritmos: `scipy`, `scikit-image` (recomendado), `matplotlib`, `pillow`
- Inferencia ONNX: `onnxruntime`

### Modelos y artefactos esperados
El backend carga:
- Detector YOLO: `config.YOLO_MODEL_PATH` (ONNX)
- Clasificador MobileNet: `config.MOBILENET_MODEL_PATH` (ONNX)
- Mapeo de clases: `config.CLASS_MAP_PATH` (`char_map.json`) o fallback EMNIST order si no existe
- Plantillas:
- Carriles por nivel: `app/templates/<level>/..._<level>.npy`
- Esqueleto plantilla: `app/templates/skeleton/..._skeleton.npy`

Nota: `generate_templates.py` genera esos archivos a partir de una fuente TTF y esqueletiza.

## Uso de memoria y eficiencia

Puntos relevantes de rendimiento basados en el código:
- Caché de plantillas:
- `endpoints.py` mantiene `_TEMPLATE_CACHE` en memoria para `carril` y `skeleton`, evitando recargar `.npy` en cada request.
- Distancia Transform (DT):
- Se construye el `dist_map` desde `skel_p` para cada evaluación.
- El mapa es del tamaño del canvas de evaluación (128x128), por lo que el coste es moderado.
- DTW (trayectoria):
- No calcula DTW “pleno” N×M.
- Usa ventana Sakoe–Chiba (`DTW_BAND_RATIO`) para limitar la banda y reduce memoria/tiempo.
- Además submuestrea secuencias a `MAX_POINTS_TRAJECTORY`.
- Visualizer:
- Genera overlay 128→512 y usa Matplotlib para el PNG final (coste mayor que el cálculo puro de métricas, pero acotado por tamaño fijo).
- `/evaluate_plana` escala linealmente con el número de caracteres:
- Cada carácter adicional añade su propio bloque de normalización + métricas + overlay.

## PIPELINE GENERAL

Flujo conceptual (aplicable tanto a `/evaluate` como a `/evaluate_plana`, cambiando la selección de template):
1. Recibir `file` (imagen) y `level` (y `target_char` en `/evaluate`).
2. Decodificar imagen.
3. Detectar caracteres con YOLO.
4. Preprocesar cada carácter:
- ROI refinement
- eliminación de líneas (HSV, grid)
- corrección de iluminación (solo fotos)
- binarización
- limpieza morfológica + eliminación de specks
- deskew
- crop y centrado a `128x128`.
5. Esqueletizar:
- Plantilla: carril/skeleton precargados.
- Alumno: esqueleto sobre `img_a`.
6. Calcular métricas de comparación:
- DT fidelity (precision + coverage + heatmap)
- SSIM / Procrustes / Hausdorff
- topología (loops)
- trayectoria (DTW banded)
- coseno por segmentos.
7. Calcular `score_final` con suma ponderada (`SCORING_WEIGHTS`).
8. Generar `feedback`.
9. Generar salida visual:
- `image_student_b64` (crop RAW)
- `template_b64` (carril guía)
- `comparison_b64` (overlay).
10. Devolver JSON con todo lo anterior.

## Conclusiones

- La API separa claramente:
- preprocesamiento (normalizer + binarización),
- comparación (métricas sobre esqueletos y masa binaria),
- scoring (ponderación configurable),
- y salida (JSON + base64 PNG).
- La “dificultad” (`level`) afecta principalmente:
- el carril/kerner de plantilla,
- y las tolerancias de DT.
- El resultado es un score interpretable pedagógicamente, acompañado por feedback y overlays visuales para diagnóstico rápido.
