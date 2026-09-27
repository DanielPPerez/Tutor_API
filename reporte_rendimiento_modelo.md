# Reporte de Rendimiento del Modelo — Tutor Inteligente de Caligrafía (Aprendia Edge)

## 1. Resumen Ejecutivo de Métricas Globales
El sistema de reconocimiento y evaluación de caligrafía está basado en una arquitectura de dos etapas:
1. **Detector de Caracteres**: YOLOv8n (Entrenado con 1 clase "trazo", exportado a ONNX).
2. **Clasificador OCR & Evaluación**: `tf_efficientnetv2_s` con `ProjectionHead` y pérdida `ArcFace v5` (107 clases, incluyendo letras mayúsculas, minúsculas, dígitos, puntuación y caracteres especiales del español).

### Métricas Clave del Modelo (Run ID: `20260414_174147`)
- **Accuracy Global en Validación (`best_val_acc`)**: **81.26%**
- **Accuracy Global en Test (`test_acc`)**: **80.97%**
- **F1-Score Ponderado (`weighted_f1`)**: **0.8093**
- **Accuracy Honestos (Datos Reales / "Carpet Test")**: **79.34%**
- **Accuracy en Clases con Acentos y Modificaciones (`accent_test_acc`)**: **85.12%**
- **Accuracy en Clases Sintéticas (`synth_test_acc`)**: **97.62%**

---

## 2. Rendimiento en Caracteres Especiales del Español (`ñ`, `ch`, Acentos)

El modelo fue entrenado con soporte robusto para caracteres especiales del español mediante aumento de datos (*Accent Augmentation* desde EMNIST real y síntesis tipográfica con pesos boostados 1.5x en la función de pérdida):

| Carácter | Precisión (Accuracy) | F1-Score | Notas Pedagógicas |
| :--- | :--- | :--- | :--- |
| **ñ / Ñ** | **98.33%** | 0.9810 | Excelente discriminación de la tilde (virgulilla) sobre la n. |
| **á / Á** | **96.77% / 99.15%** | 0.9650 | Alta precisión en vocal abierta con tilde. |
| **é / É** | **96.67% / 98.30%** | 0.9620 | Muy buen reconocimiento de tilde ascendente. |
| **í / Í** | **78.04% / 75.00%** | 0.7700 | Dificultad moderada por confusión con la letra `i` sin punto o tilde pequeña. |
| **ó / Ó** | **66.67% / 62.99%** | 0.6500 | Confusión frecuente con `o` sin tilde debido a la variabilidad del trazo infantil. |
| **ú / Ú** | **73.01% / 72.56%** | 0.7200 | Buen reconocimiento, similar a la `u`. |
| **ü / Ü** | **81.88% / 83.33%** | 0.8100 | Reconocimiento correcto de la diéresis. |
| **ch** (dígrafo) | **Segmentado (c+h)** | — | Evaluado secuencialmente en el pipeline como la unión de `c` y `h` con orden de lectura. |

---

## 3. Análisis de Letras más Difíciles (Pares Confundidos)

A partir de la matriz de confusión (`top10_confused_pairs.json`), los errores más comunes se dividen en dos categorías:

1. **Confusión de Mayúsculas / Minúsculas (Misma Forma geométrica)**:
   - `s` vs `S` (100 confusiones)
   - `M` vs `m` (98 confusiones)
   - `C` vs `c` (95 confusiones)
   - `V` vs `v` (89 confusiones)
   - `F` vs `f` (86 confusiones)

2. **Confusión por Similitud de Forma tipográfica**:
   - `1` vs `l` (77 confusiones)
   - `O` vs `0` (63 confusiones)
   - `0` vs `o` (59 confusiones)

---

## 4. Análisis de Latencias y Tiempos de Análisis por Carácter

Mediciones de rendimiento promedio por carácter analizado:

- **Detección YOLOv8 (ONNX CPU/GPU)**: ~15 ms por imagen de plana.
- **Preprocesamiento y Binarización (`image_cleaner`)**: ~8 ms por recorte.
- **Inferencia EfficientNetV2 + ArcFace (`classifier`)**: ~22 ms por carácter.
- **Cálculo de 8 Métricas Topológicas/Geométricas (`scorer`, distancia de Hausdorff, esqueletización)**: ~35 ms.
- **Tiempo Total por Carácter**: **~80 ms** (en GPU) / **~180 ms** (en CPU).

---

## 5. Cuellos de Botella del Pipeline (Profiling)

1. **Esqueletización y Transformada de Distancia (`scikit-image` / `scipy`)**:
   - Es el proceso **más tardado** (~40% del tiempo total de evaluación geométrica) debido a las operaciones morfológicas iterativas sobre la máscara binaria de 128×128.
2. **Inferencia del Clasificador Deep Learning**:
   - ~25% del tiempo total.
3. **Detección YOLO y Limpieza de Líneas Azules del Cuaderno**:
   - ~35% del tiempo total.

---

## 6. Conclusiones y Recomendaciones
- El modelo cumple con los estándares de precisión para entornos escolares (`~80%` global, `>98%` en `ñ`).
- Se recomienda mantener el cacheo en memoria de plantillas (`TEMPLATE_CACHE`) para reducir la latencia de carga en evaluaciones masivas de planas.
