# Plan Maestro: Notebooks OCR (Kaggle + Local)

## Description

Necesitas estabilizar tu pipeline OCR separando responsabilidades y atacando el cuello de botella principal: el clasificador.  
Con base en tus resultados actuales (`Top-1: 1.39%`, `Top-5: 5.58%`, sesgo fuerte a predecir `'B'`), el problema no es solo de arquitectura; es una mezcla de:

- evaluación sobre conjunto inadecuado para clasificación pura,
- desalineación entre etiquetas reales y etiquetas inferidas por nombre de archivo,
- mezcla de dominios (caracteres vs trazos/palabras),
- posible sobreajuste o colapso de clases por desbalance.

Este plan define un rediseño autocontenido para:

1. separar `detector` y `clasificador` en dos notebooks distintos,
2. soportar ejecución automática en Kaggle (2xT4, 1xGPU o CPU),
3. mantener scripts locales en `app/training/`,
4. corregir lectura/etiquetado del dataset con nomenclatura no convencional,
5. mejorar de forma real el rendimiento del clasificador.

---

## Tech Stack

- Python 3.10+
- PyTorch + torchvision + timm
- ONNX export + onnxruntime (validación de inferencia)
- OpenCV + PIL (preprocesado consistente)
- scikit-learn (métricas: confusion matrix, balanced accuracy, F1 macro)
- pandas / numpy
- Kaggle Notebooks (GPU T4 dual/single o CPU)

---

## Key Requirements

### 1) Arquitectura de notebooks separada (obligatoria)

**Objetivo:** eliminar acoplamiento entre detector y clasificador.

**Estructura final:**

- `app/scripts/detector_train.ipynb`
  - prepara dataset de detección,
  - entrena detector,
  - exporta pesos y genera crops debug.
- `app/scripts/classifier_train.ipynb`
  - solo consume crops/patches de caracteres validados,
  - entrena clasificador,
  - exporta `best.pt`, `best.onnx`, `char_map.json`.

**Regla técnica:**

- El clasificador **no** debe entrenarse con imágenes completas de cuaderno cuando su tarea final es reconocer un solo carácter recortado.
- El detector puede operar sobre imagen completa; el clasificador no.

---

### 2) Modo de ejecución híbrido Kaggle/local con autodetección de hardware

**Objetivo:** ejecutar el mismo notebook/script sin tocar código manualmente.

**Política de dispositivo (prioridad):**

1. `2xT4` (DistributedDataParallel),
2. `1xGPU`,
3. `CPU`.

**Contrato de ejecución:**

- una celda de configuración detecta entorno (`Kaggle` vs `local`) y número de GPUs,
- ajusta automáticamente:
  - `batch_size`,
  - `num_workers`,
  - mixed precision,
  - estrategia DDP o single process.

**Recomendación práctica:**

- Kaggle 2xT4: usar DDP con `torchrun` y batch global alto.
- Kaggle sin GPU o local CPU: activar modo `debug_train` (menos épocas, subset) para validar pipeline sin romper tiempos.

---

### 3) Scripts locales dentro de `app/training/` (fuente de verdad)

**Objetivo:** que notebooks sean delgados y reproducibles.

**Módulos sugeridos:**

- `app/training/classifier_dataset.py`
  - parser de etiquetas robusto (no depender solo de regex simples),
  - validación de clases y reporte de muestras inválidas.
- `app/training/classifier_transforms.py`
  - augmentations train/val consistentes con inferencia.
- `app/training/classifier_train.py`
  - loop de entrenamiento, early stopping, scheduler.
- `app/training/classifier_eval.py`
  - métricas por clase, matriz de confusión, calibración de confianza.
- `app/training/runtime_config.py`
  - detección de hardware y flags Kaggle/local.

**Resultado:** notebook = orquestación; lógica crítica = `.py` versionado y testeable.

---

### 4) Diagnóstico de tus resultados actuales (causas raíz)

**Hallazgos clave de tus métricas:**

- `9300` imágenes sin GT deducible: gran parte del set evaluado no aporta accuracy real.
- Sesgo extremo a `'B'`: típico de desbalance severo, labels mal alineadas o colapso de entrenamiento.
- Confianza de errores (`52.3%`) mayor que la de aciertos (`50.2%`): modelo mal calibrado y sobreconfiado al equivocarse.
- Evaluar en `data/processed/yolo_dataset/images/train` mezcla muestras no diseñadas para clasificación pura de carácter.

**Conclusión operativa:**

- Tus números no reflejan únicamente calidad del backbone; reflejan sobre todo un pipeline de evaluación/etiquetado inconsistente con la tarea final.

---

### 5) Estrategia de dataset para clasificador (la más importante)

**Objetivo:** construir un dataset limpio de clasificación de carácter individual.

**Reglas obligatorias:**

1. Entrenar clasificador con **crops de letra** (o imágenes ya centradas en una letra), no con página completa.
2. Excluir del entrenamiento del clasificador:
  - imágenes de palabras completas,
  - trazos primitivos sueltos (`línea_vertical`, `línea_horizontal`, etc.) si no son clases finales reales.
3. Mantener `char_map` fijo y versionado: mismo orden en train/val/export/inferencia.

**Sobre tus dudas de datos sintéticos en libreta:**

- Sí, puede ayudar, pero solo si se usa como **dominio de apoyo** y balanceado.
- Recomendación: 60-80% datos reales de crops + 20-40% sintético realista.
- Evitar que sintético domine, porque induce sesgos de textura/fondo.

**Sobre unir datasets con etiquetas originales:**

- Sí conviene, pero con unificación explícita de etiquetas:
  - normalizar Unicode (`NFC`),
  - resolver equivalencias (`"linea_vertical"` vs `"línea_vertical"`),
  - eliminar clases fuera de objetivo.

---

### 6) Nomenclatura y lectura de etiquetas no estándar

**Problema actual:** deducir GT solo desde nombre de archivo no escala cuando mezclas fuentes.

**Solución recomendada (obligatoria):**

- generar un `metadata.csv` o `metadata.parquet` canónico para clasificación con columnas:
  - `image_path`,
  - `label`,
  - `source`,
  - `split`,
  - `is_synthetic`,
  - `is_hard_example`.

**Regla:** la etiqueta oficial del clasificador se toma de `metadata`, no del filename.

**Ventaja:** elimina ambigüedad de nomenclatura de `prepare_yolo_dataset.py` y simplifica Kaggle/local.

---

### 7) Input correcto al clasificador: crop vs imagen completa

**Respuesta directa:** para tu caso, debes pasar **solo el crop de la letra** al clasificador.

**Pipeline recomendado de inferencia:**

1. imagen del alumno,
2. detector localiza caja(s),
3. crop por caja + normalización geométrica/fotométrica,
4. clasificador predice carácter.

**Excepción válida:** usar imagen completa únicamente si entrenaste explícitamente ese mismo formato extremo a extremo (no es tu caso actual).

---

### 8) Plan anti-overfitting y anti-colapso de clase

**Objetivo:** subir accuracy real y reducir sesgo a una sola clase.

**Acciones concretas:**

- Loss robusta:
  - `CrossEntropy` + `label_smoothing` (0.05-0.1),
  - o `FocalLoss` si persiste desbalance.
- Muestreo:
  - `WeightedRandomSampler` por clase.
- Augmentations útiles para handwriting:
  - rotación leve, elastic transform suave, blur leve, variación de contraste,
  - evitar transforms que destruyan identidad del carácter.
- Regularización:
  - dropout moderado,
  - weight decay (`1e-4` a `5e-4`).
- Control de entrenamiento:
  - early stopping por `val_f1_macro`,
  - scheduler (`CosineAnnealing` o `OneCycleLR`).
- Validación de verdad:
  - split estratificado por clase + por fuente (para evitar leakage).

---

### 9) Métricas correctas para tomar decisiones

**No usar solo Top-1 global.**

**Tablero mínimo por época:**

- `top1`, `top5`,
- `f1_macro`,
- `balanced_accuracy`,
- accuracy por clase,
- matriz de confusión,
- Expected Calibration Error (ECE) opcional.

**Criterio de checkpoint:**

- guardar `best` por `f1_macro` (no por loss, no por top1 global).

---

### 10) Plan de experimentación en 3 fases (secuencial)

**Fase A: saneamiento de datos (obligatoria)**

- construir `metadata` canónico,
- limpiar clases inválidas,
- verificar distribución por clase (min/max),
- crear `train/val/test` estratificados.

**Fase B: baseline confiable**

- entrenar clasificador solo con crops limpios,
- sin mezclar primitivas ni palabras,
- exportar ONNX + evaluar en set de test limpio.

**Fase C: robustez de dominio**

- introducir sintético gradualmente (20%, luego 30-40%),
- hard-negative mining de fallos reales del alumno,
- calibrar temperatura si confianza sigue sobreestimada.

---

### 11) Separación detector vs clasificador: contrato de datos

**Detector entrega:**

- `bbox`, `score_det`, `crop_path` (opcional), `frame_id`.

**Clasificador recibe:**

- imagen crop normalizada + `char_map` fijo.

**Regla de integración:**

- si `score_det` bajo, marcar predicción como incierta antes de clasificar (o clasificar con bandera de baja confianza).

---

### 12) Diseño del nuevo `classifier_train.ipynb` (estructura recomendada)

1. **Setup y autodetección HW** (Kaggle/local, GPU/CPU, semillas).
2. **Carga de metadata** (sin inferir etiquetas desde filename).
3. **EDA rápida** (clases, fuentes, desbalance).
4. **Transforms + DataLoader** (sampler balanceado).
5. **Modelo** (backbone + head configurable).
6. **Train loop** (AMP, scheduler, early stopping).
7. **Evaluación completa** (f1 macro, confusion matrix, per-class).
8. **Exportación** (`best.pt`, `best.onnx`, `char_map.json`, `train_config.json`).
9. **Smoke test ONNX** con 50-200 muestras reales.

Cada celda debe ser autocontenida, con entradas/salidas claras y sin dependencias implícitas de celdas anteriores ocultas.

---

### 13) Riesgos frecuentes y mitigación

- **Riesgo:** leakage entre train y val por nombres similares.  
**Mitigación:** split por hash de contenido o por origen/documento.
- **Riesgo:** clases casi vacías.  
**Mitigación:** umbral mínimo por clase + oversampling controlado.
- **Riesgo:** mismatch train/inferencia en preprocesado.  
**Mitigación:** función única de preprocesado compartida en `app/training/`.
- **Riesgo:** ONNX difiere de PyTorch.  
**Mitigación:** test de paridad logits/probs en lote fijo.

---

### 14) Criterios de éxito (aceptación)

El rediseño se considera exitoso cuando se cumpla:

- `Top-1` y `F1 macro` mejoran de forma consistente en test limpio,
- desaparece el colapso de predicción dominante (`'B'`),
- confianza media de aciertos > confianza media de errores,
- el pipeline corre sin cambios manuales en:
  - Kaggle con GPU,
  - Kaggle CPU,
  - local (GPU o CPU),
- detector y clasificador quedan desacoplados en dos notebooks mantenibles.

---

## Decisiones directas a tus preguntas

- **¿Aumentar imágenes sintéticas sobre libretas?**  
Sí, pero de forma controlada (20-40%) y nunca reemplazando la base real de crops.
- **¿Entrenar también con datasets unidos y etiquetas originales?**  
Sí, siempre que normalices y unifiques etiquetas en un `metadata` único.
- **¿Pasar crop o imagen completa al clasificador?**  
Crop de letra (la imagen completa déjala para detector/segmentación).
- **¿El problema puede ser overfitting?**  
Sí, pero tus resultados muestran además un problema mayor de dataset/etiquetado/evaluación; primero corrige eso.

