# Prompt por Fases para Claude (OCR Detector + Clasificador)

## Cómo usar este archivo

Copia y pega **una fase por vez** en Claude.  
No avances a la siguiente fase hasta que Claude entregue exactamente los artefactos solicitados en la fase actual.

Objetivo: construir notebooks mantenibles y reproducibles para Kaggle/local, evitando retrabajo y búsquedas innecesarias.

---

## Contexto fijo (pegar al inicio de cada fase)

```text
Proyecto OCR en español con dos etapas:
1) Detector (localiza bbox de caracteres)
2) Clasificador (predice carácter en el crop)

Estado actual:
- Detector está razonablemente bien.
- Clasificador actual rinde mal (Top-1 ~1.39%, Top-5 ~5.58%), sesgo fuerte a clase 'B'.
- Necesito separar detector y clasificador en notebooks distintos y robustecer el pipeline.

Rutas Kaggle principales:
- char_map: /kaggle/input/datasets/danielperegrinoperez/char-map/char_map.json
- dataset base: /kaggle/input/datasets/danielperegrinoperez/spanish-ocr-dataset/yolo_dataset_final/dataset.yaml

Requisito crítico:
- Debes entender y parsear nomenclatura heterogénea de imágenes.
- Debes incluir celda para descargar/unir datasets adicionales y conservar etiquetas originales cuando aplique.
- La lógica de verificación de clases debe inspirarse en app/scripts/verify_dataset_classes.py.

Regla de arquitectura:
- detector_train.ipynb y classifier_train.ipynb separados.
- Clasificador entrena con crops de letra, no imagen completa.
```

---

## Fase A — Datos, nomenclatura y metadata canónica

### Prompt (copiar/pegar en Claude)

```text
Usa el contexto fijo que te pasé.

Quiero que implementes la FASE A enfocada SOLO en datos y etiquetado para clasificación.
No entrenes modelos todavía.

Entregables obligatorios de esta fase:
1) Diseño de celdas para classifier_train.ipynb (solo bloque de datos).
2) Código de celda "bootstrap datasets" que:
   - lea:
     /kaggle/input/datasets/danielperegrinoperez/char-map/char_map.json
     /kaggle/input/datasets/danielperegrinoperez/spanish-ocr-dataset/yolo_dataset_final/dataset.yaml
   - descargue datasets extra (si no están montados) y los unifique en /kaggle/working/merged_datasets
3) Implementación de parser robusto de nomenclatura con reglas:
   - cls{idx}_*
   - prim_{slug}_*
   - char_{idx}_*
   - nombre de un carácter
   - patrones numéricos/fecha tipo 0_0_202207... y 10002_...
4) Estrategia de fallback:
   - si no se deduce etiqueta por nombre, buscar etiqueta en metadata/annotation del dataset fuente.
5) Construcción de metadata canónica (parquet o csv) con columnas:
   image_path, label, source, split, is_synthetic, parse_rule
6) Verificación de cobertura de clases (inspirado en verify_dataset_classes.py):
   - clases por dataset
   - clases cubiertas vs faltantes contra char_map
   - reporte global_missing
7) Tabla final de distribución por clase con recomendación de balance.

Restricciones:
- No uses pseudocódigo; entrega código ejecutable de notebook.
- Si hay ambigüedad de naming, prioriza metadata explícita del dataset antes que heurística.
- Mantén Unicode correcto (ñ, tildes, línea_vertical, círculo, etc.).

Formato de salida:
- Sección 1: "Celdas Fase A listas para pegar"
- Sección 2: "Checklist de validación Fase A"
- Sección 3: "Problemas esperables y mitigación"
```

### Criterio de aceptación Fase A

- Existe metadata canónica utilizable directamente por un DataLoader.
- Quedan identificadas clases faltantes y clases de baja frecuencia.
- El parser de filename no rompe con nomenclaturas mixtas.

---

## Fase B — Entrenamiento del clasificador (crop-only, anti-colapso)

### Prompt (copiar/pegar en Claude)

```text
Usa el contexto fijo que te pasé.
Asume que Fase A ya produjo metadata canónica válida.

Quiero que implementes la FASE B enfocada SOLO en entrenamiento del clasificador.
No implementes detector aquí.

Requisitos obligatorios:
1) Celdas para classifier_train.ipynb:
   - runtime detect (2xT4 DDP / 1 GPU / CPU fallback)
   - carga metadata
   - split estratificado por clase y, si posible, por source
   - transforms train/val para handwriting (sin destruir identidad)
   - DataLoader con WeightedRandomSampler
   - modelo (backbone configurable)
   - train loop con AMP + scheduler + early stopping
2) Loss y regularización:
   - CrossEntropy + label_smoothing (0.05–0.1)
   - weight_decay recomendado
   - opción FocalLoss si persiste desbalance severo
3) Métricas por época:
   - top1, top5, f1_macro, balanced_accuracy
   - accuracy por clase
4) Checkpoint:
   - guardar mejor por f1_macro (no por loss)
5) Exportables:
   - best.pt
   - train_config.json
   - class_distribution_report.json

Reglas:
- Entrenar SOLO con crops de letra (no imagen completa de cuaderno).
- Evitar leakage entre train y val.
- Incluir bloque de hard examples (opcional) para clases con peor recall.

Formato de salida:
- Sección 1: "Celdas Fase B listas para pegar"
- Sección 2: "Hiperparámetros iniciales recomendados"
- Sección 3: "Señales de colapso y acciones correctivas"
```

### Criterio de aceptación Fase B

- Entrenamiento reproducible y estable en Kaggle/local.
- Métricas por clase visibles y guardadas.
- Disminuye riesgo de colapso a una sola clase.

---

## Fase C — Evaluación, ONNX y validación end-to-end

### Prompt (copiar/pegar en Claude)

```text
Usa el contexto fijo que te pasé.
Asume que Fase B ya generó un best.pt estable.

Quiero implementar la FASE C: evaluación completa + export ONNX + smoke test real.

Requisitos obligatorios:
1) Celdas de evaluación del clasificador:
   - top1, top5, f1_macro, balanced_accuracy
   - confusion matrix por clase
   - reporte de errores más frecuentes
   - comparación confianza en aciertos vs errores
   - grafica de entrenamiento del modelo
2) Export ONNX:
   - exportar best.onnx
   - validar paridad PyTorch vs ONNX (lote fijo)
3) Artefactos finales:
   - best.onnx
   - char_map.json (el usado realmente)
   - metrics_report.json
   - onnx_parity_report.json
4) Smoke test integrado:
   - detector -> crop -> clasificador
   - mostrar 20-50 ejemplos con GT/pred/conf

Reglas:
- Si la confianza de errores > confianza de aciertos, proponer calibración (temperature scaling).
- Debes incluir una sección de "bloqueos comunes" (mismatch de preprocesado, orden de clases, shape ONNX, etc.) y cómo resolverlos.

Formato de salida:
- Sección 1: "Celdas Fase C listas para pegar"
- Sección 2: "Checklist de release"
- Sección 3: "Debug playbook (rápido)"
```

### Criterio de aceptación Fase C

- ONNX consistente con PyTorch en inferencia.
- Reportes completos guardados.
- Pipeline usable para pruebas reales de alumno.

---

## Fase D — Notebook del detector (desacoplado del clasificador)

### Prompt (copiar/pegar en Claude)

```text
Usa el contexto fijo que te pasé.
Ahora quiero SOLO el notebook del detector, separado del clasificador.

Objetivo:
- Construir detector_train.ipynb limpio, sin lógica de clasificación incrustada.

Requisitos:
1) Celdas:
   - setup runtime (Kaggle/local)
   - carga dataset de detección (dataset.yaml)
   - entrenamiento detector
   - evaluación detector (mAP, PR, ejemplos visuales)
   - export de pesos
   - generación opcional de crops debug para consumo del clasificador
2) Contrato de salida del detector:
   - bbox + score + crop opcional
3) No mezclar pérdidas/métricas de clasificación en este notebook.

Formato de salida:
- "Celdas detector_train.ipynb listas para pegar"
- "Parámetros recomendados por hardware"
- "Errores comunes de entrenamiento detector"
```

### Criterio de aceptación Fase D

- Notebook del detector independiente y mantenible.
- Entrega crops/boxes compatibles con pipeline del clasificador.

---

## Prompt de cierre (integración final)

### Prompt (copiar/pegar en Claude)

```text
Con base en las Fases A/B/C/D ya implementadas, genera:

1) Lista final de archivos a crear/modificar con ruta exacta.
2) Orden de ejecución recomendado (Kaggle y local).
3) Checklist final de verificación antes de entrenar en serio.
4) Tabla de riesgos -> síntoma -> causa probable -> fix.
5) Plan de 3 experimentos prioritarios para subir F1 macro del clasificador.

Responde de forma concreta y accionable.
```

---

## Nota práctica

Si Claude intenta mezclar todo en una sola entrega, pídele explícitamente:  
**"No avances de fase. Dame solo los entregables de esta fase."**

