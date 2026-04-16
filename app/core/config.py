# =============================================================================
# config.py — Parámetros globales del pipeline de normalización y detección
# Todos los valores ajustables están aquí; no hardcodear nada en normalizer.py
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE FORMAS / TRAYECTORIA
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SIZE           = 128
TARGET_SHAPE          = (128, 128)
MIN_BRANCH_LENGTH     = 10
MAX_POINTS_TRAJECTORY = 64
PROCRUSTES_N_POINTS   = 50
DTW_BAND_RATIO        = 0.25
HAUSDORFF_TOLERANCE   = 5
HAUSDORFF_FACTOR      = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZER — Salida final
# ─────────────────────────────────────────────────────────────────────────────
NORMALIZER_PADDING    = 15   # Margen interior (px) al centrar el caracter

# ─────────────────────────────────────────────────────────────────────────────
# CLAHE — Ecualizacion adaptativa de histograma
# ─────────────────────────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT      = 3.0
CLAHE_GRID_SIZE       = (8, 8)

# ─────────────────────────────────────────────────────────────────────────────
# ELIMINACION DE LINEAS DE LIBRETA — Filtrado morfologico
# ─────────────────────────────────────────────────────────────────────────────
GRID_LINE_MIN_WIDTH   = 20
HEAL_KERNEL_SIZE      = (3, 3)
INPAINT_RADIUS        = 3

# ─────────────────────────────────────────────────────────────────────────────
# ELIMINACION DE LINEAS POR COLOR (HSV)
# Cada entrada: ((H_low, S_low, V_low), (H_high, S_high, V_high))
# OpenCV: H en [0,179], S y V en [0,255]
# ─────────────────────────────────────────────────────────────────────────────
HSV_LINE_RANGES = [
    ((90,  40,  80), (130, 255, 255)),   # Azul (cuadernos estandar)
    ((0,   60,  80), (10,  255, 255)),   # Rojo / magenta (margen)
    ((165, 60,  80), (179, 255, 255)),   # Rojo envolvente (OpenCV cierra en 179)
    ((40,  30,  80), (80,  200, 255)),   # Verde tenue (cuadernos de contabilidad)
]
HSV_MASK_DILATE       = 3

# ─────────────────────────────────────────────────────────────────────────────
# BINARIZACION
# ─────────────────────────────────────────────────────────────────────────────
ADAPTIVE_BLOCK_SIZE   = 11   # Debe ser impar >= 3. Subir (15-21) con sombras fuertes.
ADAPTIVE_C            = 2    # Aumentar (3-5) si quedan manchas de fondo.
USE_OTSU_FALLBACK     = True
OTSU_CONTRAST_THRESHOLD = 40  # Desviacion estandar minima para activar Otsu

# ─────────────────────────────────────────────────────────────────────────────
# REFINAMIENTO DE ROI
# ─────────────────────────────────────────────────────────────────────────────
ROI_PADDING           = 12
ROI_MIN_CONTOUR_FILL  = 0.30
ROI_CANNY_LOW         = 30
ROI_CANNY_HIGH        = 120
ROI_BLUR_KSIZE        = 5
ROI_CONTOUR_MARGIN    = 4

# ─────────────────────────────────────────────────────────────────────────────
# DESKEW
# ─────────────────────────────────────────────────────────────────────────────
MAX_DESKEW_ANGLE      = 30

# ─────────────────────────────────────────────────────────────────────────────
# MORFOLOGIA
# ─────────────────────────────────────────────────────────────────────────────
MORPH_OPEN_KSIZE      = (2, 2)
MORPH_CLOSE_KSIZE     = (2, 2)
DILATE_AFTER_ROI      = 2

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS DE MODELOS ONNX
# ─────────────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH       = "app/models/classifier_artifacts/best_detector.onnx"
MOBILENET_MODEL_PATH  = "app/models/classifier_artifacts/best_classifier.onnx"
CLASS_MAP_PATH        = "app/models/char_map.json"   # generado por train_classifier.py

# ─────────────────────────────────────────────────────────────────────────────
# DETECCION / CLASIFICACION
# ─────────────────────────────────────────────────────────────────────────────
DETECTION_THRESHOLD   = 0.55  # Subido de 0.45 → reduce falsos positivos
NMS_THRESHOLD         = 0.40  # Bajado de 0.45 → elimina más cajas duplicadas
YOLO_INPUT_SIZE       = 640

# ORDEN CORRECTO DE EMNIST byclass: 0-9 primero, luego A-Z, luego a-z
# Si usas A-Z primero (orden alfabético) con un modelo entrenado en EMNIST,
# todos los índices están desplazados → clasificación incorrecta.
# processor.py usa class_map.json en lugar de este valor, pero se mantiene
# como documentación y fallback de último recurso.
CLASS_NAMES = list(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

# ─────────────────────────────────────────────────────────────────────────────
# GENERACION DE PLANTILLAS
# ─────────────────────────────────────────────────────────────────────────────

# Ruta a la fuente TTF y carpeta de salida
FONT_PATH           = "app/fonts/KGPrimaryPenmanship.ttf"
TEMPLATE_OUTPUT_DIR = "app/templates"
ALPHABET            = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyz0123456789"

# Resolucion interna de renderizado (alta para buen antialiasing antes de escalar)
TEMPLATE_RENDER_SIZE = 1024
TEMPLATE_FONT_SIZE   = 800

# Margen (px) alrededor del caracter dentro del canvas TARGET_SIZE x TARGET_SIZE
TEMPLATE_MARGIN      = int(TARGET_SIZE * 0.10)   # 10% => 12 px a cada lado

# ── Esqueletizacion ──────────────────────────────────────────────────────────
# True  → guarda tambien la plantilla esqueleto (1 px de grosor, para comparacion
#          "linea contra linea" con el trazo del alumno esqueletizado)
TEMPLATE_SAVE_SKELETON = True

# ── Niveles de dificultad (kernel de dilatacion sobre el esqueleto) ──────────
# El kernel es circular (MORPH_ELLIPSE). Cuanto mayor, mas ancho el "carril".
# Puedes agregar o quitar niveles; el script genera un PNG y NPY por nivel.
TEMPLATE_DIFFICULTY_KERNELS = {
    "principiante": 7,   # Carril ancho  — para ninos que empiezan
    "intermedio":   5,   # Carril medio
    "avanzado":     3,   # Carril estrecho — evaluacion precisa
}

# Iteraciones de dilatacion para cada nivel (normalmente 1 es suficiente)
TEMPLATE_DILATE_ITERATIONS = 1


# =============================================================================
# GENERACION DE PLANTILLAS
# =============================================================================

FONT_PATH            = "app/fonts/KGPrimaryPenmanship.ttf"
TEMPLATE_OUTPUT_DIR  = "app/templates"
ALPHABET             = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyz0123456789"

TEMPLATE_RENDER_SIZE = 1024
TEMPLATE_FONT_SIZE   = 800
TEMPLATE_MARGIN      = int(TARGET_SIZE * 0.10)   # 10% => ~12 px a cada lado
TEMPLATE_SAVE_SKELETON = True

# Niveles de dificultad: nombre -> tamano del kernel de dilatacion
# Mayor kernel => carril mas ancho => nivel mas facil
TEMPLATE_DIFFICULTY_KERNELS = {
    "principiante": 7,   # Carril ancho  (ninos que empiezan)
    "intermedio":   5,   # Carril medio
    "avanzado":     3,   # Carril estrecho (evaluacion precisa)
}
TEMPLATE_DILATE_ITERATIONS = 1

# =============================================================================
# DISTANCE TRANSFORM — Parametros de fidelidad
# =============================================================================

# Tolerancia (px) dentro de la cual el trazo del alumno se considera "correcto".
# Esto es lo que define el ancho efectivo del "carril" en la comparacion.
# Cuanto mayor, mas facil; cuanto menor, mas exigente.
DT_TOLERANCE_BY_LEVEL = {
    "principiante": 8.0,   # Muy permisivo: un trazo gordo de nino entra bien
    "intermedio":   5.0,   # Tolerancia moderada
    "avanzado":     3.0,   # Exigente: requiere precision casi perfecta
}
DT_TOLERANCE_DEFAULT  = 5.0   # Fallback si el nivel no coincide con ninguna clave

# Error promedio (px) a partir del cual la nota de precision es 0.
# Un error de 12 px promedio (casi la mitad del carril) => score_precision = 0.
DT_MAX_AVG_ERROR      = 12.0

# Pesos de la nota combinada del DT (deben sumar 1.0)
DT_WEIGHT_PRECISION   = 0.65   # Peso de "donde escribe el alumno"
DT_WEIGHT_COVERAGE    = 0.35   # Peso de "que tanto del esqueleto cubre"

# =============================================================================
# SCORING — Ponderacion de metricas (deben sumar 1.0)
# =============================================================================

SCORING_WEIGHTS = {
    "dt_precision":  0.30,  # Fidelidad de forma (Distance Transform precision)
    "dt_coverage":   0.20,  # Cobertura del esqueleto (el alumno trazo todo)
    "topology":      0.20,  # Integridad estructural (bucles/agujeros)
    "ssim":          0.12,  # Similitud estructural de masa
    "procrustes":    0.10,  # Ajuste geometrico global
    "hausdorff":     0.04,  # Penalizacion por trazos muy erraticos
    "trajectory":    0.02,  # Trayectoria estimada (orden del trazo)
    "cosine":        0.02,  # Coherencia de angulos de segmentos
}

# Score que recibe la topologia segun si coincide o no con la plantilla
SCORING_TOPO_HIT  = 100.0   # Bucles correctos
SCORING_TOPO_MISS = 30.0    # Bucles incorrectos (penalizacion fuerte)

# Factor de penalizacion por trayectoria: cada unidad de distancia DTW
# descuenta este valor en puntos de score de trayectoria
SCORING_TRAJ_FACTOR = 3.0

# =============================================================================
# NORMALIZER — Eliminacion de islas de ruido (remove_specks)
# =============================================================================

# Area minima absoluta (px) de un componente conectado para conservarlo.
# Componentes con menos pixeles que esto se consideran ruido y se eliminan.
SPECK_MIN_AREA_PX   = 15

# Fraccion del area total del ROI. Se usa el mayor entre este y SPECK_MIN_AREA_PX.
# Subir (0.002) si quedan manchas; bajar (0.0002) si se pierden trazos finos.
SPECK_AREA_RATIO    = 0.0005

# =============================================================================
# IMAGE QUALITY — Umbrales para diagnóstico adaptativo
# (usados por app/core/image_quality.py)
# =============================================================================

# Borrosidad: varianza del Laplaciano
# < BLUR_THRESHOLD → imagen considerada borrosa
BLUR_THRESHOLD          = 50.0

# Contraste: desviación estándar de píxeles
# < CONTRAST_LOW  → bajo contraste (lápiz muy suave o papel gris)
CONTRAST_LOW            = 20.0
# >= CONTRAST_HIGH → contraste suficiente para Otsu
CONTRAST_HIGH           = 35.0

# Brillo: media de píxeles
BRIGHTNESS_DARK         = 60.0    # < esto → imagen oscura
BRIGHTNESS_OVEREXPOSED  = 210.0   # > esto → sobreexpuesta / flash directo

# Sombra: variación local de iluminación respecto al fondo estimado
# > SHADOW_THRESHOLD → hay sombra significativa de mano o ángulo de celular
SHADOW_THRESHOLD        = 0.25

# =============================================================================
# BINARIZER — Sauvola (app/core/binarizer.py)
# =============================================================================

# Usar método Sauvola si scikit-image está disponible.
# Más robusto en papel cuadriculado texturizado. Más lento que adaptativo.
# Si False, usa Otsu o Adaptativo Gaussiano según la calidad de imagen.
USE_SAUVOLA             = False
SAUVOLA_WINDOW          = 25      # Tamaño de ventana local (px, impar)
SAUVOLA_K               = 0.2     # Sensibilidad (0.1=suave, 0.5=agresivo)

# =============================================================================
# ILLUMINATION — Corrección de fondo por división (app/core/illumination.py)
# =============================================================================

# El blur de fondo se calcula como max(31, lado_menor // BG_BLUR_DIVISOR)
# Valor más pequeño → blur más localizado (mejor para sombras pequeñas)
# Valor más grande  → blur más global (mejor para gradientes amplios)
BG_BLUR_DIVISOR         = 3