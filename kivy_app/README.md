# Tutor de Caligrafía — App Android (Kivy)

App móvil para evaluar trazos caligráficos usando la API del Tutor Inteligente.

## Estructura del proyecto

```
kivy_app/
├── main.py                      # Entry point, ScreenManager
├── config.py                    # Configuración global y persistencia
├── api_client.py                # Cliente HTTP (evaluate_image, evaluate_plana)
├── buildozer.spec               # Build config para APK Android
├── requirements.txt             # Dependencias Python (escritorio)
│
├── screens/
│   ├── base_screen.py           # Clase base con helpers compartidos
│   ├── home_screen.py           # Pantalla de inicio
│   ├── capture_screen.py        # Captura / galería + config de evaluación
│   ├── plana_screen.py          # Modo plana (múltiples caracteres)
│   ├── evaluate_screen.py       # Loading mientras la API responde
│   ├── result_screen.py         # Resultados: score, imágenes, métricas
│   └── config_screen.py         # URL de la API y preferencias
│
├── components/
│   └── score_gauge.py           # Widget medidor circular de puntuación
│
├── services/
│   ├── camera_service.py        # Abstracción cámara/galería (plyer)
│   ├── image_service.py         # Base64 → archivo, resize, limpieza
│   └── threading_utils.py       # Llamadas HTTP en background sin bloquear UI
│
└── assets/
    ├── fonts/                   # Roboto-Regular.ttf, Roboto-Bold.ttf
    └── images/                  # icon.png (512×512), splash.png (1024×500)
```

## Flujo de pantallas

```
HomeScreen
    ├── CaptureScreen → EvaluateScreen → ResultScreen
    ├── PlanaScreen   → EvaluateScreen → ResultScreen
    └── ConfigScreen
```

## Instalación — desarrollo en escritorio

```bash
cd kivy_app
pip install -r requirements.txt
python main.py
```

## Build APK Android

### Importante (Windows)

Buildozer / python-for-android **no compilan APK nativamente en Windows**. En Windows la forma recomendada es:

- **WSL2 (Ubuntu)** para compilar el APK
- **ADB en Windows** (o copiar el APK) para instalarlo en tu teléfono

Abajo tienes un flujo completo Windows → Android.

### Prerequisitos (Ubuntu / WSL2)

```bash
# 1. Instalar buildozer
pip install buildozer

# 2. Dependencias del sistema (Ubuntu)
sudo apt-get install -y \
    python3-pip build-essential git \
    libssl-dev libffi-dev libgmp-dev \
    autoconf automake libtool \
    zlib1g-dev openjdk-17-jdk unzip

# 3. Descargar fuentes en assets/fonts/
# Roboto: https://fonts.google.com/specimen/Roboto
# Copiar: Roboto-Regular.ttf, Roboto-Bold.ttf, Roboto-Light.ttf
```

### Compilar en Android desde Windows (WSL2)

1. Instala WSL2 (una vez) y Ubuntu:

```powershell
wsl --install -d Ubuntu
```

2. Abre Ubuntu (WSL) e instala prerequisitos:

```bash
sudo apt update
sudo apt-get install -y python3 python3-pip python3-venv build-essential git \
  libssl-dev libffi-dev libgmp-dev autoconf automake libtool \
  zlib1g-dev openjdk-17-jdk unzip

python3 -m pip install --upgrade pip
python3 -m pip install buildozer
```

3. Entra al proyecto desde WSL (tu disco E: se monta como `/mnt/e/`):

```bash
cd /mnt/e/Estadia/kivy_app
```

4. Compila el APK debug:

```bash
buildozer android debug
```

El APK quedará en `kivy_app/bin/`.

### Generar APK (debug)

```bash
cd kivy_app
buildozer android debug
# APK generado en: bin/tutorcaligrafia-1.0.0-armeabi-v7a-debug.apk
```

### Generar APK (release)

```bash
buildozer android release
```

### Instalación directa en dispositivo

```bash
buildozer android debug deploy run
```

### Instalar el APK en tu teléfono (Windows)

Opción A: copiar el APK y abrirlo en el teléfono

- Copia el archivo `kivy_app/bin/*.apk` al teléfono (USB/Drive)
- En Android habilita **“Instalar apps desconocidas”** para tu gestor de archivos/Drive
- Abre el APK e instálalo

Opción B: instalar con ADB (recomendado)

1. Activa en el teléfono:
   - Opciones de desarrollador
   - Depuración USB
2. Instala “Android Platform Tools” en Windows (incluye `adb`)
3. Conecta el teléfono por USB y ejecuta:

```powershell
adb devices
adb install -r .\kivy_app\bin\tutorcaligrafia-1.0.0-armeabi-v7a-debug.apk
```

### Cómo se lo envío a otra persona

- **Debug APK**: sirve para pruebas rápidas. Puedes enviarlo por Drive/WhatsApp/Email.
  - La otra persona debe habilitar **instalación de apps desconocidas**.
  - Android puede bloquear APKs descargados por seguridad; en ese caso, usar Drive o copiar por USB suele funcionar mejor.

- **Release APK (firmado)**: lo ideal para compartir fuera de tu teléfono.
  - Genera release con Buildozer y firma con tu keystore.
  - Comparte el APK firmado (y opcionalmente un `SHA256` para verificación).

Para obtener el hash en Windows:

```powershell
Get-FileHash .\kivy_app\bin\*.apk -Algorithm SHA256
```

## Configuración de la API

1. Abre la app → toca el ícono ⚙ en la pantalla de inicio
2. Ingresa la URL de tu servidor FastAPI  
   Ejemplo: `http://192.168.1.100:8000`
3. Toca **Probar conexión** para verificar
4. Toca **Guardar**

> La URL se persiste en `user_config.json` localmente en el dispositivo.

## Principios SOLID aplicados

| Principio | Implementación |
|---|---|
| **SRP** | `api_client.py` solo HTTP · `image_service.py` solo imágenes · `camera_service.py` solo cámara |
| **OCP** | Nuevas pantallas extienden `BaseScreen` sin modificarla |
| **LSP** | Todas las pantallas implementan el mismo protocolo `on_enter / go_to` |
| **ISP** | `api_client` expone solo `evaluate_image` y `evaluate_plana`; nada interno |
| **DIP** | Las pantallas dependen de `api_client` (abstracción), no de `requests` directamente |
