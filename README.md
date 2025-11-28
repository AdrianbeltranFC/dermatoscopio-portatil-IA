# 🏥 Dermatoscopio Portátil con IA

## 📋 Descripción

Sistema **de segmentación + clasificación** de lesiones cutáneas usando:
- **Segmentación YCbCr**: Robusta a variaciones de tono de piel
- **Clasificación EfficientNetB0**: 3 categorías (Melanoma, Nevo Benigno, Otros)
- **Validación en Raspberry Pi 5**: Con cámara en vivo

### ✨ Características

- ✅ **Segmentación YCbCr** (desacoplada de luminancia)
- ✅ **Mitigación de sesgo** con Dark Skin Simulation
- ✅ **Optimizado para todo tipo de pieles** (incluidas pieles oscuras)
- ✅ **Entrenamiento en Google Colab** (GPU gratuita)
- ✅ **Inferencia Raspberry Pi 5** (TFLite optimizado)
- ✅ **Visualización dual**: Segmentación + Clasificación

---

## 🚀 INICIO RÁPIDO: Entrenar en Google Colab

### Paso 1: Preparar datos localmente (en tu computadora)

```bash
# 1. Descargar dataset HAM10000 (~10GB)
python src/01_download_metadata.py --output data/metadata.csv

# 2. Procesar y dividir en train/val/test (7 clases → 3 clases)
python src/03_data_pipeline.py --meta data/metadata.csv --out data/processed

# 3. Crear ZIP comprimido
python create_data_zip.py --output data_processed.zip

# 4. Verificar tamaño (debe ser ~2.3GB)
ls -lh data_processed.zip
```

### Paso 2: Subir a Google Drive

1. Ve a **Google Drive**: https://drive.google.com
2. Sube `data_processed.zip` a **Mi unidad** (carpeta raíz)

### Paso 3: Entrenar en Google Colab

Ve a **Google Colab**: https://colab.research.google.com

**Copia EXACTAMENTE esto en UNA SOLA CELDA:**

```python
# Celda única para Colab - VERSIÓN CORREGIDA
import subprocess, sys, os

print("[1/6] Instalando dependencias...")
for pkg in ['tensorflow', 'scikit-learn', 'pandas', 'matplotlib', 'opencv-python']:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("✓ Listo\n")

print("[2/6] Montando Google Drive...")
from google.colab import drive
drive.mount('/content/drive')
print("✓ Listo\n")

print("[3/6] Clonando repositorio...")
os.chdir('/content')
os.system('git clone https://github.com/AdrianbeltranFC/dermatoscopio-portatil-IA.git 2>/dev/null')
os.chdir('dermatoscopio-portatil-IA')
print("✓ Listo\n")

print("[4/6] Descargando datos...")
os.system("cp '/content/drive/MyDrive/data_processed.zip' .")

# IMPORTANTE: Usar -v (verbose) para ver progreso, y verificar resultado
print("Descomprimiendo...")
result = os.system("unzip -q data_processed.zip && echo 'OK' || echo 'ERROR'")

# Verificar que se descomprimió correctamente
import glob
train_mel = glob.glob('data/processed/train/mel/*.jpg')
train_nv = glob.glob('data/processed/train/nv/*.jpg')
train_other = glob.glob('data/processed/train/other/*.jpg')

total = len(train_mel) + len(train_nv) + len(train_other)
print(f"\n✓ Imágenes en TRAIN:")
print(f"  mel: {len(train_mel)}")
print(f"  nv: {len(train_nv)}")
print(f"  other: {len(train_other)}")
print(f"  TOTAL: {total}\n")

if total == 0:
    print("❌ ERROR: No se encontraron imágenes")
    print("Intentando descompresión manual...")
    os.system("unzip data_processed.zip")
    print("Verificando de nuevo...")
    os.system("ls -la data/processed/train/")
else:
    print("[5/6] Entrenando modelo...")
    os.system('python train.py --epochs 30 --fine_tune --tflite')
    
    print("\n[6/6] Descargando modelos...")
    os.system('zip -r -q models.zip models/')
    from google.colab import files
    files.download('models.zip')
    print("✓ Descarga completada")
```

**Presiona Ctrl+Enter y espera (15-40 minutos)**

---

## 📊 Clases del Modelo

| Clase | Descripción | Muestras |
|-------|-------------|----------|
| **mel** | Melanoma (maligno) | 1,113 |
| **nv** | Nevo Benigno (Lunar) | 6,705 |
| **other** | Otras lesiones | 2,197 |

**Total:** 10,015 imágenes | **División:** 70% train, 15% val, 15% test

---

## 🎯 Pipeline Técnico Completo

### 1. Segmentación (YCbCr)

**Uso local:**
```python
from src.segmentation import SkinLesionSegmenter

segmenter = SkinLesionSegmenter(debug=True)
result = segmenter.segment("lesion.jpg", output_dir="./results/")

if result['success']:
    print(f"✓ Lesión segmentada")
    print(f"  Área: {result['area']:.0f} píxeles")
```

**¿Por qué YCbCr?**
- RGB: Luminancia y crominancia entrelazadas → **falla en pieles oscuras**
- YCbCr: Y (luminancia) **SEPARADA** de Cb-Cr (crominancia)
- Piel normal agrupa en: **Cb ∈ [77,127], Cr ∈ [133,173]** (independiente del tono)
- Lesiones se desvían de este clúster (detectable en cualquier tono)

### 2. Clasificación (EfficientNetB0)

**Uso local:**
```python
from src.inference import SkinLesionInference

inference = SkinLesionInference("models/skin_lesion_classifier.h5")
result = inference.process_image("lesion.jpg", output_dir="./results/")

if result['success']:
    print(f"✓ Clasificación: {result['class']}")
    print(f"✓ Confianza: {result['confidence']*100:.1f}%")
    for clase, prob in result['all_predictions'].items():
        print(f"  {clase}: {prob*100:.1f}%")
```

### 3. Visualización Dual

La salida muestra:
- **Imagen Original + Contorno** de la lesión
- **Máscara de Segmentación** (blanco = lesión)
- **Resultado**: Clase + Confianza (%)

---

## 📱 Uso en Raspberry Pi 5

```python
from src.raspberry_pi_app import RaspberryPiApp

# Iniciar app
app = RaspberryPiApp("models/tflite/skin_lesion_classifier_float16.tflite")
app.run(save_dir="./captures/")

# Controles en vivo:
# 's' - Capturar imagen y procesar
# 'q' - Salir
```

**Requisitos en Raspberry Pi:**
```bash
pip install tensorflow tflite-runtime opencv-python pillow numpy
```

---

## 💻 Instalación Local

```bash
# Clonar
git clone https://github.com/AdrianbeltranFC/dermatoscopio-portatil-IA.git
cd dermatoscopio-portatil-IA

# Entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

**Entrenar localmente:**
```bash
python train.py \
  --data_dir data/processed \
  --epochs 30 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --fine_tune \
  --tflite \
  --tflite_format float16
```

---

## 📂 Estructura del Repositorio