# Dermatoscopio Portátil con IA

## Descripción

Sistema de **clasificación de lesiones cutáneas mediante IA** usando el dataset HAM10000. Detecta 7 tipos de lesiones de piel con mitigación de sesgo para diferentes tonos de piel.

### Características

- ✅ **Modelo EfficientNetB0** con Transfer Learning
- ✅ **Mitigación de sesgo** (Dark Skin Simulation)
- ✅ **Optimizado para Raspberry Pi 5** (TFLite: 15 MB)
- ✅ **Entrenamiento en Google Colab** (GPU gratuita)
- ✅ **Dataset HAM10000** (10,015 imágenes de lesiones cutáneas)

---

## Inicio Rápido (Google Colab - RECOMENDADO)

### Opción A: Colab Notebook (más fácil)

```python
# Copia esto en una celda de Colab (https://colab.research.google.com)

# 1. Montar Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Instalar dependencias
!pip install -q tensorflow scikit-learn pandas matplotlib

# 3. Descargar datos del repositorio
!wget -q https://github.com/TU_USUARIO/dermatoscopio-portatil-IA/releases/download/v1.0/data_processed.zip
!unzip -q data_processed.zip

# 4. Clonar repositorio
!git clone https://github.com/TU_USUARIO/dermatoscopio-portatil-IA.git
%cd dermatoscopio-portatil-IA

# 5. Entrenar
!python train.py --epochs 50 --fine_tune --tflite
```

**Resultado:** Modelos listos en `models/` para descargar

---

### Opción B: Script Directo en Colab

```python
# Celda 1: Setup
from google.colab import drive
drive.mount('/content/drive')
!pip install -q tensorflow scikit-learn pandas matplotlib

# Celda 2: Descargar y entrenar
!cd /tmp && git clone https://github.com/TU_USUARIO/dermatoscopio-portatil-IA.git
!wget https://github.com/TU_USUARIO/dermatoscopio-portatil-IA/releases/download/v1.0/data_processed.zip
!unzip -q data_processed.zip -d /tmp/dermatoscopio-portatil-IA/
!cd /tmp/dermatoscopio-portatil-IA && python train.py --epochs 50 --fine_tune --tflite

# Celda 3: Descargar
from google.colab import files
!cd /tmp/dermatoscopio-portatil-IA && zip -r models.zip models/
files.download('models.zip')
```

---

## Instalación Local

### Requisitos

- Python 3.8+
- GPU NVIDIA (opcional, pero recomendado)
- 20 GB espacio en disco

### Pasos

```bash
# 1. Clonar repositorio
git clone https://github.com/TU_USUARIO/dermatoscopio-portatil-IA.git
cd dermatoscopio-portatil-IA

# 2. Entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Dependencias
pip install -r requirements.txt

# 4. Descargar datos (solo si entrenas localmente)
# Descarga data_processed.zip desde GitHub y descomprime en data/

# 5. Entrenar
python train.py --epochs 50 --fine_tune --tflite
```

---

## Dataset HAM10000
```
| Clase | Nombre | Cantidad | Descripción |
|-------|--------|----------|-------------|
| akiec | Actinic Keratosis | 611 | Precanceroso |
| bcc | Basal Cell Carcinoma | 514 | Cáncer de piel |
| bkl | Benign Keratosis | 1,099 | Benigno |
| df | Dermatofibroma | 115 | Fibroma |
| mel | Melanoma | 1,113 | Melanoma |
| nv | Melanocytic Nevi | 6,705 | Lunar |
| vasc | Vascular | 286 | Vasos sanguíneos |
```
**Total:** 10,015 imágenes | **Distribución:** Train 70%, Val 15%, Test 15%
---

## Arquitectura del Modelo
```
Input (224x224x3)
    ↓
[Dark Skin Simulation Augmentation]
  - RandomBrightness (factor: -0.2 to 0.1)
  - RandomContrast
  - RandomFlip, RandomRotation, RandomZoom
    ↓
Rescaling (1/255)
    ↓
EfficientNetB0 (ImageNet weights, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(7, softmax)
    ↓
Output (7 clases)
```

## Uso del Script train.py
Parámetros
```
python train.py \
  --data_dir data/processed      # Directorio de datos
  --image_size 224               # Tamaño de imagen
  --batch_size 32                # Batch size
  --epochs 50                    # Epochs
  --learning_rate 1e-3           # Learning rate
  --fine_tune                    # Habilitar fine-tuning
  --tflite                       # Exportar a TFLite
  --tflite_format float16        # float16 o int8
```
Ejemplos
```
# Entrenamiento rápido (CPU)
python train.py --epochs 10 --batch_size 16

# Entrenamiento completo (GPU)
python train.py --epochs 50 --fine_tune --tflite

# Solo exportar TFLite
python train.py --tflite --tflite_format int8
```

# Estructura del Repositorio
```
dermatoscopio-portatil-IA/
├── README.md                      # Este archivo
├── requirements.txt               # Dependencias
├── train.py                       # ⭐ Script único de entrenamiento
├── .gitignore                     # Git exclusiones
│
├── src/                           # Código fuente
│   ├── __init__.py
│   ├── 00_diagnóstico.py          # Diagnostica problemas
│   ├── 01_download_metadata.py    # Descarga dataset
│   ├── 02_eda_analysis.py         # Análisis exploratorio
│   ├── 03_data_pipeline.py        # Divide train/val/test
│   ├── model.py                   # Definición de modelos
│   ├── data_loader.py             # Cargadores
│   └── inference.py               # Predicciones
│
├── data/                          # (NO en repo, usar Drive)
│   ├── processed/
│   │   ├── train/<clase>/
│   │   ├── val/<clase>/
│   │   └── test/<clase>/
│   └── metadata.csv
│
├── models/                        # (Generados por entrenamiento)
│   ├── checkpoints/
│   ├── tflite/
│   └── skin_lesion_classifier.h5
│
└── notebooks/                     # Jupyter (opcional)
```
# Solución a problemas comunes
❌ "No se encuentran los datos"
```
# Opción 1: Descargar desde GitHub
wget https://github.com/TU_USUARIO/dermatoscopio-portatil-IA/releases/download/v1.0/data_processed.zip
unzip data_processed.zip

# Opción 2: Generar localmente
python src/01_download_metadata.py --output data/metadata.csv
python src/03_data_pipeline.py --meta data/metadata.csv --out data/processed
```
❌ "TensorFlow no detecta GPU"
```
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Si está vacío, instala CUDA/cuDNN
```
# Salidas del entrenamiento
Después de ejecutar train.py:
```
models/
├── skin_lesion_classifier.h5           # Modelo Keras completo (~160 MB)
├── checkpoints/
│   ├── best_model.h5
│   ├── training_history.png            # Gráficas
│   ├── confusion_matrix.csv
│   └── classification_report.txt
└── tflite/
    ├── skin_lesion_classifier_float16.tflite  # Para Raspberry Pi (~15 MB)
    └── skin_lesion_classifier_int8.tflite

```
# Uso en Raspberry Pi 
```
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar modelo TFLite
interpreter = tf.lite.Interpreter(
    model_path="skin_lesion_classifier_float16.tflite"
)
interpreter.allocate_tensors()

# Cargar imagen
img = Image.open("lesion.jpg").resize((224, 224))
img_array = np.array(img, dtype=np.uint8)
img_array = np.expand_dims(img_array, axis=0)

# Predicción
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"Predicción: {predicted_class} ({confidence*100:.1f}%)")
```
Requisitos en Pi:
```
pip install tensorflow tflite-runtime pillow numpy
```
#  Referencias
Dataset: HAM10000 Kaggle

Paper: The HAM10000 dataset

EfficientNet: Arxiv 1905.11946

Bias in ML: Nawaz et al. 2022

TensorFlow Lite: Documentation

