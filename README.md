# 🏥 Dermatoscopio Portátil con IA

## 📋 Descripción

Sistema de **clasificación de lesiones cutáneas mediante IA** usando el dataset HAM10000. Detecta **3 tipos principales** de lesiones de piel con mitigación de sesgo para diferentes tonos de piel.

### ✨ Características

- ✅ **Modelo EfficientNetB0** con Transfer Learning
- ✅ **Mitigación de sesgo** (Dark Skin Simulation)
- ✅ **Optimizado para Raspberry Pi 5** (TFLite: 15 MB)
- ✅ **Entrenamiento en Google Colab** (GPU gratuita)
- ✅ **Dataset HAM10000** (10,015 imágenes → 3 clases)

---

## 🚀 Inicio Rápido (Google Colab)

Copia y pega en Google Colab: https://colab.research.google.com

```python
# Celda 1: Setup básico
!pip install -q tensorflow scikit-learn pandas matplotlib
from google.colab import drive
drive.mount('/content/drive')

# Celda 2: Clonar repositorio
!git clone https://github.com/TU_USUARIO/dermatoscopio-portatil-IA.git
%cd dermatoscopio-portatil-IA

# Celda 3: Descargar datos desde Drive
# Sube manualmente data_processed.zip a tu Drive, luego:
!cp '/content/drive/MyDrive/data_processed.zip' .
!unzip -q data_processed.zip

# Celda 4: Entrenar
!python train.py --epochs 30 --fine_tune --tflite

# Celda 5: Descargar resultados
from google.colab import files
!zip -r models.zip models/
files.download('models.zip')
```

---

## 📊 Clases del Modelo

| Clase | Descripción | Muestras |
|-------|-------------|----------|
| **mel** | Melanoma (maligno) | 1,113 |
| **nv** | Lunar Benigno | 6,705 |
| **other** | Otras lesiones | 2,197 |

**Total:** 10,015 imágenes

---

## 💻 Instalación Local

```bash
git clone https://github.com/TU_USUARIO/dermatoscopio-portatil-IA.git
cd dermatoscopio-portatil-IA

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

python train.py --epochs 30 --fine_tune --tflite
```

---

## 📝 Parámetros de train.py

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
```
dermatoscopio-portatil-IA/
├── README.md
├── requirements.txt
├── train.py
├── .gitignore
├── src/
│   ├── 00_diagnóstico.py
│   ├── 01_download_metadata.py
│   ├── 02_eda_analysis.py
│   ├── 03_data_pipeline.py
│   ├── model.py
│   ├── data_loader.py
│   └── inference.py
└── data/
    └── processed/

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

