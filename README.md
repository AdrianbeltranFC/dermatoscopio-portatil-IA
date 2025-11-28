#  Dermatoscopio Portátil con IA

##  Descripción

Sistema **de segmentación + clasificación** de lesiones cutáneas usando:
- **Segmentación YCbCr**: Robusta a variaciones de tono de piel
- **Clasificación EfficientNetB0**: 3 categorías (Melanoma, Nevo (Lunar), Otro)
- **Validación en Raspberry Pi 5**: Con cámara en vivo

### Características

- ✅ **Segmentación YCbCr** (desacoplada de luminancia)
- ✅ **Mitigación de sesgo** con Dark Skin Simulation
- ✅ **Optimizado para todo tipo de pieles**
- ✅ **Entrenamiento en Google Colab** (GPU)
- ✅ **Inferencia Raspberry Pi** (TFLite)
- ✅ **Visualización dual**: Segmentación + Clasificación

---

## Pipeline Completo

### Paso 1: Segmentación (YCbCr)
```python
import cv2
import numpy as np

# Cargar imagen
img = cv2.imread("lesion.jpg")

# Convertir a YCbCr
img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Umbralizar para segmentar
_, img_segmentada = cv2.threshold(img_ycbcr[:,:,0], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Guardar resultado
cv2.imwrite("segmentacion.jpg", img_segmentada)
```
Imagen Original → YCbCr → Análisis Cb-Cr → Máscara → ROI

### Paso 2: Clasificación (EfficientNetB0)
ROI → 224×224 → EfficientNetB0 → Probabilidades → Clase
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar modelo
interpreter = tf.lite.Interpreter("models/tflite/skin_lesion_classifier_float16.tflite")
interpreter.allocate_tensors()

# Preprocesar imagen
img = Image.open("lesion.jpg").resize((224, 224))
img_array = np.array(img, dtype=np.uint8)
img_array = np.expand_dims(img_array, axis=0)

# Inferir
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

# Obtener resultados
predictions = interpreter.get_tensor(output_details[0]['index'])
class_names = ['mel', 'nv', 'other']
result = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"{result}: {confidence*100:.1f}%")
```
### Paso 3: Visualización
Original + Contorno | Máscara de Segmentación
Resultado: [Clase] ([Confianza]%)

---
## Fundamentación Técnica
¿Por qué YCbCr?

RGB: Luminancia y crominancia entrelazadas (fallos en pieles oscuras)

YCbCr: Y (luminancia) separada de Cb-Cr (crominancia)

Piel normal: Agrupa compactamente en Cb ∈ [77,127], Cr ∈ [133,173]
Lesiones: Se desvían de este clúster


## Parámetros del modelo
```
Cb: [77, 127]     # Diferencia de azul
Cr: [133, 173]    # Diferencia de rojo
Kernel: 5×5, 11×11, 21×21 ELLIPSE (morfología)
```
## Documentación y justificación completa
Ver: 
```
TECHNICAL_DOCUMENTATION.txt
```

Fundamentación teórica (YCbCr vs otros espacios)
Arquitectura del sistema
Flujo de trabajo completo
Parámetros críticos
Referencias académicas utilizadas

## Clases del Modelo

| Clase | Descripción | Muestras |
|-------|-------------|----------|
| **mel** | Melanoma (maligno) | 1,113 |
| **nv** | Lunar Benigno | 6,705 |
| **other** | Otras lesiones | 2,197 |

**Total:** 10,015 imágenes

## Entrenamiento en Colab
```
# Celda 1: Setup
!pip install -q tensorflow scikit-learn pandas matplotlib
from google.colab import drive
drive.mount('/content/drive')

# Celda 2: Clonar + Datos
!git clone https://github.com/AdrianbeltranFC/dermatoscopio-portatil-IA.git
%cd dermatoscopio-portatil-IA
!cp '/content/drive/MyDrive/data_processed.zip' .
!unzip -q data_processed.zip

# Celda 3: Entrenar
!python train.py --epochs 30 --fine_tune --tflite

# Celda 4: Descargar
from google.colab import files
!zip -r models.zip models/
files.download('models.zip')
```
---

## Instalación Local

```bash
git clone https://github.com/TU_USUARIO/dermatoscopio-portatil-IA.git
cd dermatoscopio-portatil-IA

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

python train.py --epochs 30 --fine_tune --tflite
```

---

## Parámetros de train.py

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

## Estructura del Repositorio
```
dermatoscopio-portatil-IA/
├── README.md
├── requirements.txt
├── train.py
├── .gitignore
├── src/
│   ├── 00_diagnóstico.py
│   ├── 00_make_metadata.py
│   ├── 01_eda_analysis.py
│   ├── 02_cluster_embeddings.py
│   ├── 03_data_pipeline.py
│   ├── dataset.py
│   ├── model.py
│   └── inference.py
│   ├── raspberry_pi_app.py
│   ├── segmentation.py
└── data/
    └── processed/
    └── raw/



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
├── skin_lesion_classifier.h5              # Keras completo
├── checkpoints/
│   ├── training_history.png
│   ├── confusion_matrix.csv
│   └── classification_report.txt
└── tflite/
    └── skin_lesion_classifier_float16.tflite  # Raspberry Pi

```
# Uso en Raspberry Pi 
```
import tensorflow as tf
import numpy as np
from PIL import Image

interpreter = tf.lite.Interpreter("models/tflite/skin_lesion_classifier_float16.tflite")
interpreter.allocate_tensors()

img = Image.open("lesion.jpg").resize((224, 224))
img_array = np.array(img, dtype=np.uint8)
img_array = np.expand_dims(img_array, axis=0)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

predictions = interpreter.get_tensor(output_details[0]['index'])
class_names = ['mel', 'nv', 'other']
result = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"{result}: {confidence*100:.1f}%")
```
#  Referencias
Dataset: HAM10000 Kaggle

Paper: The HAM10000 dataset

EfficientNet: Arxiv 1905.11946

Bias in ML: Nawaz et al. 2022

TensorFlow Lite: Documentation

Celebi et al. (2009): Color-based skin lesion boundary detection

Esteva et al. (2019): Dermatologist-level classification

Tan & Le (2019): EfficientNet - Rethinking Model Scaling
ITU-R BT.601: Estándar YCbCr