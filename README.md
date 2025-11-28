# 🏥 Dermatoscopio Portátil con IA

## 📋 Descripción

Sistema de **clasificación de lesiones cutáneas mediante IA** usando el dataset HAM10000. Detecta 7 tipos de lesiones de piel con mitigación de sesgo para diferentes tonos de piel.

### ✨ Características

- ✅ **Modelo EfficientNetB0** con Transfer Learning
- ✅ **Mitigación de sesgo** (Dark Skin Simulation)
- ✅ **Optimizado para Raspberry Pi 5** (TFLite: 15 MB)
- ✅ **Entrenamiento en Google Colab** (GPU gratuita)
- ✅ **Dataset HAM10000** (10,015 imágenes de lesiones cutáneas)

---

## 🚀 Inicio Rápido (Google Colab - RECOMENDADO)

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

## 💻 Instalación Local

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

## 📊 Dataset HAM10000

| Clase | Nombre | Cantidad | Descripción |
|-------|--------|----------|-------------|
| akiec | Actinic Keratosis | 611 | Precanceroso |
| bcc | Basal Cell Carcinoma | 514 | Cáncer de piel |
| bkl | Benign Keratosis | 1,099 | Benigno |
| df | Dermatofibroma | 115 | Fibroma |
| mel | Melanoma | 1,113 | Melanoma |
| nv | Melanocytic Nevi | 6,705 | Lunar |
| vasc | Vascular | 286 | Vasos sanguíneos |

**Total:** 10,015 imágenes | **Distribución:** Train 70%, Val 15%, Test 15%

---

## 🧠 Arquitectura del Modelo


