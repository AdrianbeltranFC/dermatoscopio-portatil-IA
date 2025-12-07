# 🏥 DermaScan Pro: Dermatoscopio Portátil con IA

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?style=flat&logo=tensorflow)
![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-C51A4A?style=flat&logo=raspberry-pi)
![License](https://img.shields.io/badge/License-MIT-green)

Sistema embebido de **apoyo al diagnóstico dermatológico** diseñado para clasificar lesiones cutáneas en tiempo real. El proyecto se centra en la robustez frente a variaciones de iluminación y tonos de piel (especialmente pieles oscuras), utilizando visión por computadora y Deep Learning (EfficientNetB0).

---

## 📋 Descripción del Proyecto

Este repositorio contiene el código fuente, documentación y herramientas de entrenamiento para un dermatoscopio digital inteligente. El sistema es capaz de:

1.  **Segmentar** la lesión de la piel sana utilizando algoritmos de visión artificial adaptativos.
2.  **Clasificar** la lesión en 3 categorías clínicas críticas.
3.  **Ejecutarse en el borde (Edge AI)** sobre una Raspberry Pi 5 o en PC de escritorio.

### ✨ Características Principales

* **Segmentación Híbrida:** Implementación de algoritmos **YCbCr Adaptativo** y **Canal Azul + Otsu** para aislar lunares incluso en pieles con alta melanina.
* **Modelo EfficientNetB0:** Red neuronal convolucional optimizada mediante Transfer Learning.
* **Optimización TFLite:** Inferencia cuantizada (Float16) para latencia baja (~50ms) en Raspberry Pi.
* **Interfaz Gráfica (GUI):** Aplicación de escritorio moderna (Tkinter + ttkbootstrap) con visualización de probabilidades y generación de reportes.
* **Dataset Balanceado:** Pipeline de datos customizado que mapea el dataset HAM10000 a 3 clases clínicas y corrige el desbalanceo.

---

## 🛠️ Hardware Recomendado

* **Procesador:** Raspberry Pi 5 (o PC con Windows/Linux para la versión de escritorio).
* **Cámara:** Raspberry Pi Camera Module 3 (o Webcam USB de alta resolución con macro).
* **Óptica:** Lente dermatoscópico (10x) con iluminación LED controlada.

---

## 📊 Metodología Técnica

### 1. Pipeline de Datos (HAM10000)
El script `src/03_data_pipeline.py` procesa las 7 clases originales del dataset HAM10000 y las agrupa en 3 categorías clínicas para facilitar el triaje:

| Clase Modelo | Descripción | Clases Originales HAM10000 | Acción Sugerida |
| :--- | :--- | :--- | :--- |
| **mel** | **Melanoma (Maligno)** | `mel` | 🚨 Derivar a especialista urgente |
| **nv** | **Nevo (Benigno)** | `nv` | ✅ Seguimiento rutinario |
| **other** | **Otras Lesiones** | `bkl`, `bcc`, `akiec`, `vasc`, `df` | ⚠️ Evaluar según caso |

### 2. Segmentación Adaptativa

Se abordaron los retos de iluminación variable y pieles oscuras mediante dos estrategias (ver `Interfaz/app.py`):
* **Estrategia Principal (Canal Azul + Otsu):** Aprovecha que la melanina absorbe fuertemente la luz azul. Se aplica un umbral automático (Otsu) sobre el canal B para separar la lesión.
* **Fallback (ROI Inteligente):** Si la segmentación falla, el sistema aplica un recorte central con margen del 40% para asegurar que la red neuronal reciba contexto de piel sana.

### 3. Modelo de Clasificación
Se utiliza **EfficientNetB0** debido a su eficiencia de parámetros (5.3M).
* **Entrenamiento:** Técnica de "Infinite Repeat" para balancear físicamente las clases minoritarias (Melanoma).
* **Input:** Imágenes de 224x224 píxeles (Valores crudos 0-255, sin normalización 1/255 previa, ajustado en capas internas).
* **Métricas (Validación):** Recall en Melanoma > 80% (Priorizando sensibilidad sobre especificidad).

---

## 🚀 Instalación y Uso

### Prerrequisitos
Clona el repositorio e instala las dependencias:

```bash
git clone [https://github.com/AdrianbeltranFC/dermatoscopio-portatil-IA.git](https://github.com/AdrianbeltranFC/dermatoscopio-portatil-IA.git)
cd dermatoscopio-portatil-IA

# Crear entorno virtual (Recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar librerías
pip install -r requirements.txt
```
 ### 🖥️ Opción A: Aplicación de Escritorio (PC/Laptop)
 Interfaz completa con carga de archivos, control de cámara USB y generación de reportes.
 ```
 python Interfaz/app.py
  ```
  Funciones: Zoom digital, selección de resolución, visualización de máscaras de segmentación y gráficos de confianza.

### 🍓 Opción B: Aplicación Raspberry Pi (Modo Embebido)
Versión ligera optimizada para pantalla táctil y controles físicos.
```
 # Asegúrate de tener el modelo .tflite en la carpeta correcta
python Interfaz/raspberry_pi_app.py --model models/tflite/skin_lesion_float32_FINAL.tflite
  ```
Controles: Tecla s para capturar/analizar, q para salir.

## 🧠 Entrenamiento del Modelo (Google Colab)
Si deseas re-entrenar el modelo desde cero:

Descargar Datos: Ejecuta el script de descarga de metadatos.
```
python src/01_download_metadata.py --output data/metadata.csv
  ```
Procesar Dataset: Divide y organiza las imágenes.
```
python src/03_data_pipeline.py --meta data/metadata.csv --out data/processed
  ```
Entrenar: Sube el zip procesado a Drive y utiliza el notebook proporcionado en README anterior o el script de entrenamiento principal.

## 📂 Estructura del Repositorio

```
dermatoscopio-portatil-IA/
├── data/                  # Scripts de gestión de datos y metadatos
├── Documentación técnica/ # Justificación teórica y solución de retos
├── Interfaz/              # Aplicaciones de usuario final
│   ├── app.py             # GUI Escritorio (Tkinter + Otsu)
│   └── raspberry_pi_app.py # App ligera para RPi
├── models/                # Archivos .h5 y .tflite (no incluidos en repo por peso)
├── src/                   # Código fuente del pipeline de ML
│   ├── 03_data_pipeline.py # División Train/Val/Test
│   ├── 07_inference.py     # Lógica de inferencia pura
│   ├── dataset.py          # Clases de carga de datos
│   └── segmentation.py     # Algoritmos de visión artificial base
└── requirements.txt       # Dependencias del proyecto
```


Para mas información puedes consultar la documentación técnica. 
