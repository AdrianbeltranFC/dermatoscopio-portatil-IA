# Dermatoscopio Portátil con IA

## 📋 Descripción del Proyecto

Este proyecto busca desarrollar un **dermatoscopio portátil basado en inteligencia artificial** capaz de analizar imágenes de lesiones cutáneas y clasificarlas automáticamente. Utilizamos el dataset **HAM10000**, uno de los conjuntos de datos más grandes y confiables para el diagnóstico asistido por IA de enfermedades de la piel.

### Objetivos
- Crear un modelo de IA para clasificación de lesiones cutáneas
- Desarrollar una aplicación portátil y accesible
- Utilizar el dataset HAM10000 para entrenamiento y validación
- Automatizar la descarga y filtrado de datos
- Implementar mitigación de sesgo en clasificación por tono de piel

---

##  Estructura de Carpetas

```
dermatoscopio-portatil-IA/    
├── README.md                 # Este archivo
├── data/                     # Datos del proyecto
│   ├── raw/                  # Imágenes descargadas sin procesar (no incluidas en el repo,  se descargan en automático al ejecutar los scripts)
│   ├── processed/            # Imágenes procesadas y filtradas
│   └── metadata/             # Archivos CSV con metadatos
├── download_HAM.py           # Script de descarga automática
├── src/                      # Código fuente principal
│   ├── __init__.py
│   ├── model.py              # Definición del modelo
│   ├── data_loader.py        # Cargadores de datos
│   └── inference.py          # Inferencia y predicciones
├── requirements.txt          # Dependencias del proyecto
├── .gitignore                # Archivo de exclusiones git

```

---

##  Descripción de Carpetas

### `/data`
Almacena todos los datos del proyecto:
- **raw/**: Imágenes descargadas del HAM10000 ( No se incluyen en el repositorio por su tamaño)
- **processed/**: Imágenes filtradas y preprocesadas listas para entrenamiento
- **metadata/**: Archivos CSV con información de los pacientes y clasificaciones

### `/scripts`
Contiene scripts de utilidad para automatizar tareas:
- **download_ham10000.py**: Descarga automática del dataset
- **filter_data.py**: Filtra y organiza las imágenes
- **preprocessing.py**: Normalización y augmentación de datos

### `/src`
Código fuente principal del proyecto bien organizado


---

##  Instalación

### Requisitos previos
- Python 3.8 o superior
- pip o conda

### Pasos de instalación

1. **Clonar el repositorio**
```bash
git clone <URL_DEL_REPOSITORIO>
cd dermatoscopio-portatil-IA
```

2. **Crear un entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

---

##  Dataset HAM10000

### Descarga automática

El dataset HAM10000 se descarga y procesa automáticamente ejecutando:

```bash
python scripts/download_HAM.py
```

Este script:
- ✅ Descarga las imágenes del dataset HAM10000
- ✅ Organiza los archivos en la carpeta `data/raw/`
- ✅ Genera archivos de metadatos

---

## Uso del Proyecto

### Entrenar el modelo
```bash
python src/model.py --train --config config.yaml
```

### Hacer predicciones
```bash
python src/inference.py --image <ruta_imagen> --model <modelo_entrenado>
```

### Ejecutar la aplicación
```bash
python app/main.py
```

---

## Dataset HAM10000

- **10,000 imágenes** de lesiones cutáneas
- **7 categorías** de diagnóstico
- **Múltiples fuentes** clínicas confiables
- Imágenes de alta calidad con metadatos clínicos

⚠️ **Nota**: Las imágenes no se incluyen en el repositorio. Ejecuta el script de descarga la primera vez.

---

## Licencia

Este proyecto utiliza el dataset HAM10000.

---


