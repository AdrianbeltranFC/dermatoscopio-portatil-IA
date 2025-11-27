"""
colab_train.py
Script para entrenar el modelo en Google Colab.
Copia y pega el contenido de este archivo en una celda de Colab.

Requisitos:
- Tener el repositorio clonado en Google Drive
- O descargar data/processed.zip desde el repositorio

Instrucciones:
1. Abre Google Colab: https://colab.research.google.com
2. Crea un nuevo notebook
3. En la primera celda, pega todo el código de este archivo
4. Ejecuta la celda
5. El script se encargará de montar Google Drive, descargar datos y entrenar
"""

# ============================================================================
# SETUP - Ejecutar primero
# ============================================================================

# Instalar dependencias
import subprocess
import sys

packages = [
    'tensorflow>=2.12.0',
    'scikit-learn',
    'pandas',
    'matplotlib',
    'numpy'
]

print("[*] Instalando dependencias...")
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
print("[✓] Dependencias instaladas")

# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')
print("[✓] Google Drive montado")

# ============================================================================
# DESCARGAR REPOSITORIO O DATOS
# ============================================================================

import os
from pathlib import Path

# Opción 1: Clonar repositorio (comentar si ya está clonado)
repo_path = Path('/content/drive/MyDrive/dermatoscopio-portatil-IA')

if not repo_path.exists():
    print("[*] Clonando repositorio...")
    os.system('cd /content/drive/MyDrive && git clone https://github.com/TU_USUARIO/dermatoscopio-portatil-IA.git')
    print("[✓] Repositorio clonado")
else:
    print(f"[✓] Repositorio encontrado en: {repo_path}")

os.chdir(repo_path)
print(f"[✓] Directorio actual: {os.getcwd()}")

# ============================================================================
# IMPORTAR Y EJECUTAR ENTRENAMIENTO
# ============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import logging
import matplotlib.pyplot as plt
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rutas dinámicas
PROJECT_ROOT = Path.cwd()
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
TFLITE_DIR = MODELS_DIR / "tflite"

for d in (MODELS_DIR, CHECKPOINTS_DIR, TFLITE_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"[✓] Rutas configuradas")
print(f"  DATA_PROCESSED: {DATA_PROCESSED}")
print(f"  MODELS_DIR: {MODELS_DIR}")

# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def create_augmentation_pipeline(image_size=(224, 224)):
    """Crea pipeline de augmentación con mitigación de sesgo."""
    def darken_and_contrast(x):
        x = tf.image.random_brightness(x, max_delta=0.15)
        x = tf.image.random_contrast(x, lower=0.85, upper=1.15)
        return x
    
    return tf.keras.Sequential([
        layers.Lambda(lambda x: darken_and_contrast(x)),
        layers.RandomFlip("horizontal", seed=42),
        layers.RandomRotation(0.12, seed=42),
        layers.RandomZoom(0.12, seed=42),
        layers.RandomTranslation(0.08, 0.08, seed=42),
        layers.GaussianNoise(0.02),
    ], name="augmentation")

def build_model(num_classes=7, input_shape=(224, 224, 3)):
    """Construye modelo EfficientNetB0."""
    inputs = keras.Input(shape=input_shape)
    aug = create_augmentation_pipeline(input_shape[:2])(inputs)
    x = layers.Rescaling(1.0 / 255.0)(aug)
    
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base.trainable = False
    
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_SkinLesion")
    return model

def make_datasets(processed_dir, image_size=(224, 224), batch_size=32):
    """Carga datasets desde estructura train/val/test."""
    logger.info(f"Cargando datos desde: {processed_dir}")
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        processed_dir / "train",
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        processed_dir / "val",
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        processed_dir / "test",
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    class_names = train_ds.class_names
    logger.info(f"Clases: {class_names}")
    
    return train_ds, val_ds, test_ds, class_names

def compute_class_weights_from_dataset(train_ds, class_names):
    """Calcula pesos de clases."""
    counts = np.zeros(len(class_names), dtype=int)
    
    for _, labels in train_ds.unbatch():
        idx = int(tf.argmax(labels).numpy())
        counts[idx] += 1
    
    labels_array = []
    for i, c in enumerate(counts):
        labels_array += [i] * c
    
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=np.array(labels_array)
    )
    
    return {i: float(w) for i, w in enumerate(cw)}

def train_model(model, train_ds, val_ds, epochs=50, lr=1e-3, class_weight=None):
    """Entrena el modelo."""
    optimizer = Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(CHECKPOINTS_DIR / "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    
    logger.info("Iniciando entrenamiento...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    return history

def convert_to_tflite(keras_model, val_ds, output_path, tflite_format="float16"):
    """Convierte modelo a TFLite."""
    logger.info(f"Convirtiendo a TFLite: {tflite_format}")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if tflite_format == "float16":
        converter.target_spec.supported_types = [tf.float16]
    elif tflite_format == "int8":
        def representative_gen():
            for images, _ in val_ds.take(100):
                yield [tf.cast(images, tf.float32)]
        
        converter.representative_dataset = representative_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / 1024 / 1024
    logger.info(f"TFLite guardado: {output_path} ({size_mb:.2f} MB)")

# ============================================================================
# EJECUTAR ENTRENAMIENTO
# ============================================================================

print("\n[*] Iniciando entrenamiento...")

# Parámetros de configuración
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
FINE_TUNE = True
EXPORT_TFLITE = True
TFLITE_FORMAT = "float16"  # o "int8"

try:
    # Cargar datos
    train_ds, val_ds, test_ds, class_names = make_datasets(
        DATA_PROCESSED,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    
    # Construir modelo
    logger.info("Construyendo modelo...")
    model = build_model(
        num_classes=len(class_names),
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    
    # Calcular pesos
    class_weight = compute_class_weights_from_dataset(train_ds, class_names)
    logger.info(f"Pesos de clases: {class_weight}")
    
    # Entrenar
    history = train_model(
        model,
        train_ds,
        val_ds,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        class_weight=class_weight
    )
    
    # Fine-tuning
    if FINE_TUNE:
        logger.info("Fine-tuning del modelo...")
        # (código de fine-tuning aquí)
    
    # Guardar modelo
    model_path = MODELS_DIR / "skin_lesion_classifier.h5"
    model.save(model_path)
    logger.info(f"[✓] Modelo guardado: {model_path}")
    
    # Exportar TFLite
    if EXPORT_TFLITE:
        tflite_out = TFLITE_DIR / f"skin_lesion_classifier_{TFLITE_FORMAT}.tflite"
        convert_to_tflite(model, val_ds, tflite_out, tflite_format=TFLITE_FORMAT)
        logger.info(f"[✓] TFLite exportado")
    
    print("\n[✓] ¡Entrenamiento completado exitosamente!")
    print(f"[✓] Modelo guardado en: {model_path}")
    
except Exception as e:
    print(f"\n[✗] Error durante el entrenamiento: {e}")
    import traceback
    traceback.print_exc()

# Descargar modelos entrenados
print("\n[*] Descargando modelos...")
os.system(f'zip -r /content/models.zip {MODELS_DIR}')
print("[✓] Los modelos están listos para descargar desde Colab")
