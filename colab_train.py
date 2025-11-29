"""
colab_train.py
Script para entrenar el modelo en Google Colab.
Copia y pega el contenido de este archivo en una celda de Colab.
Requisitos:
- Tener el repositorio clonado en Google Drive
Instrucciones:
1. Abre Google Colab: https://colab.research.google.com
2. Crea un nuevo notebook
3. En la primera celda, pega todo el código de este archivo
4. Ejecuta la celda
5. El script se encargará de montar Google Drive, descargar datos y entrenar
"""

# Versión final con las siguientes correciones importantes:
# 1. FIX SINTAXIS: Se corrigió el error de BatchNormalization que rompía el flujo.
# 2. FIX ESCALA: Sin capa de Rescaling (EfficientNet recibe 0-255 nativo).
# 3. FIX DATOS: Pipeline balanceado con repetición infinita (sin cortes).
# ==============================================================================

import os
import sys
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, preprocessing
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. SETUP
# ------------------------------------------------------------------------------
print("[1/6] Setup inicial...")
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

os.chdir('/content')
if os.path.exists('dermatoscopio-portatil-IA'):
    shutil.rmtree('dermatoscopio-portatil-IA')

os.system('git clone https://github.com/TU_USUARIO_DE_GITHUB/dermatoscopio-portatil-IA.git >/dev/null 2>&1')
os.chdir('dermatoscopio-portatil-IA')
#No olvides modificar la URL anterior con la de tu repositorio
# Directorios
PROJECT_ROOT = Path.cwd()
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
TFLITE_DIR = MODELS_DIR / "tflite"

for d in (MODELS_DIR, CHECKPOINTS_DIR, TFLITE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. DATOS
# ------------------------------------------------------------------------------
print("[2/6] Preparando datos...")
zip_path = '/content/drive/MyDrive/data_processed.zip'
if not os.path.exists(zip_path):
    print(f"❌ ERROR CRÍTICO: No encuentro {zip_path}")
    sys.exit(1)

os.system(f"cp '{zip_path}' . 2>/dev/null")
os.system("unzip -q data_processed.zip")

# ------------------------------------------------------------------------------
# 3. PIPELINE DE DATOS BALANCEADO
# ------------------------------------------------------------------------------
print("[3/6] Creando Pipeline...")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    return image, label

train_dir = DATA_PROCESSED / "train"

def load_class_dataset(class_name, label_idx):
    # Cargar dataset sin batching inicial para poder mezclar
    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir / class_name,
        labels=None,
        image_size=IMG_SIZE,
        batch_size=None,
        shuffle=True,
        seed=42,
        verbose=0
    )
    # Etiqueta manual one-hot
    ds = ds.map(lambda x: (x, tf.one_hot(label_idx, 3)))
    # Repetir infinitamente para evitar "ran out of data"
    ds = ds.repeat() 
    return ds

print("   - Cargando sub-datasets...")
# Aseguramos índices: mel=0, nv=1, other=2 (orden alfabético estándar)
ds_mel = load_class_dataset('mel', 0)
ds_nv = load_class_dataset('nv', 1)
ds_other = load_class_dataset('other', 2)

# Muestreo Balanceado 33% cada uno
train_ds = tf.data.Dataset.sample_from_datasets(
    [ds_mel, ds_nv, ds_other],
    weights=[1/3, 1/3, 1/3]
)

train_ds = train_ds.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Validación y Test
val_ds = preprocessing.image_dataset_from_directory(
    DATA_PROCESSED / "val",
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_ds = preprocessing.image_dataset_from_directory(
    DATA_PROCESSED / "test",
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ------------------------------------------------------------------------------
# 4. MODELO (CORREGIDO)
# ------------------------------------------------------------------------------
print("[4/6] Construyendo Modelo...")

inputs = keras.Input(shape=(224, 224, 3))

# NOTA: NO usamos Rescaling(1./255). EfficientNet espera 0-255.
# Al dejar pasar los valores crudos, evitamos el estancamiento del Loss.
x = inputs 

base_model = EfficientNetB0(weights="imagenet", include_top=False)
base_model.trainable = False 

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)

# --- CORRECCIÓN SINTAXIS AQUÍ ---
x = layers.BatchNormalization()(x) # <--- Ahora sí tiene el (x)
# --------------------------------

x = layers.Dropout(0.2)(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = models.Model(inputs, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall(name="recall")]
)

# ------------------------------------------------------------------------------
# 5. ENTRENAMIENTO
# ------------------------------------------------------------------------------
print("[5/6] Entrenando...")

# FASE 1: Transfer Learning
print("--- Fase 1: Entrenando cabezal ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    steps_per_epoch=200, 
    callbacks=[
        EarlyStopping("val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(str(CHECKPOINTS_DIR / "best_model.keras"), save_best_only=True)
    ]
)

# FASE 2: Fine-Tuning
print("\n--- Fase 2: Fine-Tuning ---")
base_model.trainable = True
# Descongelar un poco más para permitir adaptación
for layer in base_model.layers[:-60]: 
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-4), 
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall(name="recall")]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    steps_per_epoch=200,
    callbacks=[
        EarlyStopping("val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=3)
    ]
)

# ------------------------------------------------------------------------------
# 6. EXPORTAR
# ------------------------------------------------------------------------------
print("[6/6] Guardando y Exportando...")

print("Evaluando en Test Set...")
res = model.evaluate(test_ds)
print(f"📊 Test Recall FINAL: {res[2]*100:.2f}%")

model.save(MODELS_DIR / "skin_lesion_classifier.keras")

# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

tflite_path = TFLITE_DIR / "skin_lesion_classifier_float16.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("📦 Comprimiendo modelos...")
os.system(f"zip -r -q models_master.zip models/")

from google.colab import files
files.download('models_master.zip')
print("✅ ¡Descarga lista! Usa 'models_master.zip'.")