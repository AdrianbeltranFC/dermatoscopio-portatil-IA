"""
🎓 Entrenamiento de Modelo de Clasificación de Lesiones Cutáneas

Modelo: EfficientNetB0 para 3 clases (mel, nv, other)

Uso:
    python train.py --epochs 30 --fine_tune --tflite
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from pathlib import Path
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Rutas
PROJECT_ROOT = Path.cwd()
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
TFLITE_DIR = MODELS_DIR / "tflite"

for d in (MODELS_DIR, CHECKPOINTS_DIR, TFLITE_DIR):
    d.mkdir(parents=True, exist_ok=True)

def create_augmentation():
    """Augmentación con Dark Skin Simulation."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=42),
        layers.RandomRotation(0.1, seed=42),
        layers.RandomZoom(0.1, seed=42),
    ], name="augmentation")

def build_model(num_classes=3):
    """Modelo EfficientNetB0."""
    inputs = keras.Input(shape=(224, 224, 3))
    
    # Augmentación
    x = create_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)
    
    # Base model
    base = EfficientNetB0(weights="imagenet", include_top=False)
    base.trainable = False
    x = base(x, training=False)
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs, outputs, name="SkinLesionClassifier")
    return model

def load_datasets(batch_size=32):
    """Carga datasets."""
    logger.info("📂 Cargando datasets...")
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PROCESSED / "train",
        labels="inferred",
        label_mode="categorical",
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PROCESSED / "val",
        labels="inferred",
        label_mode="categorical",
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PROCESSED / "test",
        labels="inferred",
        label_mode="categorical",
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )
    
    # Optimizar
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    
    class_names = sorted(train_ds.class_names)
    logger.info(f"✓ Clases: {class_names}")
    
    return train_ds, val_ds, test_ds, class_names

def compute_class_weights(train_ds, class_names):
    """Pesos para desbalance."""
    counts = np.zeros(len(class_names), dtype=int)
    
    for _, labels in train_ds.unbatch():
        counts[np.argmax(labels)] += 1
    
    labels_array = []
    for i, c in enumerate(counts):
        labels_array.extend([i] * c)
    
    cw = compute_class_weight(
        "balanced",
        classes=np.arange(len(class_names)),
        y=np.array(labels_array)
    )
    
    return {i: float(w) for i, w in enumerate(cw)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--tflite", action="store_true", default=True)
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🎓 ENTRENAMIENTO DE MODELO")
    logger.info("=" * 80)
    
    try:
        # Cargar datos
        train_ds, val_ds, test_ds, class_names = load_datasets(args.batch_size)
        
        # Construir modelo
        logger.info("\n🏗️ Construyendo modelo...")
        model = build_model(len(class_names))
        model.summary()
        
        # Compilar
        logger.info("\n⚙️ Compilando modelo...")
        model.compile(
            optimizer=Adam(learning_rate=args.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Pesos
        class_weights = compute_class_weights(train_ds, class_names)
        logger.info(f"Pesos: {class_weights}")
        
        # Entrenar
        logger.info(f"\n🚀 Entrenando ({args.epochs} épocas)...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[
                EarlyStopping("val_loss", patience=10, restore_best_weights=True),
                ReduceLROnPlateau("val_loss", factor=0.5, patience=4),
                ModelCheckpoint(str(CHECKPOINTS_DIR / "best_model.h5"), 
                              monitor="val_loss", save_best_only=True)
            ],
            class_weight=class_weights,
            verbose=1
        )
        
        # Fine-tune
        if args.fine_tune:
            logger.info("\n🔧 Fine-tuning...")
            for layer in model.layers[-20:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
            
            model.compile(
                optimizer=Adam(1e-5),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=10,
                callbacks=[EarlyStopping("val_loss", patience=5, restore_best_weights=True)],
                verbose=1
            )
        
        # Guardar modelo
        logger.info("\n💾 Guardando modelo...")
        model_path = MODELS_DIR / "skin_lesion_classifier.h5"
        model.save(str(model_path))
        logger.info(f"✓ Guardado: {model_path}")
        
        # TFLite
        if args.tflite:
            logger.info("\n📱 Exportando TFLite...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            tflite_path = TFLITE_DIR / "skin_lesion_classifier_float16.tflite"
            
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            
            size_mb = len(tflite_model) / 1024 / 1024
            logger.info(f"✓ TFLite: {tflite_path} ({size_mb:.2f} MB)")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ¡ENTRENAMIENTO COMPLETADO!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())