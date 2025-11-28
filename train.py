"""
🎓 Entrenamiento de Modelo de Clasificación de Lesiones Cutáneas

Modelo: EfficientNetB0 para 3 clases (mel, nv, other)

Uso:
    python train.py --epochs 30 --fine_tune --tflite
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
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
    """Carga datasets con VALIDACIÓN."""
    from tensorflow.keras import preprocessing
    
    logger.info("📂 Cargando datasets...")
    
    # VERIFICAR QUE LOS DIRECTORIOS EXISTEN
    train_dir = DATA_PROCESSED / "train"
    val_dir = DATA_PROCESSED / "val"
    test_dir = DATA_PROCESSED / "test"
    
    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            raise FileNotFoundError(f"❌ Directorio no encontrado: {d}")
        
        # Verificar que tiene subdirectorios
        subdirs = list(d.iterdir())
        if not subdirs:
            raise FileNotFoundError(f"❌ {d} está vacío")
        
        logger.info(f"✓ {d.name}: {len(subdirs)} clases")
    
    try:
        logger.info("  Cargando train...")
        train_ds = preprocessing.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=True,
            seed=42
        )
        logger.info(f"  ✓ Train cargado")
        
        logger.info("  Cargando val...")
        val_ds = preprocessing.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=False,
            seed=42
        )
        logger.info(f"  ✓ Val cargado")
        
        logger.info("  Cargando test...")
        test_ds = preprocessing.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=False,
            seed=42
        )
        logger.info(f"  ✓ Test cargado")
        
    except Exception as e:
        logger.error(f"❌ Error cargando datasets: {e}")
        raise
    
    # Optimizar
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    
    class_names = sorted(train_ds.class_names)
    logger.info(f"✓ Clases encontradas: {class_names}")
    
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
    logger.info(f"Parámetros:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Fine-tune: {args.fine_tune}")
    logger.info(f"  TFLite: {args.tflite}\n")
    
    try:
        # Verificar directorios
        logger.info("1️⃣ Verificando directorios...")
        logger.info(f"   Project root: {PROJECT_ROOT}")
        logger.info(f"   Data dir: {DATA_PROCESSED}")
        
        if not DATA_PROCESSED.exists():
            raise FileNotFoundError(f"❌ {DATA_PROCESSED} no existe")
        
        # Listar contenido
        logger.info(f"   Contenido de {DATA_PROCESSED}:")
        for item in DATA_PROCESSED.iterdir():
            logger.info(f"     - {item.name}")
        
        # Cargar datos
        logger.info("\n2️⃣ Cargando datasets...")
        train_ds, val_ds, test_ds, class_names = load_datasets(args.batch_size)
        
        # Construir modelo
        logger.info("\n3️⃣ Construyendo modelo...")
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import EfficientNetB0
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = layers.Rescaling(1.0 / 255.0)(inputs)
        
        base = EfficientNetB0(weights="imagenet", include_top=False)
        base.trainable = False
        x = base(x, training=False)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(len(class_names), activation="softmax")(x)
        
        model = models.Model(inputs, outputs)
        logger.info("✓ Modelo construido")
        
        # Compilar
        logger.info("\n4️⃣ Compilando modelo...")
        from tensorflow.keras.optimizers import Adam
        
        model.compile(
            optimizer=Adam(learning_rate=args.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        logger.info("✓ Compilado")
        
        # Entrenar
        logger.info("\n5️⃣ Entrenando...")
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[
                EarlyStopping("val_loss", patience=10, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau("val_loss", factor=0.5, patience=4, verbose=1),
                ModelCheckpoint(str(CHECKPOINTS_DIR / "best_model.h5"), 
                              monitor="val_loss", save_best_only=True, verbose=1)
            ],
            verbose=1
        )
        logger.info("✓ Entrenamiento completado")
        
        # Guardar
        logger.info("\n6️⃣ Guardando modelo...")
        model_path = MODELS_DIR / "skin_lesion_classifier.h5"
        model.save(str(model_path))
        logger.info(f"✓ Guardado: {model_path}")
        
        # TFLite
        if args.tflite:
            logger.info("\n7️⃣ Exportando TFLite...")
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
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())