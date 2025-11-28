"""
🎓 Entrenamiento de Modelo de Clasificación de Lesiones Cutáneas
==================================================================

Uso LOCAL (en tu computadora):
    python train.py --data_dir data/processed --epochs 50 --fine_tune --tflite

Uso en GOOGLE COLAB:
    1. Abre: https://colab.research.google.com
    2. Crea notebook nuevo
    3. En celda 1: 
       !git clone https://github.com/TU_USUARIO/dermatoscopio-portatil-IA.git && cd dermatoscopio-portatil-IA
    4. En celda 2:
       exec(open('train.py').read())

Dataset: HAM10000 (~10,000 imágenes de lesiones cutáneas)
Modelo: EfficientNetB0 + Transfer Learning + Mitigación de Sesgo
Salida: Modelo H5 + TFLite (float16 o int8) para Raspberry Pi 5
"""

import os
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Detectar si estamos en Colab
try:
    from google.colab import drive
    IN_COLAB = True
    logger.info("✓ Detectado: Google Colab")
except:
    IN_COLAB = False
    logger.info("✓ Detectado: Local")

# Dynamic path resolution
PROJECT_ROOT = Path.cwd()
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
TFLITE_DIR = MODELS_DIR / "tflite"

for d in (MODELS_DIR, CHECKPOINTS_DIR, TFLITE_DIR):
    d.mkdir(parents=True, exist_ok=True)

logger.info(f"Project root: {PROJECT_ROOT}")

# Configuration
CONFIG = {
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-3,
    'num_classes': 7,  # HAM10000 has 7 classes
    'validation_split': 0.2,
    'test_split': 0.1,
}

# HAM10000 classes
HAM10000_CLASSES = {
    0: 'akiec',  # Actinic keratosis / Bowen's disease
    1: 'bcc',    # Basal cell carcinoma
    2: 'bkl',    # Benign keratosis-like lesions
    3: 'df',     # Dermatofibroma
    4: 'mel',    # Melanoma
    5: 'nv',     # Melanocytic nevi
    6: 'vasc',   # Vascular lesions
}


def create_augmentation_pipeline(image_size=(224, 224)):
    """
    Pipeline de augmentación con mitigación de sesgo (Dark Skin Simulation).
    
    Rationale:
    - RandomBrightness (factor: -0.2 a 0.1) → Simula piel oscura/melanada
    - RandomContrast → Bajo contraste característico de piel oscura
    - Referencia: Nawaz et al. (2022)
    """
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
    """
    EfficientNetB0 para clasificación de lesiones cutáneas.
    
    Architecture:
    - Base: EfficientNetB0 (ImageNet weights)
    - Head: GlobalAveragePooling2D → Dropout(0.3) → Dense(num_clases)
    
    Why CNN vs K-Means?
    - CNNs: Aprendizaje supervisado de características semánticas
    - K-Means: No supervisado, no captura malignidad ni variaciones de tono
    """
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
    """
    Carga datasets: train/val/test desde estructura de carpetas.
    
    Esperado:
        processed_dir/
        ├── train/<clase>/*.jpg
        ├── val/<clase>/*.jpg
        └── test/<clase>/*.jpg
    """
    processed_dir = Path(processed_dir)
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"❌ Directorio no encontrado: {processed_dir}")
    
    logger.info(f"📂 Cargando datos desde: {processed_dir}")
    
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
    
    # Optimización: cache + prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    class_names = train_ds.class_names
    logger.info(f"✓ Clases: {class_names}")
    
    return train_ds, val_ds, test_ds, class_names


def compute_class_weights_from_dataset(train_ds, class_names):
    """Calcula pesos para desbalance de clases."""
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
    
    logger.info("🚀 Iniciando entrenamiento...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    return history


def find_base_model(model):
    """Encuentra el modelo base EfficientNet."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    raise RuntimeError("❌ No se encontró modelo base EfficientNet")


def fine_tune_model(model, train_ds, val_ds, unfreeze_frac=0.2, epochs=10, lr=1e-5):
    """Fine-tuning del modelo base."""
    logger.info("🔧 Fine-tuning del modelo...")
    
    base = find_base_model(model)
    base.trainable = True
    
    num_layers = len(base.layers)
    freeze_until = int(num_layers * (1.0 - unfreeze_frac))
    
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= freeze_until
    
    logger.info(f"   Unfreezing top {unfreeze_frac*100:.1f}% ({freeze_until}/{num_layers} layers)")
    
    model.compile(
        optimizer=Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks = [EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, test_ds, class_names, out_dir):
    """Evalúa y genera reportes."""
    logger.info("📊 Evaluando modelo...")
    
    y_true = []
    y_pred = []
    
    for batch in test_ds.unbatch().batch(1):
        x, y = batch
        p = model.predict(x, verbose=0)
        y_true.append(int(tf.argmax(y[0]).numpy()))
        y_pred.append(int(np.argmax(p[0])))
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    logger.info(f"Classification Report:\n{report}")
    
    cm = confusion_matrix(y_true, y_pred)
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        Path(out_dir) / "confusion_matrix.csv"
    )
    with open(Path(out_dir) / "classification_report.txt", "w") as f:
        f.write(report)
    
    return report, cm


def convert_to_tflite(keras_model, val_ds, output_path, tflite_format="float16"):
    """Convierte a TFLite para Raspberry Pi."""
    logger.info(f"📱 Convirtiendo a TFLite ({tflite_format})...")
    
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
    logger.info(f"✓ TFLite guardado: {output_path} ({size_mb:.2f} MB)")


def plot_history(history, out_dir):
    """Plotea historia de entrenamiento."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history.get("loss", []), label="loss")
    axes[0].plot(history.history.get("val_loss", []), label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(history.history.get("accuracy", []), label="accuracy")
    axes[1].plot(history.history.get("val_accuracy", []), label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "training_history.png", dpi=100)
    plt.close()
    
    logger.info(f"✓ Gráficas guardadas en: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenar modelo EfficientNetB0 para clasificación de lesiones cutáneas"
    )
    parser.add_argument(
        "--data_dir",
        default=str(DATA_PROCESSED),
        help="Ruta a datos procesados (train/val/test)"
    )
    parser.add_argument("--image_size", type=int, default=224, help="Tamaño de imagen")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--fine_tune", action="store_true", help="Habilitar fine-tuning")
    parser.add_argument("--tflite", action="store_true", help="Exportar a TFLite")
    parser.add_argument(
        "--tflite_format",
        choices=["float16", "int8"],
        default="float16",
        help="Formato TFLite"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("🎓 Entrenamiento de Clasificación de Lesiones Cutáneas (HAM10000)")
    logger.info("=" * 80)
    
    try:
        # Cargar datos
        train_ds, val_ds, test_ds, class_names = make_datasets(
            args.data_dir,
            image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size
        )
        
        # Construir modelo
        logger.info("🏗️ Construyendo modelo EfficientNetB0...")
        model = build_model(
            num_classes=len(class_names),
            input_shape=(args.image_size, args.image_size, 3)
        )
        model.summary()
        
        # Calcular pesos de clases
        class_weight = compute_class_weights_from_dataset(train_ds, class_names)
        logger.info(f"Pesos de clases: {class_weight}")
        
        # Entrenamiento
        history = train_model(
            model,
            train_ds,
            val_ds,
            epochs=args.epochs,
            lr=args.learning_rate,
            class_weight=class_weight
        )
        
        # Gráficas
        plot_history(history, CHECKPOINTS_DIR)
        
        # Fine-tuning
        if args.fine_tune:
            fine_tune_model(model, train_ds, val_ds, epochs=10, lr=1e-5)
        
        # Evaluación
        evaluate_model(model, test_ds, class_names, CHECKPOINTS_DIR)
        
        # Guardar modelo
        model_path = MODELS_DIR / "skin_lesion_classifier.h5"
        model.save(model_path)
        logger.info(f"✓ Modelo guardado: {model_path}")
        
        # Exportar TFLite
        if args.tflite:
            tflite_out = TFLITE_DIR / f"skin_lesion_classifier_{args.tflite_format}.tflite"
            convert_to_tflite(model, val_ds, tflite_out, tflite_format=args.tflite_format)
        
        logger.info("=" * 80)
        logger.info("✅ ¡Entrenamiento completado exitosamente!")
        logger.info("=" * 80)
        
        # En Colab, preparar descarga
        if IN_COLAB:
            logger.info("\n📦 Preparando modelos para descargar...")
            os.system('cd /content && zip -r -q models.zip models/')
            logger.info("✓ Descarga: models.zip desde 'Archivos' en el panel izquierdo de Colab")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())