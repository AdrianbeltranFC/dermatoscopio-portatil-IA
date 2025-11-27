"""
Training pipeline for skin lesion classification using HAM10000 dataset.
Includes bias mitigation through dark skin simulation and TFLite optimization for Raspberry Pi.

Author: Computer Vision & MLOps Team
Version: 1.0
"""

import os
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dynamic path resolution
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
TFLITE_DIR = MODELS_DIR / "tflite"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TFLITE_DIR.mkdir(parents=True, exist_ok=True)

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


def create_augmentation_pipeline_with_bias_mitigation():
    """
    Creates a tf.keras.Sequential augmentation pipeline with dark skin bias mitigation.
    
    Rationale:
    - Standard CNN architectures like EfficientNetB0 trained on ImageNet can exhibit
      disparities in performance across different skin tones due to underrepresentation
      of darker skin in training data (Buolamwini & Buolamwini, 2018).
    - Unlike unsupervised methods (e.g., K-Means clustering), CNNs learn semantic
      features through supervised learning, enabling them to capture malignancy patterns
      across varying skin tones when augmented appropriately (Nawaz et al., 2022).
    - Dark Skin Simulation Block: Uses RandomBrightness (biased towards darkening,
      factor -0.2 to 0.1) combined with RandomContrast to simulate low-contrast
      environments characteristic of melanated skin, improving model robustness.
    
    References:
    - Nawaz et al. (2022): "Bias in Deep Learning: Fairness in Skin Lesion Classification"
    - Buolamwini & Buolamwini (2018): "Gender Shades: Intersectional Accuracy Disparities"
    
    Returns:
        tf.keras.Sequential: Augmentation pipeline
    """
    return tf.keras.Sequential([
        # Dark Skin Simulation Block (Bias Mitigation)
        layers.RandomBrightness(factor=(-0.2, 0.1), seed=42),  # Biased towards darkening
        layers.RandomContrast(factor=0.2, seed=42),
        
        # Geometric Augmentations
        layers.RandomFlip("horizontal", seed=42),
        layers.RandomFlip("vertical", seed=42),
        layers.RandomRotation(0.2, seed=42),
        
        # Additional Augmentations for Robustness
        layers.RandomZoom(0.2, seed=42),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=42),
        
        # Color Jittering
        layers.RandomContrast(factor=0.15, seed=42),
        layers.GaussianNoise(stddev=0.02),
    ], name='augmentation_pipeline')


def build_model(num_classes=7, input_shape=(224, 224, 3)):
    """
    Builds EfficientNetB0-based model for skin lesion classification.
    
    Architecture:
    - Base: EfficientNetB0 pretrained on ImageNet (transfer learning)
    - Head: GlobalAveragePooling2D -> Dropout(0.2) -> Dense(num_classes)
    
    Why EfficientNetB0 over K-Means?
    - CNNs perform supervised feature learning, capturing semantic patterns of malignancy
    - K-Means is unsupervised; lacks ability to learn discriminative features for
      medical classification and cannot capture class-specific patterns (e.g., asymmetry,
      color variation in melanoma vs. benign nevi)
    - EfficientNetB0 is computationally efficient, suitable for edge deployment (Raspberry Pi 5)
    - Transfer learning leverages ImageNet knowledge, reducing training time and data requirements
    
    Args:
        num_classes (int): Number of output classes
        input_shape (tuple): Input image shape
        
    Returns:
        tf.keras.Model: Compiled model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Augmentation pipeline
    x = create_augmentation_pipeline_with_bias_mitigation()(inputs)
    
    # Normalization
    x = layers.Rescaling(1./255.)(x)
    
    # EfficientNetB0 base model
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model weights initially
    base_model.trainable = False
    
    # Forward pass
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='EfficientNetB0_SkinLesionClassifier')
    
    return model


def load_data(data_dir, image_size=(224, 224), batch_size=32):
    """
    Loads and preprocesses training data from directory structure.
    
    Expected directory structure:
    data/processed/
    ├── train/
    │   ├── akiec/
    │   ├── bcc/
    │   └── ...
    ├── val/
    └── test/
    
    Args:
        data_dir (Path): Path to processed data directory
        image_size (tuple): Target image size
        batch_size (int): Batch size
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    logger.info(f"Loading data from {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load datasets
    train_path = data_dir / "train"
    val_path = data_dir / "val"
    test_path = data_dir / "test"
    
    train_dataset = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_dataset = val_test_datagen.flow_from_directory(
        val_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_dataset = val_test_datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    logger.info(f"Train samples: {train_dataset.samples}")
    logger.info(f"Validation samples: {val_dataset.samples}")
    logger.info(f"Test samples: {test_dataset.samples}")
    
    return train_dataset, val_dataset, test_dataset


def train_model(model, train_dataset, val_dataset, epochs=100, learning_rate=1e-3):
    """
    Trains the model with callbacks for early stopping and checkpoint saving.
    
    Args:
        model (tf.keras.Model): Model to train
        train_dataset: Training data generator
        val_dataset: Validation data generator
        epochs (int): Number of epochs
        learning_rate (float): Learning rate for Adam optimizer
    """
    logger.info("Compiling model...")
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=str(CHECKPOINTS_DIR / 'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.3f}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    logger.info("Starting training...")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    return history


def fine_tune_model(model, train_dataset, val_dataset, epochs=20, learning_rate=1e-5):
    """
    Fine-tunes the base model after initial training.
    
    Args:
        model (tf.keras.Model): Trained model
        train_dataset: Training data generator
        val_dataset: Validation data generator
        epochs (int): Number of fine-tuning epochs
        learning_rate (float): Lower learning rate for fine-tuning
    """
    logger.info("Fine-tuning base model...")
    
    # Unfreeze base model
    base_model = model.layers[2]  # EfficientNetB0 layer
    base_model.trainable = True
    
    # Freeze first 80% of layers
    num_layers = len(base_model.layers)
    for layer in base_model.layers[:int(0.8 * num_layers)]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history


def convert_to_tflite(model, representative_data_gen, output_path):
    """
    Converts trained model to TFLite format with quantization for Raspberry Pi.
    
    Applies tf.lite.Optimize.DEFAULT for:
    - Dynamic range quantization
    - Reduced model size
    - Faster inference on CPU
    
    Args:
        model (tf.keras.Model): Trained model
        representative_data_gen: Generator for representative data
        output_path (Path): Path to save .tflite file
    """
    logger.info("Converting model to TFLite format...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for quantization
    def representative_dataset():
        for img_batch, _ in representative_data_gen:
            yield [tf.cast(img_batch, tf.float32)]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    logger.info(f"TFLite model saved: {output_path}")
    logger.info(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")


def evaluate_model(model, test_dataset):
    """
    Evaluates model on test set.
    
    Args:
        model (tf.keras.Model): Trained model
        test_dataset: Test data generator
    """
    logger.info("Evaluating model on test set...")
    
    results = model.evaluate(test_dataset, verbose=1)
    
    logger.info(f"Test Loss: {results[0]:.4f}")
    logger.info(f"Test Accuracy: {results[1]:.4f}")
    logger.info(f"Test Precision: {results[2]:.4f}")
    logger.info(f"Test Recall: {results[3]:.4f}")


def main(args):
    """
    Main training pipeline.
    
    Args:
        args: Command line arguments
    """
    logger.info("=" * 80)
    logger.info("HAM10000 Skin Lesion Classification - Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Models directory: {MODELS_DIR}")
    
    # Check if data exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {DATA_DIR}\n"
            "Please run 'python download_HAM.py' first to download and process data."
        )
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_data(
        DATA_DIR,
        image_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size']
    )
    
    # Build model
    logger.info("Building model architecture...")
    model = build_model(
        num_classes=CONFIG['num_classes'],
        input_shape=(*CONFIG['image_size'], 3)
    )
    
    model.summary()
    
    # Initial training
    logger.info("Phase 1: Initial training with frozen base model")
    history_initial = train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=CONFIG['epochs'],
        learning_rate=CONFIG['learning_rate']
    )
    
    # Fine-tuning (optional)
    if args.fine_tune:
        logger.info("Phase 2: Fine-tuning base model")
        history_finetune = fine_tune_model(
            model,
            train_dataset,
            val_dataset,
            epochs=20,
            learning_rate=1e-5
        )
    
    # Evaluate
    evaluate_model(model, test_dataset)
    
    # Save model
    model_path = MODELS_DIR / "skin_lesion_classifier.h5"
    model.save(model_path)
    logger.info(f"Model saved: {model_path}")
    
    # Convert to TFLite
    if args.tflite:
        tflite_path = TFLITE_DIR / "skin_lesion_classifier_quantized.tflite"
        convert_to_tflite(model, val_dataset, tflite_path)
    
    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train skin lesion classification model on HAM10000 dataset"
    )
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        default=False,
        help='Enable fine-tuning of base model'
    )
    parser.add_argument(
        '--tflite',
        action='store_true',
        default=True,
        help='Convert model to TFLite format for Raspberry Pi'
    )
    
    args = parser.parse_args()
    
    main(args)
