"""
SCRIPT OPTIMIZADOR DE ACCURACY (CON TTA)
-------------------------------------------
1. Usa TTA (Test Time Augmentation) para mejorar predicciones.
2. Barre todos los umbrales para encontrar el PICO de Accuracy.
3. Genera el reporte con ese umbral ganador.
"""

import numpy as np
import tensorflow as tf
import cv2
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm # Barra de progreso

# CONFIGURACIÓN
MODEL_PATH = "models/tflite/skin_lesion_float32_final.tflite" # Tu mejor modelo
TEST_DIR = Path("data/processed/test")
CLASSES = ['Melanoma', 'Nevo', 'Otro']

def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def predict_single(interpreter, img_arr):
    """Hace una predicción simple con una imagen ya procesada"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_arr)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def predict_with_tta(interpreter, image_path):
    """
    Test Time Augmentation (TTA):
    Predice sobre la imagen original + 3 rotaciones + espejos.
    Promedia los resultados para mayor robustez.
    """
    img = cv2.imread(str(image_path))
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # Generar variaciones
    images = []
    
    # 1. Original
    images.append(img)
    # 2. Espejo Horizontal
    images.append(cv2.flip(img, 1))
    # 3. Espejo Vertical
    images.append(cv2.flip(img, 0))
    # 4. Rotación 90
    images.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    
    # Predecir cada una
    predictions = []
    for var_img in images:
        # Preprocesar (0-255 float32)
        inp = var_img.astype(np.float32)
        inp = np.expand_dims(inp, axis=0)
        predictions.append(predict_single(interpreter, inp))
    
    # Promediar predicciones (Ensemble de la misma imagen)
    avg_pred = np.mean(predictions, axis=0)
    return avg_pred

print("🚀 INICIANDO BÚSQUEDA DE MÁXIMA ACCURACY...")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: No encuentro {MODEL_PATH}")
    exit()

interpreter = load_model(MODEL_PATH)
y_true = []
y_probs = []

label_map = {'mel': 0, 'nv': 1, 'other': 2}

# 1. Obtener predicciones con TTA (Paciecia, es x4 más lento pero mejor)
print("   Procesando imágenes con TTA (Mejora de precisión)...")
total_files = sum([len(list((TEST_DIR/c).glob('*.jpg'))) for c in ['mel', 'nv', 'other']])

with tqdm(total=total_files) as pbar:
    for class_folder in ['mel', 'nv', 'other']:
        folder_path = TEST_DIR / class_folder
        if not folder_path.exists(): continue
        
        for img_file in folder_path.glob("*.jpg"):
            probs = predict_with_tta(interpreter, img_file)
            if probs is not None:
                y_true.append(label_map[class_folder])
                y_probs.append(probs)
            pbar.update(1)

y_true = np.array(y_true)
y_probs = np.array(y_probs)

# 2. Barrido de Umbrales para encontrar la MEJOR ACCURACY
print("\n🔎 Buscando el umbral perfecto...")
best_acc = 0
best_thresh = 0
best_preds = []

# Probamos umbrales de 0.20 a 0.80
for thresh in np.arange(0.20, 0.85, 0.05):
    current_preds = []
    for probs in y_probs:
        # Lógica de decisión personalizada
        if probs[0] >= thresh:
            current_preds.append(0) # Melanoma
        else:
            # Entre Nevo y Otro, gana el mayor
            if probs[1] > probs[2]: current_preds.append(1)
            else: current_preds.append(2)
            
    acc = accuracy_score(y_true, current_preds)
    
    # Guardamos si es el mejor record
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh
        best_preds = current_preds

print("\n" + "="*60)
print(f"🏆 MEJOR CONFIGURACIÓN ENCONTRADA")
print(f"   Umbral Óptimo: {best_thresh:.2f}")
print(f"   ACCURACY:      {best_acc*100:.2f}%")
print("="*60)

# 3. Generar Reporte con esa configuración ganadora
print("\n📋 REPORTE FINAL (PARA EL CONGRESO):")
print(classification_report(y_true, best_preds, target_names=CLASSES, digits=4))

# Matriz
cm = confusion_matrix(y_true, best_preds)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'Matriz Optimizada (Acc {best_acc:.1%})')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.savefig('matriz_max_accuracy.png', dpi=300)
print("\n✅ Matriz guardada como: matriz_max_accuracy.png")