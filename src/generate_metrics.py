import numpy as np
import tensorflow as tf
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# CONFIGURACIÓN
MODEL_PATH = "models/tflite/skin_lesion_classifier_float16.tflite"
TEST_DIR = Path("data/processed/test")
CLASSES = ['Melanoma', 'Nevo (Benigno)', 'Otro'] # Orden: mel, nv, other

def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def predict_tflite(interpreter, image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocesamiento (IGUAL AL ENTRENAMIENTO)
    img = cv2.imread(str(image_path))
    if img is None: return None
    # BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img, (224, 224))
    
    # Sin normalizar (0-255) porque quitamos Rescaling
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return np.argmax(output_data)

print("🚀 Generando Métricas para Congreso...")

interpreter = load_tflite_model(MODEL_PATH)
y_true = []
y_pred = []

# Mapeo de carpetas a índices
label_map = {'mel': 0, 'nv': 1, 'other': 2}

for class_folder in ['mel', 'nv', 'other']:
    folder_path = TEST_DIR / class_folder
    if not folder_path.exists(): continue
    
    print(f"Analizando clase: {class_folder}...")
    for img_file in folder_path.glob("*.jpg"):
        true_label = label_map[class_folder]
        pred_label = predict_tflite(interpreter, img_file)
        
        if pred_label is not None:
            y_true.append(true_label)
            y_pred.append(pred_label)

# 1. Matriz de Confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicción IA')
plt.ylabel('Diagnóstico Real')
plt.title('Matriz de Confusión (Test Set)')
plt.savefig('matriz_confusion.png', dpi=300)
print("✓ Guardada: matriz_confusion.png")

# 2. Reporte de Texto
report = classification_report(y_true, y_pred, target_names=CLASSES)
with open("metricas_finales.txt", "w") as f:
    f.write(report)
print("\n📋 REPORTE CLÍNICO:\n")
print(report)