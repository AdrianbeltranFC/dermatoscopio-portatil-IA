import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- CONFIGURACIÓN ---
MODEL_PATH = r"C:\Users\silvi\OneDrive\Documents\dermatoscopio-portatil-IA\models\tflite\skin_lesion_float32_FINAL.tflite"
TEST_DIR = "data/processed/test"
UMBRAL_MELANOMA = 0.30  # Subido a 0.30 como solicitaste

CLASS_NAMES = ['mel', 'nv', 'other'] 
CLASS_MAP = {'mel': 0, 'nv': 1, 'other': 2}

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def process_image(img_path):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img_rgb, (224, 224))

def generate_occlusion_heatmap(interpreter, img_array, true_class_idx, patch_size=32, step=16):
    """Genera un mapa de calor (Saliency Map) tapando partes de la imagen (Ideal para TFLite)"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Probabilidad base sin tapar nada
    base_input = np.expand_dims(img_array.astype(np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], base_input)
    interpreter.invoke()
    base_prob = interpreter.get_tensor(output_details[0]['index'])[0][true_class_idx]
    
    heatmap = np.zeros((224, 224), dtype=np.float32)
    
    # Deslizar un parche gris por la imagen
    for y in range(0, 224, step):
        for x in range(0, 224, step):
            img_occluded = img_array.copy()
            # Tapar la región
            y_end = min(y + patch_size, 224)
            x_end = min(x + patch_size, 224)
            img_occluded[y:y_end, x:x_end] = 128 # Color gris neutro
            
            occ_input = np.expand_dims(img_occluded.astype(np.float32), axis=0)
            interpreter.set_tensor(input_details[0]['index'], occ_input)
            interpreter.invoke()
            new_prob = interpreter.get_tensor(output_details[0]['index'])[0][true_class_idx]
            
            # Si la probabilidad cae, esa zona era muy importante
            drop = base_prob - new_prob
            heatmap[y:y_end, x:x_end] += max(0, drop)
            
    # Suavizar y normalizar el mapa de calor
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    return heatmap

def plot_paper_heatmap(img_rgb, heatmap, prob, output_path):
    """Crea una visualización de alta calidad estética nivel paper"""
    # Aplicar colormap JET al heatmap (Rojo = Alta importancia, Azul = Baja)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superponer con la imagen original
    superimposed = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Explicabilidad del Modelo (Saliency Map) - Confianza: {prob*100:.1f}%', fontsize=16)
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Mapa de Activación')
    axes[1].axis('off')
    
    axes[2].imshow(superimposed)
    axes[2].set_title('Superposición (Áreas de Interés Clínico)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs("resultados_cnib", exist_ok=True)
    
    print("Cargando modelo TFLite...")
    interpreter = load_tflite_model(MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_true = []
    y_pred_ajustado = []
    probs_melanoma = []
    
    # Para guardar un par de ejemplos de melanoma y hacerles el mapa de calor
    melanoma_samples = []

    print(f"Evaluando imágenes con Umbral = {UMBRAL_MELANOMA}...")
    
    for class_name in CLASS_NAMES:
        class_dir = Path(TEST_DIR) / class_name
        if not class_dir.exists(): continue
            
        images = list(class_dir.glob('*.jpg'))
        print(f"Clase '{class_name}': {len(images)} imágenes")
        
        for i, img_path in enumerate(tqdm(images)):
            img_resized = process_image(img_path)
            input_data = np.expand_dims(img_resized.astype(np.float32), axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
            
            y_true.append(CLASS_MAP[class_name])
            probs_melanoma.append(preds[0]) # Guardar prob cruda para la curva ROC
            
            if preds[0] >= UMBRAL_MELANOMA:
                y_pred_ajustado.append(0)
            else:
                y_pred_ajustado.append(1 if preds[1] > preds[2] else 2)
                
            # Guardar los primeros 3 melanomas que el modelo predijo correctamente para los heatmaps
            if class_name == 'mel' and preds[0] >= UMBRAL_MELANOMA and len(melanoma_samples) < 3:
                melanoma_samples.append((img_resized, preds[0]))

    # --- 1. GENERAR HEATMAPS TIPO GRAD-CAM ---
    print("\nGenerando Heatmaps de explicabilidad (Esto toma unos segundos)...")
    for idx, (img, prob) in enumerate(melanoma_samples):
        hm = generate_occlusion_heatmap(interpreter, img, true_class_idx=0)
        plot_paper_heatmap(img, hm, prob, f'resultados_cnib/heatmap_melanoma_{idx+1}.png')
    print("✅ Heatmaps guardados en 'resultados_cnib/'.")

    # --- 2. CURVA ROC PARA MELANOMA (Gráfico nivel Paper) ---
    # Binarizar las etiquetas (1 si es Melanoma, 0 si no lo es)
    y_true_bin = [1 if y == 0 else 0 for y in y_true]
    fpr, tpr, _ = roc_curve(y_true_bin, probs_melanoma)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12)
    plt.title('Curva ROC para Detección de Melanoma', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('resultados_cnib/curva_roc_melanoma.png', dpi=300, bbox_inches='tight')
    print("✅ Curva ROC guardada.")

    # --- 3. NUEVA MATRIZ DE CONFUSIÓN ---
    cm = confusion_matrix(y_true, y_pred_ajustado)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=[c.upper() for c in CLASS_NAMES], yticklabels=[c.upper() for c in CLASS_NAMES])
    plt.title(f'Matriz de Confusión (Umbral Mel > {UMBRAL_MELANOMA})')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.savefig('resultados_cnib/matriz_umbral_030.png', dpi=300, bbox_inches='tight')
    
    # --- 4. REPORTE ---
    reporte = classification_report(y_true, y_pred_ajustado, target_names=CLASS_NAMES, output_dict=True)
    with open('resultados_cnib/metricas_030.txt', 'w', encoding='utf-8') as f:
        f.write(f"=== MÉTRICAS CON UMBRAL 0.30 ===\n")
        f.write(f"Recall (Sensibilidad) Melanoma: {reporte['mel']['recall'] * 100:.2f}%\n")
        f.write(f"Precision Melanoma: {reporte['mel']['precision'] * 100:.2f}%\n")
        f.write(f"Exactitud Global (Accuracy): {reporte['accuracy'] * 100:.2f}%\n")
        f.write(f"AUC-ROC Melanoma: {roc_auc:.3f}\n")

    print("\n✅ Script finalizado. Revisa la carpeta 'resultados_cnib'.")

if __name__ == "__main__":
    main()