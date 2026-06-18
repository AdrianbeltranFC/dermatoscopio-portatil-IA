"""
src/09_generate_model_umap.py
Genera una proyección UMAP usando los 'embeddings' aprendidos por TU modelo entrenado.
Esto demuestra si la red neuronal ha aprendido a separar visualmente las clases.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import umap
import cv2
from pathlib import Path
from tqdm import tqdm

# Configuración
MODEL_PATH = "models/tflite/skin_lesion_float32_FINAL.tflite"  # Ajusta si tu .h5 tiene otro nombre
DATA_DIR = "data/processed/test"                 # Usamos el set de TEST
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_data_and_extract_features(model, data_dir):
    classes = sorted(os.listdir(data_dir))
    print(f"Clases encontradas: {classes}")
    
    # Crear un sub-modelo que devuelva la salida de la capa de Pooling (penúltima)
    # EfficientNetB0 suele tener una capa 'avg_pool' o 'global_average_pooling2d' antes de la 'dense' final.
    # Usamos output [-2] o [-3] generalmente. Para estar seguros, buscamos la capa de pooling.
    
    layer_name = None
    for layer in reversed(model.layers):
        if 'pool' in layer.name or 'flatten' in layer.name:
            layer_name = layer.name
            break
            
    if not layer_name:
        # Fallback: Usar la penúltima capa si no hallamos nombre obvio
        layer_name = model.layers[-2].name
    
    print(f"Extrayendo características de la capa: {layer_name}")
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    features = []
    labels = []
    
    image_paths = []
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls_name)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths.extend([(img, cls_name) for img in imgs])

    print(f"Procesando {len(image_paths)} imágenes...")
    
    # Procesar por lotes
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_imgs = []
        
        for p, label in batch_paths:
            img = cv2.imread(p)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                # IMPORTANTE: Usar el mismo preprocesamiento que en el entrenamiento
                # Si usaste EfficientNet con inputs 0-255, mantenlo así.
                # Si normalizaste, descomenta la linea de abajo.
                # img = img / 255.0 
                batch_imgs.append(img)
                labels.append(label)
        
        if batch_imgs:
            batch_imgs = np.array(batch_imgs, dtype=np.float32)
            batch_features = feature_extractor.predict(batch_imgs, verbose=0)
            features.extend(batch_features)
            
    return np.array(features), np.array(labels)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: No se encuentra el modelo en {MODEL_PATH}")
        return

    print("Cargando modelo...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Extrayendo embeddings...")
    X_features, y_labels = load_data_and_extract_features(model, DATA_DIR)
    
    print(f"Embeddings shape: {X_features.shape}")
    
    print("Calculando UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X_features)
    
    print("Generando gráfica...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        hue=y_labels, 
        palette='viridis', 
        s=50, 
        alpha=0.7
    )
    plt.title('UMAP Projection of Learned Features (Skin Lesion Model)', fontsize=14)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Clase', loc='best')
    
    output_path = "resultados/umap_model_trained.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica guardada en: {output_path}")
    plt.show()

if __name__ == "__main__":
    # Necesitas instalar: pip install umap-learn seaborn
    main()