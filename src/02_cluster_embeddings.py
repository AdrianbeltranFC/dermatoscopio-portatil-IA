"""
02_cluster_embeddings.py
=====================

Script para:
 - extraer embeddings (features) de imágenes dermatoscópicas usando EfficientNetB0 (pre-trained)
 - reducir dimensionalidad (PCA -> UMAP)
 - aplicar clustering no supervisado (KMeans)
 - guardar resultados (CSV y gráficos)
 - seleccionar y guardar imágenes representativas por cluster

Uso (desde la raíz del repo):
  python src/02_cluster_embeddings.py --data_dir data/raw --meta data/metadata.csv --out_dir results --sample 2000 --n_clusters 3 --batch_size 32

Notas importantes:
 - El script inserta la carpeta raíz del repo en sys.path para que puedas importar módulos de src desde Windows/PowerShell.
 - Al final del archivo hay una sección teórica extensa explicando cada paso y su propósito.
"""

# -------------------------
# Fix para importar módulos desde src/ en cualquier SO
# -------------------------
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# -------------------------
# Imports
# -------------------------
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import math
import shutil

# Importar nuestras utilidades (dataset.py en src/)
from src.dataset import load_metadata  # load_metadata devuelve filepath,label,lesion_id
# TensorFlow se usa únicamente para cargar EfficientNet y preprocesar imágenes
import tensorflow as tf

# -------------------------
# Argumentos CLI  (Interfaz de Línea de Comandos)
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Extracción de embeddings, PCA->UMAP, KMeans y visualizaciones.")
    p.add_argument("--data_dir", required=True, help="Carpeta raíz con imágenes (no usada directamente, metadata contiene filepath).")
    p.add_argument("--meta", required=True, help="CSV metadata generado (data/metadata.csv)")
    p.add_argument("--out_dir", default="results", help="Carpeta donde guardar resultados")
    p.add_argument("--n_clusters", type=int, default=3, help="Número de clusters para KMeans")
    p.add_argument("--sample", type=int, default=None, help="Número de imágenes a samplear (útil para pruebas rápidas)")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size para extracción de embeddings")
    p.add_argument("--img_size", type=int, default=224, help="Tamaño (px) para redimensionar imágenes. EfficientNetB0 usa 224.")
    p.add_argument("--reps_per_cluster", type=int, default=3, help="Número de imágenes representativas por cluster a guardar")
    return p.parse_args()

# -------------------------
# Funciones auxiliares
# -------------------------
def extract_embeddings(filepaths, batch_size=32, input_size=224):
    """
    Extrae embeddings usando EfficientNetB0 pre-entrenado en ImageNet.
    - include_top=False, pooling='avg' => produce un vector por imagen (embedding).
    - Procesa en batches para no quedarse sin memoria.
    - Preprocesa cada imagen con la función específica de EfficientNet.
    Devuelve: embeddings (N, D) y lista de indices válidos (por si hubo errores al leer imágenes).
    """
    # Cargamos EfficientNetB0 sin la cabeza de clasificación, con pooling promedio (avg) para obtener vectores.
    base = tf.keras.applications.EfficientNetB0(include_top=False, pooling='avg',
                                                weights='imagenet',
                                                input_shape=(input_size,input_size,3))
    embeddings = []
    valid_indices = []  # índices de filepaths que efectivamente se procesaron
    batch_imgs = []
    batch_idxs = []

    for i, fp in enumerate(filepaths):
        try:
            # Cargar imagen, asegurar RGB y tamaño adecuado
            img = tf.keras.preprocessing.image.load_img(fp, target_size=(input_size,input_size))
            arr = tf.keras.preprocessing.image.img_to_array(img)  # shape (H,W,3)
            # Preprocesado específico para EfficientNet: aplica escala y normalización internas
            arr = tf.keras.applications.efficientnet.preprocess_input(arr)
            batch_imgs.append(arr)
            batch_idxs.append(i)
        except Exception as e:
            # Si falla al leer una imagen, la saltamos pero informamos
            print(f"[WARN] no se pudo leer {fp}: {e}")
            continue

        # Si el batch está completo o llegamos al final -> procesar
        if len(batch_imgs) == batch_size or i == len(filepaths)-1:
            X = np.array(batch_imgs, dtype=np.float32)
            emb = base.predict(X, verbose=0)  # (batch, embed_dim)
            embeddings.append(emb)
            valid_indices.extend(batch_idxs)
            batch_imgs = []
            batch_idxs = []

    if len(embeddings) == 0:
        return np.zeros((0, base.output_shape[-1])), []

    embs = np.vstack(embeddings)  # (N_valid, embed_dim)
    return embs, valid_indices

def plot_and_save_umap(embedding_2d, clusters, labels, out_dir):
    """
    Genera y guarda:
     - clusters_umap.png : UMAP coloreado por cluster (KMeans)
     - umap_by_label.png  : UMAP coloreado por etiqueta real (label)
    """
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib.cm as cm

    # Plot por cluster
    plt.figure(figsize=(8,6))
    uniq_clusters = np.unique(clusters)
    colors = cm.tab10.colors  # paleta
    for i, c in enumerate(uniq_clusters):
        idx = clusters == c
        plt.scatter(embedding_2d[idx,0], embedding_2d[idx,1], label=f"cluster {c}", alpha=0.6, s=8, color=colors[i % len(colors)])
    plt.legend()
    plt.title("UMAP sobre embeddings (coloreado por KMeans)")
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clusters_umap.png"), dpi=200)
    plt.close()

    # Plot por label (etiqueta real)
    plt.figure(figsize=(8,6))
    unique_labels = sorted(list(set(labels)))
    for i, lab in enumerate(unique_labels):
        idx = [j for j, labj in enumerate(labels) if labj == lab]
        if len(idx) == 0: 
            continue
        plt.scatter(embedding_2d[idx,0], embedding_2d[idx,1], label=lab, alpha=0.6, s=8, color=colors[i % len(colors)])
    plt.legend(markerscale=3)
    plt.title("UMAP sobre embeddings (coloreado por etiqueta real)")
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_by_label.png"), dpi=200)
    plt.close()

def save_cluster_assignments_csv(filepaths, labels, clusters, embedding_2d, out_dir):
    """
    Guarda un CSV con (filepath,label,cluster,umap_x,umap_y)
    """
    df = pd.DataFrame({
        "filepath": filepaths,
        "label": labels,
        "cluster": clusters,
        "umap_x": embedding_2d[:,0],
        "umap_y": embedding_2d[:,1]
    })
    out_path = os.path.join(out_dir, "cluster_assignments.csv")
    df.to_csv(out_path, index=False)
    print("Guardado:", out_path)
    return df

def save_cluster_stats(df_assign, out_dir):
    """
    Guarda un CSV con conteo por (cluster,label)
    """
    stats = df_assign.groupby(["cluster","label"]).size().reset_index(name="count")
    out_stats = os.path.join(out_dir, "cluster_stats.csv")
    stats.to_csv(out_stats, index=False)
    print("Guardado:", out_stats)
    # Imprimir tabla pivoteada en consola para inspección rápida
    pivot = stats.pivot(index="cluster", columns="label", values="count").fillna(0).astype(int)
    print("Tabla de conteos por cluster / label:\n", pivot)
    return pivot

def save_representative_images(df_assign, embs_pca, clusters, kmeans_centers, out_dir, n_per_cluster=3, thumb_size=(224,224)):
    """
    Para cada cluster:
      - calcula la distancia euclidiana en espacio PCA entre cada punto del cluster y el centro del cluster
      - selecciona los n_per_cluster imágenes más cercanas (más 'representativas')
      - crea una imagen tipo grid y la guarda como representative_cluster_{k}.png
    """
    os.makedirs(out_dir, exist_ok=True)
    for c in np.unique(clusters):
        idxs = np.where(clusters == c)[0]
        if len(idxs) == 0:
            continue
        # Distancias entre cada embedding_pca y el centro
        dists = np.linalg.norm(embs_pca[idxs] - kmeans_centers[c], axis=1)
        order = np.argsort(dists)  # índices relativos ordenados por cercanía
        chosen = idxs[order[:n_per_cluster]]

        # Crear grid
        cols = n_per_cluster
        rows = 1
        w, h = thumb_size
        grid = Image.new('RGB', (cols*w, rows*h), (255,255,255))
        for i, j in enumerate(chosen):
            fp = df_assign['filepath'].iloc[j]
            try:
                im = Image.open(fp).convert('RGB').resize((w,h))
            except Exception as e:
                im = Image.new('RGB', (w,h), (200,200,200))
                print(f"[WARN] no se pudo abrir {fp} para representative: {e}")
            grid.paste(im, (i*w, 0))
        outfn = os.path.join(out_dir, f"representative_cluster_{c}.png")
        grid.save(outfn)
        print("Guardado imagen representativa:", outfn)

# -------------------------
# Main flow
# -------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print("Cargando metadata desde:", args.meta)
    df = load_metadata(args.meta)  # devuelve rutas normalizadas y filtra no-existentes
    print("Total imágenes en metadata:", len(df))
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)
        print("Sample aplicado. Imágenes a procesar:", len(df))

    # Normalizamos filepaths (por si hay backslashes en Windows)
    df['filepath'] = df['filepath'].apply(lambda p: os.path.normpath(p))

    filepaths = df['filepath'].tolist()
    labels = df['label'].tolist()

    # 1) Extraer embeddings con EfficientNetB0
    print("Extrayendo embeddings (batch_size =", args.batch_size, ") ...")
    embs, valid_indices = extract_embeddings(filepaths, batch_size=args.batch_size, input_size=args.img_size)
    if embs.shape[0] == 0:
        print("No se extrajeron embeddings. Revisa rutas de imagenes.")
        return
    print("Embeddings extraídos. Shape:", embs.shape)

    # Si hubo archivos inválidos, filtramos filepaths/labels correspondientemente
    if len(valid_indices) != len(filepaths):
        print(f"Se saltaron {len(filepaths)-len(valid_indices)} imágenes por error de lectura.")
        filepaths = [filepaths[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        df = df.iloc[valid_indices].reset_index(drop=True)

    # 2) PCA para reducir dimensionalidad antes de UMAP (acelera y quita ruido)
    pca_n = min(50, embs.shape[1])  # típicamente 1280 -> 50
    print("Aplicando PCA a", pca_n, "componentes...")
    pca = PCA(n_components=pca_n, random_state=42)
    embs_pca = pca.fit_transform(embs)
    print("PCA completado. Shape PCA:", embs_pca.shape)

    # 3) UMAP (2D) para visualización
    print("Ejecutando UMAP (2D). Esto puede tardar unos segundos-minutos según N)...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embs_pca)
    print("UMAP completado. Shape:", embedding_2d.shape)

    # 4) KMeans en espacio PCA (más estable que en espacio original)
    print("Aplicando KMeans con k =", args.n_clusters)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embs_pca)
    print("KMeans completado.")

    # 5) Guardar asignaciones, stats, plots
    df_assign = save_cluster_assignments_csv(filepaths, labels, clusters, embedding_2d, args.out_dir)
    pivot = save_cluster_stats(df_assign, args.out_dir)
    plot_and_save_umap(embedding_2d, clusters, labels, args.out_dir)

    # 6) Guardar imágenes representativas por cluster
    save_representative_images(df_assign, embs_pca, clusters, kmeans.cluster_centers_, args.out_dir, n_per_cluster=args.reps_per_cluster)

    print("Proceso completado. Revisa la carpeta:", args.out_dir)
    print("Archivos generados: clusters_umap.png, umap_by_label.png, cluster_assignments.csv, cluster_stats.csv, representative_cluster_*.png")

if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
===========================================================================
SECCIÓN TEÓRICA
===========================================================================
1) ¿Qué es un *embedding* y por qué lo usamos?
----------------------------------------------
- Un *embedding* es un vector numérico (lista de números) que resume características importantes
  de una imagen (textura, color, formas, patrones). Los modelos pre-entrenados en ImageNet
  (como EfficientNet) ya aprendieron filtros útiles y, si usamos la red sin su "top" (la
  cabeza de clasificación) con `pooling='avg'`, la salida es un vector que representa la imagen.
- Ventaja: con estos vectores podemos comparar imágenes por distancia euclidiana o usar técnicas
  clásicas de machine learning (p. ej. KMeans) sin tener que entrenar una red grande.

2) EfficientNetB0 (sin top)
---------------------------
- EfficientNetB0 es una arquitectura CNN eficiente que ofrece buen balance precisión/velocidad.
- `include_top=False` + `pooling='avg'` nos devuelve un vector por imagen (embedding).
- Preprocesamos las imágenes con `tf.keras.applications.efficientnet.preprocess_input`,
  que aplica la normalización esperada por la red.

3) ¿Por qué procesar en *batches*?
----------------------------------
- Cargar muchas imágenes y pasarlas por la red de golpe puede consumir mucha memoria (RAM/GPU).
- Procesamos en lotes (ej. 32 imágenes) para evitar quedarse sin memoria y hacerlo reproducible.

4) PCA — Análisis de Componentes Principales
--------------------------------------------
- PCA reduce la dimensionalidad preservando la mayor varianza posible.
- Por ejemplo, si los embeddings tienen 1280 componentes, aplicar PCA a 50 componentes:
  - elimina ruido, reduce redundancia y acelera algoritmos posteriores (UMAP/KMeans).
- Usamos PCA antes de UMAP porque UMAP es más costoso en dimensiones altas.

5) UMAP — Uniform Manifold Approximation and Projection
-------------------------------------------------------
- UMAP es un método de reducción de dimensionalidad para visualización (similar a t-SNE).
- Objetivo: proyectar vectores de alta dimensión a 2D o 3D de forma que vecinos cercanos en
  el espacio original sigan siendo vecinos en la proyección.
- Ventajas frente a t-SNE:
  - suele preservar mejor la estructura global,
  - suele ser más rápido y reproducible,
  - permite ver agrupamientos y relaciones entre clases.
- Interpretación de UMAP:
  - Si las imágenes de la misma clase (ej. melanoma) aparecen concentradas en una región,
    eso sugiere que la representación (embedding) distingue bien esa clase.
  - Si las clases se mezclan mucho, indica que en ese espacio las clases no son claramente separables,
    y quizás necesitemos más datos, segmentación previa, otra arquitectura o más ingeniería de features.

6) KMeans (clustering no supervisado)
-------------------------------------
- KMeans agrupa datos en K clusters minimizando la distancia a los centroides (medias).
- Es un algoritmo simple y rápido para detectar "grupos naturales".
- En este pipeline aplicamos KMeans sobre el espacio PCA (no sobre UMAP) porque:
  - PCA reduce ruido y dimensiones, haciendo KMeans más estable y eficiente.
  - UMAP es principalmente para visualización; no es ideal para clustering directo por su
    naturaleza no lineal y su énfasis en preservar estructura local.

7) Por qué combinamos todo esto
-------------------------------
- Extraer embeddings: convierte imágenes a vectores comparables.
- PCA: compacta y limpia la representación para acelerar siguientes pasos.
- UMAP: nos da una visualización intuitiva 2D para la presentación.
- KMeans: nos da una partición no supervisada para comprobar si existen grupos naturales que
  coinciden con las etiquetas reales.
- Guardar imágenes representativas: permite mostrar ejemplos reales que caracterizan cada cluster,
  útil en la defensa para explicar por qué el cluster es así.

8) Limitaciones a tener en cuenta
---------------------------------
- Los embeddings vienen de una red preentrenada en ImageNet (fotos generales), no en dermatoscopia.
  Funcionan bien como punto de partida, pero un modelo fino-tuneado con dermatoscopia mejoraría.
- KMeans asume clusters convexos y similares por tamaño; puede no captar estructuras complejas.
- UMAP proyecta información: la distancia en 2D no es exactamente la distancia en el espacio original.
  Lo usamos para intuición visual, no como medida final.

9) Qué podemos concluir después de ejecutar el script
------------------------------------------------------
- Si los clusters correlacionan bien con etiquetas: la representación es discriminativa -> buen
  indicador para entrenamiento supervisado.
- Si no hay separación clara: es señal para mejorar preprocesado (hair removal), entrenamiento de una
  red dedicada o adquisición de más datos/variedad.
- Las imágenes representativas por cluster ayudan a interpretar qué características visuales
  definen cada grupo lo cual es útil para análisis cualitativo.

"""