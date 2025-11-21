"""
dataset.py

Funciones para:
 - cargar metadata (acepta CSV con 'filepath' o construir filepath desde 'image_id' + images_dir)
 - mapear etiquetas
 - crear splits por lesion_id (evita data leakage)
 - codificar etiquetas a enteros

Nota: Este script no necesita ser ejecutado directamente, sino que sus funciones son importadas por otros scripts
 como src/02_cluster_embeddings.py o src/train_model.py.
Uso típico:
  from src.dataset import load_metadata, encode_labels, create_splits_by_lesion 

"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Mapeo por defecto (HAM10000 -> nombres legibles)
DEFAULT_LABEL_MAP = {
    "mel": "melanoma",
    "nv": "nevus",
    "bkl": "seborrheic_keratosis",
    "melanoma": "melanoma",
    "nevus": "nevus",
    "seborrheic_keratosis": "seborrheic_keratosis",
}

def load_metadata(csv_path, images_dir=None, label_col_candidates=("label","dx","dx_type")):
    """
    Carga metadata desde csv_path y devuelve DataFrame con columnas:
      - filepath (ruta normalizada a la imagen)
      - label   (nombre de clase mapeado)
      - lesion_id (si existe en el CSV; si no, None)

    Parámetros:
      - csv_path: ruta al CSV (ej. data/metadata.csv)
      - images_dir: si se proporciona, se intentará construir filepath desde image_id + images_dir
                    si no se proporciona, se buscará columna 'filepath' en el CSV.
      - label_col_candidates: nombres posibles de la columna con la etiqueta.

    Comportamiento:
      - Si el CSV contiene 'filepath', lo usará (recomendado cuando ya generaste metadata con rutas).
      - Si no contiene 'filepath' pero sí 'image_id' y images_dir fue dado, construye rutas.
      - Filtra filas con archivos inexistentes y devuelve DataFrame limpio.
    """
    df = pd.read_csv(csv_path)

    # 1) Manejo de etiqueta: detectamos la columna con label
    label_col = None
    for c in label_col_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"No se encontró columna de label en {csv_path}. Buscadas: {label_col_candidates}")

    # 2) Determinar filepath
    if 'filepath' in df.columns and df['filepath'].notna().all():
        # Si el CSV ya tiene filepath, lo usamos
        df['filepath'] = df['filepath'].astype(str).apply(lambda p: os.path.normpath(p))
    else:
        # Intentamos construirlo desde image_id + images_dir
        if images_dir is None:
            raise ValueError("El CSV no contiene 'filepath'. Debes proveer images_dir para construir rutas a partir de 'image_id'.")
        id_col = None
        for c in ("image_id","imageId","id","image"):
            if c in df.columns:
                id_col = c
                break
        if id_col is None:
            raise ValueError("No se encontró 'image_id' en el CSV y tampoco se proporcionó 'filepath'.")
        # Buscar archivo con extensiones comunes dentro de images_dir (recursivo)
        def find_file(img_id):
            for root, _, files in os.walk(images_dir):
                for f in files:
                    name, ext = os.path.splitext(f)
                    if name == img_id:
                        return os.path.normpath(os.path.join(root, f))
            return None
        df['filepath'] = df[id_col].astype(str).apply(find_file)

    # 3) Columna lesion_id opcional
    if 'lesion_id' in df.columns:
        df['lesion_id'] = df['lesion_id'].astype(str)
    else:
        df['lesion_id'] = None

    # 4) Filtrar filas sin filepath existente
    df = df.dropna(subset=['filepath']).reset_index(drop=True)
    exists_mask = df['filepath'].apply(lambda p: os.path.exists(p))
    if not exists_mask.all():
        missing = (~exists_mask).sum()
        print(f"[WARN] {missing} filas tienen 'filepath' que no existe en disco. Se descartarán.")
        df = df[exists_mask].reset_index(drop=True)

    # 5) Normalizar y mapear etiquetas
    df['label_raw'] = df[label_col].astype(str).str.lower()
    df['label'] = df['label_raw'].map(DEFAULT_LABEL_MAP).fillna(df['label_raw'])
    # Asegurar columnas finales
    return df[['filepath','label','lesion_id']].reset_index(drop=True)


def encode_labels(df, label_list=None):
    """
    Convierte labels a índices enteros y devuelve (df_copy, mapping).
    - df: DataFrame con columna 'label'
    - label_list: lista de labels en el orden deseado; si None se usa el orden alfabético.
    Devuelve:
      df_out (copia con columna 'label_idx'), mapping (dict label->idx)
    """
    df_copy = df.copy()
    if label_list is None:
        labels = sorted(df_copy['label'].unique().tolist())
    else:
        labels = label_list
    mapping = {lab: idx for idx, lab in enumerate(labels)}
    df_copy['label_idx'] = df_copy['label'].map(mapping)
    return df_copy, mapping


def create_splits_by_lesion(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Crea splits por lesion_id para evitar data leakage:
      - Primero divide lesion_ids en train+val y test.
      - Luego divide train+val en train y val.
    Retorna dict {'train': df_train, 'val': df_val, 'test': df_test}
    """
    if 'lesion_id' not in df.columns:
        raise ValueError("El DataFrame debe contener columna 'lesion_id' para hacer splits por lesión.")
    lesion_ids = df['lesion_id'].unique()
    # Si lesion_id es 'None' (string) para muchas filas, cae al split estándar por fila:
    if len(lesion_ids) < 2:
        # fallback a split por filas (si no hay lesion_id útiles)
        train_val, test = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)
        val_relative = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_relative, stratify=train_val['label'], random_state=random_state)
        return {'train': train.reset_index(drop=True), 'val': val.reset_index(drop=True), 'test': test.reset_index(drop=True)}

    train_val_lesions, test_lesions = train_test_split(lesion_ids, test_size=test_size, random_state=random_state)
    val_relative = val_size / (1 - test_size)
    train_lesions, val_lesions = train_test_split(train_val_lesions, test_size=val_relative, random_state=random_state)

    train_df = df[df['lesion_id'].isin(train_lesions)].reset_index(drop=True)
    val_df = df[df['lesion_id'].isin(val_lesions)].reset_index(drop=True)
    test_df = df[df['lesion_id'].isin(test_lesions)].reset_index(drop=True)

    return {'train': train_df, 'val': val_df, 'test': test_df}
