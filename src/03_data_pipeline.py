"""
03_data_pipeline.py
Organiza las imágenes usando data/metadata.csv en carpetas procesadas:
data/processed/train/<label>/
data/processed/val/<label>/
data/processed/test/<label>/

División:
 - stratificada por etiqueta (label), pero usando GROUP SPLIT por lesion_id
 - default: train 80%, val 10%, test 10%

Uso:
 python src/03_data_pipeline.py --meta data/metadata.csv --out data/processed --train_frac 0.8 --val_frac 0.1
"""
import argparse
import pandas as pd
import os
import shutil
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline de división train/val/test del dataset HAM10000")
    p.add_argument("--meta", required=True, help="CSV metadata generado (data/metadata.csv)")
    p.add_argument("--out", default="data/processed", help="Carpeta de salida con train/val/test")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def ensure_dir(p):
    """Crea directorio si no existe"""
    os.makedirs(p, exist_ok=True)
    logger.debug(f"Directorio asegurado: {p}")

def normalize_filepath(filepath):
    """
    Normaliza rutas de archivo para compatibilidad cross-platform
    Maneja rutas absolutas, relativas, con / o \
    """
    # Convertir a Path para normalización
    path = Path(filepath)
    # Convertir a string con separadores del sistema
    return str(path).replace('\\', os.sep).replace('/', os.sep)

def validate_file_exists(filepath):
    """Valida que un archivo exista"""
    normalized = normalize_filepath(filepath)
    if not os.path.exists(normalized):
        return False
    return True

def copy_subset(df_subset, dst_root):
    """
    Copia imágenes a carpetas por clase
    
    Args:
        df_subset: dataframe con columnas filepath, label
        dst_root: path/to/processed/train o val o test
    """
    copied_count = 0
    failed_count = 0
    
    for idx, r in df_subset.iterrows():
        label = r['label']
        src = r['filepath']
        
        # Normalizar ruta
        full_src = normalize_filepath(src)
        
        # Validar que el archivo existe
        if not os.path.exists(full_src):
            logger.debug(f"No se encontró: {full_src}")
            failed_count += 1
            continue
        
        # Crear directorio de destino por clase
        dst_dir = os.path.join(dst_root, str(label))
        ensure_dir(dst_dir)
        
        # Copiar archivo manteniendo el nombre
        fname = os.path.basename(full_src)
        dst = os.path.join(dst_dir, fname)
        
        try:
            shutil.copy2(full_src, dst)
            copied_count += 1
            if copied_count % 500 == 0:
                logger.info(f"  Progreso: {copied_count} imágenes copiadas...")
        except Exception as e:
            logger.error(f"Error al copiar {full_src} -> {dst}: {e}")
            failed_count += 1
    
    logger.info(f"✓ Copiadas {copied_count} imágenes, {failed_count} errores")
    return copied_count, failed_count

def main(args):
    logger.info("=" * 80)
    logger.info("Iniciando pipeline de división de dataset HAM10000")
    logger.info("=" * 80)
    
    # Validar que el CSV existe
    if not os.path.exists(args.meta):
        raise FileNotFoundError(f"Archivo de metadata no encontrado: {args.meta}")
    
    logger.info(f"Cargando metadata desde: {args.meta}")
    df = pd.read_csv(args.meta)
    
    # Validar columnas requeridas
    required_cols = ['label', 'filepath', 'lesion_id']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Columnas faltantes en metadata: {missing_cols}")
        logger.error(f"Columnas disponibles: {list(df.columns)}")
        raise ValueError(f"Columnas faltantes en metadata: {missing_cols}")
    
    logger.info(f"Total de imágenes en metadata: {len(df)}")
    logger.info(f"Clases: {sorted(df['label'].unique().tolist())}")
    logger.info(f"\nDistribución de clases:")
    for label, count in df['label'].value_counts().sort_index().items():
        logger.info(f"  {label}: {count} imágenes")
    
    # Validar fracciones
    total = args.train_frac + args.val_frac + args.test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train_frac + val_frac + test_frac deben sumar 1.0, pero suman {total}")
    
    logger.info(f"División: train={args.train_frac*100:.1f}%, val={args.val_frac*100:.1f}%, test={args.test_frac*100:.1f}%")
    
    # Primer split: train vs temp (val+test)
    logger.info("Realizando split train/temp (val+test)...")
    gss = GroupShuffleSplit(n_splits=1, train_size=args.train_frac, random_state=args.seed)
    groups = df['lesion_id'].values
    train_idx, temp_idx = next(gss.split(df, groups=groups))
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)
    
    logger.info(f"Train: {len(df_train)} imágenes")
    logger.info(f"Temp (val+test): {len(df_temp)} imágenes")
    
    # Segundo split: val vs test
    logger.info("Realizando split val/test...")
    rel_val_frac = args.val_frac / (args.val_frac + args.test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val_frac, random_state=args.seed)
    groups_temp = df_temp['lesion_id'].values
    val_idx_rel, test_idx_rel = next(gss2.split(df_temp, groups=groups_temp))
    
    df_val = df_temp.iloc[val_idx_rel].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx_rel].reset_index(drop=True)
    
    logger.info(f"Val: {len(df_val)} imágenes")
    logger.info(f"Test: {len(df_test)} imágenes")
    
    # Mostrar distribución por clase
    logger.info("\n--- Distribución por clase ---")
    logger.info("Train:")
    for label, count in df_train['label'].value_counts().sort_index().items():
        logger.info(f"  {label}: {count}")
    logger.info("Val:")
    for label, count in df_val['label'].value_counts().sort_index().items():
        logger.info(f"  {label}: {count}")
    logger.info("Test:")
    for label, count in df_test['label'].value_counts().sort_index().items():
        logger.info(f"  {label}: {count}")
    
    # Crear estructura de carpetas y copiar imágenes
    logger.info(f"\nCreando estructura en: {args.out}")
    out_root = Path(args.out)
    ensure_dir(out_root)
    
    total_copied = 0
    total_failed = 0
    
    for part, dframe in [('train', df_train), ('val', df_val), ('test', df_test)]:
        logger.info(f"\n--- Procesando {part.upper()} ({len(dframe)} imágenes) ---")
        dst = out_root / part
        ensure_dir(dst)
        
        # Copiar imágenes
        copied, failed = copy_subset(dframe, str(dst))
        total_copied += copied
        total_failed += failed
        
        # Guardar CSV de metadata
        csv_path = out_root / f"metadata_{part}.csv"
        dframe.to_csv(csv_path, index=False)
        logger.info(f"Metadata guardado: {csv_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completado:")
    logger.info(f"  ✓ Total de imágenes copiadas: {total_copied}")
    logger.info(f"  ✗ Total de errores: {total_failed}")
    logger.info(f"  Estructura creada en: {out_root}")
    logger.info("  Archivos metadata: metadata_train.csv, metadata_val.csv, metadata_test.csv")
    logger.info("=" * 80)
    
    if total_failed > 0:
        logger.warning(f"\n⚠️  {total_failed} imágenes no fueron encontradas.")
        logger.warning("Por favor verifica que las rutas en metadata coincidan con tu estructura de carpetas.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
