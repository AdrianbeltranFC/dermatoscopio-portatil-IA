"""
03_data_pipeline.py
Divide HAM10000 en 3 clases:
- mel (Melanoma)
- nv (Lunar Benigno)
- other (Otros: akiec, bcc, bkl, df, vasc)
"""
import argparse
import pandas as pd
import os
import shutil
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--images_root", default="data/raw")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def map_to_3_classes(label):
    """
    Mapea 7 clases HAM10000 a 3 clases clínicas.
    
    7 CLASES ORIGINALES:
    - akiec: Actinic keratosis
    - bcc: Basal cell carcinoma
    - bkl: Benign keratosis-like
    - df: Dermatofibroma
    - mel: Melanoma ✓ MANTENER
    - nv: Nevi ✓ MANTENER
    - vasc: Vascular lesions
    
    MAPEO A 3 CLASES:
    - mel → mel (Melanoma - MALIGNO)
    - nv → nv (Nevi - BENIGNO)
    - Resto → other (Otras lesiones)
    """
    if label == 'mel':
        return 'mel'      # Melanoma
    elif label == 'nv':
        return 'nv'       # Lunar benigno
    else:
        return 'other'    # Todo lo demás (akiec, bcc, bkl, df, vasc)

def normalize_path(filepath):
    """Normaliza rutas para compatibilidad Windows/Unix."""
    path = Path(filepath)
    return str(path).replace('\\', os.sep).replace('/', os.sep)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def copy_and_map_subset(df_subset, dst_root, images_root, split_name):
    """
    Copia imágenes agrupándolas en 3 clases.
    
    Args:
        df_subset: DataFrame con datos
        dst_root: Directorio destino (train/val/test)
        images_root: Raíz de imágenes
        split_name: Nombre del split (train/val/test)
    """
    logger.info(f"\n[{split_name.upper()}]")
    
    copied = 0
    failed = 0
    class_counts = {'mel': 0, 'nv': 0, 'other': 0}
    
    for idx, r in df_subset.iterrows():
        filepath = r['filepath']
        label_original = r['label']
        label_3class = map_to_3_classes(label_original)
        
        # Normalizar ruta
        normalized = normalize_path(filepath)
        
        if not os.path.exists(normalized):
            failed += 1
            continue
        
        # Crear directorio de clase
        dst_dir = os.path.join(dst_root, label_3class)
        ensure_dir(dst_dir)
        
        # Copiar archivo
        fname = os.path.basename(normalized)
        dst = os.path.join(dst_dir, fname)
        
        try:
            shutil.copy2(normalized, dst)
            copied += 1
            class_counts[label_3class] += 1
            
            if copied % 500 == 0:
                logger.info(f"  Copiadas: {copied}...")
        except Exception as e:
            logger.error(f"Error: {e}")
            failed += 1
    
    # Reporte
    logger.info(f"  ✓ Copiadas: {copied}")
    logger.info(f"  ✗ Errores: {failed}")
    logger.info(f"  Distribución clases:")
    for clase, count in class_counts.items():
        logger.info(f"    {clase}: {count}")
    
    return copied, failed

def main(args):
    logger.info("=" * 80)
    logger.info("Pipeline: HAM10000 (7 clases) → 3 clases (mel, nv, other)")
    logger.info("=" * 80)
    
    # Cargar metadata
    if not os.path.exists(args.meta):
        raise FileNotFoundError(f"metadata.csv no encontrado: {args.meta}")
    
    df = pd.read_csv(args.meta)
    
    if 'lesion_id' not in df.columns or 'label' not in df.columns:
        raise ValueError("metadata.csv debe tener: lesion_id, label, filepath")
    
    logger.info(f"\nTotal imágenes: {len(df)}")
    logger.info(f"Clases originales (7):\n{df['label'].value_counts().sort_index()}")
    
    # Mapear a 3 clases
    df['label_3class'] = df['label'].apply(map_to_3_classes)
    
    logger.info(f"\nClases después de mapeo (3):\n{df['label_3class'].value_counts()}")
    
    # Mostrar mapeo detallado
    logger.info("\nMAPEO DE 7→3 CLASES:")
    for orig_class in sorted(df['label'].unique()):
        mapped_class = map_to_3_classes(orig_class)
        count = len(df[df['label'] == orig_class])
        logger.info(f"  {orig_class:6} → {mapped_class:5} ({count:4} imágenes)")
    
    # SPLIT: Train / Temp (Val+Test)
    logger.info("\n[SPLITTING]")
    test_frac = 1.0 - args.train_frac - args.val_frac
    
    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=args.train_frac,
        random_state=args.seed
    )
    
    groups = df['lesion_id'].values
    train_idx, temp_idx = next(gss.split(df, groups=groups))
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)
    
    logger.info(f"Train: {len(df_train)} | Temp (Val+Test): {len(df_temp)}")
    
    # SPLIT: Val / Test
    rel_val_frac = args.val_frac / (args.val_frac + test_frac)
    
    gss2 = GroupShuffleSplit(
        n_splits=1,
        train_size=rel_val_frac,
        random_state=args.seed
    )
    
    groups_temp = df_temp['lesion_id'].values
    val_idx, test_idx = next(gss2.split(df_temp, groups=groups_temp))
    
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)
    
    logger.info(f"Val: {len(df_val)} | Test: {len(df_test)}")
    
    # COPIAR ARCHIVOS CON MAPEO
    logger.info("\n[COPIANDO IMÁGENES]")
    out_root = Path(args.out)
    ensure_dir(out_root)
    
    total_copied = 0
    total_failed = 0
    
    for part, dframe in [('train', df_train), ('val', df_val), ('test', df_test)]:
        dst = out_root / part
        ensure_dir(dst)
        
        # Limpiar directorio anterior si existe
        for old_class_dir in dst.glob('*'):
            if old_class_dir.is_dir():
                shutil.rmtree(old_class_dir)
        
        copied, failed = copy_and_map_subset(dframe, str(dst), args.images_root, part)
        total_copied += copied
        total_failed += failed
        
        # Guardar metadata
        dframe.to_csv(out_root / f"metadata_{part}.csv", index=False)
        logger.info(f"  ✓ metadata_{part}.csv guardado")
    
    # RESUMEN FINAL
    logger.info("\n" + "=" * 80)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 80)
    logger.info(f"✓ Imágenes copiadas: {total_copied}")
    logger.info(f"✗ Errores: {total_failed}")
    
    # Mostrar estructura final
    logger.info("\n[ESTRUCTURA FINAL]")
    for split in ['train', 'val', 'test']:
        split_dir = out_root / split
        logger.info(f"\n{split.upper()}:")
        for clase_dir in sorted(split_dir.glob('*')):
            if clase_dir.is_dir():
                file_count = len(list(clase_dir.glob('*.jpg')))
                logger.info(f"  {clase_dir.name}: {file_count} imágenes")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE COMPLETADO")
    logger.info("=" * 80)
    logger.info("\nEstructura lista para entrenamiento:")
    logger.info(f"  {out_root}/train/mel/")
    logger.info(f"  {out_root}/train/nv/")
    logger.info(f"  {out_root}/train/other/")
    logger.info(f"  {out_root}/val/...")
    logger.info(f"  {out_root}/test/...")

if __name__ == "__main__":
    main(parse_args())
