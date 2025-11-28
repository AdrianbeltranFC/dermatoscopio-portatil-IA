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
    """7 clases → 3 clases."""
    if label == 'mel':
        return 'mel'
    elif label == 'nv':
        return 'nv'
    else:
        return 'other'

def normalize_path(filepath):
    path = Path(filepath)
    return str(path).replace('\\', os.sep).replace('/', os.sep)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def copy_subset(df_subset, dst_root):
    """Copia imágenes agrupadas en 3 clases."""
    copied = 0
    failed = 0
    
    for _, r in df_subset.iterrows():
        filepath = r['filepath']
        label_original = r['label']
        label_3class = map_to_3_classes(label_original)
        
        normalized = normalize_path(filepath)
        
        if not os.path.exists(normalized):
            failed += 1
            continue
        
        dst_dir = os.path.join(dst_root, label_3class)
        ensure_dir(dst_dir)
        
        fname = os.path.basename(normalized)
        dst = os.path.join(dst_dir, fname)
        
        try:
            shutil.copy2(normalized, dst)
            copied += 1
            if copied % 500 == 0:
                logger.info(f"  Copiadas: {copied}...")
        except Exception as e:
            failed += 1
    
    return copied, failed

def main(args):
    logger.info("=" * 80)
    logger.info("Pipeline: HAM10000 → 3 clases")
    logger.info("=" * 80)
    
    # Cargar
    df = pd.read_csv(args.meta)
    
    if 'lesion_id' not in df.columns or 'label' not in df.columns:
        raise ValueError("Falta: lesion_id, label, filepath")
    
    # Mapear
    df['label_3class'] = df['label'].apply(map_to_3_classes)
    
    logger.info(f"Total imágenes: {len(df)}")
    logger.info("Agrupamiento 7→3:")
    print(df.groupby(['label', 'label_3class']).size())
    
    # Split
    test_frac = 1.0 - args.train_frac - args.val_frac
    
    gss = GroupShuffleSplit(n_splits=1, train_size=args.train_frac, random_state=args.seed)
    train_idx, temp_idx = next(gss.split(df, groups=df['lesion_id']))
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)
    
    rel_val_frac = args.val_frac / (args.val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val_frac, random_state=args.seed)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp['lesion_id']))
    
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)
    
    logger.info(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    
    # Copiar
    out_root = Path(args.out)
    ensure_dir(out_root)
    
    for part, dframe in [('train', df_train), ('val', df_val), ('test', df_test)]:
        logger.info(f"\n--- {part.upper()} ---")
        dst = out_root / part
        ensure_dir(dst)
        copied, failed = copy_subset(dframe, str(dst))
        logger.info(f"✓ {copied} copias")
        
        dframe.to_csv(out_root / f"metadata_{part}.csv", index=False)
    
    logger.info("\n✓ Pipeline completado")

if __name__ == "__main__":
    main(parse_args())
