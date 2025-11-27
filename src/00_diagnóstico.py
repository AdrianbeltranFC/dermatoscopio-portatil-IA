"""
00_diagnóstico.py
Script para diagnosticar problemas en la cadena de procesamiento de datos.
"""
import os
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_filepath(filepath):
    """Normaliza rutas para compatibilidad cross-platform"""
    path = Path(filepath)
    return str(path).replace('\\', os.sep).replace('/', os.sep)

def diagnose_structure():
    """Diagnostica la estructura de carpetas"""
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNÓSTICO 1: Estructura de Carpetas")
    logger.info("=" * 80)
    
    paths_to_check = [
        'data',
        'data/raw',
        'data/metadata.csv',
    ]
    
    for path in paths_to_check:
        exists = os.path.exists(path)
        status = "✓ EXISTE" if exists else "✗ NO EXISTE"
        logger.info(f"{status}: {path}")
        
        if exists and os.path.isdir(path):
            try:
                contents = os.listdir(path)
                logger.info(f"  Contenido ({len(contents)} items):")
                for item in contents[:10]:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        count = len(os.listdir(item_path))
                        logger.info(f"    [DIR] {item} ({count} items)")
                    else:
                        size_mb = os.path.getsize(item_path) / (1024*1024)
                        logger.info(f"    [FILE] {item} ({size_mb:.2f} MB)")
                if len(contents) > 10:
                    logger.info(f"    ... y {len(contents) - 10} items más")
            except Exception as e:
                logger.error(f"  Error listando contenido: {e}")

def diagnose_metadata():
    """Diagnostica el archivo metadata.csv"""
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNÓSTICO 2: Archivo metadata.csv")
    logger.info("=" * 80)
    
    if not os.path.exists('data/metadata.csv'):
        logger.error("✗ data/metadata.csv NO EXISTE")
        return None
    
    logger.info("✓ data/metadata.csv EXISTE")
    
    try:
        df = pd.read_csv('data/metadata.csv')
        logger.info(f"  Total de registros: {len(df)}")
        logger.info(f"  Columnas: {list(df.columns)}")
        
        logger.info("\n  Primeras 3 filas:")
        for idx, row in df.head(3).iterrows():
            logger.info(f"    [{idx}] {dict(row)}")
        
        logger.info("\n  Análisis de columna 'filepath':")
        if 'filepath' in df.columns:
            logger.info(f"    Total entries: {len(df)}")
            logger.info(f"    Ejemplos de rutas:")
            for i, fp in enumerate(df['filepath'].head(5)):
                logger.info(f"      [{i}] {fp}")
        
        return df
    except Exception as e:
        logger.error(f"✗ Error cargando metadata.csv: {e}")
        return None

def diagnose_image_paths(df):
    """Diagnostica las rutas de imágenes"""
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNÓSTICO 3: Validación de Rutas")
    logger.info("=" * 80)
    
    if df is None or 'filepath' not in df.columns:
        logger.error("No se puede diagnosticar sin metadata.csv válido")
        return
    
    logger.info("Probando primeras 10 imágenes:")
    found = 0
    not_found = 0
    
    for idx, row in df.head(10).iterrows():
        filepath = row['filepath']
        normalized = normalize_filepath(filepath)
        exists = os.path.exists(normalized)
        
        status = "✓" if exists else "✗"
        logger.info(f"  [{status}] {filepath}")
        
        if exists:
            found += 1
        else:
            not_found += 1
    
    logger.info(f"\n  Resumen: {found}/10 encontradas, {not_found}/10 no encontradas")

def diagnose_actual_images():
    """Diagnostica las imágenes que realmente existen"""
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNÓSTICO 4: Imágenes Físicamente Presentes")
    logger.info("=" * 80)
    
    images_root = 'data/raw'
    
    if not os.path.exists(images_root):
        logger.error(f"✗ {images_root} NO EXISTE")
        return
    
    logger.info(f"✓ {images_root} EXISTE")
    
    # Buscar todas las imágenes
    all_images = []
    for root, dirs, files in os.walk(images_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(file)
    
    logger.info(f"Total de imágenes encontradas: {len(all_images)}")
    
    if all_images:
        logger.info("Primeros 10 ejemplos:")
        for img in all_images[:10]:
            logger.info(f"  - {img}")
    else:
        logger.error("✗ NO SE ENCONTRARON IMÁGENES")

def main():
    logger.info("\n\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " DIAGNÓSTICO COMPLETO DEL PIPELINE DE DATOS ".center(78) + "║")
    logger.info("╚" + "=" * 78 + "╝")
    
    diagnose_structure()
    df = diagnose_metadata()
    diagnose_image_paths(df)
    diagnose_actual_images()
    
    logger.info("\n" + "=" * 80)
    logger.info("FIN DEL DIAGNÓSTICO")
    logger.info("=" * 80 + "\n")

if __name__ == "__main__":
    main()
