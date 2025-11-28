"""
Crea skin_lesion_classifier.h5 desde best_model.h5
"""

import tensorflow as tf
from pathlib import Path
import shutil

# Copiar best_model.h5 como skin_lesion_classifier.h5
source = Path("models/checkpoints/best_model.h5")
dest = Path("models/skin_lesion_classifier.h5")

if source.exists():
    shutil.copy(source, dest)
    print(f"✓ Copiado: {source} → {dest}")
    
    # Verificar
    if dest.exists():
        print(f"✓ Archivo creado exitosamente")
        print(f"  Tamaño: {dest.stat().st_size / 1024 / 1024:.2f} MB")
else:
    print(f"✗ {source} no existe")
