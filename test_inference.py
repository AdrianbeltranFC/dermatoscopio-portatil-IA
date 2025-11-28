"""
Script para probar el modelo con una imagen

Uso:
    python test_inference.py --image test_lesion.jpg
"""

import argparse
import os
from pathlib import Path
from src.inference import SkinLesionInference

def main():
    parser = argparse.ArgumentParser(description="Probar modelo con una imagen")
    parser.add_argument("--image", required=True, help="Ruta a imagen (JPG)")
    parser.add_argument("--model", default="models/skin_lesion_classifier.h5")
    parser.add_argument("--output", default="./results/")
    args = parser.parse_args()
    
    # Convertir a rutas absolutas y normalizadas
    model_path = Path(args.model).resolve()
    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()
    
    # Verificar que existen
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        print(f"Por favor, extrae models.zip primero:")
        print(f"  Expand-Archive -Path models.zip -DestinationPath .")
        return
    
    if not image_path.exists():
        print(f"❌ Imagen no encontrada: {image_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("PRUEBA DE INFERENCIA")
    print("="*80 + "\n")
    
    print(f"[1/3] Cargando modelo...")
    print(f"       Modelo: {model_path}")
    try:
        inference = SkinLesionInference(str(model_path), segmenter_debug=False)
        print("       ✓ Modelo cargado\n")
    except Exception as e:
        print(f"       ✗ Error: {e}")
        return
    
    print(f"[2/3] Procesando imagen...")
    print(f"       Imagen: {image_path.name}")
    
    try:
        result = inference.process_image(str(image_path), str(output_path))
    except Exception as e:
        print(f"       ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not result['success']:
        print(f"       ✗ Error procesando imagen")
        return
    
    print(f"       ✓ Imagen procesada\n")
    
    print(f"[3/3] RESULTADOS:")
    print(f"       Clase: {result['class']}")
    print(f"       Confianza: {result['confidence']*100:.1f}%")
    print(f"\n       Detalles por clase:")
    for clase, prob in result['all_predictions'].items():
        bar = "█" * int(prob * 30)
        print(f"         {clase:20} {prob*100:5.1f}% {bar}")
    
    print(f"\n       Visualización guardada en: {output_path}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
