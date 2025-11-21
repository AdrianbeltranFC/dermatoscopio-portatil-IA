"""
explore_metadata.py
Script de Análisis Exploratorio de Datos (EDA) para:
- Mostrar estadísticas básicas
- Guardar una cuadrícula de ejemplo por clase (9 imágenes por clase, puedes editar ese parámetro al llamar el script)
- Guardar lista de imágenes más grandes/pequeñas (si interesa)

Uso:
python src/01_explore_metadata.py --meta data/metadata.csv --out results/ --n_examples 9

"""
import argparse
import pandas as pd
import os
from PIL import Image
import math

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default="data/metadata.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--n_examples", type=int, default=9, help="Ejemplos por clase")
    return p.parse_args()

def make_grid(image_paths, outpath, grid_size=(3,3), thumb_size=(224,224)):
    cols, rows = grid_size
    w, h = thumb_size
    grid = Image.new('RGB', (cols*w, rows*h), (255,255,255))
    for i, p in enumerate(image_paths[:cols*rows]):
        try:
            im = Image.open(p).convert('RGB')
            im = im.resize((w,h))
        except Exception as e:
            print("Error leyendo:", p, e)
            im = Image.new('RGB', (w,h), (200,200,200))
        x = (i % cols) * w
        y = (i // cols) * h
        grid.paste(im, (x,y))
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    grid.save(outpath)

def main(args):
    df = pd.read_csv(args.meta)
    os.makedirs(args.out, exist_ok=True)
    print("Total imágenes:", len(df))
    print("Total lesiones únicas:", df['lesion_id'].nunique())
    print("Distribución por clase:\n", df['label'].value_counts())

    # Guardar ejemplos por clase
    labels = sorted(df['label'].unique())
    for lab in labels:
        sub = df[df['label']==lab]
        sample = sub['filepath'].tolist()[:args.n_examples]
        outfile = os.path.join(args.out, f"examples_{lab}.png")
        make_grid(sample, outfile)
        print("Guardado ejemplos:", outfile, " (n=", len(sample), ")")

    # Guardar csv reducido con counts por lesion_id (útil para ver lesion con varias imgs)
    lesion_counts = df.groupby('lesion_id').size().reset_index(name='n_images')
    lesion_counts.to_csv(os.path.join(args.out, "lesion_counts.csv"), index=False)
    print("Guardado lesion_counts.csv")

if __name__ == "__main__":
    main(parse_args())
