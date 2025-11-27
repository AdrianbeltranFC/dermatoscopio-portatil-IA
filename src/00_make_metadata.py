"""
make_metadata.py
Normaliza la metadata del HAM10000 y genera data/metadata.csv con columnas:
image_id,label,filepath,lesion_id

Filtra por clases seleccionadas (por defecto: mel, nv, bkl).
Busca imágenes recursivamente en --images_dir (útil si están en HAM10000_images_part_1 y _part_2).
Uso:
python src/make_metadata.py --src data/raw/HAM10000_metadata.csv --images_dir data/raw --out data/metadata.csv
"""

import argparse
import pandas as pd
import os

DEFAULT_KEEP = ["mel", "nv", "bkl"]

MAP_LABELS = {
    "mel": "melanoma",
    "nv": "nevus",
    "bkl": "seborrheic_keratosis",
    # si quieres mapear otras: "bcc": "basal_cell_carcinoma", ...
}

def find_file_recursive(images_root, image_id):
    """
    Busca recursivamente un archivo cuyo nombre (sin ext) sea image_id.
    Retorna ruta completa o None.
    """
    for root, _, files in os.walk(images_root):
        for f in files:
            name, ext = os.path.splitext(f)
            if name == image_id:
                return os.path.join(root, f)
    return None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="CSV original (ej. data/HAM10000_metadata.csv)")
    p.add_argument("--images_dir", required=True, help="Carpeta raíz con las imágenes (data/raw/)")
    p.add_argument("--out", default="data/metadata.csv", help="CSV de salida normalizado")
    p.add_argument("--keep", nargs="+", default=DEFAULT_KEEP,
                   help="Lista de códigos dx a conservar (ej. mel nv bkl)")
    p.add_argument("--map_labels", action="store_true", help="Mapear códigos cortos a nombres legibles")
    return p.parse_args()

def main(args):
    df = pd.read_csv(args.src)
    # columnas esperadas: lesion_id, image_id, dx
    for c in ("lesion_id","image_id","dx"):
        if c not in df.columns:
            raise SystemExit(f"Columna requerida '{c}' no encontrada en {args.src}.")

    keep_set = set([k.lower() for k in args.keep])

    rows = []
    missing = 0
    for _, r in df.iterrows():
        dx = str(r["dx"]).lower()
        image_id = str(r["image_id"])
        lesion_id = str(r["lesion_id"])
        if dx not in keep_set:
            continue  # no queremos esta clase para el MVP
        fp = find_file_recursive(args.images_dir, image_id)
        if fp is None:
            missing += 1
            continue
        label = MAP_LABELS.get(dx, dx) if args.map_labels else dx
        rows.append({
            "image_id": image_id,
            "label": label,
            "filepath": fp,
            "lesion_id": lesion_id
        })

    out_df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Metadata generado: {args.out} (filas: {len(out_df)}). Imágenes faltantes: {missing}")

if __name__ == "__main__":
    main(parse_args())
