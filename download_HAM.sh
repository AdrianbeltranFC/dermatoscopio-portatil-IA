"""
download_data.sh
Descarga HAM10000 desde Kaggle (requiere kaggle.json en ~/.kaggle/). Ver README.md para más detalles.
Advertencia: el dataset es grande (~5.20GB), por favor asegúrate de tener suficiente espacio en disco.

Uso:
Para Windows (PowerShell):
    kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw --unzip

Para Linux/Mac (Bash):
    bash download_HAM.sh

"""

set -e

OUTDIR="data/raw"
mkdir -p ${OUTDIR}

echo "Descargando HAM10000 desde Kaggle..."
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p ${OUTDIR} --unzip

echo "Contenido de ${OUTDIR}:"
ls -lh ${OUTDIR} | sed -n '1,200p'
echo "Listo. Revisa data/raw/ y coloca metadata.csv en data/"