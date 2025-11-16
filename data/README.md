# Instrucciones de datos

Este directorio está diseñado para contener los datos de entrenamiento, pero se encuentra vacío por defecto para evitar subir gigabytes de imágenes al repositorio de Git.

Para poblar este directorio y poder ejecutar el entrenamiento, sigue estos pasos.

---

## Opción A (Recomendada): Kaggle

Esta opción utiliza la API de Kaggle para descargar el conjunto de datos **HAM10000** de forma automática.

### 1. Configurar API de Kaggle (Solo se hace una vez)

1. Ve a [Kaggle.com](https://www.kaggle.com/), crea una cuenta y ve a tu perfil.  
2. En la sección **"Cuenta" -> "API"**, haz clic en **"Crear nuevo token de API"**.  
3. Esto descargará un archivo llamado `kaggle.json`.  
4. Mueve este archivo a la carpeta `.kaggle` en tu directorio de usuario. La ruta final debe ser:  
   - En **Windows**: `C:\Users\<TuUsuario>\.kaggle\kaggle.json`  
   - En **macOS/Linux**: `~/.kaggle/kaggle.json`

### 2. Descargar los Datos

Abre una terminal en la **raíz de este proyecto** (la carpeta `dermatoscopio-portatil-IA`) y ejecuta:

```bash
bash download_data.sh
```
#### Nota IMPORTANTE para usuarios de Windows (PowerShell)

Si el comando `bash` falla (es común en PowerShell), puedes ejecutar los comandos clave del script manualmente. Son estos dos:

1. Crear el directorio:
```powershell
mkdir -p data/raw
```
2. Descargar y descomprimir:
```powershell
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw --unzip
```

### 3. Preparar el Archivo de Metadatos
Una vez que las imágenes estén en data/raw/, significa que lograste descargar las imagenes y la metadata original. Para motivos de este proyecto, ahora debes ejecutar el script de Python para crear el archivo metadata.csv unificado que usará el modelo (toma el archivo crudo original de metadata y crea una versión nueva mas limpia y con imagenes únicamente de las etiquetas que se usaran para entrenar el modelo):
```
python src/make_metadata.py --src data/raw/HAM10000_metadata.csv --images_dir data/raw --out data/metadata.csv
```
Nota: Este script se debe ejecutar una única ves, ya que revisa cada archivo descargado y toma varios minutos en completarse. 
## Opción B (Alternativa): Archivo ISIC
Ve al [Archivo ISIC ](https://www.isic-archive.com/)y busca la colección HAM10000.

Descarga los datos manualmente.

Descomprime y coloca todas las imágenes dentro de la carpeta data/raw/ de este proyecto.

Asegúrate de que el archivo HAM10000_metadata.csv también esté en data/raw/.

Ejecuta el paso 3 de la Opción A para generar el data/metadata.csv.