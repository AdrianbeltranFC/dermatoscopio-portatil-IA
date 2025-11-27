#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import subprocess
import os
from datetime import datetime

# Carpeta donde se guardarán las fotos
CARPETA_FOTOS = os.path.expanduser("~/fotos")


def tomar_foto():
    os.makedirs(CARPETA_FOTOS, exist_ok=True)

    # Nombre de archivo con fecha y hora
    ahora = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"foto_{ahora}.jpg"
    ruta_completa = os.path.join(CARPETA_FOTOS, nombre_archivo)

    # Construir comando base
    comando = ["rpicam-still", "-o", ruta_completa]

    # Vista previa
    if sin_preview_var.get():
        comando.append("--nopreview")

    # Temporizador (segundos → milisegundos)
    try:
        segundos = int(temporizador_var.get())
        if segundos < 0:
            segundos = 0
    except ValueError:
        segundos = 0

    if segundos > 0:
        comando.extend(["-t", str(segundos * 1000)])

    # Resolución
    if resolucion_var.get() == "max":
        # Resolución máxima del Camera Module 3 (IMX708)
        comando.extend(["--width", "4056", "--height", "3040"])

    # Brillo (0–100 → 0.0–1.0)
    brillo_slider = brillo_var.get()
    brillo_val = brillo_slider / 100.0
    # Solo lo agregamos si no está en 50 (valor por defecto)
    if brillo_slider != 50:
        comando.extend(["--brightness", f"{brillo_val:.2f}"])

    # Zoom / ROI
    if zoom_var.get() == 1:
        # ROI: x, y, w, h → zoom centrado aprox. 2x
        comando.extend(["--roi", "0.25,0.25,0.5,0.5"])

    try:
        subprocess.run(comando, check=True)
        messagebox.showinfo("Foto tomada", f"Foto guardada en:\n{ruta_completa}")
    except subprocess.CalledProcessError as e:
        messagebox.showerror(
            "Error",
            f"No se pudo tomar la foto.\n\nComando:\n{' '.join(comando)}\n\nDetalle:\n{e}"
        )


# Ventana principal
ventana = tk.Tk()
ventana.title("Cámara Raspberry Pi (rpicam-still)")
ventana.geometry("400x350")

# ===== Sección temporizador =====
frame_temp = ttk.LabelFrame(ventana, text="Temporizador")
frame_temp.pack(fill="x", padx=10, pady=5)

temporizador_var = tk.StringVar(value="0")
tk.Label(frame_temp, text="Segundos antes de tomar la foto:").pack(anchor="w", padx=10, pady=2)
tk.Entry(frame_temp, textvariable=temporizador_var, width=10).pack(anchor="w", padx=10, pady=2)

# ===== Sección vista previa =====
frame_preview = ttk.LabelFrame(ventana, text="Vista previa")
frame_preview.pack(fill="x", padx=10, pady=5)

sin_preview_var = tk.BooleanVar(value=True)
tk.Checkbutton(
    frame_preview,
    text="Sin vista previa (modo silencioso)",
    variable=sin_preview_var
).pack(anchor="w", padx=10, pady=2)

# ===== Sección resolución =====
frame_res = ttk.LabelFrame(ventana, text="Resolución")
frame_res.pack(fill="x", padx=10, pady=5)

resolucion_var = tk.StringVar(value="auto")
tk.Radiobutton(
    frame_res,
    text="Automática",
    variable=resolucion_var,
    value="auto"
).pack(anchor="w", padx=10, pady=2)

tk.Radiobutton(
    frame_res,
    text="Máxima (4056 x 3040)",
    variable=resolucion_var,
    value="max"
).pack(anchor="w", padx=10, pady=2)

# ===== Sección brillo =====
frame_brillo = ttk.LabelFrame(ventana, text="Brillo")
frame_brillo.pack(fill="x", padx=10, pady=5)

brillo_var = tk.IntVar(value=50)  # 50% = 0.5
tk.Label(frame_brillo, text="Brillo (0 = oscuro, 100 = muy brillante):").pack(anchor="w", padx=10, pady=2)
tk.Scale(
    frame_brillo,
    from_=0,
    to=100,
    orient="horizontal",
    variable=brillo_var
).pack(fill="x", padx=10, pady=2)

# ===== Sección zoom =====
frame_zoom = ttk.LabelFrame(ventana, text="Zoom")
frame_zoom.pack(fill="x", padx=10, pady=5)

zoom_var = tk.IntVar(value=0)
tk.Checkbutton(
    frame_zoom,
    text="Zoom centrado (recorte ROI)",
    variable=zoom_var
).pack(anchor="w", padx=10, pady=2)

# ===== Botón principal =====
tk.Button(
    ventana,
    text="Tomar foto",
    command=tomar_foto,
    height=2
).pack(pady=15)

# Carpeta info
tk.Label(
    ventana,
    text=f"Las fotos se guardan en: {CARPETA_FOTOS}",
    fg="gray",
    wraplength=380
).pack(pady=5)

ventana.mainloop()