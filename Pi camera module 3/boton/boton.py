import RPi.GPIO as GPIO
import time
import subprocess
import datetime
import os

GPIO.setmode(GPIO.BCM)
PIN_BOTON = 17

GPIO.setup(PIN_BOTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def tomar_foto():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/home/pi/fotos/foto_{ts}.jpg"
    subprocess.run(["rpicam-still", "-o", filename, "--nopreview"])
    print("Foto capturada:", filename)

print("Botón listo. Presiona para tomar foto.")

try:
    while True:
        if GPIO.input(PIN_BOTON) == 0:         # 0 = presionado
            tomar_foto()
            time.sleep(0.3)                    # antirrebote
        time.sleep(0.05)

except KeyboardInterrupt:
    GPIO.cleanup()