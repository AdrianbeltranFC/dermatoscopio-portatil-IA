"""
Aplicación interactiva para Raspberry Pi 5

Características:
- Captura en tiempo real desde cámara
- Segmentación YCbCr
- Clasificación con TFLite
- Visualización en pantalla
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

from segmentation import SkinLesionSegmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RaspberryPiApp:
    """Aplicación para Raspberry Pi con cámara."""
    
    def __init__(self, tflite_model_path, camera_index=0):
        """
        Inicializa aplicación.
        
        Args:
            tflite_model_path (str): Ruta a modelo TFLite
            camera_index (int): Índice de cámara
        """
        self.tflite_model_path = Path(tflite_model_path)
        self.class_names = ['Melanoma', 'Lunar Benigno', 'Otro']
        self.segmenter = SkinLesionSegmenter(debug=False)
        
        # Cargar modelo TFLite
        self.interpreter = tf.lite.Interpreter(str(self.tflite_model_path))
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Cámara
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        logger.info("✓ Aplicación Raspberry Pi inicializada")
    
    def run(self, save_dir=None):
        """
        Ejecuta aplicación interactiva.
        
        Args:
            save_dir (Path): Directorio para guardar capturas
        
        Controles:
            's': Capturar y procesar
            'q': Salir
        """
        save_dir = Path(save_dir) if save_dir else None
        
        logger.info("\n[Aplicación Iniciada]")
        logger.info("Controles:")
        logger.info("  's' - Capturar y procesar")
        logger.info("  'q' - Salir")
        
        capture_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.error("Error capturando frame")
                break
            
            # Mostrar frame
            display_frame = frame.copy()
            cv2.putText(display_frame, "Presiona 's' para capturar", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Dermatoscopio - Presiona s para capturar', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                logger.info(f"\n{'='*60}")
                logger.info(f"Captura #{capture_count + 1}")
                logger.info(f"{'='*60}")
                
                # Guardar captura
                if save_dir:
                    capture_path = save_dir / f"capture_{capture_count:03d}.jpg"
                    cv2.imwrite(str(capture_path), frame)
                    logger.info(f"✓ Guardada: {capture_path}")
                else:
                    capture_path = Path("/tmp") / f"capture_{capture_count:03d}.jpg"
                    cv2.imwrite(str(capture_path), frame)
                
                # Procesar
                self._process_frame(frame, capture_count)
                
                capture_count += 1
            
            elif key == ord('q'):
                logger.info("\n¡Aplicación cerrada!")
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _process_frame(self, frame, capture_id):
        """Procesa un frame capturado."""
        # Guardar temporalmente
        temp_path = Path("/tmp") / f"temp_{capture_id}.jpg"
        cv2.imwrite(str(temp_path), frame)
        
        # Segmentar
        logger.info("[1/2] Segmentando...")
        seg_result = self.segmenter.segment(temp_path)
        
        if not seg_result['success']:
            logger.error("❌ Segmentación fallida")
            temp_path.unlink()
            return
        
        logger.info("✓ Segmentación exitosa")
        
        # Clasificar
        logger.info("[2/2] Clasificando...")
        roi = seg_result['lesion_roi']
        roi_resized = cv2.resize(roi, (224, 224))
        roi_normalized = roi_resized.astype('uint8')
        
        # Agregar batch
        roi_batch = np.expand_dims(roi_normalized, axis=0)
        
        # Inferencia TFLite
        self.interpreter.set_tensor(self.input_details[0]['index'], roi_batch)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        class_name = self.class_names[class_idx]
        
        logger.info(f"✓ Clasificación: {class_name}")
        
        # Mostrar resultados
        logger.info("\nResultados:")
        for name, pred in zip(self.class_names, predictions[0]):
            bar = '█' * int((pred / 255) * 30) if pred > 0 else ''
            logger.info(f"  {name:20} {bar}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ RESULTADO: {class_name} ({confidence*100/255:.1f}%)")
        logger.info(f"{'='*60}")
        
        # Limpiar
        temp_path.unlink()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ruta a modelo TFLite")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()
    
    app = RaspberryPiApp(args.model, args.camera)
    app.run(args.save_dir)
