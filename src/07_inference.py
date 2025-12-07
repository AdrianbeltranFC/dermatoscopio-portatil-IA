"""
Pipeline Completo: Segmentación + Clasificación (Soporte TFLite + Keras)

Uso: 
    python test_inference.py --image "RUTA DE TU IMAGEN" --model "models/tflite/skin_lesion_classifier_float16.tflite"

    No olvides cambiar "RUTA DE TU IMAGEN" por alguna de las rutas de imagenes de la carpeta de data
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import time

from .segmentation import SkinLesionSegmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinLesionInference:
    """Pipeline completo de segmentación y clasificación."""
    
    def __init__(self, model_path, segmenter_debug=False):
        self.model_path = Path(model_path)
        self.segmenter = SkinLesionSegmenter(debug=segmenter_debug)
        self.class_names = ['Melanoma', 'Lunar Benigno', 'Otro']
        self.model_type = 'keras' # por defecto
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Carga modelo (detecta si es .h5 o .tflite)."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        
        # Detectar tipo de modelo
        if self.model_path.suffix == '.tflite':
            self.model_type = 'tflite'
            logger.info(f"🔄 Cargando modelo TFLite: {self.model_path.name}")
            try:
                self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                logger.info("✓ Modelo TFLite cargado correctamente")
            except Exception as e:
                raise RuntimeError(f"Error cargando TFLite: {e}")
        else:
            self.model_type = 'keras'
            logger.info(f"🔄 Cargando modelo Keras: {self.model_path.name}")
            try:
                self.model = tf.keras.models.load_model(str(self.model_path))
                logger.info("✓ Modelo Keras cargado")
            except Exception as e:
                raise RuntimeError(f"Error cargando modelo Keras: {e}")
    
    def process_image(self, image_path, output_dir=None):
        """Procesa imagen completa: segmentación + clasificación."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            return {'success': False, 'error': f'Imagen no encontrada'}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Procesando: {image_path.name}")
        
        # PASO 1: Segmentación
        logger.info("[1/3] Segmentación YCbCr...")
        seg_result = self.segmenter.segment(image_path, output_dir)
        
        if not seg_result['success']:
            logger.error("❌ Segmentación fallida")
            return {'success': False, 'error': 'Segmentation failed'}
        
        # PASO 2: Preparación para el modelo
        logger.info("[2/3] Preparando imagen (Color + Resize)...")
        roi = seg_result['lesion_roi']
        
        # CRÍTICO: Convertir BGR (OpenCV) a RGB (Entrenamiento)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Resize a 224x224
        roi_resized = cv2.resize(roi_rgb, (224, 224))
        
        # Convertir a float32 (0-255)
        # NO dividir entre 255 si el modelo tiene capa de Rescaling interna
        input_data = roi_resized.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        
        # PASO 3: Inferencia
        logger.info(f"[3/3] Clasificando con motor {self.model_type.upper()}...")
        start_time = time.time()
        
        try:
            if self.model_type == 'tflite':
                # Inferencia TFLite
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            else:
                # Inferencia Keras (.h5)
                predictions = self.model.predict(input_data, verbose=0)[0]
                
        except Exception as e:
            logger.error(f"Error en inferencia: {e}")
            return {'success': False, 'error': str(e)}
            
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"⏱️ Tiempo inferencia: {elapsed:.1f}ms")

        # Procesar resultados
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        class_name = self.class_names[class_idx]
        
        # Log de resultados
        logger.info(f"\n📊 RESULTADOS:")
        for i, name in enumerate(self.class_names):
            prob = predictions[i] * 100
            bar = "█" * int(predictions[i] * 20)
            logger.info(f"   {name:<15}: {prob:5.1f}% {bar}")
        
        # PASO 4: Visualización
        visualization = self._create_visualization(seg_result, class_name, confidence)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            vis_path = output_dir / f"{image_path.stem}_result.jpg"
            cv2.imwrite(str(vis_path), visualization)
            logger.info(f"\n💾 Visualización guardada: {vis_path}")
        
        return {
            'success': True,
            'class': class_name,
            'confidence': float(confidence),
            'all_predictions': {name: float(pred) for name, pred in zip(self.class_names, predictions)},
            'visualization': visualization
        }
    
    def _create_visualization(self, seg_result, class_name, confidence):
        """Crea imagen de visualización."""
        original = seg_result['original'].copy()
        h, w = original.shape[:2]
        
        # Canvas: Izquierda (Original+Contorno), Derecha (Máscara)
        canvas = np.ones((h, w*2 + 20, 3), dtype=np.uint8) * 255
        
        # 1. Lado Izquierdo: Original + Contorno AZUL
        img_contour = original.copy()
        # Usamos AZUL (255, 0, 0) para el contorno
        cv2.drawContours(img_contour, [seg_result['contour']], -1, (255, 0, 0), 3)
        # Dibujar bounding box (opcional, verde finito)
        x, y, wr, hr = seg_result['bbox']
        cv2.rectangle(img_contour, (x, y), (x+wr, y+hr), (0, 255, 0), 1)
        
        canvas[:h, :w] = img_contour
        
        # 2. Lado Derecho: Máscara binaria
        mask_vis = cv2.cvtColor(seg_result['segmentation_mask'], cv2.COLOR_GRAY2BGR)
        canvas[:h, w+20:] = mask_vis
        
        # 3. Textos informativos
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Títulos
        cv2.putText(canvas, "Segmentacion", (10, 30), font, 0.8, (255, 0, 0), 2)
        cv2.putText(canvas, "Mascara IA", (w + 30, 30), font, 0.8, (0, 0, 0), 2)
        
        # Resultado (Fondo blanco, letras negras)
        res_text = f"{class_name} ({confidence*100:.1f}%)"
        cv2.putText(canvas, res_text, (10, h-20), font, 1.0, (0, 0, 255), 2)
        
        return canvas