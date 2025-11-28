"""
Pipeline Completo: Segmentación + Clasificación
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

from .segmentation import SkinLesionSegmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinLesionInference:
    """Pipeline completo de segmentación y clasificación."""
    
    def __init__(self, model_path, segmenter_debug=False):
        """
        Inicializa pipeline.
        
        Args:
            model_path (str/Path): Ruta al modelo H5 entrenado
            segmenter_debug (bool): Debug de segmentación
        """
        self.model_path = Path(model_path)
        self.model = None
        self.segmenter = SkinLesionSegmenter(debug=segmenter_debug)
        self.class_names = ['Melanoma', 'Lunar Benigno', 'Otro']
        
        self._load_model()
    
    def _load_model(self):
        """Carga modelo entrenado."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}\nVerifica que extrajiste models.zip")
        
        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info(f"✓ Modelo cargado: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo: {e}")
    
    def process_image(self, image_path, output_dir=None):
        """
        Procesa imagen completa: segmentación + clasificación.
        
        Args:
            image_path (str/Path): Ruta a imagen
            output_dir (str/Path): Para guardar intermedios
            
        Returns:
            dict: Resultados
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            return {'success': False, 'error': f'Imagen no encontrada: {image_path}'}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Procesando: {image_path.name}")
        logger.info(f"{'='*60}")
        
        # PASO 1: Segmentación
        logger.info("\n[1/3] Segmentación YCbCr...")
        try:
            seg_result = self.segmenter.segment(image_path, output_dir)
        except Exception as e:
            logger.error(f"Error en segmentación: {e}")
            return {'success': False, 'error': f'Segmentation failed: {e}'}
        
        if not seg_result['success']:
            logger.error("❌ Segmentación fallida")
            return {'success': False, 'error': 'Segmentation failed'}
        
        logger.info("✓ Segmentación exitosa")
        
        # PASO 2: Preparar imagen para clasificación
        logger.info("\n[2/3] Preparando imagen para clasificación...")
        roi = seg_result['lesion_roi']
        
        # Redimensionar a 224x224
        roi_resized = cv2.resize(roi, (224, 224))
        
        # Normalizar a [0, 1]
        roi_normalized = roi_resized.astype('float32') / 255.0
        
        # Agregar batch dimension
        roi_batch = np.expand_dims(roi_normalized, axis=0)
        
        logger.info("✓ Imagen preparada: 224x224")
        
        # PASO 3: Clasificación
        logger.info("\n[3/3] Clasificando...")
        try:
            predictions = self.model.predict(roi_batch, verbose=0)
        except Exception as e:
            logger.error(f"Error en clasificación: {e}")
            return {'success': False, 'error': f'Classification failed: {e}'}
        
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        class_name = self.class_names[class_idx]
        
        logger.info(f"✓ Clasificación: {class_name} ({confidence*100:.1f}%)")
        
        # Mostrar todas las predicciones
        logger.info("\nDetalle de predicciones:")
        for i, (name, pred) in enumerate(zip(self.class_names, predictions[0])):
            bar = '█' * int(pred * 30)
            logger.info(f"  {name:20} {pred*100:5.1f}% {bar}")
        
        # PASO 4: Crear visualización completa
        logger.info("\n[4/4] Creando visualización...")
        visualization = self._create_visualization(seg_result, class_name, confidence)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            vis_path = output_dir / f"{image_path.stem}_result.jpg"
            cv2.imwrite(str(vis_path), visualization)
            logger.info(f"✓ Visualización guardada: {vis_path}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ RESULTADO: {class_name} ({confidence*100:.1f}%)")
        logger.info(f"{'='*60}\n")
        
        return {
            'success': True,
            'image_path': image_path,
            'class': class_name,
            'confidence': float(confidence),
            'all_predictions': {name: float(pred) 
                              for name, pred in zip(self.class_names, predictions[0])},
            'segmentation_mask': seg_result['segmentation_mask'],
            'visualization': visualization,
            'roi': roi,
            'roi_resized': roi_resized
        }
    
    def _create_visualization(self, seg_result, class_name, confidence):
        """Crea imagen de visualización completa."""
        original = seg_result['original'].copy()
        h, w = original.shape[:2]
        
        # Canvas grande
        canvas = np.ones((h, w*2 + 20), dtype=np.uint8) * 255
        
        # Imagen original + contorno
        img_seg = original.copy()
        cv2.drawContours(img_seg, [seg_result['contour']], 0, (0, 255, 0), 3)
        canvas[:h, :w] = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
        
        # Máscara de segmentación
        canvas[:h, w+20:] = seg_result['segmentation_mask']
        
        # Texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Segmentacion YCbCr", (10, 30),
                   font, 1, (0, 0, 0), 2)
        cv2.putText(canvas, "Mascara", (w + 30, 30),
                   font, 1, (0, 0, 0), 2)
        cv2.putText(canvas, f"Resultado: {class_name}", (10, h-20),
                   font, 1.2, (0, 0, 0), 2)
        cv2.putText(canvas, f"Confianza: {confidence*100:.1f}%", (10, h+30),
                   font, 0.9, (0, 0, 0), 2)
        
        # Convertir a BGR para cv2.imwrite
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    def process_directory(self, image_dir, output_dir=None):
        """Procesa todas las imágenes en directorio."""
        image_dir = Path(image_dir)
        results = []
        
        for image_file in image_dir.glob('*.jpg'):
            try:
                result = self.process_image(image_file, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Error procesando {image_file}: {e}")
        
        return results

# Script de uso
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Ruta a imagen")
    parser.add_argument("--model", default="models/skin_lesion_classifier.h5")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    
    # Crear pipeline
    inference = SkinLesionInference(args.model, segmenter_debug=True)
    
    # Procesar imagen
    result = inference.process_image(args.image, args.output_dir)
    
    if result['success']:
        print(f"\n✅ {result['class']}: {result['confidence']*100:.1f}%")
