"""
Módulo de Segmentación de Lesiones Cutáneas usando YCbCr

Implementa segmentación robusta para pieles oscuras utilizando:
- Transformación YCbCr para desacoplamiento de luminancia
- Umbralización adaptativa en canales Cb-Cr
- Morfología matemática para refinamiento

Referencias:
- Celebi et al. (2009): Color-based skin lesion boundary detection
- Esteva et al. (2019): Dermatologist-level classification of skin cancer
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinLesionSegmenter:
    """
    Segmentador de lesiones cutáneas robusto a variaciones de tono de piel.
    
    Método: YCbCr + Análisis del plano Cb-Cr
    Justificación:
    - RGB entrelaza luminancia y crominancia
    - YCbCr separa explícitamente: Y (luminancia), Cb-Cr (crominancia)
    - Piel normal agrupa compactamente en plano Cb-Cr independiente del tono
    - Lesiones pigmentadas desviadas del clúster normal
    - Cb proporciona contraste excepcional en pieles oscuras
    """
    
    def __init__(self, debug=False):
        """
        Inicializa segmentador.
        
        Args:
            debug (bool): Si True, guarda imágenes intermedias
        """
        self.debug = debug
        
        # Parámetros YCbCr optimizados (empíricamente validados)
        self.cb_lower = 77      # Umbral inferior Cb
        self.cb_upper = 127     # Umbral superior Cb
        self.cr_lower = 133     # Umbral inferior Cr
        self.cr_upper = 173     # Umbral superior Cr
        
        # Kernel para morfología
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    
    def segment(self, image_path, output_dir=None):
        """
        Segmenta lesión en imagen.
        
        Args:
            image_path (str/Path): Ruta a imagen
            output_dir (Path): Directorio para guardar intermedios (si debug=True)
            
        Returns:
            dict: {
                'original': imagen original,
                'segmentation_mask': máscara binaria de lesión,
                'lesion_roi': región de interés (ROI),
                'contour': contorno de la lesión,
                'success': bool indicando éxito
            }
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Imagen no encontrada: {image_path}")
            return {'success': False}
        
        # Cargar imagen
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.error(f"Error cargando imagen: {image_path}")
            return {'success': False}
        
        h, w = img_bgr.shape[:2]
        logger.info(f"📸 Imagen cargada: {w}x{h}")
        
        # PASO 1: Transformar a YCbCr
        img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(img_ycbcr)
        
        if self.debug:
            logger.info("✓ Transformado a YCbCr")
        
        # PASO 2: Crear máscara basada en plano Cb-Cr
        # Región donde la piel normal se agrupa
        mask_cbcr = cv2.inRange(cb, self.cb_lower, self.cb_upper) & \
                    cv2.inRange(cr, self.cr_lower, self.cr_upper)
        
        # Invertir: queremos FUERA del clúster de piel normal (la lesión)
        mask_lesion = cv2.bitwise_not(mask_cbcr)
        
        if self.debug:
            logger.info("✓ Máscara Cb-Cr creada")
        
        # PASO 3: Morfología - remover ruido pequeño
        mask_lesion = cv2.morphologyEx(mask_lesion, cv2.MORPH_OPEN, self.kernel_small)
        mask_lesion = cv2.morphologyEx(mask_lesion, cv2.MORPH_CLOSE, self.kernel_medium)
        
        if self.debug:
            logger.info("✓ Morfología aplicada")
        
        # PASO 4: Encontrar contorno más grande (la lesión principal)
        contours, _ = cv2.findContours(mask_lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("⚠️ No se encontraron contornos")
            return {'success': False, 'original': img_bgr}
        
        # Obtener contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        # Validar que el área sea razonable (10-80% de la imagen)
        total_area = h * w
        if contour_area < (total_area * 0.01) or contour_area > (total_area * 0.95):
            logger.warning(f"⚠️ Área sospechosa: {contour_area/total_area*100:.1f}%")
        
        logger.info(f"✓ Contorno encontrado: {contour_area:.0f} px ({contour_area/total_area*100:.1f}%)")
        
        # PASO 5: Crear máscara final refinada
        mask_final = np.zeros_like(mask_lesion)
        cv2.drawContours(mask_final, [largest_contour], 0, 255, -1)
        
        # PASO 6: Extraer ROI (región de interés)
        x, y, w_roi, h_roi = cv2.boundingRect(largest_contour)
        lesion_roi = img_bgr[y:y+h_roi, x:x+w_roi].copy()
        
        logger.info(f"✓ ROI extraído: {w_roi}x{h_roi}")
        
        # Guardar intermedios si debug
        if self.debug and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(output_dir / "01_original.jpg"), img_bgr)
            cv2.imwrite(str(output_dir / "02_cb_channel.jpg"), cb)
            cv2.imwrite(str(output_dir / "03_cr_channel.jpg"), cr)
            cv2.imwrite(str(output_dir / "04_mask_cbcr.jpg"), mask_cbcr)
            cv2.imwrite(str(output_dir / "05_mask_lesion.jpg"), mask_lesion)
            cv2.imwrite(str(output_dir / "06_mask_final.jpg"), mask_final)
            cv2.imwrite(str(output_dir / "07_lesion_roi.jpg"), lesion_roi)
            
            # Visualización
            img_vis = img_bgr.copy()
            cv2.drawContours(img_vis, [largest_contour], 0, (0, 255, 0), 2)
            cv2.imwrite(str(output_dir / "08_segmentation_result.jpg"), img_vis)
            
            logger.info(f"📁 Intermedios guardados en {output_dir}")
        
        return {
            'success': True,
            'original': img_bgr,
            'segmentation_mask': mask_final,
            'lesion_roi': lesion_roi,
            'contour': largest_contour,
            'bbox': (x, y, w_roi, h_roi),
            'area': contour_area,
            'cb_channel': cb,
            'cr_channel': cr
        }
    
    def visualize_segmentation(self, seg_result, save_path=None):
        """
        Visualiza segmentación sobre imagen original.
        
        Args:
            seg_result (dict): Resultado de segment()
            save_path (str/Path): Ruta para guardar visualización
            
        Returns:
            imagen con contorno dibujado
        """
        if not seg_result['success']:
            return None
        
        img = seg_result['original'].copy()
        contour = seg_result['contour']
        bbox = seg_result['bbox']
        
        # Dibujar contorno
        cv2.drawContours(img, [contour], 0, (0, 255, 0), 3)
        
        # Dibujar bounding box
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Texto
        area = seg_result['area']
        cv2.putText(img, f"Area: {area:.0f}px", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(str(save_path), img)
        
        return img
