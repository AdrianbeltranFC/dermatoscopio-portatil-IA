"""
Módulo de Segmentación de Lesiones Cutáneas - YCbCr Adaptativo

Implementa segmentación robusta respetando la teoría de que la piel
se agrupa en clusters de Cb-Cr, pero calculando dichos clusters
dinámicamente para cada imagen (calibración automática).
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinLesionSegmenter:
    """
    Segmentador YCbCr Adaptativo.
    1. Calibra el color de piel sano muestreando las esquinas.
    2. Define rangos dinámicos de Cb-Cr para esa foto específica.
    3. Usa una máscara circular (ROI) solo para limpiar bordes del microscopio.
    """
    
    def __init__(self, debug=False):
        """
        Inicializa segmentador.
        Args:
            debug (bool): Si True, guarda imágenes intermedias (opcional)
        """
        self.debug = debug
        
        # Kernels para limpieza morfológica
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    
    def segment(self, image_path, output_dir=None):
        """
        Segmenta lesión en imagen usando YCbCr Adaptativo + Limpieza de Bordes.
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Imagen no encontrada: {image_path}")
            return {'success': False}
        
        # 1. Cargar imagen
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.error(f"Error cargando imagen: {image_path}")
            return {'success': False}
            
        h, w = img_bgr.shape[:2]
        
        # 2. Transformar a YCbCr
        img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        
        # 3. CALIBRACIÓN AUTOMÁTICA DE PIEL
        # En lugar de números fijos [77, 127], muestreamos la piel real de esta foto.
        # Tomamos 4 zonas seguras en las esquinas interiores.
        
        sample_size = 20
        samples_cb = []
        samples_cr = []
        
        # Coordenadas de muestreo (Esquinas interiores para evitar el borde negro)
        # Usamos 1/4 y 3/4 del ancho/alto
        offsets = [
            (h//4, w//4), (h//4, 3*w//4), 
            (3*h//4, w//4), (3*h//4, 3*w//4)
        ]
        
        for r, c in offsets:
            # Extraer parche de piel
            patch = img_ycbcr[r:r+sample_size, c:c+sample_size, :]
            samples_cb.extend(patch[:,:,1].flatten())
            samples_cr.extend(patch[:,:,2].flatten())
            
        # Calculamos la mediana (más robusta que el promedio)
        if len(samples_cb) > 0:
            median_cb = np.median(samples_cb)
            median_cr = np.median(samples_cr)
        else:
            # Fallback por seguridad si algo falla
            median_cb, median_cr = 100, 150
        
        logger.info(f"🎨 Calibración Piel -> Cb: {median_cb:.0f}, Cr: {median_cr:.0f}")
        
        # 4. Definir rangos elásticos
        # Un ancho de banda de +/- 25 suele cubrir la variación natural de la piel
        delta = 25 
        cb_lower = int(max(0, median_cb - delta))
        cb_upper = int(min(255, median_cb + delta))
        cr_lower = int(max(0, median_cr - delta))
        cr_upper = int(min(255, median_cr + delta))
        
        # 5. Segmentación YCbCr con rangos adaptados
        y, cr, cb = cv2.split(img_ycbcr)
        
        # Máscara de PIEL SANA
        mask_cbcr = cv2.inRange(cb, cb_lower, cb_upper) & \
                    cv2.inRange(cr, cr_lower, cr_upper)
        
        # Invertir: Lo que NO es piel sana, es posible lesión
        mask_lesion = cv2.bitwise_not(mask_cbcr)
        
        # 6. Limpieza de Bordes (ROI Circular)
        # Eliminamos las esquinas negras del microscopio que NO son lesión
        mask_roi = np.zeros((h, w), dtype=np.uint8)
        # Radio del 48% del lado menor (cubre casi todo el centro)
        cv2.circle(mask_roi, (w//2, h//2), int(min(h, w) * 0.48), 255, -1)
        
        # Intersección: Lesión detectada Y que esté dentro del círculo válido
        mask_lesion = cv2.bitwise_and(mask_lesion, mask_roi)
        
        # 7. Morfología (Limpiar ruido)
        mask_lesion = cv2.morphologyEx(mask_lesion, cv2.MORPH_OPEN, self.kernel_small)
        mask_lesion = cv2.morphologyEx(mask_lesion, cv2.MORPH_CLOSE, self.kernel_medium)
        
        # 8. Selección de Contorno (Con Fallback)
        contours, _ = cv2.findContours(mask_lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_contour = None
        use_fallback = False
        
        if not contours:
            use_fallback = True
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            # Si el área es menor al 1% de la imagen, es ruido -> Usar Fallback
            if cv2.contourArea(largest_contour) < (h * w * 0.01):
                use_fallback = True
        
        # Plan B: Recorte Central si la segmentación falla
        if use_fallback:
            logger.warning("⚠️ Segmentación por color no clara. Usando recorte central.")
            sz = int(min(h, w) * 0.5) # 50% del tamaño
            x, y = (w//2 - sz//2), (h//2 - sz//2)
            # Crear cuadrado artificial
            largest_contour = np.array([[[x,y]], [[x+sz,y]], [[x+sz,y+sz]], [[x,y+sz]]])
            
        # 9. Generar resultados finales
        mask_final = np.zeros_like(mask_lesion)
        cv2.drawContours(mask_final, [largest_contour], 0, 255, -1)
        
        # Extraer ROI con un poco de padding (margen)
        x, y, w_roi, h_roi = cv2.boundingRect(largest_contour)
        
        pad = 10 # 10 píxeles de aire alrededor para que la IA vea bordes
        x = max(0, x - pad)
        y = max(0, y - pad)
        w_roi = min(w - x, w_roi + 2*pad)
        h_roi = min(h - y, h_roi + 2*pad)
        
        lesion_roi = img_bgr[y:y+h_roi, x:x+w_roi].copy()
        
        logger.info(f"✓ ROI generado: {w_roi}x{h_roi}")
        
        # Guardar debug si se solicitó
        if self.debug and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / "debug_01_roi_mask.jpg"), mask_roi)
            cv2.imwrite(str(output_dir / "debug_02_lesion_raw.jpg"), mask_lesion)
            cv2.imwrite(str(output_dir / "debug_03_final.jpg"), mask_final)

        return {
            'success': True,
            'original': img_bgr,
            'segmentation_mask': mask_final,
            'lesion_roi': lesion_roi,
            'contour': largest_contour,
            'bbox': (x, y, w_roi, h_roi),
            'area': cv2.contourArea(largest_contour) if largest_contour is not None else 0
        }

    def visualize_segmentation(self, seg_result, save_path=None):
        """
        Visualiza segmentación sobre imagen original.
        """
        if not seg_result['success']:
            return None
        
        img = seg_result['original'].copy()
        contour = seg_result['contour']
        bbox = seg_result['bbox']
        
        # Dibujar contorno
        cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)
        
        # Dibujar bounding box
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(str(save_path), img)
        
        return img