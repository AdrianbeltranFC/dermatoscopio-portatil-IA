import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import threading
import datetime
import queue
import os
import numpy as np
import tensorflow as tf
from tkinter import filedialog
import time
import traceback

# ====================================================
# SISTEMA DE SEGMENTACIÓN MEJORADO (Canal Azul + Otsu)
# ====================================================
class SkinLesionSegmenter:
    def __init__(self, debug=False):  # CORREGIDO: __init__
        self.debug = debug
        
    def segment(self, image_path, output_dir=None):
        """
        Segmentación mejorada para dermatología:
        1. Usa el canal AZUL (donde la melanina resalta más).
        2. Aplica Thresholding de Otsu (automático).
        3. Si falla, hace un recorte central (Fallback).
        """
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'No se pudo cargar la imagen'}
            
            orig_h, orig_w = image.shape[:2]
            
            # --- ESTRATEGIA 1: Segmentación por Canal Azul (Mejor para lunares) ---
            # Los lunares absorben mucha luz azul, por lo que se ven muy oscuros en el canal B
            b_channel = image[:, :, 0] 
            
            # Suavizar para quitar pelos/ruido
            blur = cv2.GaussianBlur(b_channel, (5, 5), 0)
            
            # Invertir (queremos que el lunar sea blanco y la piel negra para la máscara)
            # Otsu encuentra el umbral óptimo automáticamente
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Limpieza morfológica (quitar ruido pequeño)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por tamaño y posición
            valid_contours = []
            img_center = np.array([orig_w // 2, orig_h // 2])
            min_area = (orig_w * orig_h) * 0.005 # 0.5% del total
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    # Preferir contornos cercanos al centro
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # Distancia al centro
                        dist = np.linalg.norm(img_center - np.array([cX, cY]))
                        # Si es lo suficientemente grande y no está en el borde extremo
                        valid_contours.append((cnt, area, dist))

            final_mask = np.zeros_like(b_channel)
            roi = None
            main_contour = None
            
            success = False
            
            if valid_contours:
                # Ordenar: Priorizar área grande y cercanía al centro
                main_contour = sorted(valid_contours, key=lambda x: x[1], reverse=True)[0][0]
                
                # 1. Crear máscara para VISUALIZACIÓN (lo que ves en la app)
                cv2.drawContours(final_mask, [main_contour], -1, 255, -1)
                segmented_image = cv2.bitwise_and(image, image, mask=final_mask)
                
                # 2. Extraer ROI para el MODELO (Cuadrado y con Piel)
                x, y, w, h = cv2.boundingRect(main_contour)
                
                # --- AQUÍ ESTÁ EL TRUCO: AUMENTAR EL MARGEN (PADDING) ---
                # Aumentamos un 40% el tamaño del recuadro para incluir piel sana
                pad_w = int(w * 0.40)
                pad_h = int(h * 0.40)
                
                # Calcular nuevas coordenadas asegurando no salirnos de la imagen
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(orig_w, x + w + pad_w)
                y2 = min(orig_h, y + h + pad_h)
                
                # IMPORTANTE: Recortamos de 'image' (original), NO de 'segmented_image' (negra)
                roi = image[y1:y2, x1:x2] 
                
                success = True
            
            # --- ESTRATEGIA 2: FALLBACK (Si falla la detección) ---
            if not success or roi is None or roi.size == 0:
                print("⚠ No se detectó contorno claro. Usando recorte central (Fallback).")
                # Tomar el 60% central de la imagen
                c_x, c_y = orig_w // 2, orig_h // 2
                w_crop, h_crop = int(orig_w * 0.6), int(orig_h * 0.6)
                x1 = max(0, c_x - w_crop // 2)
                y1 = max(0, c_y - h_crop // 2)
                roi = image[y1:y1+h_crop, x1:x1+w_crop]
                
                # Máscara dummy (todo visible en el centro)
                final_mask[y1:y1+h_crop, x1:x1+w_crop] = 255
                
                # Crear un contorno artificial cuadrado para visualización
                main_contour = np.array([[x1, y1], [x1+w_crop, y1], [x1+w_crop, y1+h_crop], [x1, y1+h_crop]])
                success = True # Consideramos éxito porque tenemos datos para el modelo

            # Generar imágenes de visualización
            segmented_image = cv2.bitwise_and(image, image, mask=final_mask)
            
            image_with_contours = image.copy()
            if main_contour is not None:
                cv2.drawContours(image_with_contours, [main_contour], -1, (0, 255, 0), 3)

            result = {
                'success': True,
                'original_image': image,
                'segmented_image': segmented_image,
                'contour_image': image_with_contours,
                'roi': roi, # ESTA ES LA IMAGEN CLAVE PARA EL MODELO
                'area': cv2.contourArea(main_contour) if main_contour is not None else 0,
                'contours_found': len(valid_contours)
            }

            if self.debug:
                print(f"✅ Segmentación completada. ROI shape: {roi.shape}")

            return result
            
        except Exception as e:
            print(f"❌ Error crítico en segmentación: {e}")
            import traceback
            traceback.print_exc()
            # Retorno de emergencia: devolver imagen original
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'No se pudo cargar la imagen'}
                
            return {
                'success': True, 
                'original_image': image,
                'segmented_image': image,
                'roi': image,
                'contour_image': image,
                'area': 0,
                'contours_found': 0
            }

# ====================================================
# MODELO DE MACHINE LEARNING CON PIPELINE COMPLETO
# ====================================================
class DermatologyModel:
    def __init__(self, debug=False):  # CORREGIDO: __init__
        self.model = None
        self.input_details = None
        self.output_details = None
        self.segmenter = SkinLesionSegmenter(debug=debug)
        self.debug = debug
        
        # Clases del modelo
        self.class_names = ['mel', 'nv', 'other']
        self.class_descriptions = {
            'mel': 'MALIGNO - Melanoma',
            'nv': 'BENIGNO - Nevus',
            'other': 'BENIGNO - Otras lesiones'
        }
        
        self.load_model()
    
    def load_model(self):
        """Cargar modelo TFLite"""
        try:
            # RUTAS DONDE BUSCAR TU MODELO - CORREGIDO: Coma faltante
            possible_paths = [
                "models/tflite/skin_lesion_float32_FINAL.tflite",  # COMA AGREGADA
                "models/skin_lesion_float32_FINAL.tflite",
                "models/filite/skin_lesion_float32_FINAL.tflite", 
                "skin_lesion_float32_FINAL.tflite",
                "model.tflite",
                "../models/skin_lesion_float32_FINAL.tflite",
                "./skin_lesion_float32_FINAL.tflite"
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"✅ Modelo encontrado en: {path}")
                    break
            
            if not model_path:
                current_dir = os.path.cwd()
                print(f"❌ MODELO NO ENCONTRADO")
                print(f"📂 Directorio actual: {current_dir}")
                print(f"🔍 Buscado en:")
                for path in possible_paths:
                    print(f"   - {os.path.abspath(path)}")
                raise FileNotFoundError(
                    "Modelo no encontrado. Por favor:\n"
                    "1. Descarga tu modelo .tflite\n"
                    "2. Colócalo en una de las rutas mostradas arriba\n"
                    "3. O especifica la ruta manualmente")
            
            print("🔄 Cargando modelo de IA...")
            
            # Cargar el modelo
            self.model = tf.lite.Interpreter(model_path=model_path)
            self.model.allocate_tensors()
            
            # Obtener detalles del modelo
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()
            
            print("✅ Modelo de IA cargado exitosamente!")
            print(f"📊 Input shape: {self.input_details[0]['shape']}")
            print(f"📊 Input dtype: {self.input_details[0]['dtype']}")
            print(f"📊 Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"❌ Error cargando el modelo: {e}")
            raise
    
    def preprocess_image(self, image):
        """Preprocesar imagen para el modelo"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Imagen vacía o None recibida")
            
            # Redimensionar a 224x224
            image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Convertir a RGB
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            
            # CORREGIDO: Sangría correcta
            if self.input_details[0]['dtype'] == np.float32:
                # EfficientNet usa 0-255, NO DIVIDIR
                image_normalized = image_rgb.astype(np.float32)
            elif self.input_details[0]['dtype'] == np.uint8:
                # Sin normalización para uint8
                image_normalized = image_rgb.astype(np.uint8)
            else:
                raise ValueError(f"Dtype no soportado: {self.input_details[0]['dtype']}")
            
            # Agregar dimensión del batch
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            if self.debug:
                print(f"🎯 Imagen procesada: {image_batch.shape}")
                print(f"   Min: {image_batch.min():.3f}, Max: {image_batch.max():.3f}")
            
            return image_batch
            
        except Exception as e:
            print(f"❌ Error procesando imagen: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def predict(self, image_path, output_dir=None):
        """Pipeline completo: Segmentación + Clasificación - VERSIÓN CORREGIDA"""
        try:
            print("\n" + "="*60)
            print("🔬 INICIANDO PIPELINE COMPLETO DE ANÁLISIS")
            print("="*60)

            # 1. SEGMENTACIÓN MEJORADA
            print("📐 Ejecutando segmentación mejorada (Canal Azul + Otsu)...")
            segmentation_result = self.segmenter.segment(image_path, output_dir)
            
            # Inicializar variables para todos los casos
            original_image = None
            image_to_classify = None
            contour_image = None
            area = 0
            has_segmentation = False
            contours_found = 0

            if not segmentation_result['success']:
                print(f"⚠  Segmentación falló: {segmentation_result.get('error', 'Desconocido')}")
                print("   Usando imagen original completa...")
                original_image = cv2.imread(image_path)

                if original_image is None:
                    raise ValueError(f"No se pudo cargar la imagen: {image_path}")
                image_to_classify = original_image
                contour_image = original_image.copy()
                has_segmentation = False
            else:
                original_image = segmentation_result['original_image']
                contour_image = segmentation_result.get('contour_image', original_image.copy())
                area = segmentation_result.get('area', 0)
                contours_found = segmentation_result.get('contours_found', 0)
                has_segmentation = True
                print(f"✅ Segmentación completada")
                
                # CORRECCIÓN VITAL: Priorizar el ROI (Recorte) para la predicción
                if 'roi' in segmentation_result and segmentation_result['roi'] is not None:
                    image_to_classify = segmentation_result['roi']
                    print("🚀 Usando ROI (Recorte del lunar) para el modelo")
                    print(f"   ROI shape: {image_to_classify.shape}")
                else:
                    image_to_classify = original_image
                    print("⚠ Usando imagen completa (sin recortar)")

            # Validar tamaño antes de pasar al modelo
            if image_to_classify is None or image_to_classify.size == 0:
                print("⚠ Imagen a clasificar es inválida, usando imagen original")
                image_to_classify = original_image
            
            # 2. PREPROCESAMIENTO
            print("🔄 Preprocesando imagen para clasificación...")
            input_data = self.preprocess_image(image_to_classify)
            if input_data is None:
                raise ValueError("Error en preprocesamiento de imagen")
            
            # 3. CLASIFICACIÓN
            print("🧠 Ejecutando clasificación con IA...")
            self.model.set_tensor(self.input_details[0]['index'], input_data)
            self.model.invoke()
            
            # Obtener resultados
            output_data = self.model.get_tensor(self.output_details[0]['index'])
            predictions = output_data[0]
            
            if predictions.max() > 1.0:
                print("   Aplicando softmax a las predicciones...")
                exp_preds = np.exp(predictions - np.max(predictions))
                predictions = exp_preds / exp_preds.sum()
            
            print(f"📊 Predicciones raw: {predictions}")

            # 4. POST-PROCESAMIENTO
            print("\n📈 PASO 4: Interpretación de resultados...")
            predicted_class_idx = np.argmax(predictions)
            confidence = predictions[predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            class_description = self.class_descriptions.get(predicted_class, predicted_class)
            
            result_text = f"{class_description} (Confianza: {confidence:.1%})"
            
            # 5. PREPARAR RESULTADOS COMPLETOS
            all_predictions = {}
            for i, class_name in enumerate(self.class_names):
                all_predictions[class_name] = float(predictions[i])
                print(f"   {class_name:8s}: {predictions[i]:.1%}")

            # Información de segmentación
            segmentation_info = {
                'area_pixels': area,
                'contours_found': contours_found,
                'has_contour': has_segmentation,
                'contour_image': contour_image if contour_image is not None else original_image.copy(),
                'original_image': original_image.copy() if original_image is not None else None,
                'segmented_image': segmentation_result.get('segmented_image', None) if has_segmentation else None,
                'success': has_segmentation
            }
            
            print("\n" + "="*60)
            print(f"✅ RESULTADO FINAL: {result_text}")
            print("="*60 + "\n")
            
            return result_text, confidence, all_predictions, segmentation_info
            
        except Exception as e:
            print(f"\n❌ ERROR EN PIPELINE: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise



# ====================================================
# VENTANA DE ANÁLISIS MEJORADA CON VISUALIZACIÓN
# ====================================================
class AnalysisWindow:
    def __init__(self, parent, image_path, ml_model):
        self.parent = parent
        self.image_path = image_path
        self.ml_model = ml_model
        self.analysis_in_progress = False
        self.segmentation_results = None
        self.original_photo = None
        self.contour_photo = None
        self.segmented_photo = None
        self.prob_photo = None
        self.create_window()
        
    def create_window(self):
        self.window = tb.Toplevel(self.parent)
        self.window.title("DermaScan Pro - Análisis con Segmentación Mejorada")
        self.window.geometry("1400x850")  # Aumenté un poco el tamaño
        self.window.configure(padx=20, pady=20)
        
        # Frame principal
        main_frame = tb.Frame(self.window)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Título
        title_label = tb.Label(
            main_frame,
            text="ANÁLISIS DERMATOLÓGICO CON SEGMENTACIÓN MEJORADA IA",
            font=('Arial', 18, 'bold'),
            bootstyle='primary'
        )
        title_label.pack(pady=(0, 20))
        
        # Contenido en dos columnas
        content_frame = tb.Frame(main_frame)
        content_frame.pack(fill=BOTH, expand=True)
        
        # COLUMNA IZQUIERDA - VISUALIZACIONES
        left_frame = tb.Frame(content_frame)
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 20))
        
        # Panel de visualizaciones
        viz_panel = tb.Labelframe(
            left_frame,
            text="VISUALIZACIONES DEL ANÁLISIS",
            bootstyle='info',
            padding=10
        )
        viz_panel.pack(fill=BOTH, expand=True)
        
        # Frame para las imágenes en grid 2x2
        image_grid = tb.Frame(viz_panel)
        image_grid.pack(fill=BOTH, expand=True)
        
        # Configurar grid para que se expanda
        image_grid.columnconfigure(0, weight=1, uniform="equal")
        image_grid.columnconfigure(1, weight=1, uniform="equal")
        image_grid.rowconfigure(0, weight=1, uniform="equal")
        image_grid.rowconfigure(1, weight=1, uniform="equal")
        
        # Imagen original
        original_frame = tb.Labelframe(image_grid, text="IMAGEN ORIGINAL", bootstyle='secondary')
        original_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        original_frame.grid_propagate(False)
        
        self.original_canvas = tk.Canvas(original_frame, bg='#2c3e50', highlightthickness=0)
        self.original_canvas.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Imagen con contorno
        contour_frame = tb.Labelframe(image_grid, text="SEGMENTACIÓN MEJORADA", bootstyle='success')
        contour_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        contour_frame.grid_propagate(False)
        
        self.contour_canvas = tk.Canvas(contour_frame, bg='#2c3e50', highlightthickness=0)
        self.contour_canvas.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Imagen segmentada
        segmented_frame = tb.Labelframe(image_grid, text="LESIÓN AISLADA", bootstyle='warning')
        segmented_frame.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        segmented_frame.grid_propagate(False)
        
        self.segmented_canvas = tk.Canvas(segmented_frame, bg='#2c3e50', highlightthickness=0)
        self.segmented_canvas.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Gráfico de probabilidades
        prob_frame = tb.Labelframe(image_grid, text="PROBABILIDADES", bootstyle='danger')
        prob_frame.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
        prob_frame.grid_propagate(False)
        
        self.prob_canvas = tk.Canvas(prob_frame, bg='white', highlightthickness=0)
        self.prob_canvas.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # COLUMNA DERECHA - CONTROLES Y RESULTADOS
        right_frame = tb.Frame(content_frame, width=400)  # Aumenté el ancho
        right_frame.pack(side=RIGHT, fill=Y)
        right_frame.pack_propagate(False)
        
        # Panel de controles
        control_panel = tb.Labelframe(
            right_frame,
            text="CONTROLES DE ANÁLISIS",
            bootstyle='primary',
            padding=15
        )
        control_panel.pack(fill=X, pady=(0, 20))
        
        # Estado del modelo
        model_status = "❌ NO DISPONIBLE" if self.ml_model and self.ml_model.model else "UWU"
        status_label = tb.Label(
            control_panel,
            text=f"Modelo IA: {model_status}",
            font=('Arial', 10, 'bold'),
            bootstyle='success' if self.ml_model and self.ml_model.model else 'danger'
        )
        status_label.pack(fill=X, pady=5)
        
        # Botones de análisis
        buttons = [
            ("🔬 ANALIZAR CON SEGMENTACIÓN MEJORADA", 'info', self.analyze_image),
            ("💾 GUARDAR REPORTE COMPLETO", 'success', self.save_report),
            ("📊 MÉTRICAS AVANZADAS", 'secondary', self.show_technical_details)
        ]
        
        for text, style, command in buttons:
            btn = tb.Button(
                control_panel,
                text=text,
                bootstyle=style,
                command=command,
                padding=(15, 10),
                width=20
            )
            btn.pack(fill=X, pady=5)
        
        # Panel de resultados
        result_panel = tb.Labelframe(
            right_frame,
            text="RESULTADO DEL DIAGNÓSTICO",
            bootstyle='success',
            padding=15
        )
        result_panel.pack(fill=BOTH, expand=True)
        
        # Etiqueta de diagnóstico
        self.diagnosis_label = tb.Label(
            result_panel,
            text="ESPERANDO ANÁLISIS",
            font=('Arial', 16, 'bold'),
            anchor=CENTER,
            justify=CENTER,
            wraplength=350
        )
        self.diagnosis_label.pack(fill=X, pady=10)
        
        # Etiqueta de confianza
        self.confidence_label = tb.Label(
            result_panel,
            text="Confianza: --%",
            font=('Arial', 12),
            anchor=CENTER
        )
        self.confidence_label.pack(fill=X, pady=5)
        
        # Información de segmentación mejorada
        self.segmentation_label = tb.Label(
            result_panel,
            text="Área: -- píxeles | Contornos: --",
            font=('Arial', 10),
            anchor=CENTER,
            bootstyle='secondary'
        )
        self.segmentation_label.pack(fill=X, pady=5)
        
        # Información adicional
        info_text = """
🎯 PIPELINE MEJORADO:
1. Canal Azul + Otsu
2. Validación multicriterio
3. ROI con margen 40%
4. Post-procesamiento avanzado

📊 MÉTRICAS:
• Área de lesión
• Número de contornos
• Confianza IA

⚠️ Sistema de apoyo al diagnóstico
   No sustituye evaluación médica.
        """
        
        info_label = tb.Label(
            result_panel,
            text=info_text,
            font=('Arial', 9),
            justify=LEFT,
            bootstyle='secondary',
            wraplength=350
        )
        info_label.pack(fill=X, pady=(20, 0))
        
        # Configurar tamaño mínimo para frames de imagen
        self.window.update()
        for frame in [original_frame, contour_frame, segmented_frame, prob_frame]:
            frame.config(width=300, height=250)
        
        # Cargar imagen original
        self.load_original_image()
    
    def resize_image_to_fit(self, image, target_width, target_height):
        """Redimensiona una imagen para que se ajuste al tamaño del recuadro manteniendo la proporción"""
        if not image:
            return None
            
        img_width, img_height = image.size
        img_ratio = img_width / img_height
        target_ratio = target_width / target_height
        
        if img_ratio > target_ratio:
            # Imagen más ancha que el recuadro
            new_width = target_width
            new_height = int(target_width / img_ratio)
        else:
            # Imagen más alta que el recuadro
            new_height = target_height
            new_width = int(target_height * img_ratio)
        
        # Redimensionar manteniendo calidad
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Crear una imagen nueva del tamaño exacto del recuadro con fondo negro
        final_image = Image.new('RGB', (target_width, target_height), (44, 62, 80))
        
        # Centrar la imagen redimensionada
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        final_image.paste(resized_image, (x_offset, y_offset))
        
        return final_image
    
    def load_original_image(self):
        """Cargar imagen original en el canvas - AJUSTADA AL TAMAÑO DEL RECUADRO"""
        if self.image_path and os.path.exists(self.image_path):
            try:
                image = Image.open(self.image_path)
                
                # Quitar filtro azul (reduce canal azul)
                #img_np = np.array(image)

                # Convertir a RGB si viene en otro formato
                #if img_np.shape[2] == 3:
                    # Baja la intensidad del canal azul
                    #img_np[:, :, 2] = 0

                #image = Image.fromarray(img_np.astype('uint8'))
                # Obtener tamaño del canvas
                canvas_width = self.original_canvas.winfo_width() or 300
                canvas_height = self.original_canvas.winfo_height() or 250
                
                # Redimensionar imagen para ajustarse al recuadro
                final_image = self.resize_image_to_fit(image, canvas_width, canvas_height)
                
                if final_image:
                    self.original_photo = ImageTk.PhotoImage(final_image)
                    
                    # Centrar la imagen en el canvas
                    self.original_canvas.delete("all")
                    self.original_canvas.create_image(
                        canvas_width // 2,
                        canvas_height // 2,
                        image=self.original_photo,
                        anchor='center'
                    )
                    
                print(f"✅ Imagen original cargada y ajustada al recuadro")
                
            except Exception as e:
                print(f"Error cargando imagen original: {e}")
                self.original_canvas.delete("all")
                self.original_canvas.create_text(
                    150, 125,
                    text="Error cargando\nimagen",
                    fill="red",
                    font=('Arial', 10),
                    justify=CENTER
                )
    
    def update_canvas_image(self, canvas, image_array, canvas_width, canvas_height, bg_color='#2c3e50'):
        """Actualizar un canvas con una imagen de numpy array, ajustando al tamaño del recuadro"""
        try:
            if image_array is None or image_array.size == 0:
                raise ValueError("Imagen vacía")
            
            # Convertir numpy array a PIL Image
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Convertir BGR a RGB si es necesario
                if image_array[0, 0, 0] > image_array[0, 0, 2]:  # Heurística simple para detectar BGR
                    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image_array
                pil_image = Image.fromarray(image_rgb)
            else:
                # Imagen en escala de grises
                pil_image = Image.fromarray(image_array)
            
            # Redimensionar para ajustarse al recuadro
            final_image = self.resize_image_to_fit(pil_image, canvas_width, canvas_height)
            
            if final_image:
                photo = ImageTk.PhotoImage(final_image)
                canvas.delete("all")
                canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    image=photo,
                    anchor='center'
                )
                # Mantener referencia para evitar garbage collection
                if canvas == self.contour_canvas:
                    self.contour_photo = photo
                elif canvas == self.segmented_canvas:
                    self.segmented_photo = photo
                elif canvas == self.prob_canvas:
                    self.prob_photo = photo
                    
        except Exception as e:
            print(f"Error actualizando canvas: {e}")
            canvas.delete("all")
            canvas.create_text(
                canvas_width // 2,
                canvas_height // 2,
                text="Error\nvisualización",
                fill="white" if bg_color == '#2c3e50' else "red",
                font=('Arial', 10),
                justify=CENTER
            )
    
    def update_visualizations(self, segmentation_info, all_predictions):
        """Actualizar todas las visualizaciones - AJUSTADAS AL TAMAÑO DEL RECUADRO"""
        try:
            # Obtener tamaños actuales de los canvas
            contour_width = self.contour_canvas.winfo_width() or 300
            contour_height = self.contour_canvas.winfo_height() or 250
            
            segmented_width = self.segmented_canvas.winfo_width() or 300
            segmented_height = self.segmented_canvas.winfo_height() or 250
            
            prob_width = self.prob_canvas.winfo_width() or 300
            prob_height = self.prob_canvas.winfo_height() or 250
            
            # 1. Imagen con contorno
            if segmentation_info and segmentation_info.get('contour_image') is not None:
                contour_img = segmentation_info['contour_image']
                self.update_canvas_image(
                    self.contour_canvas, 
                    contour_img, 
                    contour_width, 
                    contour_height
                )
            else:
                self.contour_canvas.delete("all")
                self.contour_canvas.create_text(
                    contour_width // 2,
                    contour_height // 2,
                    text="Sin datos de\nsegmentación",
                    fill="yellow",
                    font=('Arial', 10),
                    justify=CENTER
                )
            
            # 2. Imagen segmentada
            if segmentation_info and segmentation_info.get('segmented_image') is not None:
                seg_img = segmentation_info['segmented_image']
                self.update_canvas_image(
                    self.segmented_canvas, 
                    seg_img, 
                    segmented_width, 
                    segmented_height
                )
            else:
                # Cargar imagen original como fallback
                try:
                    original_img = cv2.imread(self.image_path)
                    if original_img is not None:
                        self.update_canvas_image(
                            self.segmented_canvas, 
                            original_img, 
                            segmented_width, 
                            segmented_height
                        )
                    else:
                        raise ValueError("No se pudo cargar imagen original")
                except:
                    self.segmented_canvas.delete("all")
                    self.segmented_canvas.create_text(
                        segmented_width // 2,
                        segmented_height // 2,
                        text="Imagen no\ndisponible",
                        fill="orange",
                        font=('Arial', 10),
                        justify=CENTER
                    )
            
            # 3. Gráfico de probabilidades
            if all_predictions and len(all_predictions) > 0:
                self.draw_probability_chart(all_predictions, prob_width, prob_height)
            else:
                self.prob_canvas.delete("all")
                self.prob_canvas.create_text(
                    prob_width // 2,
                    prob_height // 2,
                    text="Sin datos\nprobabilísticos",
                    fill="black",
                    font=('Arial', 10),
                    justify=CENTER
                )
                
        except Exception as e:
            print(f"Error actualizando visualizaciones: {e}")
    
    def draw_probability_chart(self, predictions, width, height):
        """Dibujar gráfico de barras de probabilidades ajustado al tamaño del recuadro"""
        try:
            canvas = self.prob_canvas
            canvas.delete("all")
            
            # Configuración del gráfico
            margin = 30
            bar_width = 40
            
            # Fondo blanco
            canvas.config(bg='white')
            canvas.create_rectangle(0, 0, width, height, fill='white', outline='')
            
            # Verificar predicciones
            if not predictions:
                raise ValueError("Sin predicciones")
                
            max_prob = max(predictions.values())
            if max_prob == 0:
                max_prob = 1.0
            
            # Colores y descripciones
            colors = {'mel': '#e74c3c', 'nv': '#2ecc71', 'other': '#f39c12'}
            descriptions = {'mel': 'Melanoma', 'nv': 'Nevus', 'other': 'Otras'}
            classes = list(predictions.keys())
            
            # Calcular espacio disponible
            available_width = width - 2 * margin
            bar_spacing = (available_width - (len(classes) * bar_width)) / (len(classes) + 1)
            
            # Dibujar ejes
            axis_y = height - margin
            canvas.create_line(margin, margin, margin, axis_y, width=2, fill='black')
            canvas.create_line(margin, axis_y, width - margin, axis_y, width=2, fill='black')
            
            # Dibujar barras
            max_bar_height = axis_y - margin - 20
            
            for i, class_name in enumerate(classes):
                prob = predictions.get(class_name, 0)
                bar_height = (prob / max_prob) * max_bar_height if max_prob > 0 else 0
                
                # Asegurar altura mínima para visibilidad
                if bar_height < 5 and prob > 0:
                    bar_height = 5
                
                # Calcular posición
                x0 = margin + bar_spacing + i * (bar_width + bar_spacing)
                y0 = axis_y
                y1 = y0 - bar_height
                
                # Dibujar barra
                canvas.create_rectangle(x0, y1, x0 + bar_width, y0, 
                                      fill=colors.get(class_name, '#3498db'), 
                                      outline='black')
                
                # Etiqueta de probabilidad
                canvas.create_text(x0 + bar_width/2, y1 - 10, text=f"{prob:.1%}", 
                                 font=('Arial', 8, 'bold'), fill='black')
                
                # Etiqueta de clase
                canvas.create_text(x0 + bar_width/2, axis_y + 15, 
                                 text=descriptions.get(class_name, class_name), 
                                 font=('Arial', 8), fill='black')
            
            # Título
            canvas.create_text(width/2, 15, text="PROBABILIDADES", 
                             font=('Arial', 10, 'bold'), fill='black')
                             
        except Exception as e:
            print(f"Error dibujando gráfico: {e}")
            canvas.delete("all")
            canvas.create_rectangle(0, 0, width, height, fill='white', outline='')
            canvas.create_text(width//2, height//2, text="Error en gráfico", 
                             fill="red", font=('Arial', 10))
    
    # Resto de los métodos permanecen igual (analyze_image, perform_analysis, etc.)
    def analyze_image(self):
        """Ejecutar análisis completo con segmentación mejorada"""
        if self.analysis_in_progress:
            return
            
        if not self.ml_model or not self.ml_model.model:
            self.show_message("Error: Modelo de IA no disponible", 'error')
            return
            
        self.analysis_in_progress = True
        self.diagnosis_label.config(text="EJECUTANDO ANÁLISIS MEJORADO...", bootstyle='info')
        self.confidence_label.config(text="Segmentación Mejorada + Clasificación IA")
        self.segmentation_label.config(text="Procesando...")
        
        # Limpiar visualizaciones
        for canvas in [self.contour_canvas, self.segmented_canvas, self.prob_canvas]:
            canvas.delete("all")
            canvas.create_text(
                canvas.winfo_width() // 2 if canvas.winfo_width() > 0 else 150,
                canvas.winfo_height() // 2 if canvas.winfo_height() > 0 else 125,
                text="Procesando...", 
                fill="white", 
                font=('Arial', 10), 
                justify=CENTER
            )
        
        # Ejecutar en hilo separado
        threading.Thread(target=self.perform_analysis, daemon=True).start()
    
    def perform_analysis(self):
        """Realizar análisis completo en hilo separado"""
        try:
            # Crear directorio para resultados
            os.makedirs("analysis_results", exist_ok=True)
            
            # Ejecutar pipeline completo
            result_text, confidence, all_predictions, segmentation_info = self.ml_model.predict(
                self.image_path, "analysis_results")
            
            # Actualizar interfaz en el hilo principal
            self.window.after(0, self.show_analysis_results, result_text, confidence, all_predictions, segmentation_info)
            
        except Exception as e:
            error_msg = f"Error en análisis: {str(e)}"
            print(f"Error en perform_analysis: {error_msg}")
            self.window.after(0, self.show_analysis_error, error_msg)
    
    def show_analysis_results(self, result_text, confidence, all_predictions, segmentation_info):
        """Mostrar resultados del análisis completo"""
        self.analysis_in_progress = False
        
        # Determinar estilo según resultado
        if "MALIGNO" in result_text:
            bootstyle = 'danger'
        else:
            bootstyle = 'success'
        
        self.diagnosis_label.config(text=result_text, bootstyle=bootstyle)
        self.confidence_label.config(text=f"Confianza: {confidence:.1%}")
        
        # Actualizar información de segmentación mejorada
        if segmentation_info:
            area = segmentation_info.get('area_pixels', 0)
            contours_found = segmentation_info.get('contours_found', 0)
            self.segmentation_label.config(text=f"Área: {area:.0f} px | Contornos: {contours_found}")
        
        # Actualizar visualizaciones
        self.update_visualizations(segmentation_info, all_predictions)
        
        # Guardar log mejorado
        self.save_analysis_log(result_text, confidence, segmentation_info)
        
        print("✅ Análisis completado y visualizaciones actualizadas")
    
    def show_analysis_error(self, error_msg):
        """Mostrar error en análisis"""
        self.analysis_in_progress = False
        self.diagnosis_label.config(text=error_msg, bootstyle='danger')
        self.confidence_label.config(text="Error en análisis")
        self.segmentation_label.config(text="---")
        print(f"❌ Error mostrado en interfaz: {error_msg}")
    
    def save_analysis_log(self, result, confidence, segmentation_info):
        """Guardar log del análisis mejorado"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"analysis_logs/analisis_{timestamp}.txt"
            
            os.makedirs("analysis_logs", exist_ok=True)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=== DERMASCAN PRO - ANÁLISIS MEJORADO ===\n")
                f.write(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Imagen: {self.image_path}\n")
                f.write(f"Resultado: {result}\n")
                f.write(f"Confianza: {confidence:.1%}\n")
                if segmentation_info:
                    f.write(f"Área de lesión: {segmentation_info.get('area_pixels', 0):.0f} píxeles\n")
                    f.write(f"Contornos detectados: {segmentation_info.get('contours_found', 0)}\n")
                f.write("Tecnología: Segmentación Canal Azul + EfficientNetB0\n")
                f.write("=" * 60 + "\n")
                
            print(f"📝 Log guardado: {log_file}")
                
        except Exception as e:
            print(f"Error guardando log: {e}")
    
    def save_report(self):
        """Guardar reporte completo mejorado"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"reportes/reporte_{timestamp}.txt"
            
            os.makedirs("reportes", exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=== REPORTE DERMATOLÓGICO MEJORADO - DERMASCAN PRO ===\n\n")
                f.write(f"Fecha del análisis: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Imagen analizada: {os.path.basename(self.image_path)}\n\n")
                f.write("RESULTADOS:\n")
                f.write(f"- Diagnóstico: {self.diagnosis_label.cget('text')}\n")
                f.write(f"- Nivel de confianza: {self.confidence_label.cget('text')}\n")
                f.write(f"- {self.segmentation_label.cget('text')}\n\n")
                f.write("METODOLOGÍA MEJORADA:\n")
                f.write("- Canal Azul + Thresholding Otsu\n")
                f.write("- ROI con margen del 40%\n")
                f.write("- Validación multicriterio de contornos\n")
                f.write("- Clasificación: Red Neuronal EfficientNetB0\n\n")
                f.write("NOTA: Este reporte es de apoyo al diagnóstico y debe ser\n")
                f.write("validado por un especialista médico certificado.\n")
            
            self.show_message(f"Reporte guardado: {report_file}", 'success')
            
        except Exception as e:
            self.show_message(f"Error guardando reporte: {str(e)}", 'error')
    
    def show_technical_details(self):
        """Mostrar detalles técnicos del pipeline mejorado"""
        details = """
🔬 PIPELINE MEJORADO - TECNOLOGÍA AVANZADA:

1. SEGMENTACIÓN POR CANAL AZUL
   • La melanina absorbe luz azul
   • Los lunares aparecen oscuros en canal B
   • Umbral automático con Otsu
   • Limpieza morfológica

2. VALIDACIÓN MULTICRITERIO
   • Tamaño mínimo: 0.5% del área
   • Proximidad al centro
   • Área y distancia ponderadas

3. ROI INTELIGENTE
   • Recorte del contorno principal
   • Margen del 40% para piel sana
   • Ajuste a bordes de imagen
   • Fallback a recorte central

4. CLASIFICACIÓN IA
   • EfficientNetB0 optimizado
   • 3 clases: Mel, NV, Other
   • Preprocesamiento específico

        """
        self.show_message(details, 'info')
    
    def show_message(self, message, style):
        """Mostrar mensaje emergente"""
        mb = tb.dialogs.Messagebox
        mb.show_info(message, title="DermaScan Pro", parent=self.window)


# ====================================================
# APLICACIÓN PRINCIPAL (MANTIENE LA CÁMARA ORIGINAL)
# ====================================================
class DermatoscopeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DermaScan Pro v3.2 - Dermatoscopio con Segmentación YCbCr Mejorada")
        self.root.geometry("1400x900")
        
        # Inicializar modelo de ML con pipeline completo YCbCr mejorado
        try:
            self.ml_model = DermatologyModel(debug=True)
        except Exception as e:
            self.show_error(f"Error crítico: {str(e)}")
            return
        
        # Configurar estilo
        self.style = tb.Style("darkly")
        self.style.configure('Title.TLabel', font=('Arial', 24, 'bold'))
        self.style.configure('Subtitle.TLabel', font=('Arial', 11))
        self.style.configure('PanelTitle.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Status.TLabel', font=('Arial', 10, 'bold'))
        
        # Variables de control de cámara
        self.frame_queue = queue.Queue(maxsize=1)
        self.cap = None
        self.camera_running = False
        self.current_image_path = None
        self.zoom_level = 1.0
        self.resolution = "1280x720"

        self.create_modern_layout()
        self.update_display()

    def create_modern_layout(self):
        # Frame principal
        main_container = tb.Frame(self.root, padding=15)
        main_container.pack(fill=BOTH, expand=True)

        # Header
        header_frame = tb.Frame(main_container)
        header_frame.pack(fill=X, pady=(0, 20))

        # Logo y título
        title_frame = tb.Frame(header_frame)
        title_frame.pack(side=LEFT, fill=Y)

        tb.Label(
            title_frame, 
            text="DERMASCAN PRO v3.2",
            style='Title.TLabel'
        ).pack(anchor=W)
        
        tb.Label(
            title_frame, 
            text="Sistema de Análisis Dermatológico con Segmentación YCbCr Mejorada",
            style='Subtitle.TLabel'
        ).pack(anchor=W, pady=(2, 0))

        # Estado del sistema
        status_frame = tb.Frame(header_frame)
        status_frame.pack(side=RIGHT, fill=Y)
        
        self.status_label = tb.Label(
            status_frame, 
            text="● SISTEMA LISTO",
            style='Status.TLabel',
            bootstyle='success'
        )
        self.status_label.pack(anchor=E)

        # Contenido principal
        content_frame = tb.Frame(main_container)
        content_frame.pack(fill=BOTH, expand=True)

        # COLUMNA IZQUIERDA - VISTA DE CÁMARA
        left_column = tb.Frame(content_frame)
        left_column.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 15))

        # Panel de cámara
        cam_panel = tb.Labelframe(
            left_column, 
            text="CÁMARA DERMATOSCÓPICA",
            bootstyle='primary',
            padding=12
        )
        cam_panel.pack(fill=BOTH, expand=True)

        # Controles de cámara
        control_frame = tb.Frame(cam_panel)
        control_frame.pack(fill=X, pady=(0, 10))

        # Botones de control
        cam_controls = [
            ("📷 Iniciar Cámara", 'outline-primary', self.start_camera),
            ("⏹️ Detener", 'outline-secondary', self.stop_camera),
        ]

        for text, style, command in cam_controls:
            btn = tb.Button(
                control_frame,
                text=text,
                bootstyle=style,
                command=command,
                padding=(12, 6)
            )
            btn.pack(side=LEFT, padx=(0, 8))

        # Controles de imagen
        image_controls = tb.Frame(control_frame)
        image_controls.pack(side=RIGHT)

        # Control de resolución
        resolution_frame = tb.Frame(image_controls)
        resolution_frame.pack(side=LEFT, padx=(0, 8))
        
        tb.Label(resolution_frame, text="Resolución:").pack(side=LEFT, padx=(0, 5))
        
        self.resolution_var = tb.StringVar(value="1280x720")
        resolution_menu = tb.Menubutton(
            resolution_frame,
            textvariable=self.resolution_var,
            bootstyle='outline-primary',
            padding=(8, 4)
        )
        resolution_menu.pack(side=LEFT)
        
        # Menu de resoluciones
        resolution_menu_menu = tk.Menu(resolution_menu, tearoff=0)
        resolutions = [
            ("640x480", "640x480"),
            ("800x600", "800x600"), 
            ("1024x768", "1024x768"),
            ("1280x720", "1280x720"),
            ("1920x1080", "1920x1080")
        ]
        
        for text, value in resolutions:
            resolution_menu_menu.add_radiobutton(
                label=text,
                variable=self.resolution_var,
                value=value,
                command=self.change_resolution
            )
        
        resolution_menu['menu'] = resolution_menu_menu

        # Control de zoom (CORREGIDO: bolita al extremo izquierdo con valor inicial 0.5)
        zoom_frame = tb.Frame(image_controls)
        zoom_frame.pack(side=LEFT)
        
        tb.Label(zoom_frame, text="Zoom:").pack(side=LEFT, padx=(0, 5))
        
        self.zoom_scale = tb.Scale(
            zoom_frame,
            from_=0.5,
            to=3.0,
            value=0.5,  # VALOR INICIAL A LA IZQUIERDA
            command=self.update_zoom,
            length=100,
            bootstyle='primary',
            orient='horizontal'
        )
        self.zoom_scale.pack(side=LEFT)
        self.zoom_level = 0.5  # Establecer el nivel de zoom inicial

        # Vista de cámara
        cam_container = tb.Frame(cam_panel, bootstyle='dark', relief='sunken', height=500)
        cam_container.pack(fill=BOTH, expand=True, pady=5)
        cam_container.pack_propagate(False)

        self.cam_label = tb.Label(
            cam_container, 
            text="CÁMARA NO INICIADA\n\nHaga clic en 'INICIAR CÁMARA' para comenzar",
            anchor=CENTER,
            bootstyle='secondary'
        )
        self.cam_label.pack(fill=BOTH, expand=True, padx=2, pady=2)

        # Información de la cámara
        info_frame = tb.Frame(cam_panel)
        info_frame.pack(fill=X, pady=(10, 0))

        self.cam_info = tb.Label(
            info_frame, 
            text="Resolución: --- | FPS: ---",
            font=('Arial', 9)
        )
        self.cam_info.pack(side=LEFT)

        self.capture_info = tb.Label(
            info_frame, 
            text="Última captura: ---",
            font=('Arial', 9)
        )
        self.capture_info.pack(side=RIGHT)

        # COLUMNA DERECHA - PANEL DE ACCIONES
        right_column = tb.Frame(content_frame, width=350)
        right_column.pack(side=RIGHT, fill=Y)
        right_column.pack_propagate(False)

        # ====================================================
        # PANEL DE ACCIONES PRINCIPALES
        # ====================================================
        action_panel = tb.Labelframe(
            right_column, 
            text="ACCIONES",
            bootstyle='info',
            padding=15
        )
        action_panel.pack(fill=X, pady=(0, 15))

        # Botones de acción principales
        actions = [
            ("📸 CAPTURAR IMAGEN", 'success', self.capture),
            ("📁 CARGAR ARCHIVO", 'secondary', self.open_file),
            ("🔬 ABRIR ANÁLISIS MEJORADO", 'primary', self.open_analysis),
        ]

        for text, style, command in actions:
            btn = tb.Button(
                action_panel, 
                text=text,
                bootstyle=style,
                command=command,
                padding=(15, 12)
            )
            btn.pack(fill=X, pady=8)

        # ====================================================
        # PANEL DE INFORMACIÓN DEL SISTEMA
        # ====================================================
        info_panel = tb.Labelframe(
            right_column, 
            text="INFORMACIÓN DEL SISTEMA MEJORADO",
            bootstyle='secondary',
            padding=15
        )
        info_panel.pack(fill=BOTH, expand=True)

        # Estado de conexión
        conn_frame = tb.Frame(info_panel)
        conn_frame.pack(fill=X, pady=(0, 15))

        tb.Label(conn_frame, text="Conexión Raspberry Pi:", font=('Arial', 10, 'bold')).pack(anchor=W)
        
        self.conn_status = tb.Label(
            conn_frame, 
            text="● No  CONECTADO",
            bootstyle='success',
            font=('Arial', 9, 'bold')
        )
        self.conn_status.pack(anchor=W, pady=(2, 0))

        # Información de tecnología mejorada
        tech_info = """
TECNOLOGÍA MEJORADA:
• Segmentación YCbCr Adaptativa
• Muestreo inteligente de piel
• Validación multicriterio
• Fallback inteligente 3 niveles

HARDWARE:
• Raspberry Pi 5
• Cámara HQ 12MP
• Iluminación LED regulable
• Lente dermatoscópico 10x

SOFTWARE:
• OpenCV 4.8.0
• TensorFlow 2.13
• Python 3.11
• Tkinter/ttkbootstrap
"""

        tb.Label(
            info_panel,
            text=tech_info,
            font=('Arial', 9),
            justify=LEFT,
            bootstyle='secondary'
        ).pack(anchor=W)

        # ====================================================
        # BARRA DE ESTADO
        # ====================================================
        statusbar = tb.Frame(main_container)
        statusbar.pack(fill=X, pady=(15, 0))

        # Información de versión
        version_label = tb.Label(
            statusbar, 
            text="DermaScan Pro v3.2 - Sistema YCbCr Mejorado con Validación Multicriterio",
            font=('Arial', 9)
        )
        version_label.pack(side=LEFT)

        # Reloj
        self.clock_label = tb.Label(
            statusbar, 
            text="",
            font=('Arial', 9)
        )
        self.clock_label.pack(side=RIGHT)

        self.update_clock()

    def change_resolution(self):
        """Cambiar resolución de la cámara"""
        if self.camera_running:
            self.stop_camera()
            self.start_camera()

    def update_clock(self):
        """Actualizar reloj"""
        now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.clock_label.config(text=now)
        self.root.after(1000, self.update_clock)

    def update_display(self):
        """Actualizar pantalla desde el hilo principal"""
        try:
            if not self.frame_queue.empty():
                imgtk = self.frame_queue.get_nowait()
                self.cam_label.imgtk = imgtk
                self.cam_label.config(image=imgtk, text="")
        except queue.Empty:
            pass
        finally:
            self.root.after(30, self.update_display)

    def start_camera(self):
        if self.camera_running:
            return
        
        self.camera_running = True
        self.status_label.config(text="● CÁMARA ACTIVA", bootstyle='success')
        self.cam_label.config(text="INICIANDO CÁMARA...", bootstyle='info')

        # Obtener resolución seleccionada
        resolution = self.resolution_var.get()
        width, height = map(int, resolution.split('x'))
        
        try:
            # FORZAR BACKEND DSHOW (Windows)
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                # Intentar sin backend específico
                self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                self.show_error("Error: No se pudo conectar a la cámara")
                self.camera_running = False
                return
            
            # Configuración básica sin forzar demasiado
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Reducir FPS para mayor estabilidad
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            print("✅ Cámara configurada correctamente")
                
        except Exception as e:
            self.show_error(f"Error de conexión: {str(e)}")
            self.camera_running = False
            return

        threading.Thread(target=self.capture_frames, daemon=True).start()

    def capture_frames(self):
        """Hilo secundario: captura frames"""
        frame_count = 0
        start_time = datetime.datetime.now()
        error_count = 0
        max_errors = 5
        
        while self.camera_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    error_count += 1
                    if error_count >= max_errors:
                        print("❌ Demasiados errores de cámara, reiniciando...")
                        self.root.after(0, lambda: self.show_error("Error de cámara. Reinicie la cámara."))
                        break
                    continue
                
                # Resetear contador de errores si hay frame válido
                error_count = 0
                
                # Aplicar zoom normal (con bolita al extremo izquierdo = 0.5 = zoom out)
                if self.zoom_level != 1.0:
                    frame = self.apply_zoom_normal(frame, self.zoom_level)

                # Actualizar información de FPS
                frame_count += 1
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                if elapsed >= 1:
                    fps = frame_count / elapsed
                    resolution = self.resolution_var.get()
                    self.root.after(0, lambda: self.cam_info.config(
                        text=f"Resolución: {resolution} | FPS: {fps:.1f} | Zoom: {self.zoom_level:.1f}x"))
                    frame_count = 0
                    start_time = datetime.datetime.now()

                # Procesar frame para visualización
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Redimensionar manteniendo aspecto
                h, w = frame.shape[:2]
                target_h = 500
                target_w = int(w * target_h / h)
                
                img = Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Poner el frame en la cola
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(imgtk)
                except queue.Full:
                    pass
                    
            except Exception as e:
                print(f"Error en captura: {e}")
                error_count += 1
                if error_count >= max_errors:
                    break
                continue

    def apply_zoom_normal(self, frame, zoom_level):
        """Aplicar zoom NORMAL con bolita al extremo izquierdo = zoom out"""
        if zoom_level == 1.0:
            return frame
            
        h, w = frame.shape[:2]
        
        # Con bolita a la izquierda (0.5) = zoom out
        # Con bolita a la derecha (3.0) = zoom in
        if zoom_level < 1.0:
            # Zoom out: agregar padding
            scale_factor = 1.0 / zoom_level
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # Crear fondo negro
            zoomed = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            
            # Calcular posición para centrar la imagen original
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            
            # Colocar la imagen original en el centro
            zoomed[start_y:start_y+h, start_x:start_x+w] = frame
            
            # Redimensionar al tamaño original
            zoomed = cv2.resize(zoomed, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Zoom in: recortar y redimensionar
            new_h, new_w = int(h / zoom_level), int(w / zoom_level)
            
            # Calcular región de recorte
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            
            # Recortar y redimensionar
            cropped = frame[start_y:start_y+new_h, start_x:start_x+new_w]
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return zoomed

    def update_zoom(self, value):
        """Actualizar nivel de zoom - bolita al extremo izquierdo"""
        try:
            self.zoom_level = float(value)
        except ValueError:
            self.zoom_level = 0.5

    def capture(self):
        """Capturar imagen"""
        if not self.camera_running:
            self.show_error("ERROR: ENCIENDA LA CÁMARA PRIMERO")
            return

        ret, frame = self.cap.read()
        if ret:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captura_{timestamp}.jpg"
            
            cv2.imwrite(filename, frame)
            self.current_image_path = filename
            
            capture_time = datetime.datetime.now().strftime("%H:%M:%S")
            self.capture_info.config(text=f"Última captura: {capture_time}")
            self.show_message(f"Imagen capturada: {filename}", 'success')

    def open_file(self):
        """Cargar archivo y abrir automáticamente la ventana de análisis"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.show_message(f"Imagen cargada: {os.path.basename(file_path)}", 'success')
            # Abrir automáticamente la ventana de análisis
            self.open_analysis()

    def open_analysis(self):
        """Abrir ventana de análisis YCbCr mejorado (solo abre ventana, NO ejecuta análisis)"""
        if not self.current_image_path:
            self.show_error("Capture o cargue una imagen primero")
            return
            
        AnalysisWindow(self.root, self.current_image_path, self.ml_model)

    def show_message(self, message, style='info'):
        """Mostrar mensaje"""
        mb = tb.dialogs.Messagebox
        mb.show_info(message, title="DermaScan Pro", parent=self.root)
#-------------------------
    def show_error(self, message):
        """Mostrar error"""
        mb = tb.dialogs.Messagebox
        mb.show_error(message, title="Error", parent=self.root)

    def stop_camera(self):
        """Detener cámara"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        
        self.cam_label.config(
            image="", 
            text="CÁMARA NO INICIADA\n\nHaga clic en 'INICIAR CÁMARA' para comenzar",
            bootstyle='secondary'
        )
        self.status_label.config(text="● SISTEMA EN ESPERA", bootstyle='secondary')
        self.cam_info.config(text="Resolución: --- | FPS: ---")


# ====================================================
# EJECUCIÓN
# ====================================================
if __name__ == "__main__":
    root = tb.Window(themename="darkly")  
    app = DermatoscopeApp(root)
    root.mainloop()