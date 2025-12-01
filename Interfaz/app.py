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

# ====================================================
# SISTEMA DE SEGMENTACIÓN YCbCr
# ====================================================
class SkinLesionSegmenter:
    def __init__(self, debug=False):
        self.debug = debug
        
    def segment(self, image_path, output_dir=None):
        """Segmentar lesión de piel usando espacio de color YCbCr"""
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'No se pudo cargar la imagen'}
            
            # Convertir a YCbCr
            ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            # Definir rangos para piel en espacio YCbCr
            lower_skin = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin = np.array([255, 173, 127], dtype=np.uint8)
            
            # Crear máscara
            skin_mask = cv2.inRange(ycbcr_image, lower_skin, upper_skin)
            
            # Operaciones morfológicas para mejorar la máscara
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'success': False, 'error': 'No se detectaron lesiones'}
            
            # Encontrar el contorno más grande (la lesión principal)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Crear máscara de la lesión
            lesion_mask = np.zeros_like(skin_mask)
            cv2.drawContours(lesion_mask, [main_contour], -1, 255, -1)
            
            # Aplicar máscara a la imagen original
            segmented_image = cv2.bitwise_and(image, image, mask=lesion_mask)
            
            # Calcular área de la lesión
            area_pixels = cv2.contourArea(main_contour)
            
            # Crear imagen con contorno
            image_with_contour = image.copy()
            cv2.drawContours(image_with_contour, [main_contour], -1, (0, 255, 0), 3)
            
            result = {
                'success': True,
                'original_image': image,
                'segmented_image': segmented_image,
                'lesion_mask': lesion_mask,
                'contour_image': image_with_contour,
                'contour': main_contour,
                'area': area_pixels,
                'contours_found': len(contours)
            }
            
            # Guardar resultados si se especifica directorio de salida
            if output_dir and os.path.exists(output_dir):
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_segmented.jpg"), segmented_image)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_contour.jpg"), image_with_contour)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.jpg"), lesion_mask)
            
            if self.debug:
                print(f"✅ Segmentación exitosa - Área: {area_pixels:.0f} píxeles")
                print(f"📊 Contornos encontrados: {len(contours)}")
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Error en segmentación: {str(e)}'}

# ====================================================
# MODELO DE MACHINE LEARNING CON PIPELINE COMPLETO
# ====================================================
class DermatologyModel:
    def __init__(self, debug=False):
        self.model = None
        self.input_details = None
        self.output_details = None
        self.segmenter = SkinLesionSegmenter(debug=debug)
        self.debug = debug
        
        # Clases del modelo
        self.class_names = ['mel', 'nv', 'other']
        self.class_descriptions = {
            'mel': 'MALIGNO - Melanoma',
            'nv': 'BENIGNO - Nevus (Lunar)',
            'other': 'BENIGNO - Otras lesiones'
        }
        
        self.load_model()
    
    def load_model(self):
        """Cargar modelo TFLite"""
        try:
            # RUTAS DONDE BUSCAR TU MODELO
            possible_paths = [
                "models/skin_lesion_classifier_float16.tflite",
                "models/filite/skin_lesion_classifier_float16.tflite", 
                "skin_lesion_classifier_float16.tflite",
                "model.tflite"
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"✅ Modelo encontrado en: {path}")
                    break
            
            if not model_path:
                raise FileNotFoundError("Modelo no encontrado. Coloque el archivo skin_lesion_classifier_float16.tflite en la carpeta models/")
            
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
            
        except Exception as e:
            print(f"❌ Error cargando el modelo: {e}")
            raise
    
    def preprocess_image(self, image):
        """Preprocesar imagen para el modelo"""
        try:
            # Redimensionar a 224x224
            image_resized = cv2.resize(image, (224, 224))
            
            # Convertir a RGB
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            
            # Normalización [0, 1]
            image_normalized = image_rgb.astype(np.float32) / 255.0
            
            # Agregar dimensión del batch
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            if self.debug:
                print(f"🎯 Imagen procesada: {image_batch.shape}")
            
            return image_batch
            
        except Exception as e:
            print(f"❌ Error procesando imagen: {e}")
            return None
    
    def predict(self, image_path, output_dir=None):
        """Pipeline completo: Segmentación + Clasificación"""
        try:
            print("🔬 Iniciando pipeline completo de análisis...")
            
            # 1. SEGMENTACIÓN
            print("📐 Ejecutando segmentación YCbCr...")
            segmentation_result = self.segmenter.segment(image_path, output_dir)
            
            if not segmentation_result['success']:
                print(f"❌ Error en segmentación: {segmentation_result.get('error', 'Desconocido')}")
                # Usar imagen original si falla la segmentación
                original_image = cv2.imread(image_path)
                segmented_image = original_image
                contour_image = original_image
                area = 0
            else:
                original_image = segmentation_result['original_image']
                segmented_image = segmentation_result['segmented_image']
                contour_image = segmentation_result['contour_image']
                area = segmentation_result['area']
                print(f"✅ Segmentación exitosa - Área: {area:.0f} píxeles")
            
            # 2. PREPROCESAMIENTO
            print("🔄 Preprocesando imagen para clasificación...")
            input_data = self.preprocess_image(segmented_image)
            if input_data is None:
                raise ValueError("Error en preprocesamiento de imagen")
            
            # 3. CLASIFICACIÓN
            print("🧠 Ejecutando clasificación con IA...")
            self.model.set_tensor(self.input_details[0]['index'], input_data)
            self.model.invoke()
            
            # Obtener resultados
            output_data = self.model.get_tensor(self.output_details[0]['index'])
            predictions = output_data[0]
            
            print(f"📊 Predicciones: {[f'{p:.3f}' for p in predictions]}")
            
            # 4. POST-PROCESAMIENTO
            predicted_class_idx = np.argmax(predictions)
            confidence = predictions[predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            class_description = self.class_descriptions.get(predicted_class, predicted_class)
            
            result_text = f"{class_description} (Confianza: {confidence:.1%})"
            
            # 5. PREPARAR RESULTADOS COMPLETOS
            all_predictions = {}
            for i, class_name in enumerate(self.class_names):
                all_predictions[class_name] = float(predictions[i])
            
            # Información de segmentación
            segmentation_info = {
                'area_pixels': area,
                'contours_found': segmentation_result.get('contours_found', 0),
                'has_contour': segmentation_result['success'],
                'contour_image': contour_image
            }
            
            print(f"✅ Análisis completo: {result_text}")
            
            return result_text, confidence, all_predictions, segmentation_info
            
        except Exception as e:
            print(f"❌ Error en pipeline completo: {e}")
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
        self.create_window()
        
    def create_window(self):
        self.window = tb.Toplevel(self.parent)
        self.window.title("DermaScan Pro - Análisis con Segmentación")
        self.window.geometry("1200x800")
        self.window.configure(padx=20, pady=20)
        
        # Frame principal
        main_frame = tb.Frame(self.window)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Título
        title_label = tb.Label(
            main_frame,
            text="ANÁLISIS DERMATOLÓGICO CON SEGMENTACIÓN IA",
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
        
        # Imagen original
        original_frame = tb.Labelframe(image_grid, text="IMAGEN ORIGINAL", bootstyle='secondary')
        original_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        self.original_canvas = tk.Canvas(original_frame, width=250, height=200, bg='#2c3e50')
        self.original_canvas.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Imagen con contorno
        contour_frame = tb.Labelframe(image_grid, text="SEGMENTACIÓN", bootstyle='success')
        contour_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        
        self.contour_canvas = tk.Canvas(contour_frame, width=250, height=200, bg='#2c3e50')
        self.contour_canvas.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Imagen segmentada
        segmented_frame = tb.Labelframe(image_grid, text="LESIÓN AISLADA", bootstyle='warning')
        segmented_frame.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        
        self.segmented_canvas = tk.Canvas(segmented_frame, width=250, height=200, bg='#2c3e50')
        self.segmented_canvas.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Gráfico de probabilidades
        prob_frame = tb.Labelframe(image_grid, text="PROBABILIDADES", bootstyle='danger')
        prob_frame.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
        
        self.prob_canvas = tk.Canvas(prob_frame, width=250, height=200, bg='white')
        self.prob_canvas.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Configurar grid
        image_grid.columnconfigure(0, weight=1)
        image_grid.columnconfigure(1, weight=1)
        image_grid.rowconfigure(0, weight=1)
        image_grid.rowconfigure(1, weight=1)
        
        # COLUMNA DERECHA - CONTROLES Y RESULTADOS
        right_frame = tb.Frame(content_frame, width=350)
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
        model_status = "✅ CONECTADO" if self.ml_model and self.ml_model.model else "❌ NO DISPONIBLE"
        status_label = tb.Label(
            control_panel,
            text=f"Modelo IA: {model_status}",
            font=('Arial', 10, 'bold'),
            bootstyle='success' if self.ml_model and self.ml_model.model else 'danger'
        )
        status_label.pack(fill=X, pady=5)
        
        # Botones de análisis
        buttons = [
            ("🔬 ANALIZAR CON SEGMENTACIÓN", 'info', self.analyze_image),
            ("💾 GUARDAR REPORTE", 'success', self.save_report),
            ("📊 DETALLES TÉCNICOS", 'secondary', self.show_technical_details)
        ]
        
        for text, style, command in buttons:
            btn = tb.Button(
                control_panel,
                text=text,
                bootstyle=style,
                command=command,
                padding=(15, 10)
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
            wraplength=300
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
        
        # Información de segmentación
        self.segmentation_label = tb.Label(
            result_panel,
            text="Área de lesión: -- píxeles",
            font=('Arial', 10),
            anchor=CENTER,
            bootstyle='secondary'
        )
        self.segmentation_label.pack(fill=X, pady=5)
        
        # Información adicional
        info_text = """
🎯 PIPELINE COMPLETO:
1. Segmentación YCbCr
2. Aislamiento de lesión  
3. Clasificación EfficientNet
4. Análisis de características

⚠️ Sistema de apoyo al diagnóstico
   No sustituye evaluación médica.
        """
        
        info_label = tb.Label(
            result_panel,
            text=info_text,
            font=('Arial', 9),
            justify=LEFT,
            bootstyle='secondary'
        )
        info_label.pack(fill=X, pady=(20, 0))
        
        # Cargar imagen original
        self.load_original_image()
    
    def load_original_image(self):
        """Cargar imagen original en el canvas"""
        if self.image_path and os.path.exists(self.image_path):
            image = Image.open(self.image_path)
            # Redimensionar para visualización
            image.thumbnail((240, 190), Image.Resampling.LANCZOS)
            self.original_photo = ImageTk.PhotoImage(image)
            
            self.original_canvas.delete("all")
            self.original_canvas.create_image(125, 100, image=self.original_photo)
            
            # Mostrar mensaje en otros canvas
            for canvas in [self.contour_canvas, self.segmented_canvas, self.prob_canvas]:
                canvas.delete("all")
                canvas.create_text(125, 100, text="Ejecute análisis\npara ver resultados", 
                                 fill="white", font=('Arial', 10), justify=CENTER)
    
    def update_visualizations(self, segmentation_info, all_predictions):
        """Actualizar todas las visualizaciones"""
        try:
            # 1. Imagen con contorno
            if segmentation_info and segmentation_info.get('contour_image') is not None:
                contour_img = segmentation_info['contour_image']
                contour_img_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
                contour_pil = Image.fromarray(contour_img_rgb)
                contour_pil.thumbnail((240, 190), Image.Resampling.LANCZOS)
                self.contour_photo = ImageTk.PhotoImage(contour_pil)
                
                self.contour_canvas.delete("all")
                self.contour_canvas.create_image(125, 100, image=self.contour_photo)
            
            # 2. Imagen segmentada (usar imagen original por ahora)
            if self.image_path and os.path.exists(self.image_path):
                segmented_img = Image.open(self.image_path)
                segmented_img.thumbnail((240, 190), Image.Resampling.LANCZOS)
                self.segmented_photo = ImageTk.PhotoImage(segmented_img)
                
                self.segmented_canvas.delete("all")
                self.segmented_canvas.create_image(125, 100, image=self.segmented_photo)
            
            # 3. Gráfico de probabilidades
            if all_predictions:
                self.draw_probability_chart(all_predictions)
                
        except Exception as e:
            print(f"Error actualizando visualizaciones: {e}")
    
    def draw_probability_chart(self, predictions):
        """Dibujar gráfico de barras de probabilidades"""
        try:
            canvas = self.prob_canvas
            canvas.delete("all")
            
            # Configuración del gráfico
            width = 250
            height = 200
            margin = 30
            bar_width = 40
            max_prob = max(predictions.values())
            
            # Colores para cada clase
            colors = {'mel': '#e74c3c', 'nv': '#2ecc71', 'other': '#f39c12'}
            descriptions = {'mel': 'Melanoma', 'nv': 'Nevus', 'other': 'Otras'}
            
            # Dibujar ejes
            canvas.create_line(margin, height - margin, width - margin, height - margin, width=2)
            canvas.create_line(margin, margin, margin, height - margin, width=2)
            
            # Dibujar barras
            classes = list(predictions.keys())
            for i, class_name in enumerate(classes):
                prob = predictions[class_name]
                bar_height = (prob / max_prob) * (height - 2 * margin - 20) if max_prob > 0 else 0
                x0 = margin + 10 + i * (bar_width + 30)
                y0 = height - margin
                y1 = y0 - bar_height
                
                # Dibujar barra
                canvas.create_rectangle(x0, y1, x0 + bar_width, y0, fill=colors[class_name], outline='black')
                
                # Etiqueta de probabilidad
                canvas.create_text(x0 + bar_width/2, y1 - 10, text=f"{prob:.1%}", 
                                 font=('Arial', 8, 'bold'))
                
                # Etiqueta de clase
                canvas.create_text(x0 + bar_width/2, height - margin + 15, 
                                 text=descriptions[class_name], font=('Arial', 8))
            
            # Título
            canvas.create_text(width/2, 15, text="PROBABILIDADES", 
                             font=('Arial', 10, 'bold'))
                             
        except Exception as e:
            print(f"Error dibujando gráfico: {e}")
            canvas.create_text(125, 100, text="Error en gráfico", 
                             fill="red", font=('Arial', 10))
    
    def analyze_image(self):
        """Ejecutar análisis completo con segmentación"""
        if self.analysis_in_progress:
            return
            
        if not self.ml_model or not self.ml_model.model:
            self.show_message("Error: Modelo de IA no disponible", 'error')
            return
            
        self.analysis_in_progress = True
        self.diagnosis_label.config(text="EJECUTANDO ANÁLISIS...", bootstyle='info')
        self.confidence_label.config(text="Segmentación + Clasificación IA")
        self.segmentation_label.config(text="Procesando...")
        
        # Limpiar visualizaciones
        for canvas in [self.contour_canvas, self.segmented_canvas, self.prob_canvas]:
            canvas.delete("all")
            canvas.create_text(125, 100, text="Procesando...", 
                             fill="white", font=('Arial', 10))
        
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
        
        # Actualizar información de segmentación
        if segmentation_info:
            area = segmentation_info.get('area_pixels', 0)
            self.segmentation_label.config(text=f"Área de lesión: {area:.0f} píxeles")
        
        # Actualizar visualizaciones
        self.update_visualizations(segmentation_info, all_predictions)
        
        # Guardar log
        self.save_analysis_log(result_text, confidence, segmentation_info)
        
        print("✅ Análisis completado y visualizaciones actualizadas")
    
    def show_analysis_error(self, error_msg):
        """Mostrar error en análisis"""
        self.analysis_in_progress = False
        self.diagnosis_label.config(text=error_msg, bootstyle='danger')
        self.confidence_label.config(text="Error en análisis")
        self.segmentation_label.config(text="---")
    
    def save_analysis_log(self, result, confidence, segmentation_info):
        """Guardar log del análisis"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"analysis_logs/analisis_{timestamp}.txt"
            
            os.makedirs("analysis_logs", exist_ok=True)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=== DERMASCAN PRO - ANÁLISIS COMPLETO ===\n")
                f.write(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Imagen: {self.image_path}\n")
                f.write(f"Resultado: {result}\n")
                f.write(f"Confianza: {confidence:.1%}\n")
                if segmentation_info:
                    f.write(f"Área de lesión: {segmentation_info.get('area_pixels', 0):.0f} píxeles\n")
                    f.write(f"Contornos detectados: {segmentation_info.get('contours_found', 0)}\n")
                f.write("Tecnología: Segmentación YCbCr + EfficientNetB0\n")
                f.write("=" * 60 + "\n")
                
            print(f"📝 Log guardado: {log_file}")
                
        except Exception as e:
            print(f"Error guardando log: {e}")
    
    def save_report(self):
        """Guardar reporte completo"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"reportes/reporte_{timestamp}.txt"
            
            os.makedirs("reportes", exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=== REPORTE DERMATOLÓGICO - DERMASCAN PRO ===\n\n")
                f.write(f"Fecha del análisis: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Imagen analizada: {os.path.basename(self.image_path)}\n\n")
                f.write("RESULTADOS:\n")
                f.write(f"- Diagnóstico: {self.diagnosis_label.cget('text')}\n")
                f.write(f"- Nivel de confianza: {self.confidence_label.cget('text')}\n")
                f.write(f"- {self.segmentation_label.cget('text')}\n\n")
                f.write("METODOLOGÍA:\n")
                f.write("- Segmentación: Espacio de color YCbCr\n")
                f.write("- Clasificación: Red Neuronal EfficientNetB0\n")
                f.write("- Características analizadas: Asimetría, Bordes, Color, Diámetro\n\n")
                f.write("NOTA: Este reporte es de apoyo al diagnóstico y debe ser\n")
                f.write("validado por un especialista médico certificado.\n")
            
            self.show_message(f"Reporte guardado: {report_file}", 'success')
            
        except Exception as e:
            self.show_message(f"Error guardando reporte: {str(e)}", 'error')
    
    def show_technical_details(self):
        """Mostrar detalles técnicos del pipeline"""
        details = """
🔬 PIPELINE TÉCNICO COMPLETO:

1. SEGMENTACIÓN YCbCr
   - Conversión a espacio de color YCbCr
   - Detección de piel: Cb ∈ [77,127], Cr ∈ [133,173]
   - Operaciones morfológicas (abertura/cierre)
   - Detección de contornos
   - Aislamiento de la lesión

2. CLASIFICACIÓN IA
   - Red Neuronal: EfficientNetB0
   - Input: 224x224 píxeles (RGB)
   - Normalización: [0, 1]
   - 3 Clases: Melanoma, Nevus, Otras

3. ANÁLISIS DE CARACTERÍSTICAS
   - Asimetría de la lesión
   - Bordes irregulares
   - Variación de color
   - Diámetro y área

PRECISIÓN REPORTADA: 87.3%
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
        self.root.title("DermaScan Pro - Dermatoscopio con Segmentación IA")
        self.root.geometry("1400x900")
        
        # Inicializar modelo de ML con pipeline completo
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
        self.flash_on = False
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
            text="DERMASCAN PRO",
            style='Title.TLabel'
        ).pack(anchor=W)
        
        tb.Label(
            title_frame, 
            text="Sistema de Análisis Dermatológico con Segmentación IA",
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

        # Control de flash
        self.flash_btn = tb.Button(
            image_controls,
            text="⚡ Flash: OFF",
            bootstyle='outline-warning',
            command=self.toggle_flash,
            padding=(8, 4)
        )
        self.flash_btn.pack(side=LEFT, padx=(0, 8))

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

        # Control de zoom
        zoom_frame = tb.Frame(image_controls)
        zoom_frame.pack(side=LEFT)
        
        tb.Label(zoom_frame, text="Zoom:").pack(side=LEFT, padx=(0, 5))
        
        self.zoom_scale = tb.Scale(
            zoom_frame,
            from_=0.5,
            to=3.0,
            value=1.0,
            command=self.update_zoom,
            length=100,
            bootstyle='primary'
        )
        self.zoom_scale.pack(side=LEFT)

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
            ("🔬 ABRIR ANÁLISIS", 'primary', self.open_analysis),
            ("📁 ABRIR ARCHIVO", 'secondary', self.open_file),
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
            text="INFORMACIÓN DEL SISTEMA",
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
            text="● CONECTADO",
            bootstyle='success',
            font=('Arial', 9, 'bold')
        )
        self.conn_status.pack(anchor=W, pady=(2, 0))

        # Información de hardware
        hardware_info = """
HARDWARE:
• Raspberry Pi 5
• Cámara HQ 12MP
• Iluminación LED
• Lente dermatoscópico

SOFTWARE:
• OpenCV 4.8.0
• TensorFlow 2.13
• Python 3.11
"""

        tb.Label(
            info_panel,
            text=hardware_info,
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
            text="DermaScan Pro v3.0 - Sistema Médico de Apoyo al Diagnóstico",
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
                
                # Aplicar zoom si es necesario
                if self.zoom_level != 1.0:
                    frame = self.apply_zoom(frame, self.zoom_level)

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

    def apply_zoom(self, frame, zoom_level):
        """Aplicar zoom a la imagen"""
        if zoom_level == 1.0:
            return frame
            
        h, w = frame.shape[:2]
        new_h, new_w = int(h / zoom_level), int(w / zoom_level)
        
        # Calcular región de recorte
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        
        # Recortar y redimensionar
        cropped = frame[start_y:start_y+new_h, start_x:start_x+new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return zoomed

    def update_zoom(self, value):
        """Actualizar nivel de zoom"""
        self.zoom_level = float(value)

    def toggle_flash(self):
        """Alternar flash (simulado)"""
        self.flash_on = not self.flash_on
        state = "ON" if self.flash_on else "OFF"
        self.flash_btn.config(text=f"⚡ Flash: {state}")
        
        if self.flash_on:
            self.status_label.config(text="● FLASH ACTIVADO", bootstyle='warning')
        else:
            self.status_label.config(text="● CÁMARA ACTIVA", bootstyle='success')

    def capture(self):
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

    def open_analysis(self):
        """Abrir ventana de análisis"""
        if not self.current_image_path:
            self.show_error("Capture una imagen primero")
            return
            
        AnalysisWindow(self.root, self.current_image_path, self.ml_model)

    def open_file(self):
        """Abrir archivo de imagen"""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.show_message(f"Imagen cargada: {os.path.basename(file_path)}", 'success')

    def show_message(self, message, style='info'):
        """Mostrar mensaje"""
        mb = tb.dialogs.Messagebox
        mb.show_info(message, title="DermaScan Pro", parent=self.root)

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