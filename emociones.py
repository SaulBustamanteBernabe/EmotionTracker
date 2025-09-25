import cv2
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from keras.src.saving import load_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict, Counter

from tensorflow.python.ops.signal.shape_ops import frame

from consejo import consejo

custom_emotion_model = load_model("my_emotion_model.h5")
class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector Avanzado de Emociones")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")
        
        self.detector = DeepFace
        
        # Variables de estado
        self.cap = None
        self.streaming = False
        self.current_image = None
        
        # Variables para estadísticas de emociones
        self.emotion_history = []
        self.emotion_stats = defaultdict(list)
        self.chart_frame = None
        self.chart_canvas = None
        
        # Estilos para botones
        self.btn_style = {
            "font": ("Helvetica", 14, "bold"),
            "fg": "white",
            "bg": "#2980b9",
            "activebackground": "#3498db",
            "activeforeground": "white",
            "bd": 0,
            "width": 15,
            "height": 2
        }
        
        # Marco superior para botones
        self.top_frame = tk.Frame(self.root, bg="#2c3e50")
        self.top_frame.pack(pady=20)
        
        self.btn_image = tk.Button(
            self.top_frame,
            text="📁 Cargar Imagen",
            command=self.load_image,
            **self.btn_style
        )
        self.btn_image.grid(row=0, column=0, padx=20)
        
        self.btn_webcam = tk.Button(
            self.top_frame,
            text="📷 Iniciar Webcam",
            command=self.toggle_webcam,
            **self.btn_style
        )
        self.btn_webcam.grid(row=0, column=1, padx=20)
        
        # Marco central para mostrar imagen o video
        self.display_frame = tk.Frame(self.root, bg="#34495e", bd=2, relief="sunken")
        self.display_frame.pack(pady=10, expand=True, fill="both")
        
        self.display_label = tk.Label(self.display_frame, bg="#34495e")
        self.display_label.pack(expand=True)
        
        # Marco inferior para texto de estado
        self.bottom_frame = tk.Frame(self.root, bg="#2c3e50")
        self.bottom_frame.pack(pady=10)
        
        self.result_label = tk.Label(
            self.bottom_frame,
            text="Estado: Esperando acción...",
            font=("Helvetica", 16),
            fg="white",
            bg="#2c3e50"
        )
        self.result_label.pack()

    def load_image(self):
        # Si la webcam está activa, la detenemos
        if self.streaming:
            self.stop_webcam()
        
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            self.result_label.config(text="Error: No se pudo cargar la imagen.")
            return
        
        # Analizar emoción con DeepFace
        try:
            resultado = DeepFace.analyze(
                img_path=img_bgr,
                actions=['emotion'],
                models={"emotion": custom_emotion_model},  # << aquí tu modelo
                enforce_detection=False
            )
            if isinstance(resultado, list):
                res = resultado[0]
            else:
                res = resultado
            
            dominant = res['dominant_emotion']
            confidence = res['emotion'][dominant] * 100 if 'emotion' in res else 0.0
            texto = f"{dominant.capitalize()}"
            porcentaje = f"{confidence:05.2f}%"
        except Exception as e:
            texto = "Error al analizar"
            porcentaje = "00.00%"
        
        # Convertir la imagen a RGB y redimensionarla para el display
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = self.resize_for_display(img_rgb)
        
        # Dibujar la caja y el texto mejorado
        img_completo = self.draw_emotion_display(img_rgb, texto, porcentaje)
        
        img_pil = Image.fromarray(img_completo)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.current_image = img_tk
        self.display_label.config(image=img_tk)
        self.result_label.config(text=f"Imagen: {texto} - {porcentaje}")

    def toggle_webcam(self):
        if not self.streaming:
            self.start_webcam()
        else:
            self.stop_webcam()

    def start_webcam(self):
        self.streaming = True
        # Limpiar estadísticas anteriores
        self.emotion_history.clear()
        self.emotion_stats.clear()
        self.hide_chart()
        
        self.btn_webcam.config(
            text="❌ Detener Webcam",
            bg="#c0392b",
            activebackground="#e74c3c"
        )
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def stop_webcam(self):
        self.streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Mostrar estadísticas si hay datos
        if self.emotion_history:
            self.show_emotion_statistics()
            consejo(self.root, self.emotion_history)
        else:
            self.display_label.config(image='')
            
        self.btn_webcam.config(
            text="📷 Iniciar Webcam",
            bg="#2980b9",
            activebackground="#3498db"
        )
        self.result_label.config(text="Estado: Webcam detenida")

    def update_frame(self):
        if not self.streaming or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("ret:", ret)
            self.result_label.config(text="Error al acceder a la cámara.")
            self.stop_webcam()
            return
        
        # Analizar emoción en el frame
        try:
            resultado = self.detector.analyze(
                frame,
                actions=['emotion'],
                #models={"emotion": custom_emotion_model},
                #detector_backend="opencv",
                enforce_detection=False
            )
            if isinstance(resultado, list):
                res = resultado[0]
            else:
                res = resultado
            
            dominant = res['dominant_emotion']
            confidence = res['emotion'][dominant] * 100 if 'emotion' in res else 0.0
            texto = f"{dominant.capitalize()}"
            porcentaje = f"{confidence:05.2f}%"
            
            # Guardar estadísticas
            self.emotion_history.append(dominant)
            for emotion, value in res['emotion'].items():
                self.emotion_stats[emotion].append(value * 100)
                
        except:
            texto = "Sin detección"
            porcentaje = "--"
        
        # Convertir a RGB y redimensionar para display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = self.resize_for_display(frame_rgb)
        
        # Dibujar la caja y el texto mejorado
        frame_con_texto = self.draw_emotion_display(frame_rgb, texto, porcentaje)
        
        img_pil = Image.fromarray(frame_con_texto)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.current_image = img_tk
        self.display_label.config(image=img_tk)
        self.result_label.config(text=f"Webcam: {texto} - {porcentaje}")
        
        # Volver a llamar en 30 ms
        self.root.after(30, self.update_frame)

    def draw_emotion_display(self, img, emocion, porcentaje):
        """
        Dibuja un display elegante con la emoción detectada y su porcentaje
        """
        h, w = img.shape[:2]
        overlay = img.copy()
        
        # Configuración de fuentes y tamaños
        font_emotion = cv2.FONT_HERSHEY_DUPLEX
        font_percentage = cv2.FONT_HERSHEY_SIMPLEX
        
        emotion_scale = 1.2
        percentage_scale = 0.8
        thickness = 2
        
        # Obtener dimensiones del texto
        (emotion_w, emotion_h), _ = cv2.getTextSize(emocion, font_emotion, emotion_scale, thickness)
        (percent_w, percent_h), _ = cv2.getTextSize(porcentaje, font_percentage, percentage_scale, thickness)
        
        # Configuración del contenedor principal
        padding = 20
        container_height = emotion_h + percent_h + padding * 3
        container_width = max(emotion_w, percent_w) + padding * 2
        
        # Centrar el contenedor horizontalmente
        container_x = (w - container_width) // 2
        container_y = 15
        
        # Dibujar contenedor principal con gradiente simulado
        # Capa base (más oscura)
        cv2.rectangle(overlay, 
                     (container_x - 5, container_y - 5), 
                     (container_x + container_width + 5, container_y + container_height + 5), 
                     (30, 30, 30), -1)
        
        # Capa principal
        cv2.rectangle(overlay, 
                     (container_x, container_y), 
                     (container_x + container_width, container_y + container_height), 
                     (45, 45, 45), -1)
        
        # Línea superior decorativa (color según emoción)
        color_emotion = self.get_emotion_color(emocion.lower())
        cv2.rectangle(overlay, 
                     (container_x, container_y), 
                     (container_x + container_width, container_y + 4), 
                     color_emotion, -1)
        
        # Aplicar transparencia
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Posiciones del texto
        emotion_x = container_x + (container_width - emotion_w) // 2
        emotion_y = container_y + padding + emotion_h
        
        percent_x = container_x + (container_width - percent_w) // 2
        percent_y = emotion_y + padding + percent_h
        
        # Dibujar texto de emoción con sombra
        # Sombra
        cv2.putText(img, emocion, (emotion_x + 2, emotion_y + 2), 
                   font_emotion, emotion_scale, (0, 0, 0), thickness + 1)
        # Texto principal
        cv2.putText(img, emocion, (emotion_x, emotion_y), 
                   font_emotion, emotion_scale, (255, 255, 255), thickness)
        
        # Dibujar porcentaje con color de la emoción
        # Sombra
        cv2.putText(img, porcentaje, (percent_x + 1, percent_y + 1), 
                   font_percentage, percentage_scale, (0, 0, 0), thickness)
        # Texto principal
        cv2.putText(img, porcentaje, (percent_x, percent_y), 
                   font_percentage, percentage_scale, color_emotion, thickness)
        
        return img

    def get_emotion_color(self, emotion):
        """
        Retorna un color BGR específico para cada emoción
        """
        colors = {
            'happy': (0, 215, 255),      # Naranja brillante
            'sad': (255, 100, 100),      # Azul suave
            'angry': (0, 50, 255),       # Rojo
            'fear': (128, 0, 255),       # Púrpura
           # 'surprise': (0, 255, 255),   # Amarillo
            'disgust': (0, 128, 0),      # Verde
           # 'neutral': (200, 200, 200),  # Gris claro
            'contempt': (100, 150, 200), # Marrón claro
        }
        return colors.get(emotion, (255, 255, 255))  # Blanco por defecto

    def show_emotion_statistics(self):
        """
        Muestra una gráfica con las estadísticas de emociones detectadas
        """
        # Ocultar la imagen actual
        self.display_label.config(image='')
        
        # Crear frame para la gráfica si no existe
        if self.chart_frame is None:
            self.chart_frame = tk.Frame(self.display_frame, bg="#34495e")
        
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Calcular estadísticas
        emotion_counts = Counter(self.emotion_history)
        total_detections = len(self.emotion_history)
        
        # Calcular promedios de confianza
        avg_confidences = {}
        for emotion, values in self.emotion_stats.items():
            if values:
                avg_confidences[emotion] = np.mean(values)
        
        # Crear la figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#34495e')
        
        # Colores para las emociones (RGB normalizados)
        emotion_colors_rgb = {
            'happy': (1.0, 0.84, 0.0),      # Amarillo-naranja
            'sad': (0.39, 0.39, 1.0),       # Azul
            'angry': (1.0, 0.2, 0.2),       # Rojo
            'fear': (0.5, 0.0, 1.0),        # Púrpura
            #'surprise': (1.0, 1.0, 0.0),    # Amarillo
            'disgust': (0.0, 0.5, 0.0),     # Verde
            #'neutral': (0.78, 0.78, 0.78),  # Gris
            'contempt': (0.6, 0.4, 0.2),    # Marrón
        }
        
        # Gráfica 1: Frecuencia de emociones (barras)
        emotions = list(emotion_counts.keys())
        frequencies = [emotion_counts[emotion] for emotion in emotions]
        percentages = [(count/total_detections)*100 for count in frequencies]
        
        colors1 = [emotion_colors_rgb.get(emotion, (0.8, 0.8, 0.8)) for emotion in emotions]
        
        bars = ax1.bar(emotions, percentages, color=colors1, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax1.set_title('Frecuencia de Emociones Detectadas', color='white', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Emociones', color='white', fontsize=12)
        ax1.set_ylabel('Porcentaje (%)', color='white', fontsize=12)
        ax1.set_facecolor('#2c3e50')
        ax1.tick_params(colors='white', rotation=45)
        ax1.grid(True, alpha=0.3, color='white')
        
        # Agregar valores en las barras
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{percentage:.1f}%', ha='center', va='bottom', color='white', fontweight='bold')
        
        # Gráfica 2: Confianza promedio (gráfica circular)
        if avg_confidences:
            # Filtrar solo emociones que fueron detectadas
            detected_emotions = [emotion for emotion in avg_confidences.keys() if emotion in emotions]
            avg_values = [avg_confidences[emotion] for emotion in detected_emotions]
            colors2 = [emotion_colors_rgb.get(emotion, (0.8, 0.8, 0.8)) for emotion in detected_emotions]
            
            wedges, texts, autotexts = ax2.pie(avg_values, labels=detected_emotions, autopct='%1.1f%%',
                                              colors=colors2, startangle=90, textprops={'color': 'white'})
            
            # Mejorar el texto
            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            ax2.set_title('Confianza Promedio por Emoción', color='white', fontsize=14, fontweight='bold')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Crear canvas y mostrarlo
        if self.chart_canvas:
            self.chart_canvas.destroy()
            
        self.chart_canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Crear botón para cerrar estadísticas
        close_btn = tk.Button(
            self.chart_frame,
            text="❌ Cerrar Estadísticas",
            command=self.hide_chart,
            font=("Helvetica", 12, "bold"),
            fg="white",
            bg="#c0392b",
            activebackground="#e74c3c",
            activeforeground="white",
            bd=0,
            width=20,
            height=2
        )
        close_btn.pack(pady=10)
        
        # Actualizar texto de estado
        total_time = len(self.emotion_history) * 0.03  # Aproximadamente 30ms por frame
        self.result_label.config(
            text=f"Estadísticas: {total_detections} detecciones en {total_time:.1f}s - Emoción más frecuente: {max(emotion_counts, key=emotion_counts.get).capitalize()}"
        )


    def hide_chart(self):
        """
        Oculta la gráfica de estadísticas
        """
        if self.chart_frame:
            self.chart_frame.destroy()
            self.chart_frame = None
        if self.chart_canvas:
            self.chart_canvas = None

    def resize_for_display(self, img):
        """
        Redimensiona 'img' (RGB) para que quepa dentro de display_frame,
        manteniendo proporción.
        """
        frame_h, frame_w = img.shape[:2]
        disp_w = self.display_frame.winfo_width()
        disp_h = self.display_frame.winfo_height()
        if disp_w <= 0 or disp_h <= 0:
            return img  # Aún no se ha renderizado el frame
        scale = min(disp_w / frame_w, disp_h / frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)
        return cv2.resize(img, (new_w, new_h))

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()