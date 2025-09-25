import tkinter as tk
import numpy as np
from collections import Counter
from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print
import threading


class consejo(tk.Toplevel):
    def __init__(self, master, emotion_history: list):
        super().__init__(master)
        self.title("Consejo y Recomendaciones")
        self.geometry("900x450")
        self.configure(bg="#34495e")
        self.resizable(True, True)
        self.transient(master)

        # Configuración del bot
        self.llm_cfg = {
            "model": "qwen3:1.7b",
            "model_server": "http://localhost:11434/v1",
        }
        self.system_message = """
        Eres un Psicólogo experto en análisis de emociones y un consejero empático. Tu objetivo principal es ayudar a las personas a entender y gestionar sus estados emocionales.

        Cuando se te presente un sconjunto de emociones, debes:
        1.  Validar y reconocer las emociones del usuario de forma empática.
        2.  Analizar la combinación de emociones para identificar posibles causas subyacentes o interacciones.
        3.  Proporcionar una serie de consejos cortos y concisos para que el usuario pueda tomar para gestionar esas emociones de forma constructiva.

        Tu tono debe ser siempre comprensivo, alentador y profesional.
        """
        self.bot = Assistant(llm=self.llm_cfg, system_message=self.system_message)

        # Calcular estadísticas
        emotion_counts = Counter(emotion_history)
        total_detections = len(emotion_history)
        # Gráfica 1: Frecuencia de emociones (barras)
        emotions = list(emotion_counts.keys())
        frequencies = [emotion_counts[emotion] for emotion in emotions]
        percentages = [(count/total_detections)*100 for count in frequencies]
        # Diccionario de emociones
        emotion_percentages = dict(zip(emotions, percentages))
        for emotion, percentage in emotion_percentages.items():
            emotion_percentages[emotion] = f"{percentage:.2f}%"

        # Widget de texto solo lectura
        self.texto = tk.Text(self, wrap="word", font=("Helvetica", 12), bg="#2c3e50", fg="white", height=10, width=35, borderwidth=0)
        self.texto.pack(padx=15, pady=20, fill="both", expand=True)
        # Configurar como solo lectura
        self.texto.config(state="disabled")

        # Prompt para el bot
        prompt = f"{emotion_percentages}"

        # Obtener respuesta del bot en segundo plano
        def obtener_consejo():
            response_plain_text = ''
            for response in self.bot.run(messages=[{'role': 'user', 'content': prompt}]):
                response_plain_text = typewriter_print(response, response_plain_text)
                # Insertar texto de consejo en el hilo principal
                self.texto.after(0, lambda: [
                self.texto.config(state="normal"),
                self.texto.delete("1.0", "end"),
                self.texto.insert("1.0", f"{response_plain_text}"),
                self.texto.config(state="disabled")
                ])

        threading.Thread(target=obtener_consejo, daemon=True).start()