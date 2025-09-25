from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print

llm_cfg = {
    "model": "qwen3:1.7b",
    "model_server": "http://localhost:11434/v1",
}

system_message = """
Eres un Psicólogo experto en análisis de emociones y un consejero empático. Tu objetivo principal es ayudar a las personas a entender y gestionar sus estados emocionales.

Cuando se te presente un conjunto de emociones, debes:
1.  Validar y reconocer las emociones del usuario de forma empática.
2.  Analizar la combinación de emociones para identificar posibles causas subyacentes o interacciones.
3.  Proporcionar una serie de consejos cortos y concisos para que el usuario pueda tomar para gestionar esas emociones de forma constructiva.

Tu tono debe ser siempre comprensivo, alentador y profesional.
"""

bot = Assistant(llm=llm_cfg, system_message=system_message)

messages = [{'role': 'user', 'content': "{'neutral': '38.33%', 'happy': '22.47%', 'sad': '13.22%', 'fear': '12.78%', 'angry': '12.78%', 'surprise': '0.44%'}"}]

response_plain_text = ''
print('bot response:')
for response in bot.run(messages=messages):
    response_plain_text = typewriter_print(response, response_plain_text)