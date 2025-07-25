# Demo rápido del chatbot sin cargar modelo completo
import torch

print("=== Demo Chatbot (simulado) ===")
print("Modelo: DialoGPT optimizado para GPU")
print(f"GPU disponible: {torch.cuda.is_available()}")

# Simulación de conversación
conversations = [
    ("Hola", "¡Hola! ¿Cómo estás hoy?"),
    ("¿Qué puedes hacer?", "Puedo mantener conversaciones, responder preguntas y ayudarte con tareas básicas."),
    ("¿Te gusta la programación?", "¡Sí! La programación es fascinante. Hay mucho que aprender."),
    ("Gracias", "¡De nada! ¿En qué más puedo ayudarte?")
]

print("\n--- Ejemplo de Conversación ---")
for user_msg, bot_response in conversations:
    print(f"Tú: {user_msg}")
    print(f"Bot: {bot_response}\n")

print("Nota: Chatbot completo requiere modelo DialoGPT cargado")
if torch.cuda.is_available():
    print(f"GPU lista para acelerar inferencia")