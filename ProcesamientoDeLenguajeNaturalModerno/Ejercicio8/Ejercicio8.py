# Ejercicio 8: Chatbot básico con Transformers - OPTIMIZADO
# Objetivo: Crear una interfaz conversacional usando un modelo conversacional.
# Herramientas: transformers
# Modelo: microsoft/DialoGPT-medium (optimizado para GPU)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Verificar GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "GPU" if torch.cuda.is_available() else "CPU"

print(f"Cargando modelo DialoGPT en {device_name}...")

try:
    # Usar modelo más pequeño si no hay GPU suficiente
    model_name = "microsoft/DialoGPT-small" if not torch.cuda.is_available() else "microsoft/DialoGPT-medium"
    print(f"Usando modelo: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True
    )
    model = model.to(device)
    
    # Añadir pad_token si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    exit(1)

chat_history_ids = None

def respond(message):
    global chat_history_ids
    
    # Codificar mensaje de entrada
    new_user_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')
    new_user_input_ids = new_user_input_ids.to(device)
    
    # Concatenar con historial si existe
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids
    
    # Generar respuesta
    with torch.no_grad():
        chat_history_ids = model.generate(
            bot_input_ids, 
            max_length=min(1000, bot_input_ids.shape[-1] + 50),
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decodificar solo la nueva respuesta
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
        skip_special_tokens=True
    )
    
    return response.strip()

# Modo consola en lugar de Gradio para evitar dependencias
print("\n=== Chatbot Listo ===")
print("Escribe 'quit' para salir\n")

while True:
    user_input = input("Tú: ")
    
    if user_input.lower() in ['quit', 'exit', 'salir']:
        print("\u00a1Hasta luego!")
        break
    
    if user_input.strip():
        try:
            response = respond(user_input)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Error generando respuesta: {e}\n")
    else:
        print("Por favor, escribe algo.\n")

if torch.cuda.is_available():
    print(f"Memoria GPU utilizada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# Resultado esperado: Interfaz web funcional con respuestas básicas tipo chatbot.

# Preguntas:
# 1. ¿Qué diferencia a un chatbot basado en reglas de uno basado en modelos generativos
# como DialoGPT?
# 2. ¿Cómo maneja el modelo el historial de la conversación?
# 3. ¿Qué problemas encontraste en la coherencia de las respuestas?
# 4. ¿Qué harías para mejorar la fluidez y precisión del chatbot?
# 5. ¿Qué otros modelos podrías probar en lugar de DialoGPT?

# Sugerencias de mejora:
# • Implementar almacenamiento del historial de conversación en history y mostrarlo en UI.
# • Comentar los límites de coherencia de modelos pequeños como DialoGPT.