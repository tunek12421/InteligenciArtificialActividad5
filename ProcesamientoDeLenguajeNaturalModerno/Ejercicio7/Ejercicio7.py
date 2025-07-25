# Ejercicio 7: Fine-tuning de BERT en tareas de QA - OPTIMIZADO
# Objetivo: Ajustar un modelo BERT para responder preguntas sobre contexto.
# Dataset: SQuAD 2.0 (o propio)
# Herramientas: transformers, datasets

import torch
from transformers import pipeline

# Optimizar para GPU si está disponible
device = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if device == 0 else "CPU"

print(f"Cargando modelo de Question Answering en {device_name}...")

try:
    qa = pipeline(
        "question-answering", 
        model="distilbert-base-uncased-distilled-squad",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    print("Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Múltiples contextos y preguntas para pruebas
qa_pairs = [
    {
        'context': 'La Universidad Técnica de Oruro fue fundada en 1892. Es una de las universidades más antiguas de Bolivia y se especializa en ingeniería y tecnología.',
        'question': '¿Cuándo fue fundada la Universidad Técnica de Oruro?'
    },
    {
        'context': 'La Universidad Técnica de Oruro fue fundada en 1892. Es una de las universidades más antiguas de Bolivia y se especializa en ingeniería y tecnología.',
        'question': '¿En qué se especializa la universidad?'
    },
    {
        'context': 'La inteligencia artificial es una rama de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Esto incluye el aprendizaje, la percepción, el razonamiento y la toma de decisiones.',
        'question': '¿Qué es la inteligencia artificial?'
    },
    {
        'context': 'Python es un lenguaje de programación de alto nivel creado por Guido van Rossum en 1991. Es conocido por su sintaxis simple y su versatilidad en diferentes áreas como desarrollo web, ciencia de datos y aprendizaje automático.',
        'question': '¿Quién creó Python y cuándo?'
    }
]

print("\n--- Sistema de Preguntas y Respuestas ---")

for i, qa_pair in enumerate(qa_pairs, 1):
    print(f"\n{i}. Pregunta: {qa_pair['question']}")
    
    try:
        result = qa(qa_pair)
        
        answer = result['answer']
        confidence = result['score']
        start_pos = result['start']
        end_pos = result['end']
        
        print(f"   Respuesta: {answer}")
        print(f"   Confianza: {confidence:.4f}")
        print(f"   Posición en texto: {start_pos}-{end_pos}")
        
        # Mostrar contexto con la respuesta resaltada
        context = qa_pair['context']
        highlighted = context[:start_pos] + f"**{answer}**" + context[end_pos:]
        print(f"   Contexto: {highlighted[:100]}...")
        
    except Exception as e:
        print(f"   Error procesando pregunta: {e}")

if torch.cuda.is_available():
    print(f"\nMemoria GPU utilizada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# Resultado esperado: Respuestas precisas con alta confianza

# Preguntas:
# 1. ¿Qué hace el modelo para identificar la respuesta dentro del contexto?
# 2. ¿Por qué es útil tener un modelo preentrenado en SQuAD?
# 3. ¿Qué tan preciso fue el modelo en tus pruebas?
# 4. ¿Qué desafíos enfrentarías si quisieras entrenar tu propio modelo de QA?
# 5. ¿Puedes imaginar una aplicación de esta técnica en tu entorno profesional?

# Sugerencias de mejora:
# • Usar múltiples preguntas sobre un mismo contexto para evaluar comprensión.
# • Sugerir prueba con textos propios (p. ej. artículos académicos).