# Ejercicio 6: Análisis de sentimientos con DistilBERT - OPTIMIZADO
# Objetivo: Detectar sentimientos usando distilbert-base-uncased-finetuned-sst-2-english

import torch
from transformers import pipeline

# Optimizar para GPU si está disponible
device = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if device == 0 else "CPU"

print(f"Cargando modelo de análisis de sentimientos en {device_name}...")

try:
    sentiment = pipeline(
        "sentiment-analysis",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    print("Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    sentiment = pipeline("sentiment-analysis")

# Múltiples frases para análisis en lote (más eficiente)
test_sentences = [
    "This new laptop is amazing!",
    "This was the worst customer service ever.",
    "The movie was okay, nothing special.",
    "I absolutely love this product!",
    "The weather is nice today.",
    "I'm feeling sad about the news.",
    "This restaurant has excellent food!",
    "The service was slow and disappointing."
]

print("\n--- Análisis de Sentimientos en Lote ---")
# Procesar en lote para mejor rendimiento
results = sentiment(test_sentences)

for sentence, result in zip(test_sentences, results):
    label = result['label']
    score = result['score']
    print(f"\"{sentence}\"")
    print(f"   Sentimiento: {label} (confianza: {score:.3f})\n")

# Resultado esperado: Sentimiento Positivo o Negativo con nivel de confianza.

# Preguntas:
# 1. ¿Qué ventajas tiene DistilBERT sobre BERT completo?
# 2. ¿Qué tipo de tareas reales puedes resolver con análisis de sentimientos?
# 3. ¿Qué nivel de confianza obtuviste para las frases positivas/negativas?
# 4. ¿En qué casos podría fallar un modelo de sentimiento?
# 5. ¿Qué cambios podrías hacer para adaptarlo a un nuevo idioma?

# Sugerencias de mejora:
# • Agregar visualización del score de sentimiento (por ejemplo, en barra).
# • Sugerir evaluación sobre varios ejemplos en lote (batch).