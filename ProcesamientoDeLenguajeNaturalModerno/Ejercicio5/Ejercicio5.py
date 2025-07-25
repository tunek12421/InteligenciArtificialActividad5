# Ejercicio 5: Resumen automático de texto con BART - OPTIMIZADO
# Objetivo: Generar resúmenes de texto con un modelo preentrenado.
# Herramientas: transformers, pipeline
# Texto libre o noticias

import torch
from transformers import pipeline

# Verificar GPU disponible para acelerar inferencia
device = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if device == 0 else "CPU"

print(f"Cargando modelo BART en {device_name}...")

# Usar modelo más pequeño para velocidad o BART completo si hay GPU
model_name = "facebook/bart-large-cnn" if torch.cuda.is_available() else "facebook/bart-base"
print(f"Usando modelo: {model_name}")

try:
    summarizer = pipeline(
        "summarization", 
        model=model_name,
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    print("Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error cargando modelo completo: {e}")
    print("Usando modelo base más pequeño...")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

# Textos de ejemplo para probar
texts = [
    """La inteligencia artificial está transformando múltiples industrias a un ritmo sin precedentes. Desde la medicina hasta las finanzas, pasando por el transporte y la educación, los sistemas de IA están automatizando tareas complejas, mejorando la precisión en el diagnóstico médico, optimizando las rutas de transporte y personalizando la experiencia educativa. Los algoritmos de machine learning procesan enormes cantidades de datos para identificar patrones que los humanos no pueden detectar fácilmente. Sin embargo, esta revolución tecnológica también plantea importantes desafíos éticos y sociales, incluyendo preocupaciones sobre el empleo, la privacidad de los datos y la equidad algorítmica.""",
    
    """El cambio climático representa uno de los mayores desafíos de nuestro tiempo. Las temperaturas globales continúan aumentando, causando derretimiento de glaciares, aumento del nivel del mar y eventos climáticos extremos más frecuentes. Los científicos han demostrado que las actividades humanas, especialmente la quema de combustibles fósiles, son la principal causa de este calentamiento. Las consecuencias incluyen pérdida de biodiversidad, inseguridad alimentaria y desplazamiento de poblaciones. Es crucial que tomemos medidas inmediatas para reducir las emisiones de carbono y adoptar energías renovables."""
]

print("Generando resúmenes...")

for i, text in enumerate(texts, 1):
    print(f"\n--- Texto {i} ---")
    print(f"Texto original ({len(text)} caracteres)")
    
    try:
        # Optimizar parámetros según longitud del texto
        max_len = min(100, len(text.split()) // 2)
        min_len = max(20, max_len // 3)
        
        summary = summarizer(
            text, 
            max_length=max_len, 
            min_length=min_len, 
            do_sample=False,
            truncation=True
        )
        
        result = summary[0]['summary_text']
        print(f"Resumen ({len(result)} caracteres): {result}")
    except Exception as e:
        print(f"Error procesando texto: {e}")

if torch.cuda.is_available():
    print(f"\nMemoria GPU utilizada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# Resultado esperado: Un resumen claro y conciso del texto original.

# Preguntas:
# 1. ¿Cómo se diferencia el resumen extractivo del resumen abstractivo?
# 2. ¿Por qué usamos facebook/bart-large-cnn para esta tarea?
# 3. ¿Qué limitaciones encontraste en los resúmenes generados?
# 4. ¿Cómo podrías ajustar el modelo para resúmenes más cortos o más largos?
# 5. ¿En qué aplicaciones reales sería útil esta técnica?

# Sugerencias de mejora:
# • Probar también t5-small para ver diferencias entre modelos.
# • Mostrar cómo ajustar max_length, min_length, do_sample para distintos objetivos.