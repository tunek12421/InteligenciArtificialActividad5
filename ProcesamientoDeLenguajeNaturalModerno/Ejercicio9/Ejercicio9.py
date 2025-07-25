# Ejercicio 9: Proyecto final integrador - OPTIMIZADO
# Objetivo: Cargar un texto largo, analizar su sentimiento, clasificar su tema y resumirlo.
# Herramientas: transformers, pipeline
# Pipeline completo de NLP con optimizaciones GPU

import torch
from transformers import pipeline
import time

# Optimizar para GPU si está disponible
device = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if device == 0 else "CPU"

print(f"=== Pipeline NLP Integrado en {device_name} ===")
print("Cargando modelos...")

# Textos de ejemplo
textos = [
    "El avance de la inteligencia artificial está cambiando el mundo. Los modelos de machine learning están revolucionando industrias enteras, desde la medicina hasta el transporte. Sin embargo, también plantean desafíos éticos importantes sobre el futuro del trabajo y la privacidad de los datos. Es crucial que desarrollemos estas tecnologías de manera responsable para maximizar sus beneficios y minimizar los riesgos potenciales para la sociedad.",
    
    "La situación económica mundial muestra signos de recuperación tras la crisis. Los indicadores financieros han mejorado considerablemente en los últimos meses. Sin embargo, la inflación sigue siendo una preocupación para muchos países. Los bancos centrales están implementando políticas monetarias cuidadosas para mantener la estabilidad. Es fundamental que los gobiernos continúen con reformas estructurales para asegurar un crecimiento sostenible a largo plazo.",
    
    "Los nuevos descubrimientos médicos ofrecen esperanza para el tratamiento de enfermedades raras. Los investigadores han desarrollado terapias génicas innovadoras que muestran resultados prometedores en ensayos clínicos. La medicina personalizada está revolucionando cómo abordamos el tratamiento de pacientes. Estos avances representan un paso importante hacia tratamientos más efectivos y menos invasivos para condiciones que antes eran incurables."
]

# Cargar pipelines con optimizaciones
try:
    # Usar modelos más pequeños si no hay GPU
    summarizer_model = "facebook/bart-large-cnn" if torch.cuda.is_available() else "sshleifer/distilbart-cnn-12-6"
    
    summarizer = pipeline(
        "summarization", 
        model=summarizer_model,
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    classifier = pipeline(
        "zero-shot-classification",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    print("Todos los modelos cargados exitosamente!\n")
except Exception as e:
    print(f"Error cargando modelos optimizados: {e}")
    print("Usando modelos estándar...")
    summarizer = pipeline("summarization")
    sentiment_analyzer = pipeline("sentiment-analysis")
    classifier = pipeline("zero-shot-classification")

# Procesar cada texto
for i, texto in enumerate(textos, 1):
    print(f"\n{'='*50}")
    print(f"TEXTO {i} - Análisis Completo")
    print(f"{'='*50}")
    print(f"Texto original ({len(texto)} caracteres):")
    print(f"{texto[:100]}...\n")
    
    start_time = time.time()
    
    try:
        # 1. Resumen
        print("RESUMEN:")
        max_len = min(80, len(texto.split()) // 2)
        min_len = max(20, max_len // 3)
        
        resumen = summarizer(texto, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
        print(f"   {resumen}\n")
        
        # 2. Sentimiento
        print("SENTIMIENTO:")
        sentimiento = sentiment_analyzer(texto)[0]
        print(f"   {sentimiento['label']} (confianza: {sentimiento['score']:.3f})\n")
        
        # 3. Clasificación temática
        print("CLASIFICACIÓN TEMÁTICA:")
        labels = ["tecnología", "política", "salud", "economía", "educación", "medio ambiente"]
        result = classifier(texto, candidate_labels=labels)
        
        print(f"   Tema principal: {result['labels'][0]} (confianza: {result['scores'][0]:.3f})")
        print(f"   Temas secundarios:")
        for label, score in zip(result['labels'][1:3], result['scores'][1:3]):
            print(f"      - {label}: {score:.3f}")
        
        processing_time = time.time() - start_time
        print(f"\nTiempo de procesamiento: {processing_time:.2f} segundos")
        
    except Exception as e:
        print(f"Error procesando texto {i}: {e}")

if torch.cuda.is_available():
    print(f"\nMemoria GPU utilizada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

print("\nPipeline NLP completo ejecutado exitosamente!")

# Resultado esperado: Todo el pipeline NLP funcionando como una miniapp.

# Preguntas:
# 1. Qué tarea resultó más precisa: ¿el resumen, la clasificación o el análisis de sentimiento?
# 2. ¿Qué tan bien se adaptaron los modelos preentrenados a tu texto personalizado?
# 3. ¿Cómo integrarías este pipeline en una aplicación web real?
# 4. ¿Qué parte del pipeline automatizarías o optimizarías con otra herramienta?
# 5. ¿Qué mejoras podrías hacer si el texto estuviera en otro idioma o jerga regional?

# Sugerencias de mejora:
# • Dar más detalles sobre el modelo de clasificación zero-shot.
# • Proponer extensión del proyecto como miniAPI web o app de análisis textual.