# Ejercicio 3: Crear embeddings personalizados de un corpus - OPTIMIZADO
# Objetivo: Generar embeddings personalizados a partir de texto local (dataset propio).
# Herramientas: gensim, nltk
# Dataset: Medium Articles (with Content) - AI/ML/DS de Kaggle

import pandas as pd
import nltk
import re
import multiprocessing
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec

print("Verificando recursos de NLTK...")
nltk.download('punkt', quiet=True)

# Verificar que el dataset existe
dataset_path = '../Dataset/Medium_AggregatedData.csv'
try:
    print("Cargando dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset cargado: {len(df)} artículos")
except FileNotFoundError:
    print(f"ERROR: No se encontró {dataset_path}")
    print("Usando texto de ejemplo para demostración...")
    df = pd.DataFrame({
        'text': [
            "Machine learning algorithms are transforming artificial intelligence applications.",
            "Deep learning models require large datasets for optimal performance.",
            "Natural language processing enables computers to understand human language.",
            "Neural networks are inspired by biological brain structures.",
            "Data science combines statistics, programming, and domain expertise."
        ] * 100  # Repetir para crear un corpus más grande
    })

# Optimizar muestra para velocidad
sample_size = min(500, len(df))  # Máximo 500 artículos
df_sample = df.head(sample_size)

print(f"Procesando {len(df_sample)} artículos...")

# Limpiar y procesar texto de forma más eficiente
def clean_text(text):
    if pd.isna(text):
        return ""
    # Limpiar texto básico
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Procesar en lotes para mejor rendimiento
corpus_texts = []
for i, text in enumerate(df_sample['text']):
    if i % 100 == 0:
        print(f"Procesado: {i}/{len(df_sample)}")
    clean = clean_text(text)
    if clean:
        corpus_texts.append(clean)

print("Tokenizando corpus...")
tokens = []
for text in corpus_texts[:100]:  # Limitar para demo rápida
    sentences = sent_tokenize(text)
    for sent in sentences:
        words = [word for word in word_tokenize(sent) 
                if word.isalpha() and len(word) > 2]
        if len(words) > 3:  # Solo oraciones con suficientes palabras
            tokens.append(words)

print(f"Corpus preparado: {len(tokens)} oraciones")

# Optimizar entrenamiento
workers = multiprocessing.cpu_count()
print(f"Entrenando modelo con {workers} workers...")

model_custom = Word2Vec(
    tokens, 
    vector_size=100,  # Aumentado para mejor calidad
    window=5, 
    min_count=2,  # Mínimo 2 apariciones
    workers=workers,
    epochs=10,
    sg=1  # Skip-gram
)

print("Modelo entrenado exitosamente!")
print(f"Vocabulario: {len(model_custom.wv)} palabras")

# Probar similitudes con manejo de errores
test_words = ['intelligence', 'artificial', 'algorithm', 'machine', 'learning', 'data']

for word in test_words:
    try:
        print(f"\nPalabras similares a '{word}':")
        similar = model_custom.wv.most_similar(word, topn=3)
        for sim_word, score in similar:
            print(f"  {sim_word}: {score:.3f}")
    except KeyError:
        print(f"  Palabra '{word}' no encontrada en vocabulario")

# Mostrar muestra del vocabulario si no hay palabras conocidas
if len(model_custom.wv) > 0:
    print(f"\nMuestra del vocabulario entrenado:")
    vocab_sample = list(model_custom.wv.index_to_key)[:10]
    print(vocab_sample)

# Resultado esperado: Relación entre términos como "intelligence", "artificial", "algorithm".

# Preguntas:
# 1. ¿Por qué podrías preferir entrenar tus propios embeddings en vez de usar GloVe?
# 2. ¿Qué características del texto pueden afectar la calidad de los embeddings?
# 3. ¿Cómo se refleja el dominio del texto en los vectores obtenidos?
# 4. ¿Qué cambios harías para mejorar la calidad de tus embeddings?
# 5. ¿Qué usos prácticos tendría este modelo dentro de una empresa?

# Sugerencias de mejora:
# • Añadir análisis de frecuencia de palabras del corpus como preprocesamiento.
# • Comentar la limpieza del texto antes de tokenizar.