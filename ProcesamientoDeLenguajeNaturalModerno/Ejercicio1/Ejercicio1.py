# Ejercicio 1: Entrenamiento de Word2Vec desde cero - OPTIMIZADO
# Objetivo: Aprender a construir un modelo de Word Embeddings con Word2Vec.
# Herramientas: gensim, nltk
# Dataset: Corpus de reseñas de películas (nltk.corpus.movie_reviews)

import os
import multiprocessing
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import nltk

print("Descargando recursos de NLTK (si es necesario)...")
nltk.download('movie_reviews', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("Preparando corpus...")
# Usar solo una muestra para entrenamiento más rápido
fileids = movie_reviews.fileids()[:500]  # Reducido para velocidad
sentences = []

print(f"Procesando {len(fileids)} archivos...")
for i, fileid in enumerate(fileids):
    if i % 100 == 0:
        print(f"Procesado: {i}/{len(fileids)}")
    # Tokenizar y limpiar en un solo paso
    tokens = [token.lower() for token in word_tokenize(movie_reviews.raw(fileid)) 
              if token.isalpha() and len(token) > 2]
    sentences.append(tokens)

print("Entrenando modelo Word2Vec...")
# Usar todos los cores disponibles para acelerar
workers = multiprocessing.cpu_count()
model = Word2Vec(
    sentences, 
    vector_size=100, 
    window=5, 
    min_count=2, 
    workers=workers,
    epochs=5,
    sg=1  # Skip-gram es más rápido para corpus pequeños
)

print("Modelo entrenado exitosamente!")
print(f"Vocabulario: {len(model.wv)} palabras")

# Ejemplo: palabras similares a "good"
try:
    print("\nPalabras similares a 'good':")
    similar_words = model.wv.most_similar("good", topn=5)
    for word, similarity in similar_words:
        print(f"  {word}: {similarity:.3f}")
except KeyError:
    print("La palabra 'good' no está en el vocabulario. Probando con otras palabras...")
    # Mostrar algunas palabras del vocabulario
    vocab_sample = list(model.wv.index_to_key)[:10]
    print(f"Muestra del vocabulario: {vocab_sample}")
    if vocab_sample:
        test_word = vocab_sample[0]
        print(f"\nPalabras similares a '{test_word}':")
        similar_words = model.wv.most_similar(test_word, topn=5)
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.3f}")

print(f"\nTiempo de entrenamiento optimizado usando {workers} workers")

# Resultado esperado: vector de palabras similares a "good", como "great", "nice", etc.

# Preguntas:
# 1. ¿Qué representa un vector de palabras en Word2Vec?
# 2. ¿Cuál es la diferencia entre el enfoque CBOW y Skip-Gram?
# 3. ¿Qué significa que dos palabras tengan vectores "cercanos"?
# 4. ¿Cómo influye el parámetro window en el entrenamiento?
# 5. ¿Por qué es necesario hacer tokenización antes de entrenar?

# Sugerencias de mejora:
# • Agregar visualización de los vectores con TSNE o PCA para mayor comprensión.
# • Mencionar si se usa CBOW o Skip-Gram por defecto.