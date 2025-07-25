# Ejercicio 2: Cargar GloVe y realizar similitud semántica - OPTIMIZADO
# Objetivo: Utilizar embeddings preentrenados con GloVe para comparar palabras.
# Herramientas: gensim, numpy
# Dataset: glove.6B.100d.txt (disponible en Kaggle)

import os
import time
from gensim.models import KeyedVectors

glove_path = '../Dataset/glove.6B.100d.txt'

# Verificar que el archivo existe
if not os.path.exists(glove_path):
    print(f"ERROR: No se encontró el archivo {glove_path}")
    print("Asegúrate de haber descargado glove.6B.100d.txt en la carpeta Dataset/")
    exit(1)

print("Cargando modelo GloVe... (esto puede tardar unos momentos)")
start_time = time.time()

# Cargar con configuraciones optimizadas
glove_model = KeyedVectors.load_word2vec_format(
    glove_path, 
    binary=False,
    no_header=True,
    limit=100000  # Limitar vocabulario para carga más rápida
)

load_time = time.time() - start_time
print(f"Modelo cargado en {load_time:.2f} segundos")
print(f"Vocabulario: {len(glove_model)} palabras")

# Similitud entre pares de palabras con manejo de errores
word_pairs = [
    ('king', 'queen'),
    ('cat', 'banana'),
    ('computer', 'technology'),
    ('happy', 'joy'),
    ('car', 'vehicle')
]

print("\n--- Análisis de Similitudes ---")
for word1, word2 in word_pairs:
    try:
        similarity = glove_model.similarity(word1, word2)
        print(f"Similitud {word1}-{word2}: {similarity:.4f}")
    except KeyError as e:
        print(f"Palabra no encontrada en vocabulario: {e}")

# Ejemplo de analogía: king - man + woman ≈ queen
print("\n--- Analogías ---")
try:
    result = glove_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    print(f"king - man + woman = {result[0][0]} (score: {result[0][1]:.4f})")
except KeyError as e:
    print(f"Error en analogía: {e}")

# Resultado esperado: Diferencias semánticas evidentes.

# Preguntas:
# 1. ¿Cuál es la diferencia entre GloVe y Word2Vec en cuanto a su forma de entrenamiento?
# 2. ¿Por qué usamos KeyedVectors en este ejercicio?
# 3. ¿Qué resultados obtuviste al comparar "king" y "queen"? ¿Qué interpretas?
# 4. ¿Puedes mencionar un caso donde el análisis semántico con GloVe sería útil en la
# industria?
# 5. ¿Qué limitaciones tienen los embeddings estáticos como GloVe?

# Sugerencias de mejora:
# • Mostrar un ejemplo de analogía (king - man + woman ≈ queen) sería didáctico.
# • Agregar comentarios sobre la dimensionalidad de los vectores.