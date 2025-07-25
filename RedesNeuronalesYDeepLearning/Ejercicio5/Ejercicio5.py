# Ejercicio 5: Clasificación de Sentimientos con LSTM
# Contexto: Análisis de opiniones de clientes en una tienda online (positivo/negativo).
# Conceptos clave:
# • Procesamiento secuencial de texto
# • Tokenización y secuencias
# • Embeddings y redes LSTM
#
# Descripción del desarrollo:
# Uso de Tokenizer, pad_sequences y modelo LSTM para texto.

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

texts = ['Me encantó el producto', 'No me gustó nada', 'Excelente servicio', 'Terrible atención']
labels = [1, 0, 1, 0]

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=10)
y = np.array(labels)

model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=10),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10)

# Preguntas para reforzar el aprendizaje:
# • ¿Por qué es importante el preprocesamiento del texto?
# • ¿Qué significa la capa Embedding?