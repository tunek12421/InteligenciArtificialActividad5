# Ejercicio 2: Clasificación de Imágenes con CNN (MNIST)
# Contexto: Reconocimiento automático de dígitos manuscritos para sistemas bancarios.
# Conceptos clave:
# • Redes convolucionales
# • Pooling y flattening
# • Softmax para clasificación multiclase
#
# Descripción del desarrollo:
# CNN simple entrenado con el dataset MNIST.

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_split=0.2)

# Preguntas para reforzar el aprendizaje:
# • ¿Qué ventajas ofrece el uso de convoluciones en vez de capas densas?
# • ¿Qué pasaría si eliminamos la capa de pooling?