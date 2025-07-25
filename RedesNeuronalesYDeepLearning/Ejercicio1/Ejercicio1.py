# Ejercicio 1: Clasificación Binaria con Perceptrón Multicapa (MLP)
# Contexto: Clasificación de correos electrónicos como spam o no spam.
# Conceptos clave:
# • Perceptrón multicapa (MLP)
# • Funciones de activación (ReLU, Sigmoid)
# • Función de pérdida binaria (binary_crossentropy)
#
# Descripción del desarrollo:
# Modelo MLP con TensorFlow/Keras usando datos simulados con sklearn.

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(16, activation='relu', input_shape=(20,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Preguntas para reforzar el aprendizaje:
# • ¿Por qué se usa la función sigmoid en la salida?
# • ¿Cómo afecta la arquitectura (número de capas y neuronas) al rendimiento?