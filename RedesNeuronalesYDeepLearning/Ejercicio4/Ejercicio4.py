# Ejercicio 4: Predicción de Ventas con LSTM
# Contexto: Predicción mensual de ventas en un supermercado.
# Conceptos clave:
# • Series temporales
# • LSTM para datos secuenciales
# • Normalización de datos
#
# Descripción del desarrollo:
# LSTM que predice valores futuros en una serie generada artificialmente.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generar datos simulados de ventas (serie temporal)
np.random.seed(0)
data = np.sin(np.linspace(0, 100, 300)) + np.random.normal(0, 0.1, 300)

# Escalar los datos entre 0 y 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.reshape(-1, 1))

# Crear ventanas de tiempo (X: 10 pasos anteriores, y: siguiente paso)
X, y = [], []
for i in range(len(scaled_data) - 10):
    X.append(scaled_data[i:i+10])
    y.append(scaled_data[i+10])
X, y = np.array(X), np.array(y)

# Ajustar la forma para LSTM: (muestras, pasos de tiempo, características)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Crear modelo LSTM
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(10, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(X, y, epochs=20, batch_size=16, validation_split=0.2)

# Visualizar pérdida
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante entrenamiento LSTM')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Preguntas para reforzar el aprendizaje:
# • ¿Cuál es la diferencia práctica entre RNN y LSTM?
# • ¿Por qué es importante normalizar series temporales?