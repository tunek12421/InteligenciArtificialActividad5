# Ejercicio 7: Visualización de Función de Pérdida y Activaciones
# Contexto: Comprender el comportamiento interno del modelo durante el entrenamiento.
# Conceptos clave:
# • Visualización de activaciones
# • Historial de pérdida y precisión
# • Interpretabilidad de modelos
#
# Descripción del desarrollo:
# Durante el entrenamiento de redes neuronales, es fundamental monitorear el desempeño del
# modelo y entender el comportamiento interno de las capas. Este ejercicio permite visualizar
# cómo aprende un modelo, detectar sobreajuste y observar activaciones intermedias.
# Requisitos
# • TensorFlow / Keras
# • Matplotlib
# • Numpy
# • Dataset simulado (o puede integrarse con ejercicios anteriores)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Generar datos simulados
X, y = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Normalización
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Construcción del modelo
model = Sequential([
    Dense(32, activation='relu', input_shape=(20,), name="capa_1"),
    Dropout(0.2),
    Dense(16, activation='relu', name="capa_2"),
    Dense(1, activation='sigmoid', name="salida")
])
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 4. Entrenamiento y visualización de pérdida
history = model.fit(X_train, y_train, validation_split=0.2, epochs=15)

# Gráfico de pérdida
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.title('Evolución de la Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()

# 5. Visualización de activaciones internas (solo para una muestra)
from tensorflow.keras import backend as K

# Crear un modelo que devuelva salidas intermedias
layer_outputs = [layer.output for layer in model.layers if 'Dense' in layer.__class__.__name__]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Seleccionar una muestra aleatoria
sample = X_test[0].reshape(1, -1)

# Obtener activaciones
activations = activation_model.predict(sample)

# Mostrar activaciones por capa
for i, activation in enumerate(activations):
    plt.figure(figsize=(6, 1))
    plt.title(f"Activación de la capa {i+1}")
    plt.imshow(activation, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Preguntas para reforzar el aprendizaje
# 1. ¿Qué parones observas en la evolución de la pérdida? ¿Hay indicios de sobreajuste?
# 2. ¿Cómo interpretas las activaciones internas? ¿Qué ocurre si la entrada cambia?
# 3. ¿Cuál es el impacto del Dropout en las activaciones?
# 4. ¿Cómo podrías visualizar activaciones en una red convolucional?