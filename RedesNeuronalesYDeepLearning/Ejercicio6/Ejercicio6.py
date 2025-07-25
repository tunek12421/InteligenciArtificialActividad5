# Ejercicio 6: Técnicas de Regularización y Validación
# Contexto: Evitar el sobreajuste en la predicción de abandono de clientes.
# Conceptos clave:
# • Dropout
# • Early Stopping
# • Batch Normalization
#
# Descripción del desarrollo:
# Modelo con regularización usando técnicas combinadas en Keras.

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Generar datos simulados
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit(X_train, y_train, validation_split=0.2, epochs=20, callbacks=[early_stop])

# Preguntas para reforzar el aprendizaje:
# • ¿Qué técnica fue más efectiva para evitar el sobreajuste?
# • ¿Qué significa el parámetro patience en EarlyStopping?