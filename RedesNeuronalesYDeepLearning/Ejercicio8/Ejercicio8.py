# Ejercicio 8: Comparación de Optimizadores y Funciones de Pérdida
# Contexto: Determinar qué combinación produce mejores resultados.
# Conceptos clave:
# • Adam vs SGD vs RMSprop
# • Funciones de pérdida: MSE, MAE, Categorical CE
# • Evaluación comparativa
#
# Descripción del desarrollo:
# Modelo entrenado múltiples veces cambiando optimizador y función de pérdida.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Datos simulados
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Combinaciones a evaluar
optimizers = ['adam', 'sgd', 'rmsprop']
losses = ['binary_crossentropy', 'mean_squared_error', 'mean_absolute_error']
results = []

# Entrenamiento y evaluación
for opt in optimizers:
    for loss_fn in losses:
        model = Sequential([
            Dense(32, activation='relu', input_shape=(20,)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, verbose=0)
        final_val_acc = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        print(f"Optimizador: {opt}, Pérdida: {loss_fn}, Val_Acc: {final_val_acc:.4f}, Val_Loss: {final_val_loss:.4f}")
        results.append((opt, loss_fn, final_val_acc, final_val_loss))

# Visualización sugerida (requiere pandas, seaborn)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(results, columns=["Optimizador", "Pérdida", "Precisión", "Pérdida final"])
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="Optimizador", y="Precisión", hue="Pérdida")
plt.title("Comparación de optimizadores y funciones de pérdida")
plt.ylim(0, 1)
plt.show()

# Preguntas para reforzar el aprendizaje:
# • ¿Qué combinación fue más rápida en converger?
# • ¿Cuál obtuvo mejor precisión final?