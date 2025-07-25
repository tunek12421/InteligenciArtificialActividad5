# Ejercicio 9: Evaluación Avanzada del Modelo
# Contexto: Evaluación de un clasificador multiclase aplicado a imágenes.
# Conceptos clave:
# • Matriz de confusión
# • Curvas ROC y AUC
# • Precisión, recall y F1-score
#
# Descripción del desarrollo:
# Cálculo de métricas con sklearn.metrics y visualización.

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Cargar y preparar datos MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Crear y entrenar modelo simple
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=2, validation_split=0.1, verbose=0)  # Solo 2 epochs para demo

# Supongamos que tenemos 10 clases como en MNIST
n_classes = 10

# Suponiendo que y_test es categorical y y_pred_proba es la predicción por clase
y_pred_proba = model.predict(X_test) # salida softmax
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = y_test  # y_test ya son las etiquetas originales (0-9)

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Reporte de métricas
print(classification_report(y_true, y_pred))

# ROC AUC multiclase
# Necesitamos binarizar las etiquetas
y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
y_pred_bin = y_pred_proba # ya viene como probabilidades por clase

auc_score = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')
print("ROC AUC (macro average):", auc_score)

# Preguntas para reforzar el aprendizaje:
# • ¿Qué clases son más difíciles de predecir?
# • ¿Cómo interpretar un bajo F1-score?