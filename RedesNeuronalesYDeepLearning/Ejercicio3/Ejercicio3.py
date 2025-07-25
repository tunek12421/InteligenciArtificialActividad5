# Ejercicio 3: Reconocimiento de Patrones Médicos con CNN
# Contexto: Clasificación de radiografías pulmonares (neumonía vs normal).
# Conceptos clave:
# • CNN en imágenes médicas
# • Clasificación binaria en imágenes
# • Uso de generadores de datos
#
# Descripción del desarrollo:
# Uso de ImageDataGenerator para cargar imágenes médicas clasificadas.
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
train_generator = train_datagen.flow_from_directory(
    'archive/chest_xray/train', target_size=(150, 150), batch_size=32, class_mode='binary')

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10)

# Preguntas para reforzar el aprendizaje:
# • ¿Qué medidas éticas deben considerarse en el uso de IA médica?
# • ¿Cómo se mejora la precisión usando aumento de datos?