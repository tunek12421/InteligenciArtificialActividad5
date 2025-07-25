# Ejercicio 4: Clasificación de texto con BERT (Hugging Face) - OPTIMIZADO PARA GPU
# Objetivo: Clasificar reseñas de películas usando bert-base-uncased.
# Herramientas: transformers, datasets, sklearn
# Dataset: IMDb (disponible en datasets)

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np

# Verificar disponibilidad de GPU y configurar device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)  # Ultra rápido: tokens reducidos

encoded = dataset.map(tokenize_fn, batched=True)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Mover modelo a GPU
model = model.to(device)

# Función para calcular métricas
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Argumentos de entrenamiento optimizados para GPU GTX 1650 (4GB VRAM)
training_args = TrainingArguments(
    output_dir="./results", 
    per_device_train_batch_size=4,   # Reducido para GPU con poca memoria
    per_device_eval_batch_size=4,    # Reducido para evaluación
    gradient_accumulation_steps=4,   # Simula batch size de 16
    num_train_epochs=1,              # Ultra rápido: solo 1 época
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,                # Ultra rápido: log cada 10 pasos
    warmup_steps=5,                  # Ultra rápido: solo 5 pasos warmup
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    dataloader_pin_memory=True,      # Optimización para GPU
    fp16=True,                       # Precisión mixta para acelerar entrenamiento
    dataloader_num_workers=2,        # Optimización adicional
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=encoded["train"].select(range(100)),   # Ultra rápido: solo 100 muestras
    eval_dataset=encoded["test"].select(range(50)),      # Ultra rápido: solo 50 muestras
    compute_metrics=compute_metrics
)

# Limpiar memoria GPU antes del entrenamiento
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

print("Iniciando entrenamiento...")
trainer.train()

# Evaluar el modelo
print("\nEvaluando modelo...")
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results.get('eval_accuracy', 'N/A')}")
print(f"Loss: {eval_results.get('eval_loss', 'N/A')}")

# Mostrar información adicional sobre el entrenamiento
print(f"\nInformación del entrenamiento:")
print(f"Dispositivo utilizado: {device}")
if torch.cuda.is_available():
    print(f"Memoria GPU utilizada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memoria GPU máxima: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

# Resultado esperado: Modelo entrenado capaz de predecir sentimiento positivo/negativo.

# Preguntas:
# 1. ¿Cuál es el propósito del proceso de fine-tuning en BERT?
# 2. ¿Qué diferencias encontraste entre entrenar con un subconjunto pequeño vs. el dataset
# completo?
# 3. ¿Qué hace el tokenizer en el pipeline de Hugging Face?
# 4. ¿Qué métrica usarías para evaluar este modelo?
# 5. ¿Por qué es más eficaz BERT que un modelo tradicional como Naive Bayes para
# clasificación de texto?

# Sugerencias de mejora:
# • Agregar impresión de métricas (accuracy, f1, etc.) luego del trainer.evaluate().
# • Ofrecer opción con pipeline() para alumnos con menos experiencia.