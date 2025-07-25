# Test rápido para verificar que GPU funciona correctamente
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Dataset pequeño para test rápido
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

# Solo 100 ejemplos para test rápido
encoded = dataset.map(tokenize_fn, batched=True)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model = model.to(device)

# Configuración mínima para test
training_args = TrainingArguments(
    output_dir="./test_results", 
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    eval_strategy="epoch",
    logging_steps=10,
    fp16=True,
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=encoded["train"].select(range(100)),
    eval_dataset=encoded["test"].select(range(50))
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Iniciando test de entrenamiento...")
trainer.train()

print("\nTest completado exitosamente!")
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results.get('eval_accuracy', 'N/A')}")

if torch.cuda.is_available():
    print(f"Memoria GPU utilizada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")