from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# 1. Veri ve Model Hazırlığı
dataset = load_dataset("savasy/ttc4900")
model_name = "dbmdz/bert-base-turkish-cased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Tokenizasyon ve Etiket Hazırlığı
def tokenize_function(examples):
    # Metni işle
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True)
    # BERT 'labels' isminde bir sütun bekler, 'category' değerlerini oraya aktarıyoruz
    tokenized["labels"] = examples["category"]
    return tokenized

print("Veriler BERT için hazırlanıyor...")
# Sütun isimlerini BERT'e göre ayarlıyoruz
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Eğitimi hızlandırmak için küçük bir parça seçiyoruz
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(300)) # 500'den 300'e indirdim daha hızlı olsun
small_eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(300, 400))

# 3. Modeli Yükleme
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)

# 4. Metrik Hesaplama
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. Eğitim Ayarları
training_args = TrainingArguments(
    output_dir="test_trainer", 
    eval_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=4, # Hafıza hatası almamak için 8'den 4'e düşürdüm
    save_strategy="no"
)

# 6. Eğitici (Trainer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

print("BERT Fine-tuning başlıyor (Lütfen bekleyin)...")
trainer.train()

print("\nBERT Başarı Sonucu:")
print(trainer.evaluate())