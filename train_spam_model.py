import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# 1. Load Dataset
# Load from CSV file
df = pd.read_csv("data/classification/enron_spam_data.csv")
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

# 2. Preprocess Data
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Convert labels to numeric (ham=0, spam=1)
def prepare_dataset(example):
    example["label"] = 1 if example["Spam/Ham"].lower() == "spam" else 0
    return example

dataset = dataset.map(prepare_dataset)

def preprocess_function(examples):
    # Handle None/empty messages by replacing with empty string
    messages = [msg if msg is not None else "" for msg in examples["Message"]]
    return tokenizer(messages, truncation=True, padding="max_length")

# Remove rows with empty messages or missing labels
dataset = dataset.filter(lambda x: x["Message"] is not None and len(str(x["Message"]).strip()) > 0)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Map labels
id2label = {0: "HAM", 1: "SPAM"}
label2id = {"HAM": 0, "SPAM": 1}

# 3. Load Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

# 4. Define Metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7. Train
print("Starting training...")
trainer.train()

# 8. Evaluate
print("Evaluating...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 9. Save Model
trainer.save_model("./spam_classifier_model")
print("Model saved to ./spam_classifier_model")
