import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

# 1. Load Synthetic Data
print("Loading synthetic dataset...")
try:
    df = pd.read_csv("data/email_reply_dataset/synthetic_reply_dataset.csv")
except FileNotFoundError:
    print("Error: 'synthetic_reply_dataset.csv' not found. Please run 'create_reply_dataset.py' first!")
    exit(1)

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1)

# 2. Prepare Model & Tokenizer
# We use DistilGPT-2 for a lightweight generation model
model_name = "gpt2"
print(f"Loading model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 doesn't have a pad token by default, so we use the EOS token
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    inputs = examples["Message"]
    targets = examples["reply"]
    
    # Format the text for Causal Language Modeling (CLM)
    # The model learns to predict the reply given the email
    # Format: "Email: [Body] \n\nReply: [Reply]"
    full_texts = []
    for i, t in zip(inputs, targets):
        # Ensure strings
        i = str(i) if i is not None else ""
        t = str(t) if t is not None else ""
        
        text = f"Email: {i[:512]}\n\nReply: {t}{tokenizer.eos_token}"
        full_texts.append(text)
    
    return tokenizer(full_texts, truncation=True, padding="max_length", max_length=256)

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Training Arguments
training_args = TrainingArguments(
    output_dir="./reply_model_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Keep small for commodity hardware
    per_device_eval_batch_size=4,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
)

# 4. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 5. Train
print("Starting training...")
trainer.train()

# 6. Save
output_dir = "./reply_generator_model"
print(f"Saving model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done!")
