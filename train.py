# train.py

import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    logging as hf_logging
)
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np

hf_logging.set_verbosity_info()

def tokenize_function(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding = False
    )

def compute_metrics(eval_pred, tokenizer, rouge_metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Decode predictions and labels (skip padding)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Filter out empty strings
    decoded_preds = [pred.strip() for pred in decoded_preds if pred.strip()]
    decoded_labels = [label.strip() for label in decoded_labels if label.strip()]
    
    if not decoded_preds or not decoded_labels:
        return {"rougeL": 0.0}
    
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"rougeL": result["rougeL"].mid.fmeasure}

def main(args):
    print(f"Loading dataset from {args.data_dir}")
    dataset = load_from_disk(args.data_dir)

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Critical for GPT-2

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto"  # 自动使用 GPU（如果可用）
    )

    # Apply LoRA
    if args.use_lora:
        print("Applying LoRA...")
        if "distilgpt2" in args.model_name.lower():
                target_modules = ["q_lin", "v_lin"]  
        else:
            target_modules = None  # 让 PEFT 自动处理 gpt2

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,  
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
        cache_file_name="./data/tokenized_dataset.arrow",
        num_proc=1
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # Causal LM, not masked LM
    )

    # Load ROUGE for evaluation
    rouge_metric = evaluate.load("rouge")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy="no",  # 不保存 checkpoint（节省磁盘）
        # evaluation_strategy="steps",
        evaluation_strategy="no",
        # eval_steps=50,
        fp16=args.fp16,
        report_to="none",
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  # 简化：用同一数据做 eval（实际应分 train/val）
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=lambda pred: compute_metrics(pred, tokenizer, rouge_metric)
        compute_metrics=None
    )

    print("Starting training...")
    if args.use_lora:
        print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    trainer.train()

    # Save final model
    if args.use_lora:
        model.save_pretrained(f"{args.output_dir}/lora_adapter")
        tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")
        print(f"LoRA adapter saved to {args.output_dir}/lora_adapter")
    else:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-2 model with LoRA for email reply generation")
    parser.add_argument("--data_dir", type=str, default="./data/email_reply_dataset", help="Path to preprocessed dataset")
    parser.add_argument("--model_name", type=str, default="gpt2", choices=["distilgpt2", "gpt2"], help="Base model")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory") 
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for PEFT")    # 注意 如果训练时候要使用lora 要显示指定 python train.py --use_lora  若不加则会默认全参数微调
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (recommended for GPU)")

    args = parser.parse_args()
    main(args)