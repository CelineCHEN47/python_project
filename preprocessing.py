# data_preprocess.py

from datasets import load_dataset, Dataset
import argparse
import os

def create_email_style_pairs(dialogs, style="support"):
    """
    将对话列表转换为 email-style prompt-response 对。
    
    Args:
        dialogs: List of dialog lists, e.g., [["Hi", "Hello"], ["How are you?", "Fine"]]
        style: "support" (Customer/Support) or "chat" (User/Assistant)
    
    Returns:
        List of {"text": "..."} for causal LM training
    """
    examples = []
    for dialog in dialogs:
        if len(dialog) < 2:
            continue
        for i in range(len(dialog) - 1):
            if style == "support":
                input_text = f"Customer: {dialog[i]}"
                response_text = f"Support: {dialog[i + 1]}"
            else:
                input_text = f"User: {dialog[i]}"
                response_text = f"Assistant: {dialog[i + 1]}"
            
            # 合并为一条训练样本（模型学习生成 response）
            full_text = f"{input_text}\n{response_text}"
            examples.append({"text": full_text})
    return examples

def main(args):
    print(f"Loading dataset: {args.dataset_name}")
    
    # 加载数据集
    raw_dataset = load_dataset(args.dataset_name)
    
    # 提取训练集对话（假设字段名为 'dialog'）
    train_dialogs = [ex["dialog"] for ex in raw_dataset["train"]]
    
    # 转换为 email-style pairs
    print("Converting to email-style prompt-response pairs...")
    processed_examples = create_email_style_pairs(train_dialogs, style=args.style)
    
    # 创建新 Dataset
    processed_dataset = Dataset.from_list(processed_examples)
    
    # 保存到磁盘
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    processed_dataset.save_to_disk(output_dir)
    
    print(f"✅ Saved {len(processed_dataset)} samples to {output_dir}")
    print("Sample:")
    print(processed_dataset[0]["text"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DailyDialog for email reply generation")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="kmyoo/dailydialog-tiny",
        help="Hugging Face dataset name (e.g., 'daily_dialog' or 'kmyoo/dailydialog-tiny')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/larger_dataset",
        help="Directory to save processed dataset"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="support",
        choices=["support", "chat"],
        help="Prompt style: 'support' (Customer/Support) or 'chat' (User/Assistant)"
    )
    args = parser.parse_args()
    main(args)