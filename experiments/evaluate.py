# evaluate.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Optional

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

from openai import OpenAI
import openai


# =========================================================
#            LLM API 调用：Qwen (DashScope 兼容模式)
# =========================================================

def call_llm_api(prompt: str, model_name: str, api_key: str) -> str:
    """
    调用 Qwen 模型进行评分，API Key 通过命令行传入。
    """
    if not api_key:
        raise ValueError("API Key is empty. Please pass --api_key <key>")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Qwen3 模型需要 extra_body={"enable_thinking": False}
    extra_kwargs = {}
    if model_name.startswith("qwen3"):
        extra_kwargs["extra_body"] = {"enable_thinking": False}

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=512,
        **extra_kwargs,
    )

    message = completion.choices[0].message
    content = message.content

    if isinstance(content, list):  # 兼容某些模型返回列表结构
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)

    return content


# =========================================================
#                 Part 1: Spam Classification
# =========================================================

def run_spam_pipeline(
    model_dir: str,
    test_csv: str,
    pred_csv: str,
    text_column: str = "text",
    label_column: str = "label",
    max_length: int = 128,
    batch_size: int = 32,
):
    """
    整体流程：
    1）加载微调好的 BERT（或其他 HF 分类模型）
    2）对测试集 test_csv 做预测，写出 pred_csv（新增 pred_label 列）
    3）计算 Accuracy & Macro F1，并打印

    参数：
        model_dir: 保存了模型权重和 tokenizer 的目录（save_pretrained 结果）
        test_csv: 测试集 CSV，必须包含 text_column 和 label_column
        pred_csv: 输出预测结果的 CSV 路径
        text_column: 文本列名（例如 "data" 或 "text"）
        label_column: 标签列名（例如 "Spam/Ham" 或 "label"）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Spam] Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    print(f"[Spam] Loading test data from: {test_csv}")
    df = pd.read_csv(test_csv)

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Test CSV must contain columns '{text_column}' and '{label_column}'. "
            f"Found columns: {df.columns.tolist()}"
        )

    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].tolist()

    all_preds = []

    print(f"[Spam] Running inference on {len(df)} examples...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i: i + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

        all_preds.extend(preds)

    # 保存预测结果
    df["pred_label"] = all_preds
    os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
    df.to_csv(pred_csv, index=False, encoding="utf-8")
    print(f"[Spam] Saved predictions to: {pred_csv}")

    # 计算指标（支持字符串标签或数字标签）
    acc = accuracy_score(labels, all_preds)
    macro_f1 = f1_score(labels, all_preds, average="macro")

    print(f"[Spam] Accuracy : {acc:.4f}")
    print(f"[Spam] Macro F1 : {macro_f1:.4f}")

    return {"accuracy": acc, "macro_f1": macro_f1}


# =========================================================
#           Part 2: Ham Reply Generation + Scoring
# =========================================================

def build_scoring_prompt(
    incoming_email: str,
    model_reply: str,
    reference_reply: Optional[str] = None,
) -> str:
    """
    构造给 Qwen 的评分 prompt。
    让模型对 model_reply 从三个维度打 1–5 分：
      - Fluency
      - Relevance
      - Politeness
    返回要求 JSON 格式，便于解析。
    """
    ref_part = ""
    if reference_reply is not None and isinstance(reference_reply, str) and reference_reply.strip():
        ref_part = f"\n\nReference human reply (for your reference, DO NOT score this):\n{reference_reply}"

    prompt = f"""
You are an expert email writing evaluator.

I will give you:
1) An incoming email.
2) A model-generated reply.
{ '3) A reference human reply.' if ref_part else '' }

Please evaluate ONLY the model-generated reply on three dimensions, using a 1–5 point Likert scale (integers only):

- Fluency: Is the reply grammatically correct and natural?
- Relevance: Does the reply correctly address the content and intent of the incoming email?
- Politeness: Is the reply polite and appropriate in tone?

Return your answer as a JSON object ONLY, with this exact format:

{{
  "fluency": <integer 1-5>,
  "relevance": <integer 1-5>,
  "politeness": <integer 1-5>
}}

Do not add any explanation or extra text outside the JSON.

Incoming email:
{incoming_email}

Model-generated reply:
{model_reply}
{ref_part}
"""
    return prompt.strip()


def parse_llm_scores(raw_text: str) -> dict:
    """
    解析 Qwen 返回的 JSON 字符串，例如：

    {
      "fluency": 4,
      "relevance": 5,
      "politeness": 4
    }

    返回一个 dict，键为 fluency / relevance / politeness，值为 int(1–5)。
    """
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # 尝试从第一个 { 到最后一个 } 之间截断
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = raw_text[start: end + 1]
            data = json.loads(json_str)
        else:
            raise

    scores = {
        "fluency": int(data["fluency"]),
        "relevance": int(data["relevance"]),
        "politeness": int(data["politeness"]),
    }
    return scores


def run_ham_reply_pipeline(
    gpt2_model_dir: str,
    input_csv: str,
    output_csv: str,
    llm_model_name: str,
    api_key: str,
    spam_label_column: str = "Spam/Ham",
    text_column: str = "data",
    max_input_length: int = 256,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_samples_for_scoring: Optional[int] = None,
):
    """
    针对“混合 spam/ham 测试集”（例如 6k.csv）的专用 pipeline：

    1）从 input_csv 中读取所有样本；
    2）只保留标签为 ham 的行：
       - 如果 spam_label_column 是字符串列，则筛选值为 "ham"（忽略大小写）
       - 如果是数字列，则默认假定 0=ham, 1=spam（如有不同，可以改这里逻辑）
    3）用 GPT-2 对每条 ham 的 text_column 生成回复 -> 写入 'response'；
    4）调用 Qwen 对每条 response 打 Fluency / Relevance / Politeness（三列）；
    5）保存结果到 output_csv。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 读取测试集并筛选 ham
    print(f"[HamReply] Loading test data from: {input_csv}")
    df = pd.read_csv(input_csv)

    if spam_label_column not in df.columns or text_column not in df.columns:
        raise ValueError(
            f"Input CSV must contain columns '{spam_label_column}' and '{text_column}'. "
            f"Found columns: {df.columns.tolist()}"
        )

    label_series = df[spam_label_column]

    # 根据列类型自动判断如何筛选 ham
    if label_series.dtype == object:
        # 字符串标签：例如 "ham"/"spam"
        df_ham = df[label_series.str.lower() == "ham"].copy()
    else:
        # 数字标签：假设 0=ham, 1=spam（如果你用的是别的约定，可以改这里）
        df_ham = df[label_series == 0].copy()

    df_ham.reset_index(drop=True, inplace=True)

    # 固定随机抽样 400 条（用于 reply generation 和 scoring）
    SAMPLE_SIZE = 400
    if len(df_ham) > SAMPLE_SIZE:
        df_ham = df_ham.sample(n=SAMPLE_SIZE, random_state=42, replace=False).copy()
        df_ham = df_ham.reset_index(drop=True)

    print(f"[HamReply] Using {len(df_ham)} ham emails (random sample of 400).")
    # 如果一个 ham 都没有，就不往下跑了，避免除以 0
    if len(df_ham) == 0:
        print("[HamReply] No ham examples found. Please check your label column and values.")
        # 仍然输出一个空的 CSV，方便你检查
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_ham.to_csv(output_csv, index=False, encoding="utf-8")
        return

    # 2. 加载 GPT-2，用于生成回复
    print(f"[HamReply] Loading GPT-2 model from: {gpt2_model_dir}")
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_dir)
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_dir)
    gpt2_model.to(device)
    gpt2_model.eval()

    responses = []

    print("[HamReply] Generating responses with GPT-2...")
    for _, row in tqdm(df_ham.iterrows(), total=len(df_ham)):
        incoming = str(row[text_column])

        prompt = (
            "Email:\n"
            + incoming
            + "\n\nReply:\n"
        )

        enc = gpt2_tokenizer(
            prompt,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            output_ids = gpt2_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=gpt2_tokenizer.pad_token_id,
            )

        generated_text = gpt2_tokenizer.decode(
            output_ids[0][enc["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        response = generated_text.strip()
        responses.append(response)

    df_ham["response"] = responses

    # 先把仅包含 response 的中间结果落盘，避免数据丢失
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_ham.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[HamReply] Saved intermediate HAM + responses (no scores yet) to: {output_csv}")


    # 3. 调用 Qwen 对每条 response 打分（3 个维度）
    fluency_scores = []
    relevance_scores = []
    politeness_scores = []

    print(f"[HamReply] Scoring responses via LLM: {llm_model_name}")
    for idx, row in tqdm(df_ham.iterrows(), total=len(df_ham)):
        incoming = str(row[text_column])
        model_reply = str(row["response"])

        prompt = build_scoring_prompt(
            incoming_email=incoming,
            model_reply=model_reply,
            reference_reply=None,  # 对 6k 测试集通常没有参考人工回复
        )

        try:
            raw = call_llm_api(prompt, model_name=llm_model_name, api_key=api_key)
            scores = parse_llm_scores(raw)

            fluency_scores.append(scores["fluency"])
            relevance_scores.append(scores["relevance"])
            politeness_scores.append(scores["politeness"])

        except openai.BadRequestError as e:
            # 这里专门处理 data_inspection_failed 或其他请求错误
            print(f"[HamReply][WARN] Skipping sample at index {idx} due to API error:")
            print(f"  {e}")
            # 用 None 占位，后面计算均值时会忽略这些 None
            fluency_scores.append(None)
            relevance_scores.append(None)
            politeness_scores.append(None)
            continue

    df_ham["fluency_score"] = fluency_scores
    df_ham["relevance_score"] = relevance_scores
    df_ham["politeness_score"] = politeness_scores

    # 4. 保存结果
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_ham.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[HamReply] Saved ham + responses + scores to: {output_csv}")

    # 计算均值时忽略 None / Compute means ignoring None
    valid_flu = [s for s in fluency_scores if s is not None]
    valid_rel = [s for s in relevance_scores if s is not None]
    valid_pol = [s for s in politeness_scores if s is not None]

    print("[HamReply] Mean scores on ham emails (excluding skipped samples):")
    if valid_flu:
        print(f"  Fluency    : {sum(valid_flu) / len(valid_flu):.3f}")
    else:
        print("  Fluency    : N/A")

    if valid_rel:
        print(f"  Relevance  : {sum(valid_rel) / len(valid_rel):.3f}")
    else:
        print("  Relevance  : N/A")

    if valid_pol:
        print(f"  Politeness : {sum(valid_pol) / len(valid_pol):.3f}")
    else:
        print("  Politeness : N/A")

# =========================================================
#                         CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run inference + evaluation for spam classification and ham reply generation."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    # -------- spam classification pipeline --------
    spam_p = subparsers.add_parser(
        "spam",
        help="Run spam classification pipeline (predict + evaluate)",
    )
    spam_p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to fine-tuned classifier model dir",
    )
    spam_p.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to test CSV (e.g., 6k.csv)",
    )
    spam_p.add_argument(
        "--pred_csv",
        type=str,
        required=True,
        help="Where to save predictions CSV",
    )
    spam_p.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column in test_csv",
    )
    spam_p.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label column in test_csv",
    )
    spam_p.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length for the classifier",
    )
    spam_p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for classifier inference",
    )

    # -------- ham reply generation + scoring pipeline --------
    ham_reply_p = subparsers.add_parser(
        "ham_reply",
        help="Use HAM emails from mixed spam/ham test set to generate GPT-2 responses and score with Qwen.",
    )
    ham_reply_p.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="DashScope (Qwen) API Key",
    )
    ham_reply_p.add_argument(
        "--gpt2_model_dir",
        type=str,
        required=True,
        help="Path to fine-tuned GPT-2 (or other causal LM) model dir",
    )
    ham_reply_p.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to mixed spam/ham test CSV (e.g., 6k.csv)",
    )
    ham_reply_p.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Where to save HAM emails + generated responses + scores CSV",
    )
    ham_reply_p.add_argument(
        "--llm_model_name",
        type=str,
        required=True,
        help="Qwen model name for scoring, e.g. 'qwen3-14b'",
    )
    ham_reply_p.add_argument(
        "--spam_label_column",
        type=str,
        default="Spam/Ham",
        help="Column name indicating spam/ham labels in input_csv",
    )
    ham_reply_p.add_argument(
        "--text_column",
        type=str,
        default="data",
        help="Column name containing email text in input_csv",
    )
    ham_reply_p.add_argument(
        "--max_input_length",
        type=int,
        default=256,
        help="Max sequence length for GPT-2 input",
    )
    ham_reply_p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens for GPT-2 generation",
    )
    ham_reply_p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for GPT-2 generation",
    )
    ham_reply_p.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p for GPT-2 generation",
    )
    ham_reply_p.add_argument(
        "--max_samples_for_scoring",
        type=int,
        default=None,
        help="Optionally limit the number of HAM emails to score (for quick tests)",
    )

    args = parser.parse_args()

    if args.task == "spam":
        run_spam_pipeline(
            model_dir=args.model_dir,
            test_csv=args.test_csv,
            pred_csv=args.pred_csv,
            text_column=args.text_column,
            label_column=args.label_column,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )

    elif args.task == "ham_reply":
        run_ham_reply_pipeline(
            gpt2_model_dir=args.gpt2_model_dir,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            llm_model_name=args.llm_model_name,
            api_key=args.api_key,
            spam_label_column=args.spam_label_column,
            text_column=args.text_column,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_samples_for_scoring=args.max_samples_for_scoring,
        )

    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
