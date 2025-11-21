# data_augmentation.py
# -*- coding: utf-8 -*-

import argparse
import random
import pandas as pd
from typing import List, Tuple

import nltk
from nltk.corpus import wordnet, stopwords

# 第一次用时需要手动在 Python 里运行一次：
# import nltk
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")
# nltk.download("stopwords")


# ==============================
#  通用：简单 EDA 文本增强类
# ==============================

class TextAugmenter:
    """
    简单的英文文本增强器：
    - 同义词替换 (synonym replacement)
    - 随机删除 (random deletion)
    """

    def __init__(self,
                 language: str = "english",
                 random_seed: int = 42):
        random.seed(random_seed)
        self.stop_words = set(stopwords.words(language))

    def _tokenize(self, text: str) -> List[str]:
        return nltk.word_tokenize(text)

    def _detokenize(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def _get_synonyms(self, word: str) -> List[str]:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ").lower()
                # 去掉和原词一样的 / 带奇怪符号的
                if name != word.lower() and name.isalpha():
                    synonyms.add(name)
        return list(synonyms)

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        将句子中最多 n 个非停用词替换为同义词
        """
        tokens = self._tokenize(text)
        if not tokens:
            return text

        candidate_indices = [
            i for i, w in enumerate(tokens)
            if w.lower() not in self.stop_words and w.isalpha()
        ]
        if not candidate_indices:
            return text

        random.shuffle(candidate_indices)
        replaced = 0

        for idx in candidate_indices:
            word = tokens[idx]
            syns = self._get_synonyms(word)
            if not syns:
                continue
            tokens[idx] = random.choice(syns)
            replaced += 1
            if replaced >= n:
                break

        return self._detokenize(tokens)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        以概率 p 随机删除每个词，保证至少保留 1 个
        """
        tokens = self._tokenize(text)
        if len(tokens) <= 1:
            return text

        new_tokens = [w for w in tokens if random.random() > p]
        if not new_tokens:
            new_tokens = [random.choice(tokens)]
        return self._detokenize(new_tokens)

    def augment_eda(self,
                    text: str,
                    num_aug: int = 1,
                    synonym_n: int = 1,
                    deletion_p: float = 0.1) -> List[str]:
        """
        基于 EDA 策略生成多条增强样本
        """
        augmented = []
        for _ in range(num_aug):
            choice = random.choice(["synonym", "delete", "both"])
            new_text = text
            if choice in ("synonym", "both"):
                new_text = self.synonym_replacement(new_text, n=synonym_n)
            if choice in ("delete", "both"):
                new_text = self.random_deletion(new_text, p=deletion_p)
            augmented.append(new_text)
        return augmented


# =====================================================
#  Part 1: 垃圾邮件分类数据增强（enron_spam_data.csv）
# =====================================================

def augment_enron_spam(
    input_path: str,
    output_path: str,
    num_aug_per_example: int = 1,
    synonym_n: int = 1,
    deletion_p: float = 0.1,
    seed: int = 42,
):
    """
    对 enron_spam_data.csv 做数据增强：
    - 输入列格式固定为：
      ['Message ID', 'Subject', 'Message', 'Spam/Ham', 'Date']
    - 增强策略：
      只对 'Message'（正文）做 EDA 增强
    - 输出：
      同样的列结构，但样本数量增加
    """
    random.seed(seed)
    df = pd.read_csv(input_path)
    required_cols = ["Message ID", "Subject", "Message", "Spam/Ham", "Date"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {input_path}")

    augmenter = TextAugmenter(random_seed=seed)

    # 新的行存放在列表里，最后 concat 成 DataFrame
    new_rows = []

    # 原始数据先全部保留
    max_id = df["Message ID"].max() if "Message ID" in df.columns else 0
    next_id = int(max_id) + 1

    for _, row in df.iterrows():
        # 原始样本
        new_rows.append(row.to_dict())

        original_message = str(row["Message"]) if not pd.isna(row["Message"]) else ""

        # 增强样本
        aug_messages = augmenter.augment_eda(
            original_message,
            num_aug=num_aug_per_example,
            synonym_n=synonym_n,
            deletion_p=deletion_p,
        )

        for aug_msg in aug_messages:
            aug_row = row.to_dict()
            # 为增强样本分配新的 Message ID，避免重复
            if "Message ID" in aug_row:
                aug_row["Message ID"] = next_id
                next_id += 1
            aug_row["Message"] = aug_msg
            new_rows.append(aug_row)

    out_df = pd.DataFrame(new_rows, columns=df.columns)  # 保持列顺序
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[ENRON SPAM] Original: {len(df)}, Augmented: {len(out_df)}")
    print(f"Saved augmented spam dataset to: {output_path}")


# =======================================================
#  Part 2: 邮件回复生成数据增强（synthetic_reply_dataset.csv）
# =======================================================

def augment_reply_dataset(
    input_path: str,
    output_path: str,
    num_aug_per_example: int = 1,
    augment_mode: str = "eda_incoming",
    synonym_n: int = 1,
    deletion_p: float = 0.1,
    seed: int = 42,
):
    """
    对 synthetic_reply_dataset.csv 做数据增强：
    - 输入列格式固定为：['Message', 'reply']
      Message: 来信内容
      reply:   回复内容（ground truth）
    - 增强模式（当前实现了一个轻量版 EDA）：
      - eda_incoming: 只对 Message 做 EDA 增强，reply 不变
      （如果你们之后想用 back-translation，可以在这里再扩展）

    输出：与输入相同的列 ['Message', 'reply']，但行数增加
    """
    random.seed(seed)
    df = pd.read_csv(input_path)
    required_cols = ["Message", "reply"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {input_path}")

    augmenter = TextAugmenter(random_seed=seed)

    new_messages = []
    new_replies = []

    for _, row in df.iterrows():
        msg = str(row["Message"]) if not pd.isna(row["Message"]) else ""
        rep = str(row["reply"]) if not pd.isna(row["reply"]) else ""

        # 原始样本
        new_messages.append(msg)
        new_replies.append(rep)

        # 增强样本
        if augment_mode == "eda_incoming":
            aug_msgs = augmenter.augment_eda(
                msg,
                num_aug=num_aug_per_example,
                synonym_n=synonym_n,
                deletion_p=deletion_p,
            )
            for aug_msg in aug_msgs:
                new_messages.append(aug_msg)
                new_replies.append(rep)
        else:
            raise ValueError(f"Unknown augment_mode: {augment_mode}")

    out_df = pd.DataFrame({
        "Message": new_messages,
        "reply": new_replies,
    })
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[REPLY DATASET] Original: {len(df)}, Augmented: {len(out_df)}")
    print(f"Saved augmented reply dataset to: {output_path}")


# ==============================
#           CLI 入口
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="Data augmentation for Enron spam classification and synthetic email reply dataset."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["spam", "reply"],
        help="Which dataset to augment: 'spam' for enron_spam_data.csv, 'reply' for synthetic_reply_dataset.csv",
    )
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")

    parser.add_argument("--num_aug", type=int, default=1, help="Number of augmented samples per original")
    parser.add_argument("--synonym_n", type=int, default=1, help="Max synonym replacements per sample")
    parser.add_argument("--deletion_p", type=float, default=0.1, help="Random deletion probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # 专门给 reply 数据集用的模式（目前只实现 eda_incoming）
    parser.add_argument(
        "--augment_mode",
        type=str,
        default="eda_incoming",
        help="Augmentation mode for reply dataset (default: eda_incoming)",
    )

    args = parser.parse_args()

    if args.task == "spam":
        augment_enron_spam(
            input_path=args.input,
            output_path=args.output,
            num_aug_per_example=args.num_aug,
            synonym_n=args.synonym_n,
            deletion_p=args.deletion_p,
            seed=args.seed,
        )
    elif args.task == "reply":
        augment_reply_dataset(
            input_path=args.input,
            output_path=args.output,
            num_aug_per_example=args.num_aug,
            augment_mode=args.augment_mode,
            synonym_n=args.synonym_n,
            deletion_p=args.deletion_p,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
