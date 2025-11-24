# visualize_results.py
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


# =============================
# 配置：根据你自己的路径调整
# =============================
SPAM_PRED_CSV = "results/spam_pred.csv"
HAM_REPLY_SCORED_CSV = "results/ham_reply_scored.csv"

# 对于 spam_pred.csv：
# 真实标签列名 & 预测标签列名
TRUE_LABEL_COL = "Spam/Ham"    # 真实标签
PRED_LABEL_COL = "pred_label"  # evaluate.py 写出的预测标签列名


def load_classification_results(csv_path: str):
    df = pd.read_csv(csv_path)
    if TRUE_LABEL_COL not in df.columns or PRED_LABEL_COL not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{TRUE_LABEL_COL}' and '{PRED_LABEL_COL}'. "
            f"Found: {df.columns.tolist()}"
        )

    y_true = df[TRUE_LABEL_COL].tolist()
    y_pred = df[PRED_LABEL_COL].tolist()

    # 确保两边都是字符串，方便混淆矩阵显示
    y_true = [str(x) for x in y_true]
    y_pred = [str(x) for x in y_pred]

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    # 获取类别顺序（保持稳定）
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")

    # x, y 轴标签
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # 在格子中写数字
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Viz] Saved confusion matrix to {save_path}")

    return fig, ax


def plot_classification_metrics(y_true, y_pred, save_path=None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    metrics_names = ["Accuracy", "Macro F1"]
    values = [acc, macro_f1]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(metrics_names, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics")

    # 在柱子顶部标数值
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Viz] Saved classification metrics bar chart to {save_path}")

    return fig, ax


def load_reply_scores(csv_path: str):
    df = pd.read_csv(csv_path)

    score_cols = ["fluency_score", "relevance_score", "politeness_score"]
    for col in score_cols:
        if col not in df.columns:
            raise ValueError(
                f"CSV must contain column '{col}'. Found: {df.columns.tolist()}"
            )
        # 转成 numeric，出问题的设为 NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 丢掉有 NaN 的行（如果有）
    df = df.dropna(subset=score_cols)

    return df, score_cols


def plot_score_histograms(df, score_cols, save_path=None):
    """
    为每个评分维度画一个直方图。
    """
    num_scores = len(score_cols)
    fig, axes = plt.subplots(1, num_scores, figsize=(4 * num_scores, 4), sharey=True)

    if num_scores == 1:
        axes = [axes]

    for ax, col in zip(axes, score_cols):
        scores = df[col].values
        ax.hist(scores, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], rwidth=0.8)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Score")
        ax.set_title(col.replace("_", " ").title())

    axes[0].set_ylabel("Count")

    fig.suptitle("Distribution of LLM Scores", y=1.02)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Viz] Saved score histograms to {save_path}")

    return fig, axes


def plot_score_means(df, score_cols, save_path=None):
    """
    画三维评分的平均值条形图（带标准差误差线）。
    """
    means = df[score_cols].mean()
    stds = df[score_cols].std()

    x = np.arange(len(score_cols))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, means.values, yerr=stds.values, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in score_cols])
    ax.set_ylim(0.0, 5.5)
    ax.set_ylabel("Score (1-5)")
    ax.set_title("Average LLM Scores (with Std)")

    # 在柱子顶部标记平均分
    for i, v in enumerate(means.values):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Viz] Saved score means bar chart to {save_path}")

    return fig, ax


def main():
    # ===== Part 1: Spam classification visualization =====
    print("[Viz] Loading classification results...")
    y_true, y_pred = load_classification_results(SPAM_PRED_CSV)

    print("[Viz] Plotting confusion matrix...")
    plot_confusion_matrix(
        y_true,
        y_pred,
        save_path="figs/confusion_matrix.png",
    )

    print("[Viz] Plotting classification metrics...")
    plot_classification_metrics(
        y_true,
        y_pred,
        save_path="figs/classification_metrics.png",
    )

    # ===== Part 2: Reply generation scores visualization =====
    print("[Viz] Loading reply scores...")
    df_scores, score_cols = load_reply_scores(HAM_REPLY_SCORED_CSV)

    print("[Viz] Plotting score histograms...")
    plot_score_histograms(
        df_scores,
        score_cols,
        save_path="figs/score_histograms.png",
    )

    print("[Viz] Plotting score means...")
    plot_score_means(
        df_scores,
        score_cols,
        save_path="figs/score_means.png",
    )

    # 最后一次性展示所有图（也可以注释掉，仅保存为文件）
    plt.show()


if __name__ == "__main__":
    main()
