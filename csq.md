# 📂 **Project Structure**

```
project/
│
├──data/
│   ├── classification/
│   ├── email_reply_dataset/
│   └── eval/
│
├── data_augmentation.py        # Data augmentation script
├── evaluate.py                 # Model inference + evaluation pipeline
├── visualization.py            # Result visualization
│
├── results/                    # Output CSVs (predictions and scores)
├── figs/                       # Saved plots (confusion matrix, histograms)
└── models/
	├── spam_classifier_model/
    └── reply_generator_model/ 

```

---

# ⚙️ **Environment Setup**

```bash
pip install pandas numpy matplotlib scikit-learn transformers tqdm openai nltk
```

Download required NLTK resources (first run):

```python
import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("omw-1.4")
```

---

# 📌 **1. Data Augmentation**

Implemented in **`data_augmentation.py`**

### ✔ Supported Strategies

The script uses lightweight EDA-style augmentation:

- **Synonym Replacement** — replace up to *n* non-stopwords with WordNet synonyms
- **Random Deletion** — remove each word with probability *p*
- **Combined EDA** — randomly choose synonym / deletion / both

### ✔ Augmentation Modes

- `spam`: augment only the **Message** field in *enron_spam_data.csv*
- `reply`: augment only the **incoming Message** (not the reply) for reply-generation dataset

### ✔ Example Command

```bash
# For the spam classification dataset
python data_augmentation.py \
  --task spam \
  --input data/classification/enron_spam_data.csv \
  --output data/classification/enron_spam_data_augmented.csv \
  --num_aug 1 \
  --synonym_n 1 \
  --deletion_p 0.1

# For the reply generation dataset
python data_augmentation.py \
  --task reply \
  --input data/classification/synthetic_reply_dataset.csv \
  --output data/classification/synthetic_reply_dataset_augmented.csv \
  --num_aug 1 \
  --augment_mode eda_incoming \
  --synonym_n 1 \
  --deletion_p 0.1
```

---

# 📌 **2. Model Evaluation Pipeline**

Implemented in **`evaluate.py`**

Two evaluation pipelines are supported:

---

## 🔹 **A. Spam Classification Evaluation**

Steps:

1. Load fine-tuned classifier (BERT or any HF sequence-classification model)
2. Run inference
3. Save predictions (`pred_label`)
4. Compute **Accuracy & Macro F1**

### Example Command:

```bash
python evaluate.py spam \
  --model_dir models\spam_classifier_model\spam_classifier_model \
  --test_csv data/eval/span_ham_dataset.csv \
  --pred_csv results/spam_pred.csv \
  --text_column data \
  --label_column "Spam/Ham"

```

Output includes:

- `results/spam_pred.csv`
- Printed metrics:
    - Accuracy
    - Macro F1

---

## 🔹 **B. Reply Generation + LLM Scoring**

The pipeline:

1. Read mixed spam/ham dataset (e.g., `span_ham_dataset.csv`)
2. **Select only HAM emails**
3. Randomly sample **400 emails**
4. Generate replies using fine-tuned GPT-2
5. Score replies using a large model (e.g., `qwen3-14b`)
    - Fluency
    - Relevance
    - Politeness

✦ Scores are **1–5 Likert scale**.

### Example Comman:

```bash
python evaluate.py ham_reply \
  --gpt2_model_dir models/reply_generator_model/reply_generator_model \
  --input_csv data/eval/span_ham_dataset.csv \
  --output_csv results/ham_reply_scored.csv \
  --llm_model_name qwen3-14b \
  --api_key "YOUR_API_KEY"
```

Output:

- `response` (GPT-2 reply)
- `fluency_score`
- `relevance_score`
- `politeness_score`

Problematic samples filtered by Qwen (content moderation) are safely skipped.

---

# 📌 **3. Visualization**

Implemented in **`visualization.py`**

And also available as Jupyter notebook:

`notebooks/visualize_results.ipynb`

### Visualizations include:

- **Confusion Matrix** (spam classifier)
- **Accuracy & Macro F1 bar chart**
- **LLM score histograms** (fluency / relevance / politeness)
- **Average score with standard deviation**

### Run:

```bash
python visualization.py

```

Plots saved to `figs/`.

---

# 📌 **4. Reproducibility Notes**

- Random seeds are fixed in augmentation, sampling, and GPT-2 generation.
- Exactly **400 HAM samples** are evaluated each run (stable dataset).
- LLM scoring includes structured error handling for content moderation.