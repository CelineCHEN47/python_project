# SmartMail - An integrated system for spam detection and email reply​

## Team Members
- Ghanibhuti Gogoi (50014104)
- Yuhan Chen (50013198)
- Siqi Chen (50012493)

## Project Overview
This project analyzes emails for spam detection and generates intelligent replies using fine-tuned LLMs. It includes backend AI models (BERT, GPT-2), comprehensive evaluation pipelines, and a user-friendly Streamlit WebUI.

---

## 📂 **Project Structure**

```
python_project/
│
├── data/
│   ├── classification/         # Spam/Ham datasets
│   ├── email_reply_dataset/    # Reply generation datasets
│   └── eval/                   # Evaluation datasets
│
├── experiments/                # Experiment scripts & notebooks
│   ├── data_augmentation.py    # Data augmentation script
│   ├── evaluate.py             # Model inference + evaluation pipeline
│   ├── visualization.py        # Result visualization
│   ├── lora_experiment.ipynb
│   └── zero-shot-few-shot.ipynb
│
├── models/                     # Saved models (created during training)
│   ├── spam_classifier_model/
│   └── reply_generator_model/
│
├── results/                    # Output CSVs (created during evaluation)
│
├── app.py                      # Streamlit WebUI
├── create_reply_dataset.py     # Dataset creation script
├── train_reply_model.py        # Reply generator training script
├── train_spam_model.py         # Spam classifier training script
└── requirements.txt
```

---

## ⚙️ **Environment Setup**

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # Or manually:
   pip install pandas numpy matplotlib scikit-learn transformers tqdm openai nltk streamlit datasets peft torch torchvision evaluate
   ```

2. **Download NLTK Resources**
   ```python
   import nltk
   nltk.download("punkt")
   nltk.download("wordnet")
   nltk.download("stopwords")
   nltk.download("omw-1.4")
   ```

---

## 🚀 **Quick Start (Pre-trained Models)**

If you prefer not to train the models from scratch, you can download our pre-trained models:

1.  **Download**: [models.zip](https://hkustgz-my.sharepoint.com/:u:/g/personal/ggogoi543_connect_hkust-gz_edu_cn/IQCEOZxluCUBQ7AKz1GVn0tiAcXZNh4icJRpPz3ARy_F8y0?e=2tMrln)
2.  **Unzip**: Extract the contents into the project root. Ensure the `models/` folder is created.
    ```bash
    unzip models.zip
    ```
3.  **Run App**: You can now directly launch the WebUI:
    ```bash
    streamlit run app.py
    ```

---

## 📌 **Data Augmentation**

Implemented in **`experiments/data_augmentation.py`**

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
python experiments/data_augmentation.py \
  --task spam \
  --input data/classification/enron_spam_data.csv \
  --output data/classification/enron_spam_data_augmented.csv \
  --num_aug 1 \
  --synonym_n 1 \
  --deletion_p 0.1

# For the reply generation dataset
python experiments/data_augmentation.py \
  --task reply \
  --input data/classification/synthetic_reply_dataset.csv \
  --output data/classification/synthetic_reply_dataset_augmented.csv \
  --num_aug 1 \
  --augment_mode eda_incoming \
  --synonym_n 1 \
  --deletion_p 0.1
```

---

## 📌 **1. Model Training & Experiments**

### **A. Local Training**
Run the training scripts directly. No arguments are required.

**1. Train Spam Classifier (DistilBERT)**
```bash
python train_spam_model.py
```
- Input: `data/classification/enron_spam_data.csv`
- Output: `./models/spam_classifier_model`

**2. Train Reply Generator (GPT-2)**
```bash
python train_reply_model.py
```
- Input: `data/email_reply_dataset/synthetic_reply_dataset.csv`
- Output: `./models/reply_generator_model`

#### **Training on Augmented Data**
To evaluate the impact of data augmentation, you can train a separate model on the augmented dataset (provided in the `data/` folder).

1.  **Edit the Script**: Open `train_spam_model.py` or `train_reply_model.py`.
2.  **Update Data Path**: Manually change the `pd.read_csv(...)` path to point to the augmented CSV file (e.g., `data/classification/enron_spam_data_augmented.csv`).
3.  **Run Training**: Execute the script again.
    *   *Tip: You may also want to change the `output_dir` in the script to save the augmented model to a new folder (e.g., `./models/spam_classifier_model_augmented`) to avoid overwriting the original model.*

### **B. Kaggle Experiments**
The `experiments/` folder contains notebooks designed to run on Kaggle.

#### **`lora_experiment.ipynb`**
- **Setup**: Copy to Kaggle. Mount datasets to `/kaggle/input`.
- **Important**: Click **Factory Reset** after every cell execution to avoid OOM errors.
- **Dependencies**: Run `!pip install evaluate` after every reset.
- **Paths**:
    - Input: `/kaggle/input/python_project/data/...`
    - Output: `/kaggle/working/...` (Save output manually before session ends).

#### **`zero-shot-few-shot.ipynb`**
- **Setup**: Copy to Kaggle. Mount datasets.
- **Note**: No Factory Reset needed usually.

### **C. Loading the LoRA Model**
The LoRA model consists of the base model and the adapter.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load LoRA adapter
lora_path = "./models/reply_generator_model/lora_adapter"
merged_model = PeftModel.from_pretrained(model, lora_path)

# Generate
inputs = tokenizer("Hello", return_tensors="pt")
outputs = merged_model.generate(**inputs, max_new_tokens=50)
```

---

## 📌 **2. Model Evaluation Pipeline**

Implemented in **`experiments/evaluate.py`**.

### **A. Spam Classification Evaluation**
Calculates **Accuracy & Macro F1**.

```bash
python experiments/evaluate.py spam \
  --model_dir models/spam_classifier_model/spam_classifier_model \
  --test_csv data/eval/span_ham_dataset.csv \
  --pred_csv results/spam_pred.csv \
  --text_column data \
  --label_column "Spam/Ham"
```

### **B. Reply Generation + LLM Scoring**
Generates replies using fine-tuned GPT-2 and scores them using a large LLM (e.g., Qwen) on **Fluency, Relevance, and Politeness** (1-5 scale).

```bash
python experiments/evaluate.py ham_reply \
  --gpt2_model_dir models/reply_generator_model/reply_generator_model \
  --input_csv data/eval/span_ham_dataset.csv \
  --output_csv results/ham_reply_scored.csv \
  --llm_model_name qwen3-14b \
  --api_key "YOUR_API_KEY"
```

---

## 📌 **3. Visualization**

Implemented in **`experiments/visualization.py`**.

Generates:
- Confusion Matrix (Spam Classifier)
- Accuracy & Macro F1 charts
- LLM score histograms

```bash
python experiments/visualization.py
```
Plots are saved to `figs/`.

---

## 📌 **4. Web Application (Streamlit)**

The `app.py` provides a comprehensive interface for the project.

### **Features**
- **Email Analysis**: Real-time spam detection (BERT) and reply generation.
- **History**: Tracks past analyses.
- **Configuration**: Model selection and thresholds.

### **Running the App**
```bash
streamlit run app.py
```

---

## 📌 **5. Reproducibility Notes**
- Random seeds are fixed in augmentation, sampling, and generation.
- Evaluation uses exactly **400 HAM samples** for stability.
- LLM scoring includes error handling for content moderation.
