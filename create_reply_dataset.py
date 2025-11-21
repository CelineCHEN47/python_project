import pandas as pd
from langchain_ollama import OllamaLLM
import re
from tqdm import tqdm
import sys

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv("enron_spam_data.csv")

# Filter for legitimate emails only (Ham)
ham_df = df[df["Spam/Ham"] == "ham"].copy()

# IMPORTANT: Generating text with a 32B model can be slow depending on hardware.
# Set SAMPLE_SIZE to None to generate for all emails, or a number to limit.
SAMPLE_SIZE = None  # Change to a number like 50, 100, 500 to limit
if SAMPLE_SIZE:
    print(f"Taking a sample of {SAMPLE_SIZE} emails from {len(ham_df)} total ham emails...")
    ham_df = ham_df.head(SAMPLE_SIZE)
else:
    print(f"Generating replies for all {len(ham_df)} ham emails...")
    print("Warning: This may take a very long time depending on your hardware!")

# 2. Setup Ollama with LangChain
MODEL_NAME = "neural-chat:7b"

print(f"Using Ollama model: {MODEL_NAME}")

# Initialize the LLM
try:
    print("Initializing Ollama LLM...")
    llm = OllamaLLM(model=MODEL_NAME)
    print(f"Model '{MODEL_NAME}' initialized successfully.")
except Exception as e:
    print(f"Error initializing Ollama: {e}")
    print("Please ensure Ollama is running. You can start it with 'ollama serve' in a terminal.")
    sys.exit(1)

def generate_reply_ollama(email_body):
    # Prompt designed for DeepSeek R1 to generate a clean reply
    prompt = f"Write a short, professional email reply to this message. Output ONLY the reply body. Do not include subject lines or placeholders.\n\nMessage:\n{email_body[:1000]}"
    
    try:
        response = llm.invoke(prompt)
        text = str(response)
        
        # Remove <think> blocks if present (DeepSeek R1 feature)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()
    except Exception as e:
        print(f"Error generating reply: {e}")
        return ""

# 3. Generate Replies
replies = []
print("Generating synthetic replies with Ollama...")

for message in tqdm(ham_df["Message"]):
    # Handle empty messages
    if message is None or not isinstance(message, str):
        replies.append("")
        continue
        
    output = generate_reply_ollama(message)
    replies.append(output)
    
    # Log the first 10 examples
    if len(replies) <= 10:
        print(f"\n--- Example {len(replies)} ---")
        print(f"Email: {message[:100]}...")
        print(f"Reply: {output}")
        print("-----------------------")

# 4. Save New Dataset
ham_df["reply"] = replies

# Filter out rows where generation failed or was empty
ham_df = ham_df[ham_df["reply"].str.len() > 0]

output_file = "synthetic_reply_dataset.csv"
# Save only the columns we need with proper quoting
ham_df[["Message", "reply"]].to_csv(output_file, index=False, quoting=1)  # quoting=1 is csv.QUOTE_ALL

print(f"\nSuccess! Saved {len(ham_df)} synthetic reply pairs to {output_file}")
print("You can now use this file to fine-tune your GPT-2 model.")
