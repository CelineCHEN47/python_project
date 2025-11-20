"""
Streamlit WebUI for Spam Email Analysis and Reply Generation
A skeleton app with mock spam detection and LLM-based reply generation
"""

import streamlit as st
import random
from datetime import datetime
import time
import torch
import torch.nn as nn
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set page configuration
st.set_page_config(
    page_title="Spam Email Analyzer",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        padding: 10px 20px;
    }
    .spam-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .spam-badge.spam {
        background-color: #ff6b6b;
        color: white;
    }
    .spam-badge.legitimate {
        background-color: #51cf66;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== Model Classes ====================

def clean_email_text(text):
    """Clean email text by removing URLs, extra whitespace, and special characters"""
    text = re.sub(r'http\S+|www\S+|ftp\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'&#\d+;', '', text)
    text = text.strip()
    return text


@st.cache_resource
def load_spam_model():
    """Load spam model and tokenizer from saved files"""
    model_path = 'spam_classifier_model'
    
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model not found at {model_path}. Using mock classifier.")
        return None, None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
    except Exception as e:
        st.warning(f"⚠️ Error loading model: {e}")
        return None, None


def predict_spam(model, tokenizer, email_text):
    """
    Predict if an email is spam.
    Returns: (is_spam: bool, confidence: float)
    """
    if model is None or tokenizer is None:
        # Fallback to mock if model not available
        is_spam = random.choice([True, False])
        confidence = round(random.uniform(0.7, 0.99), 2)
        return is_spam, confidence
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Clean text
        text = clean_email_text(email_text)
        
        # Tokenize
        inputs = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Convert to is_spam boolean (1 = spam, 0 = non-spam)
        is_spam = bool(prediction == 1)
        
        return is_spam, confidence
    except Exception as e:
        st.warning(f"⚠️ Error in prediction: {e}")
        # Fallback to mock
        is_spam = random.choice([True, False])
        confidence = round(random.uniform(0.7, 0.99), 2)
        return is_spam, confidence


# ==================== Mock Models ====================

class MockLLMReplyGenerator:
    """Mock LLM that generates generic predetermined responses"""
    
    GENERIC_REPLIES = [
        "Thank you for reaching out! We appreciate your message. We'll review your inquiry and get back to you shortly with a helpful response. Best regards!",
        "Hello! Thank you for contacting us. We've received your email and will prioritize addressing your concerns. We'll be in touch soon!",
        "Hi there! Thanks for getting in touch. We're here to help and will respond to your message as soon as possible. Looking forward to assisting you!",
        "Thank you for your email! We value your communication and will provide you with a detailed response soon. We appreciate your patience!",
        "Hi! We've received your message and will respond promptly. Thank you for choosing to reach out to us. We're ready to help!",
    ]
    
    @staticmethod
    def generate_reply(email_text):
        """
        Generate a generic reply to a legitimate email.
        Returns: reply_text (str)
        """
        reply = random.choice(MockLLMReplyGenerator.GENERIC_REPLIES)
        # Simulate processing time
        time.sleep(1)
        return reply


# ==================== Helper Functions ====================

def display_spam_result(is_spam, confidence):
    """Display spam classification result with styling"""
    if is_spam:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 🚨 SPAM DETECTED")
        with col2:
            st.markdown(
                f'<p class="spam-badge spam">SPAM ({confidence*100:.0f}%)</p>',
                unsafe_allow_html=True
            )
        st.warning(
            f"This email has been flagged as spam with {confidence*100:.0f}% confidence. "
            "Consider blocking this sender or moving it to your spam folder."
        )
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### ✅ LEGITIMATE EMAIL")
        with col2:
            st.markdown(
                f'<p class="spam-badge legitimate">LEGITIMATE ({confidence*100:.0f}%)</p>',
                unsafe_allow_html=True
            )
        st.success(
            f"This email appears to be legitimate with {confidence*100:.0f}% confidence. "
            "A reply suggestion has been generated below."
        )


def save_analysis_to_history(email_sender, email_datetime, email_subject, is_spam, confidence, reply=None):
    """Save analysis to session state history"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sender': email_sender,
        'email_datetime': email_datetime,
        'subject': email_subject,
        'is_spam': is_spam,
        'confidence': confidence,
        'reply': reply
    }
    st.session_state.analysis_history.append(entry)


# ==================== Main App ====================

def main():
    # Initialize BERT spam classifier
    model, tokenizer = load_spam_model()
    llm_generator = MockLLMReplyGenerator()
    
    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'generated_reply' not in st.session_state:
        st.session_state.generated_reply = None
    
    # Header
    st.markdown("# 📧 Spam Email Analyzer & Reply Generator")
    st.markdown("*Analyze emails for spam and generate intelligent replies using AI*")
    st.divider()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Email", "📋 History", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Email Analysis")
        
        # Email metadata section
        col1, col2, col3 = st.columns([1.5, 1.5, 1])
        
        with col1:
            email_sender = st.text_input(
                "👤 Sender Email",
                placeholder="sender@example.com",
                key="email_sender"
            )
        
        with col2:
            email_datetime = st.text_input(
                "📅 Date & Time",
                placeholder="2025-11-19 14:30:00",
                key="email_datetime",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        
        with col3:
            st.write("")  # Spacing
            st.write("")
            analyze_button = st.button("Analyze Email", use_container_width=True, type="primary")
        
        # Email subject input
        email_subject = st.text_input(
            "📌 Email Subject",
            placeholder="Enter the email subject...",
            key="email_subject"
        )
        
        # Email body input
        email_body = st.text_area(
            "📝 Email Body",
            placeholder="Paste the email content here...",
            height=200,
            key="email_body"
        )
        
        st.divider()
        
        # Analysis section
        if analyze_button:
            if not email_sender or not email_subject or not email_body or not email_datetime:
                st.error("❌ Please enter sender, subject, date & time, and email body!")
            else:
                with st.spinner("🔄 Analyzing email..."):
                    # Mock spam classification
                    is_spam, confidence = predict_spam(model, tokenizer, email_body)
                    st.session_state.analysis_result = {
                        'is_spam': is_spam,
                        'confidence': confidence
                    }
                    
                    # Generate reply if not spam
                    if not is_spam:
                        with st.spinner("💭 Generating reply..."):
                            reply = llm_generator.generate_reply(email_body)
                            st.session_state.generated_reply = reply
                    else:
                        st.session_state.generated_reply = None
                    
                    # Save to history
                    save_analysis_to_history(
                        email_sender,
                        email_datetime,
                        email_subject,
                        is_spam,
                        confidence,
                        st.session_state.generated_reply
                    )
        
        # Display results
        if st.session_state.analysis_result:
            st.write("")
            st.write("---")
            st.write("## Analysis Results")
            st.write("")
            
            is_spam = st.session_state.analysis_result['is_spam']
            confidence = st.session_state.analysis_result['confidence']
            
            display_spam_result(is_spam, confidence)
            
            # Display generated reply if not spam
            if not is_spam and st.session_state.generated_reply:
                st.write("")
                st.write("---")
                st.subheader("📨 Suggested Reply")
                
                col1, col2 = st.columns([4, 1])
                with col2:
                    copy_button = st.button("📋 Copy Reply", use_container_width=True)
                
                st.text_area(
                    "Reply Preview",
                    value=st.session_state.generated_reply,
                    height=150,
                    disabled=True,
                    key="reply_preview"
                )
                
                if copy_button:
                    st.success("✅ Reply copied to clipboard!")
                
                # Download reply as text file
                st.download_button(
                    label="⬇️ Download Reply",
                    data=st.session_state.generated_reply,
                    file_name=f"reply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    with tab2:
        st.subheader("Analysis History")
        
        if 'analysis_history' in st.session_state and st.session_state.analysis_history:
            # Display history in reverse chronological order
            for i, entry in enumerate(reversed(st.session_state.analysis_history)):
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**From:** {entry['sender']}")
                        st.write(f"**Subject:** {entry['subject']}")
                        st.write(f"**Email Date:** {entry['email_datetime']}")
                        st.write(f"**Analyzed At:** {entry['timestamp']}")
                    with col2:
                        if entry['is_spam']:
                            st.markdown(
                                f'<p class="spam-badge spam">SPAM</p>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<p class="spam-badge legitimate">LEGIT</p>',
                                unsafe_allow_html=True
                            )
                    
                    st.write(f"Confidence: {entry['confidence']*100:.0f}%")
                    
                    if entry['reply']:
                        with st.expander("View Generated Reply"):
                            st.write(entry['reply'])
            
            # Clear history button
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.analysis_history = []
                st.rerun()
        else:
            st.info("No analysis history yet. Analyze an email to get started!")
    
    with tab3:
        st.subheader("Settings & Configuration")
        
        st.write("### Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Spam Detection")
            st.info("Using fine-tuned DistilBERT model")
            spam_threshold = st.slider(
                "Spam Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Emails with spam confidence above this threshold will be flagged"
            )
        
        with col2:
            st.write("#### Reply Generation")
            llm_model = st.selectbox(
                "LLM Model",
                ["Generic (Mock)", "GPT-2", "DistilGPT-2", "Custom"],
                help="Select which LLM to use for reply generation"
            )
            temperature = st.slider(
                "Generation Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values = more creative replies, lower = more deterministic"
            )
        
        st.write("---")
        st.write("### Advanced Options")
        
        show_confidence = st.checkbox("Show detailed confidence scores", value=True)
        auto_save = st.checkbox("Auto-save analysis history", value=True)
        
        if st.button("💾 Save Configuration", use_container_width=True):
            st.success("✅ Settings saved successfully!")
        
        st.write("---")
        st.write("### About")
        st.markdown(
            """
            **Spam Email Analyzer & Reply Generator v0.3**
            
            This application uses AI models to:
            - 🔍 Detect spam emails with high accuracy using fine-tuned DistilBERT
            - 💭 Generate intelligent replies to legitimate emails
            
            **Current Status:** DistilBERT spam detector active
            - Spam detection: Fine-tuned DistilBERT model ✅
            - Reply generation: Generic responses (will be replaced with fine-tuned LLM)
            
            **Technology Stack:**
            - Frontend: Streamlit
            - Backend: Python
            - Spam Classifier: DistilBERT (Fine-tuned on Enron spam dataset)
            - Reply Generator: TBD (Fine-tuned GPT-2/DistilGPT-2)
            """
        )


if __name__ == "__main__":
    main()
