import streamlit as st
import tensorflow as tf
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- CONFIGURATION ---
MAX_LENGTH = 250
# IMPORTANT: Pointing to the file you just saved
MODEL_PATH = 'best_model.h5'      
TOKENIZER_PATH = 'tokenizer.pickle'

# --- LOAD RESOURCES ---
@st.cache_resource # This makes the app load faster by remembering the model
def load_artifacts():
    try:
        # Load the trained model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load the tokenizer
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    except FileNotFoundError:
        st.error("Error: Model or Tokenizer file not found. Make sure 'best_model.h5' and 'tokenizer.pickle' are in this folder.")
        return None, None

model, tokenizer = load_artifacts()

# --- PREPROCESSING ---
def clean_text(text):
    """
    Same cleaning logic as training.
    """
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = text.lower()
    return text

def predict_sentiment(review_text):
    if model is None or tokenizer is None:
        return 0.5 # Default fallback
        
    # 1. Clean
    cleaned_text = clean_text(review_text)
    
    # 2. Tokenize
    seq = tokenizer.texts_to_sequences([cleaned_text])
    
    # 3. Pad
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    # 4. Predict
    prediction = model.predict(padded)[0][0]
    return prediction

# --- FRONTEND (USER INTERFACE) ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Type a review below to see if our AI thinks it is **Positive** or **Negative**.")

# Input Area
user_input = st.text_area("Your Review:", height=150, placeholder="Example: The cinematography was great, but the story was boring...")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please type a review first!")
    else:
        with st.spinner("Analyzing..."):
            score = predict_sentiment(user_input)
            
            # Convert score (0.0 - 1.0) to class
            if score > 0.5:
                label = "POSITIVE"
                color = "#4CAF50" # Green
                emoji = "😊"
            else:
                label = "NEGATIVE"
                color = "#F44336" # Red
                emoji = "😠"
            
            # Display Results
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### Prediction:")
                st.markdown(f"<h2 style='color: {color};'>{label} {emoji}</h2>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"### Confidence Score:")
                st.info(f"{score:.4f}")
                
            # Progress bar visualization
            st.caption("Sentiment Scale (0 = Negative, 1 = Positive)")
            st.progress(float(score))