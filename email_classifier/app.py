import streamlit as st
import pandas as pd
import re
import string
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Title and description
st.set_page_config(page_title="Email Classifier", layout="centered")
st.title("📧 Email Classifier")
st.write("Classify emails into categories using machine learning")

# Train or load model
@st.cache_resource
def get_model():
    model_path = "email_model.pkl"
    
    # Load existing model if available
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    # Otherwise train new model
    df = None
    for fname in ("emails.csv", "test_emails.csv"):
        try:
            df = pd.read_csv(fname)
            st.success(f"Loaded {fname}")
            break
        except FileNotFoundError:
            df = None
    
    if df is None:
        st.error("Could not find emails.csv or test_emails.csv")
        return None
    
    # Handle CSV parsing
    if "email_text" not in df.columns or "label" not in df.columns:
        if df.shape[1] == 1:
            raw = pd.read_csv(fname, header=None, dtype=str, skip_blank_lines=True)[0].astype(str)
            raw = raw.str.strip().str.strip('"').str.strip("'")
            if raw.size and raw.iloc[0].lower().startswith('email_text'):
                raw = raw.iloc[1:]
            parsed = raw.str.rsplit(',', n=1, expand=True)
            parsed.columns = ['email_text', 'label']
            df = parsed
        else:
            st.error("CSV must contain 'email_text' and 'label' columns.")
            return None
    
    # Clean text
    df["email_text"] = df["email_text"].astype(str).apply(clean_text)
    X = df["email_text"]
    y = df["label"]
    
    # Train model
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    
    model.fit(X, y)
    st.success("✅ Model trained successfully!")
    
    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    return model

# Get the model
model = get_model()

if model:
    # User input
    st.write("---")
    st.subheader("Enter an email to classify:")
    
    email_text = st.text_area("Email text:", height=150, placeholder="Paste your email here...")
    
    if st.button("Classify Email", type="primary"):
        if email_text.strip():
            cleaned_text = clean_text(email_text)
            prediction = model.predict([cleaned_text])[0]
            confidence = model.predict_proba([cleaned_text]).max()
            
            st.write("---")
            st.subheader("📊 Result:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Category", prediction)
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
        else:
            st.warning("Please enter some email text to classify.")
    
    # Example section
    st.write("---")
    st.subheader("💡 Try an example:")
    if st.button("Classify 'Special offer - 50% off!'"):
        example = clean_text("Special offer - 50% off!")
        prediction = model.predict([example])[0]
        confidence = model.predict_proba([example]).max()
        st.success(f"**Category:** {prediction} | **Confidence:** {confidence:.1%}")
