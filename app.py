import streamlit as st
import pandas as pd
import re
import string
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


# -------------------------------------------------
# Text cleaning
# -------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------------------------
# Streamlit config
# -------------------------------------------------
st.set_page_config(page_title="Email Classifier", layout="centered")
st.title("📧 Email Classifier")
st.write("Real-world text classification using LinearSVC with calibrated confidence")


# -------------------------------------------------
# Train / Load model
# -------------------------------------------------
@st.cache_resource
def get_model():
    model_path = "email_model.pkl"

    # Load model if already trained
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    # Load dataset
    df = None
    for fname in ("emails.csv", "test_emails.csv"):
        try:
            df = pd.read_csv(fname)
            break
        except FileNotFoundError:
            pass

    if df is None:
        st.error("❌ Training data not found")
        return None

    if "email_text" not in df.columns or "label" not in df.columns:
        st.error("❌ CSV must contain 'email_text' and 'label' columns")
        return None

    # Clean text
    df["email_text"] = df["email_text"].astype(str).apply(clean_text)

    X = df["email_text"]
    y = df["label"]

    # Base classifier (BEST for text)
    base_svc = LinearSVC()

    # Calibrate probabilities (REQUIRED)
    clf = CalibratedClassifierCV(
        estimator=base_svc,   # NEW sklearn syntax
        method="sigmoid",
        cv=5
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )),
        ("clf", clf)
    ])

    model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    st.success("✅ Model trained successfully")
    return model


model = get_model()


# -------------------------------------------------
# Prediction UI
# -------------------------------------------------
if model:
    st.write("---")
    st.subheader("Enter email text")

    email_text = st.text_area(
        "Email text",
        height=150,
        placeholder="Paste your email here..."
    )

    if st.button("Classify Email", type="primary"):
        if email_text.strip():
            cleaned = clean_text(email_text)

            probs = model.predict_proba([cleaned])[0]
            labels = model.classes_

            best_idx = probs.argmax()
            prediction = labels[best_idx]
            confidence = probs[best_idx]

            st.write("---")
            st.subheader("📊 Result")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Category", prediction)
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")

            if confidence < 0.4:
                st.warning("⚠️ Low confidence prediction")

        else:
            st.warning("Please enter some text")


    # -------------------------------------------------
    # Example
    # -------------------------------------------------
    st.write("---")
    st.subheader("💡 Example")
    if st.button("Classify: Login button not working on Safari on iPhone"):
        example = clean_text("Login button not working on Safari on iPhone")

        probs = model.predict_proba([example])[0]
        labels = model.classes_

        idx = probs.argmax()
        st.success(
            f"**Category:** {labels[idx]} | **Confidence:** {probs[idx]:.1%}"
        )
