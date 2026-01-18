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
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Email Classifier", layout="centered")
st.title("📧 Email Classifier")
st.write("LinearSVC + calibrated confidence (industry standard for text)")


# -------------------------------------------------
# Train / Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "email_model.pkl"

    # Load cached model
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
        st.error("❌ Training CSV not found")
        return None

    if not {"email_text", "label"}.issubset(df.columns):
        st.error("❌ CSV must contain 'email_text' and 'label'")
        return None

    df["email_text"] = df["email_text"].astype(str).apply(clean_text)

    X = df["email_text"]
    y = df["label"]

    # Base classifier
    base_svc = LinearSVC()

    # Calibrated classifier
    clf = CalibratedClassifierCV(
        estimator=base_svc,
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


model = load_model()


# -------------------------------------------------
# Prediction UI
# -------------------------------------------------
if model:
    st.write("---")
    text = st.text_area(
        "Enter email text",
        height=150,
        placeholder="Paste email here..."
    )

    if st.button("Classify", type="primary"):
        if text.strip():
            cleaned = clean_text(text)

            probs = model.predict_proba([cleaned])[0]
            labels = model.classes_

            idx = probs.argmax()
            prediction = labels[idx]
            confidence = probs[idx]

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
            st.warning("Please enter text")


# -------------------------------------------------
# Example
# -------------------------------------------------
st.write("---")
if st.button("Example: Login button not working on Safari on iPhone"):
    sample = clean_text("Login button not working on Safari on iPhone")
    probs = model.predict_proba([sample])[0]
    labels = model.classes_

    idx = probs.argmax()
    st.success(
        f"**Category:** {labels[idx]} | **Confidence:** {probs[idx]:.1%}"
    )
