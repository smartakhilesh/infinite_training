import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data (try common filenames)
df = None
for fname in ("emails.csv", "test_emails.csv"):
    try:
        df = pd.read_csv(fname)
        print(f"Loaded {fname}")
        break
    except FileNotFoundError:
        df = None
if df is None:
    raise FileNotFoundError("Could not find emails.csv or test_emails.csv in workspace.")
# Handle malformed CSV where entire line is quoted (single column like 'email_text,label')
if "email_text" not in df.columns or "label" not in df.columns:
    # try to parse lines manually by splitting on the last comma
    if df.shape[1] == 1:
        raw = pd.read_csv(fname, header=None, dtype=str, skip_blank_lines=True)[0].astype(str)
        raw = raw.str.strip().str.strip('"').str.strip("'")
        if raw.size and raw.iloc[0].lower().startswith('email_text'):
            raw = raw.iloc[1:]
        parsed = raw.str.rsplit(',', n=1, expand=True)
        parsed.columns = ['email_text', 'label']
        df = parsed
    else:
        raise ValueError("CSV must contain 'email_text' and 'label' columns.")

df["email_text"] = df["email_text"].astype(str).apply(clean_text)

X = df["email_text"]
y = df["label"]

# Train model
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

print("Model trained successfully!\n")

# Interactive testing
print("Type an email (type 'exit' to stop):")
while True:
    text = input(">> ")
    if text.lower() == "exit":
        break
    prediction = model.predict([clean_text(text)])[0]
    print("Predicted category:", prediction)
