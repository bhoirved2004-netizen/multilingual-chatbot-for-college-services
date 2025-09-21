# app.py - Multilingual chatbot (TF-IDF + Classifier) + LLM response generation (robust)
import matplotlib
matplotlib.use("Agg")  # avoid "main thread" issues with matplotlib in some environments

import streamlit as st
import pandas as pd
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# transformers (LLM)
from transformers import pipeline
# optional: detect torch/cuda
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data(path="college_chatbot_dataset_500.csv"):
    """Load dataset from a relative CSV path. Make sure the CSV is in same folder as app.py."""
    return pd.read_csv(path)

df = load_data()

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.title("Model & Dataset Controls")
classifier_choice = st.sidebar.selectbox("Classifier", ["LogisticRegression", "SVM (linear)"])
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20)
random_state = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))
show_examples = st.sidebar.checkbox("Show sample Q&A", value=True)
llm_model_name = st.sidebar.selectbox("LLM model (HF)", ["google/flan-t5-small", "google/flan-t5-base"])

st.set_page_config(page_title="Multilingual College Chatbot", layout="wide", page_icon="🎓")
st.title("🤖 Multilingual College Chatbot (NLP + LLM)")
st.markdown("This demo uses TF-IDF + classifier to detect intent, and an LLM to generate natural answers.")

if show_examples:
    st.subheader("Sample dataset overview")
    st.dataframe(df.sample(8).reset_index(drop=True))

# -------------------------------
# Prepare training data (NLP)
# -------------------------------
questions = pd.concat([df['question_en'], df['question_hi'], df['question_mr']], ignore_index=True)
intents = pd.concat([df['intent'], df['intent'], df['intent']], ignore_index=True)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(questions)
y = intents

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(test_size / 100.0), random_state=random_state, stratify=y
)

if classifier_choice == "LogisticRegression":
    clf = LogisticRegression(max_iter=1000)
else:
    clf = SVC(kernel='linear', probability=True)

with st.spinner("Training the NLP classifier..."):
    clf.fit(X_train, y_train)

# Evaluate and show basic metrics
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.metric("Test Accuracy", f"{acc:.3f}")

st.subheader("Classification Report (sample)")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}), height=220)

# -------------------------------
# LLM loader (cached, robust)
# -------------------------------
@st.cache_resource
def load_llm(model_name="google/flan-t5-small"):
    """Attempts to load HF text2text-generation pipeline. Returns pipeline or None if loading fails."""
    try:
        device = 0 if (_HAS_TORCH and torch.cuda.is_available()) else -1
        pipe = pipeline("text2text-generation", model=model_name, device=device)
        return pipe
    except Exception as e:
        st.warning(f"Could not load LLM '{model_name}' (will fallback to canned answers). Error: {e}")
        return None

llm = load_llm(llm_model_name)

# -------------------------------
# Chat UI (stateful)
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of (speaker, text)

st.subheader("💬 Chat with the bot")
user_input = st.text_input("Ask about admissions, exams, timetable, results or fees:", key="user_input")

def generate_response(query_text):
    # Detect language
    try:
        lang = detect(query_text)
    except Exception:
        lang = "en"

    # NLP: predict intent
    Xq = vectorizer.transform([query_text])
    pred_intent = clf.predict(Xq)[0]

    # Base canned answer (from dataset)
    cand = df[df['intent'] == pred_intent].sample(1).iloc[0]
    base_answer_en = str(cand.get('answer_en', ''))
    base_answer_hi = str(cand.get('answer_hi', ''))
    base_answer_mr = str(cand.get('answer_mr', ''))

    # If LLM available
    if llm is not None:
        lang_label = "English"
        if str(lang).startswith("hi"):
            lang_label = "Hindi"
        elif str(lang).startswith("mr"):
            lang_label = "Marathi"

        prompt = (
            f"User query: {query_text}\n"
            f"Detected intent: {pred_intent}\n"
            f"Reference answer (English): {base_answer_en}\n"
            f"Please generate a concise helpful reply in {lang_label}."
        )
        try:
            out = llm(prompt, max_length=150, do_sample=False)
            generated = out[0].get("generated_text", "").strip()
            if generated:
                return generated, pred_intent, lang
        except Exception as e:
            st.warning(f"LLM generation failed, using canned answer. Error: {e}")

    # Fallback: dataset answer in detected language
    if str(lang).startswith("hi"):
        return base_answer_hi, pred_intent, lang
    elif str(lang).startswith("mr"):
        return base_answer_mr, pred_intent, lang
    else:
        return base_answer_en, pred_intent, lang

# Chat interaction
if st.button("Send") and user_input:
    response, intent, lang = generate_response(user_input.strip())
    st.session_state.history.append(("You", user_input.strip()))
    st.session_state.history.append(("Bot", response))
    st.session_state.user_input = ""

if st.session_state.history:
    st.markdown("### Conversation")
    for speaker, text in st.session_state.history[::-1]:
        if speaker == "You":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")

# Sidebar download
with open("college_chatbot_dataset_500.csv", "rb") as f:
    data_bytes = f.read()
st.sidebar.download_button("Download dataset (CSV)", data_bytes, file_name="college_chatbot_dataset_500.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.markdown("**Notes:**\n- If the LLM fails to load, the app falls back to dataset answers.\n- LLM models are large; use `flan-t5-small` for local testing.")
