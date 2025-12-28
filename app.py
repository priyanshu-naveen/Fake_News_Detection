import streamlit as st
import tensorflow as tf
import pickle
import re
import numpy as np
import os
import gdown
from newspaper import Article
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

# ---------------- CONFIG ----------------
MAX_LEN = 300
REAL_THRESHOLD = 0.70
FAKE_THRESHOLD = 0.30

MODEL_PATH = "fake_news_model.h5"
GDRIVE_FILE_ID = "1GX0HxcBWhz4FpB6DPGVaKk8wZ0cMj-zO"   
st.set_page_config(page_title="Fake News Detector", layout="centered")

# ---------------- DOWNLOAD MODEL FROM GDRIVE ----------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
            MODEL_PATH,
            quiet=False
        )
# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ---------------- FUNCTIONS ----------------
def extract_news_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def get_top_words(text, n=8):
    words = text.split()
    common = Counter(words).most_common(n)
    return [w for w, _ in common]

def predict_news(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob = model.predict(padded, verbose=0)[0][0]
    return prob

# ---------------- UI ----------------
st.title("📰 Fake News Detection")
st.write("Paste a **news article URL** to analyze it.")

st.info(
    "⚠️ This model analyzes **writing patterns**, not factual correctness "
    "or source credibility. Predictions may vary."
)

news_url = st.text_input("🔗 Enter News URL")

if st.button("Analyze News"):
    if news_url.strip() == "":
        st.warning("Please enter a valid news URL.")
    else:
        try:
            with st.spinner("Fetching and analyzing article..."):
                article_text = extract_news_from_url(news_url)

                if len(article_text.strip()) < 200:
                    st.error("Article text too short to analyze reliably.")
                else:
                    cleaned_text = clean_text(article_text)
                    prob = predict_news(cleaned_text)
                    keywords = get_top_words(cleaned_text)

                    # ---------------- DECISION LOGIC ----------------
                    if prob >= REAL_THRESHOLD:
                        confidence = prob * 100
                        explanation = (
                            "The article uses structured language, neutral tone, "
                            "and informational wording commonly seen in real news."
                        )
                        st.success(f"✅ REAL NEWS ({confidence:.2f}% confidence)")

                    elif prob <= FAKE_THRESHOLD:
                        confidence = (1 - prob) * 100
                        explanation = (
                            "The article shows linguistic patterns such as emotional wording, "
                            "sensational phrases, or repetition commonly found in fake news."
                        )
                        st.error(f"🚨 FAKE NEWS ({confidence:.2f}% confidence)")

                    else:
                        confidence = 100 - abs(50 - prob * 100) * 2
                        explanation = (
                            "The article contains mixed linguistic signals. "
                            "The model is not confident enough to classify it reliably."
                        )
                        st.warning(f"⚠️ UNCERTAIN ({confidence:.2f}% confidence)")

                    # ---------------- EXPLANATION ----------------
                    st.subheader("🧠 Why this prediction?")
                    st.write(explanation)

                    st.subheader("🔍 Influential Words Detected")
                    st.write(", ".join(keywords))

        except Exception as e:
            st.error("Failed to extract article. Try a different link.")

