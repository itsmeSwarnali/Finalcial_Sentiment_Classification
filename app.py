import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = "swarnaliM/financial-sentiment-distillbert"

# Same label meaning used in your project
ID2LABEL = {
    0: "Bearish (Pessimistic Market)",
    1: "Bullish (Optimistic Market)",
    2: "Neutral"
}


st.set_page_config(
    page_title="Financial Sentiment Classifier",
    page_icon="📈",
    layout="centered"
)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_sentiment(text: str):
    tokenizer, model, device = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    latency_ms = (time.perf_counter() - start) * 1000

    pred_id = int(probs.argmax())
    pred_label = ID2LABEL[pred_id]
    confidence = float(probs[pred_id])

    return pred_label, confidence, probs, latency_ms


st.title("📈 Financial News Sentiment Classifier")
st.write(
    "This app uses a fine-tuned DistilBERT model to classify financial text as "
    "**Bearish**, **Bullish**, or **Neutral**."
)

with st.sidebar:
    st.header("Project Info")
    st.write("Model: Fine-tuned DistilBERT")
    st.write("Task: Financial sentiment classification")
    st.write("Labels: Bearish, Bullish, Neutral")

    results_path = Path("results_finetune.json")
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        st.subheader("Saved Test Results")
        st.write(f"Accuracy: `{results['metrics']['accuracy']:.4f}`")
        st.write(f"Macro-F1: `{results['metrics']['f1_macro']:.4f}`")
        st.write(f"Latency: `{results['latency_ms_per_sample']:.2f} ms/sample`")


example_texts = [
    "Tesla shares rise after stronger than expected earnings report.",
    "The company missed revenue expectations and cut its full-year outlook.",
    "The stock remained unchanged after the announcement.",
]

example = st.selectbox("Choose an example text:", example_texts)

text = st.text_area(
    "Enter a financial tweet/news headline:",
    value=example,
    height=120
)

if st.button("Classify Sentiment"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        label, confidence, probs, latency_ms = predict_sentiment(text)

        st.subheader("Prediction")

        if label == "Bullish":
            st.success(f"Prediction: {label}")
        elif label == "Bearish":
            st.error(f"Prediction: {label}")
        else:
            st.info(f"Prediction: {label}")

        st.write(f"Confidence: **{confidence:.4f}**")
        st.write(f"Inference time: **{latency_ms:.2f} ms**")

        prob_df = pd.DataFrame({
            "Sentiment": ["Bearish", "Bullish", "Neutral"],
            "Probability": probs
        })

        st.subheader("Class Probabilities")
        st.bar_chart(prob_df.set_index("Sentiment"))


st.divider()

st.subheader("Batch Prediction from CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV file with a column named `text`",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("Your CSV must contain a column named `text`.")
    else:
        predictions = []
        confidences = []

        for text_item in df["text"].astype(str):
            label, confidence, _, _ = predict_sentiment(text_item)
            predictions.append(label)
            confidences.append(confidence)

        df["predicted_sentiment"] = predictions
        df["confidence"] = confidences

        st.write(df.head(20))

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="distilbert_sentiment_predictions.csv",
            mime="text/csv"
        )