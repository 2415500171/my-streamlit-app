import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import streamlit as st
from langdetect import detect, DetectorFactory
import io

DetectorFactory.seed = 0

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        background: linear-gradient(135deg, #f8ffae 0%, #43c6ac 100%);
    }
    .main-title {
        font-size: 2.9em !important;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #1c3b3f;
        margin-bottom: 0.7em;
    }
    .card {
        background: white;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgb(67 198 172 / 15%);
        padding: 1.2em 1.5em;
        margin-bottom: 1.2em;
        border: 1px solid #e3e3e3;
    }
    .emoji-label {
        font-size: 2.3em;
        font-weight: 600;
        margin-right: 0.3em;
    }
    .result-stars {
        font-size: 1.5em;
        font-weight: 500;
        color: #43c6ac;
        margin-top: 0.7em;
    }
    .footer {
        font-size: 1.1em;
        color: #333;
        margin-top: 1.6em;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #43c6ac 0%, #f8ffae 100%);
        color: #2d373b;
        border-radius: 40px;
        font-weight: 600;
        font-size: 1.1em;
        border: none;
        padding: 0.7em 2em;
        box-shadow: 0px 4px 12px rgb(67 198 172 / 12%);
        transition: 0.3s;
    }
    .stButton>button:hover {
        filter: brightness(1.07);
        background: linear-gradient(90deg, #f8ffae 0%, #43c6ac 100%);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if torch.cuda.is_available():
        model.to('cuda')
    return tokenizer, model

tokenizer, model = load_model()

def analyze_sentiment(text):
    if not text.strip():
        return None
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    if torch.cuda.is_available():
        encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input)
    scores = output.logits[0].detach().cpu().numpy()
    scores = softmax(scores)
    predicted_star = np.argmax(scores) + 1
    sentiment_map = {
        5: ("🤩", "Very Positive", "#43c6ac"),
        4: ("😊", "Positive", "#90ee90"),
        3: ("😐", "Neutral", "#f7b731"),
        2: ("😞", "Negative", "#fa7268"),
        1: ("😡", "Very Negative", "#eb3b5a")
    }
    sentiment_emoji, sentiment_label, color = sentiment_map[predicted_star]
    sentiment_scores = {
        '1_star': scores[0],
        '2_star': scores[1],
        '3_star': scores[2],
        '4_star': scores[3],
        '5_star': scores[4]
    }
    return {
        "text": text,
        "predicted_star_rating": predicted_star,
        "sentiment_emoji": sentiment_emoji,
        "sentiment_label": sentiment_label,
        "confidence_scores": sentiment_scores,
        "color": color
    }

st.sidebar.image(
    "https://emojipedia-us.s3.amazonaws.com/source/skype/289/globe-with-meridians_1f310.png", width=72
)
st.sidebar.markdown("### 🌍 Multilingual Sentiment Analyzer")
st.sidebar.info("Analyze text sentiment in 100+ languages using a powerful AI transformer model. Built with 💚 by Divyanshu & Gaurav.")

st.set_page_config(page_title="Beautiful Sentiment Analysis 🌍", layout="wide")

st.markdown('<div class="main-title">🌎 Multilingual Sentiment Analysis Pipeline</div>', unsafe_allow_html=True)
st.markdown("---")

st.markdown('<div class="card">This application analyzes the sentiment of text in multiple languages using a pre-trained Transformers model, displaying results with beautiful emojis and styled cards.</div>', unsafe_allow_html=True)

input_method = st.radio(
    "🔹 <b>Choose your input method:</b>",
    ("Enter Text Manually", "Upload a Text File (.txt, .csv)"),
    index=0,
    help="You can enter text directly or analyze a whole file of texts.",
    format_func=lambda x: f"📋 {x}" if x == "Enter Text Manually" else f"📁 {x}"
)

if input_method == "Enter Text Manually":
    user_input = st.text_area(
        "✍️ Enter text for sentiment analysis:",
        "Project is as Good as it has to be!",
        height=130,
        help="Paste any text in any language here."
    )
    analyze_btn = st.button("🔍 Analyze Sentiment")
    if analyze_btn:
        if user_input:
            try:
                detected_lang = detect(user_input)
                st.info(f"Detected Language: **{detected_lang.upper()}**")
                result = analyze_sentiment(user_input)
                if result:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(
                        f"""<span class="emoji-label" style="color:{result['color']}">{result['sentiment_emoji']}</span>
                        <span style="font-size:1.3em; color:{result['color']}"><b>{result['sentiment_label']}</b></span>
                        <div class="result-stars">⭐ {result['predicted_star_rating']} Stars</div>
                        """, unsafe_allow_html=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.subheader("Confidence Scores:")
                    scores_df = pd.DataFrame([result['confidence_scores']]).T.rename(columns={0: 'Probability'})
                    st.dataframe(scores_df.style.format("{:.2%}"), use_container_width=True)
                    st.json({
                        "text": result["text"],
                        "predicted_star_rating": result["predicted_star_rating"],
                        "sentiment_emoji": result["sentiment_emoji"],
                        "sentiment_label": result["sentiment_label"],
                        "confidence_scores": result["confidence_scores"],
                        "language": detected_lang
                    })
                else:
                    st.warning("Please enter some text to analyze.")
            except Exception as e:
                st.error(f"Error during analysis or language detection: {e}. Please try again.")
        else:
            st.warning("Please enter some text to analyze.")
else:
    uploaded_file = st.file_uploader(
        "📤 Upload a text file (.txt or .csv) for batch sentiment analysis:",
        type=["txt", "csv"],
        help="TXT: one sentence per line. CSV: select the text column."
    )
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        texts_to_analyze = []
        if file_extension == "txt":
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            texts_to_analyze = [line.strip() for line in stringio if line.strip()]
            st.success(f"Loaded {len(texts_to_analyze)} lines from the text file.")
        elif file_extension == "csv":
            df = pd.read_csv(uploaded_file)
            st.write("---")
            st.subheader("CSV File Preview:")
            st.dataframe(df.head())
            st.warning("Please select the column containing text for sentiment analysis.")
            text_column = st.selectbox("Select the text column from your CSV:", df.columns)
            if text_column:
                texts_to_analyze = df[text_column].dropna().astype(str).tolist()
                st.success(f"Loaded {len(texts_to_analyze)} texts from column '{text_column}'.")
            else:
                st.error("No text column selected. Please select a column to proceed.")
                texts_to_analyze = []
        if texts_to_analyze and st.button("🔍 Analyze Sentiment (File Upload)"):
            st.subheader("Batch Analysis Results:")
            progress_bar = st.progress(0)
            total_texts = len(texts_to_analyze)
            all_results = []
            for i, text in enumerate(texts_to_analyze):
                try:
                    detected_lang = detect(text)
                    result = analyze_sentiment(text)
                    if result:
                        result_for_df = {k: v for k, v in result.items() if k != 'color'}
                        result_for_df['detected_language'] = detected_lang
                        all_results.append(result_for_df)
                except Exception as e:
                    st.warning(f"Could not analyze text '{text[:50]}...' due to error: {e}")
                progress_bar.progress((i + 1) / total_texts)
            if all_results:
                results_df_full = pd.DataFrame(all_results)
                scores_expanded = pd.json_normalize(results_df_full['confidence_scores'])
                results_df_final = pd.concat([results_df_full.drop(columns=['confidence_scores']), scores_expanded], axis=1)
                st.dataframe(results_df_final, use_container_width=True)
                csv_output = results_df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_output,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                )
            else:
                st.error("No valid results to display. Check your file content.")
        elif not texts_to_analyze and uploaded_file is not None:
            st.info("Upload a file and select the text column (if CSV) to enable analysis.")
    else:
        st.info("Upload a `.txt` file with one sentence per line, or a `.csv` file with a text column.")

st.markdown("---")
st.markdown(
    '<div class="footer">Pipeline built by <b>Divyanshu & Gaurav</b> for a college project using <b>Streamlit</b>, <b>Hugging Face Transformers</b>, and <b>Langdetect</b>.<br>'
    'For any issue contact <a href="mailto:divyanshubuisness@gmail.com">divyanshubuisness@gmail.com</a></div>',
    unsafe_allow_html=True
)
