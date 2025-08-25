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

@st.cache_resource
def load_model():
    """Loads the pre-trained model and tokenizer, caching them for Streamlit."""
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model.to('cuda')
    return tokenizer, model

tokenizer, model = load_model()

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text using the pre-trained multilingual model.
    Returns a dictionary with sentiment scores (1-5 stars), predicted sentiment emoji & label.
    """
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
    sentiment_emoji_label = ""
    color_style = ""
    if predicted_star == 5:
        sentiment_emoji_label = "🤩 Very Positive"
        color_style = "color: green;"
    elif predicted_star == 4:
        sentiment_emoji_label = "😊 Positive"
        color_style = "color: lightgreen;"
    elif predicted_star == 3:
        sentiment_emoji_label = "😐 Neutral"
        color_style = "color: orange;"
    elif predicted_star == 2:
        sentiment_emoji_label = "😞 Negative"
        color_style = "color: salmon;"
    else: 
        sentiment_emoji_label = "😡 Very Negative"
        color_style = "color: red;"
    
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
        "sentiment_emoji_label": sentiment_emoji_label,  # New field
        "confidence_scores": sentiment_scores,
        "color_style": color_style  # For Streamlit display
    }

st.set_page_config(page_title="Multilingual Sentiment Analysis", layout="wide")

st.title("🌎 Multilingual Sentiment Analysis Pipeline")
st.markdown("---")

st.write("This application analyzes the sentiment of text in multiple languages using a pre-trained Transformers model, displaying results with amazing emojis.")

input_method = st.radio(
    "**Choose your input method👇🏻:**",
    ("Enter Text Manually", "Upload a Text File (.txt, .csv)"),
    index=0
)

if input_method == "Enter Text Manually":
    user_input = st.text_area("Enter text for sentiment analysis:", "Project is as Good as it has to be!", height=150)
    if st.button("**Analyze Sentiment**"):
        if user_input:
            try:
                
                detected_lang = detect(user_input)
                st.info(f"Detected Language: **{detected_lang.upper()}**")

                result = analyze_sentiment(user_input)
                if result:
                    st.markdown("### Analysis Result:")
                    
                    
                    st.markdown(f"**Sentiment:** <span style='font-size: 24px; {result['color_style']} font-weight: bold;'>{result['sentiment_emoji_label']} ({result['predicted_star_rating']} stars)</span>", unsafe_allow_html=True)
                    
                    st.subheader("Confidence Scores:")
                    scores_df = pd.DataFrame([result['confidence_scores']]).T.rename(columns={0: 'Probability'})
                    st.dataframe(scores_df.style.format("{:.2%}"), use_container_width=True)

                    
                    st.json({"text": result["text"], 
                              "predicted_star_rating": result["predicted_star_rating"], 
                              "sentiment_emoji_label": result["sentiment_emoji_label"], 
                              "confidence_scores": result["confidence_scores"]})
                else:
                    st.warning("Please enter some text to analyze.")
            except Exception as e:
                st.error(f"Error during analysis or language detection: {e}. Please try again.")
        else:
            st.warning("Please enter some text to analyze.")


else:
    uploaded_file = st.file_uploader("Upload a text file (.txt or .csv) for batch sentiment analysis:", type=["txt", "csv"])

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
                texts_to_analyze = []  # Clear if no column selected

        if texts_to_analyze and st.button("Analyze Sentiment (File Upload)"):
            st.subheader("Batch Analysis Results:")
            progress_bar = st.progress(0)
            total_texts = len(texts_to_analyze)
            
            all_results = []
            for i, text in enumerate(texts_to_analyze):
                try:
                    detected_lang = detect(text)
                    result = analyze_sentiment(text)
                    if result:
                        
                        result_for_df = {k: v for k, v in result.items() if k != 'color_style'}
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
                    label="Download Results as CSV",
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
st.markdown("Pipeline build by **Divyanshu and Gaurav** for a college project using ( Streamlit, Hugging Face Transformers, and Langdetect )")
st.markdown("For any issue contact **divyanshubuisness@gmail.com**")
