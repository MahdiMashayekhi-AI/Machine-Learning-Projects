from preprocessor import TextPreprocessor, TextCleaner, StopwordsRemover, Stemmer
import streamlit as st
import joblib
import pandas as pd
import base64

preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("best_extratrees_model.pkl")

st.set_page_config(page_title="📨 SMS Spam Classifier",
                   page_icon="📩", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📨 SMS Spam Classifier</h1>",
            unsafe_allow_html=True)
st.markdown("##### Paste an SMS message below and find out if it's Spam or Not.")

user_input = st.text_area("✍️ Type your message here:", height=150)

if st.button("🔍 Detect Spam"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        text_series = pd.Series([user_input])
        vectorized_input = preprocessor.transform(text_series)

        pred = model.predict(vectorized_input)[0]
        prob = model.predict_proba(vectorized_input)[0][1]

        if pred == 1:
            st.error(f"🚫 Spam Detected! (Confidence: {prob:.2f})")
        else:
            st.success(
                f"✅ This is a Ham (Not Spam). (Confidence: {1 - prob:.2f})")

st.markdown("""
<hr>
<div style='text-align:center; color: gray'>
Built with ❤️ using Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)
