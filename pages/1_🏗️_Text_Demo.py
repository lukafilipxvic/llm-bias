import streamlit as st
from typing import Generator
from groq import Groq
import json
import random
import requests
import time


st.set_page_config(page_icon="🏗️", layout="centered",
                   page_title="Text Demo",
                   initial_sidebar_state="expanded")


headers = {"Authorization": f"Bearer {st.secrets["BIAS_DETECTION_API_KEY"]}"
           }

with open('bias-models.json', 'r') as file:
    bias_models = json.load(file)

st.header('🏗️ Text Demo')
st.write('Test the barebone bias detection models.')

bias_model_option = st.sidebar.selectbox(
    "Choose a bias detection model:",
    options=list(bias_models.keys()),
    format_func=lambda x: bias_models[x]["name"],
    index=1 # Default to d4data
)

st.sidebar.text('2024 Build Club AI: Hackathon')

def analyze_bias(bias_model, text):
    """Analyze the bias of the given text using Hugging Face's NLP API."""
    retries = 5  # Maximum number of retries
    delay = 20  # Delay in seconds (as suggested by the error message)
    
    for attempt in range(retries):
        response = requests.post(bias_model, headers=headers, json={"inputs": text})
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503 and "currently loading" in response.text:
            if attempt < retries - 1:
                time.sleep(delay)  # Wait for the estimated time before retrying
            else:
                return {"error": "Model loading timeout", "message": "Failed to load model after several attempts"}
        else:
            return {"error": response.status_code, "message": response.text}

test_text = st.chat_input()

if test_text:
    st.write(test_text)
    with st.spinner('Analyzing bias...'):
        text_analysis = analyze_bias(bias_model_option, test_text)
    bias_data = []
    for bias in text_analysis[0]:
        bias_data.append({"Bias Type": bias['label'].capitalize(), "Prompt Bias (%)": f"{bias['score'] * 100:.1f}"})
    st.dataframe(bias_data)