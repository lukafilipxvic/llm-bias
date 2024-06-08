import streamlit as st
from groq import Groq
import requests

import os

client = Groq(
    api_key=f"{st.secrets["GROQ_API_KEY"]}",
)
api_url = "https://api-inference.huggingface.co/models/d4data/bias-detection-model" # https://api-inference.huggingface.co/models/D1V1DE/bias-detection
headers = {"Authorization": f"Bearer {st.secrets["BIAS_DETECTION_API_KEY"]}"
           }

def analyze_bias(payload):
	response = requests.post(api_url, headers=headers, json=payload)
	return response.json()

def groq_query(prompt, model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"prompt",
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content


import time

def analyze_bias(text):
    """Analyze the bias of the given text using Hugging Face's NLP API."""
    retries = 5  # Maximum number of retries
    delay = 20  # Delay in seconds (as suggested by the error message)

    for attempt in range(retries):
        response = requests.post(api_url, headers=headers, json={"inputs": text})
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503 and "currently loading" in response.text:
            if attempt < retries - 1:
                time.sleep(delay)  # Wait for the estimated time before retrying
            else:
                return {"error": "Model loading timeout", "message": "Failed to load model after several attempts"}
        else:
            return {"error": response.status_code, "message": response.text}

def main():
    st.title("Bias Checker for Large Language Model")
    st.write("Enter a prompt and its response to check for bias.")

    model = st.selectbox(label="Select LLM Model", options=['gemma-7b-it','llama3-70b-8192','llama3-8b-8192','mixtral-8x7b-32768'])
    prompt = st.text_area("Prompt")
    response = st.text_area("Response")

    if st.button("Check Bias"):
        if prompt and response:
            with st.spinner("Analyzing..."):
                prompt_analysis = analyze_bias(prompt)
                response_analysis = analyze_bias(response)

            st.subheader("Prompt Analysis")
            st.json(prompt_analysis)

            st.subheader("Response Analysis")
            st.json(response_analysis)
        else:
            st.error("Please enter both a prompt and a response.")

if __name__ == "__main__":
    main()
