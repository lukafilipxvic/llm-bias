import streamlit as st
from typing import Generator
from groq import Groq
import json
import requests
import time


st.set_page_config(page_icon="🔎", layout="centered",
                   page_title="LLM Bias Detection",
                   initial_sidebar_state="expanded")


client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)
 
headers = {"Authorization": f"Bearer {st.secrets["BIAS_DETECTION_API_KEY"]}"}

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define bias and llm model details from files
with open('groq-models.json', 'r') as file:
    models = json.load(file)

with open('bias-models.json', 'r') as file:
    bias_models = json.load(file)

with st.sidebar:
    st.header('LLM Bias Detection 🔎')
    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=2  # Default to llama3-8b
    )
    bias_model_option = st.selectbox(
        "Choose a bias detection model:",
        options=list(bias_models.keys()),
        format_func=lambda x: bias_models[x]["name"],
        index=0 # Default to d4data
    )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

with st.sidebar:
    # Adjust max_tokens slider dynamically based on the selected model
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,  # Minimum value to allow some flexibility
        max_value=max_tokens_range,
        # Default value or max allowed if less
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
    )
    temperature = st.slider(
        "Temperature:",
        min_value=0.00,
        max_value=2.0,
        value=1.00,
        step=0.01,
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

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

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def main():
    with st.spinner("Loading bias detection model..."):
        analyze_bias(bias_model_option,"test")

    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👨‍💻'):
            st.markdown(prompt)

        # Fetch response from Groq API
        try:
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=[
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            # Use the generator function with st.write_stream
            with st.chat_message("assistant", avatar="🤖"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)
        except Exception as e:
            st.error(e, icon="🚨")

        # Append the full response to session_state.messages
        if isinstance(full_response, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_response})
            
        with st.spinner("Analyzing..."):
            prompt_analysis = analyze_bias(bias_model_option, prompt)
            response_analysis = analyze_bias(bias_model_option, full_response)

        st.write(f'Prompt bias: {prompt_analysis[0][0]['score'] * 100:.1f}%')
        try:
            st.write(f'Response bias: {response_analysis[0][0]['score'] * 100:.1f}%')
        except KeyError:
            st.write("Response bias: Response too long to analyse.")

if __name__ == "__main__":
    main()