import streamlit as st
from typing import Generator
from groq import Groq
import json
import random
import requests
import time


st.set_page_config(page_icon="⚔️", layout="wide",
                   page_title="Elo Arena",
                   initial_sidebar_state="expanded")


client = Groq(
    api_key=f"{st.secrets["GROQ_API_KEY"]}",
)
headers = {"Authorization": f"Bearer {st.secrets["BIAS_DETECTION_API_KEY"]}"
           }

# Initialize chat history and selected model
if "messages1" not in st.session_state:
    st.session_state.messages1 = []
if "messages2" not in st.session_state:
    st.session_state.messages2 = []


if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define bias and llm model details from files
with open('groq-models.json', 'r') as file:
    models = json.load(file)

with open('bias-models.json', 'r') as file:
    bias_models = json.load(file)

st.header('⚔️ Elo Arena')
st.write('Battle the bias of 2 random language models. Select which model is more bias.')

bias_model_option = st.sidebar.selectbox(
    "Choose a bias detection model:",
    options=list(bias_models.keys()),
    format_func=lambda x: bias_models[x]["name"],
    index=1 # Default to d4data
)

temperature = st.sidebar.slider(
        "Temperature:",
        min_value=0.00,
        max_value=2.0,
        value=1.00,
        step=0.01,
)

st.sidebar.text('2024 Build Club AI: Hackathon')

col1, col2 = st.columns([1,1])

model_keys = list(models.keys())
random.shuffle(model_keys)  # Shuffle the model keys randomly

model_option1 = model_keys[0]  # Select the first model after shuffling
model_option2 = model_keys[1]  # Select the second model after shuffling
selected_models = [model_option1, model_option2]

col1.write(f'LLM A:')
col2.write(f'LLM B:')

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
    # Check if a button was pressed in the previous run and display the toast
    if 'button_pressed' in st.session_state:
        if st.session_state['button_pressed'] == 'A':
            st.toast(f'selected A: {st.session_state["model_option1"]}')
            time.sleep(0.5)
            st.toast(f'Model B was {st.session_state["model_option2"]}')
        elif st.session_state['button_pressed'] == 'B':
            st.toast(f'selected B: {st.session_state["model_option2"]}')
            time.sleep(0.5)
            st.toast(f'Model A was {st.session_state["model_option1"]}')
        # Reset the button press state
        del st.session_state['button_pressed']

    with st.spinner("Loading bias detection model..."):
        analyze_bias(bias_model_option, "test")

    if prompt := st.chat_input("Enter your prompt here..."):
        # Append the user's prompt to session_state.messages for both models

        st.session_state.messages1.append({"role": "user", "content": prompt})
        st.session_state.messages2.append({"role": "user", "content": prompt})

        # Display the user's prompt in chat for both models
        with st.chat_message("user", avatar='👨‍💻'):
            st.markdown(prompt)

        # Fetch and display responses from Groq API for each selected model
        for index, model_option in enumerate(selected_models):
            col = col1 if index == 0 else col2  # Choose the column based on the model index
            messages = st.session_state.messages1 if index == 0 else st.session_state.messages2

            try:
                chat_completion = client.chat.completions.create(
                    model=model_option,
                    messages=[
                        {
                            "role": m["role"],
                            "content": m["content"]
                        }
                        for m in messages
                    ],
                    temperature=temperature,
                    stream=True
                )

                with col.chat_message("assistant", avatar="🤖"):
                    chat_responses_generator = generate_chat_responses(chat_completion)
                    full_response = col.write_stream(chat_responses_generator)

                if isinstance(full_response, str):
                    messages.append(
                        {"role": "assistant", "content": full_response})
                else:
                    # Handle the case where full_response is not a string
                    combined_response = "\n".join(str(item) for item in full_response)
                    messages.append(
                        {"role": "assistant", "content": combined_response})

            except Exception as e:
                col.error(e, icon="🚨")

        colA, colB = st.columns([1,1])
        if colA.button(label='Model A is more bias 👈'):
            st.session_state['button_pressed'] = 'A'
            st.session_state['model_option1'] = model_option1
            st.session_state['model_option2'] = model_option2
        if colB.button(label='Model B is more bias👉'):
            st.session_state['button_pressed'] = 'B'
            st.session_state['model_option1'] = model_option1
            st.session_state['model_option2'] = model_option2


if __name__ == "__main__":
    main()