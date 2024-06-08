# LLM Bias Analyzer

## Problem
From talking to collegues at a law & tech consulting firm, there is a common problem amongst users with the biases in LLM responses.
There is no real way of comparing language models from the perspective of biases. This affects everyone using language models to help to work better.

## Solution
That's why I built a simple bias benchmarking tool for prompts and responses.
Think of it as a spend towards solving the bigger problem at large!

## How does it work?
The bias is analysed using small transformer models for text classificiation.
These models were fine-trained as a multi-label classifier designed to specifically detect biases within in job descriptions.
But I what if we applied it to consumer language models?

The benchmarking tool is similar to your standard conversation AI, but with the extra insights of analyzing for potential biases in the prompt and response.
You can also compare the biases of Groq LLMs responses in the one UI.

## Run it locally
```
pip install -r requirements.txt
```

Create a ```secrets.toml``` file in the ```.streamlit``` folder.
The tool requires GroqCloud and HuggingFace API keys.
Use the following template:
```
GROQ_API_KEY = ""
BIAS_DETECTOR_API_KEY = ""
```

Start the app:
```
streamlit run Home.py
```
