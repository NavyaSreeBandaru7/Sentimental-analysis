# -*- coding: utf-8 -*-
import sys
import os
import re
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from transformers import pipeline
import streamlit as st
import torch
import spacy
from wordcloud import WordCloud
import pandas as pd
from collections import defaultdict

# --- Fix Permission Issues ---
USER_HOME = os.path.expanduser("~")
SPACY_MODEL_DIR = os.path.join(USER_HOME, ".spacy_models")
os.makedirs(SPACY_MODEL_DIR, exist_ok=True)

# --- Streamlit Config ---
st.set_page_config(
    page_title="National Park Review Analyzer",
    page_icon="🏞️",
    layout="wide"
)

# --- NLP Setup with Permissions Fix ---
@st.cache_resource
def load_nlp_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except (OSError, IOError):
        try:
            import subprocess
            subprocess.run([
                sys.executable, 
                "-m", "spacy", "download", 
                "--user",  # Install to user directory
                "en_core_web_sm"
            ], check=True)
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"""SpaCy model installation failed. Manually install with:
                    \n```
                    python -m spacy download en_core_web_sm --user
                    ```\nError: {str(e)}""")
            raise

    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )
    return nlp, sentiment_analyzer

try:
    nlp, sentiment_analyzer = load_nlp_models()
except Exception as e:
    st.error(f"Critical Error: {str(e)}")
    st.stop()

# --- Rest of the original code remains unchanged ---
# [Keep all your existing class definitions and analysis logic here]
