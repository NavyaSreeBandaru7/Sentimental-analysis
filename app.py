# -*- coding: utf-8 -*-
import sys
import re
import time
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from transformers import pipeline
import streamlit as st
import torch
import spacy
from wordcloud import WordCloud
import pandas as pd
from collections import defaultdict

# --- System Check ---
try:
    import distutils
except ImportError:
    st.error("Missing distutils package. Install with: sudo apt-get install python3-distutils")

# --- Streamlit Config ---
st.set_page_config(
    page_title="National Park Review Analyzer",
    page_icon="🏞️",
    layout="wide"
)

# --- Dependency Check ---
@st.cache_resource
def check_dependencies():
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        import selenium
        import torch
        return True
    except ImportError as e:
        st.error(f"Missing dependency: {str(e)}")
        return False

# --- Enhanced NLP Setup ---
@st.cache_resource
def load_nlp_models():
    # Ensure spaCy model is installed
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Configure sentiment analyzer
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )
    return nlp, sentiment_analyzer

try:
    nlp, sentiment_analyzer = load_nlp_models()
except Exception as e:
    st.error(f"Initialization Error: {str(e)}")
    st.stop()

# --- Constants and Class Definitions ---
# (Keep the rest of your existing code here without changes)
