# -*- coding: utf-8 -*-
import sys
import re
import os
import requests
import time
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
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from transformers import pipeline
import streamlit as st
import torch
import spacy
from wordcloud import WordCloud
import pandas as pd
from collections import defaultdict

# Fix inotify watch limit
os.system('sysctl fs.inotify.max_user_watches=524288')

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="National Park Review Analyzer",
    page_icon="🏞️",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/navyasreebandaru7/sentimental-analysis',
        'About': "National Park Review Sentiment Analysis Tool"
    }
)

# --- System Dependencies Check ---
@st.cache_resource
def check_system_deps():
    try:
        import distutils
        from selenium import webdriver
        return True
    except (ImportError, ModuleNotFoundError) as e:
        st.error(f"Missing system dependency: {str(e)}")
        return False

# --- Enhanced NLP Setup ---
@st.cache_resource
def load_nlp_models():
    try:
        # Install spaCy model with user permissions
        if not os.path.exists(os.path.expanduser('~/.local/lib/python3.12/site-packages/en_core_web_sm')):
            os.system('python3 -m spacy download en_core_web_sm --user')
            
        nlp = spacy.load("en_core_web_sm")
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        return nlp, sentiment_analyzer
    except Exception as e:
        st.error(f"Error initializing NLP: {str(e)}")
        st.stop()

nlp, sentiment_analyzer = load_nlp_models()

# --- Constants ---
CATEGORIES = {
    'hiking': ['hiking', 'trail', 'hike'],
    'fees': ['fee', 'price', 'cost'],
    'facilities': ['restroom', 'bathroom', 'parking']
}

class ParkScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        })
        
    def extract_content(self, url):
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            
            # Simplified extraction logic
            reviews = [elem.text for elem in driver.find_elements(By.CSS_SELECTOR, '.review, .comment')]
            
            return {
                'reviews': reviews,
                'title': driver.title
            }
        except Exception as e:
            return {'error': str(e)}
        finally:
            if 'driver' in locals():
                driver.quit()

def analyze_content(url):
    scraper = ParkScraper()
    data = scraper.extract_content(url)
    
    if 'error' in data:
        return None, data['error']
    
    try:
        sentiments = []
        for review in data['reviews'][:100]:  # Limit to 100 reviews for stability
            result = sentiment_analyzer(review[:512])[0]
            sentiments.append(result)
            
        return {
            'positive': len([s for s in sentiments if s['label'] == 'POSITIVE']),
            'negative': len([s for s in sentiments if s['label'] == 'NEGATIVE']),
            'total': len(sentiments)
        }, None
    except Exception as e:
        return None, str(e)

# --- Streamlit Interface ---
st.title("National Park Review Analyzer")

url_input = st.text_input("Enter Park Review URL:", placeholder="https://example.com/reviews")

if st.button("Analyze"):
    if not check_system_deps():
        st.stop()
        
    with st.spinner("Analyzing..."):
        report, error = analyze_content(url_input)
        
        if error:
            st.error(f"Error: {error}")
        else:
            st.success(f"Analyzed {report['total']} reviews")
            st.write(f"Positive: {report['positive']}")
            st.write(f"Negative: {report['negative']}")

st.sidebar.markdown("""
**System Requirements:**
