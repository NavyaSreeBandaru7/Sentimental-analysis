# -*- coding: utf-8 -*-
import re
import sys
import time
import requests
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from transformers import pipeline
import streamlit as st
import torch
import spacy
from wordcloud import WordCloud
import pandas as pd
from collections import defaultdict

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Advanced Park Review Analyzer",
    page_icon="🏞️",
    layout="wide"
)

# --- Dependency Validation ---
@st.cache_resource
def validate_dependencies():
    try:
        # Check critical dependencies
        from webdriver_manager.chrome import ChromeDriverManager
        import selenium
        import torch
        return True
    except ImportError as e:
        st.error(f"Missing critical dependency: {str(e)}")
        st.stop()
        return False

# --- NLP Setup ---
@st.cache_resource
def load_nlp_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
    
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )
    return nlp, sentiment_analyzer

try:
    nlp, sentiment_analyzer = load_nlp_models()
except Exception as e:
    st.error(f"Error loading NLP models: {str(e)}")
    nlp, sentiment_analyzer = None, None

# --- Constants ---
CATEGORIES = {
    'hiking': ['hiking', 'trail', 'hike', 'trek', 'trekking', 'paths', 'walk', 'walking'],
    'fees': ['fee', 'price', 'cost', 'payment', 'dollar', 'money', 'expensive', 'cheap', 'affordable'],
    'equipment': ['equipment', 'gear', 'supplies', 'tent', 'backpack', 'boots', 'poles', 'shoes'],
    'water': ['water', 'lake', 'river', 'stream', 'pond', 'waterfall', 'creek', 'swimming'],
    'facilities': ['facilities', 'restroom', 'bathroom', 'shower', 'toilet', 'visitor center', 'parking']
}

class ParkScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        
    def validate_url(self, url):
        return True  # Domain restriction removed
    
    def extract_content(self, url, use_selenium=True):
        if use_selenium:
            return self._selenium_extractor(url)
        return self._basic_extractor(url)
    
    def _selenium_extractor(self, url):
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            driver.get(url)
            time.sleep(3)
            
            title = self._get_title(driver)
            reviews = self._paginated_review_extraction(driver)
            
            page_content = driver.page_source
            soup = BeautifulSoup(page_content, 'html.parser')
            driver.quit()
            
            return {
                'reviews': reviews,
                'fees': self._extract_fees(soup),
                'facilities': self._extract_facilities(soup),
                'activities': self._extract_activities(soup),
                'title': title
            }
        except Exception as e:
            return {'error': f"Selenium error: {str(e)}"}

    # Rest of the class methods remain same as previous version...

# Rest of the analysis and UI code remains same as previous version...

# Streamlit Interface
st.title("🌲 Advanced Park Review Analyzer")
st.write("""
This tool analyzes reviews from any park/travel website. 
Supports multi-page review scraping using advanced browser automation.
""")

# Dependency check before proceeding
validate_dependencies()

url_input = st.text_input(
    "Enter Park/Travel URL", 
    placeholder="https://www.example.com/park-reviews"
)

use_selenium = st.checkbox("Enable Complete Review Scraping (Recommended)", value=True,
                          help="Uses browser automation to scrape all review pages")

if st.button("Analyze", type="primary"):
    if not url_input:
        st.error("Please enter a URL to analyze")
    else:
        with st.spinner("Analyzing... This may take 2-3 minutes for large sites"):
            report, error = analyze_content(url_input, use_selenium=use_selenium)
            
            # Display logic remains same as previous version...
