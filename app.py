# -*- coding: utf-8 -*-
import sys
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
from webdriver_manager.chrome import ChromeDriverManager
from transformers import pipeline
import streamlit as st
import torch
import spacy
from wordcloud import WordCloud

# --- Permanent Fix for inotify Error ---
os.environ['STREAMLIT_SERVER_ENABLE_WATCHER'] = 'false'  # Disable file watcher
try:
    os.system('echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf')
    os.system('sudo sysctl -p')
except Exception as e:
    st.error(f"System configuration failed: {str(e)}")

# --- Streamlit Config ---
st.set_page_config(
    page_title="Park Review Analyzer",
    page_icon="🏞️",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/navyasreebandaru7/sentimental-analysis',
        'About': "Sentiment Analysis Tool for Park Reviews"
    }
)

# --- Dependencies Check ---
@st.cache_resource
def check_dependencies():
    try:
        import distutils
        from selenium import webdriver
        return True
    except ImportError as e:
        st.error(f"Missing dependency: {str(e)}\n\nRun: sudo apt-get install python3-distutils")
        return False

# --- NLP Setup ---
@st.cache_resource
def load_nlp():
    try:
        nlp = spacy.load("en_core_web_sm")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        return nlp, sentiment_pipeline
    except Exception as e:
        st.error(f"NLP initialization failed: {str(e)}")
        st.stop()

nlp, sentiment_analyzer = load_nlp()

# --- Web Scraper Class ---
class ReviewScraper:
    def __init__(self):
        self.options = Options()
        self.options.add_argument("--headless=new")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        
    def scrape_reviews(self, url):
        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=self.options
            )
            
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            
            reviews = [elem.text for elem in driver.find_elements(By.CSS_SELECTOR, '.review, .comment, [itemprop="review"]')]
            return {
                'reviews': reviews[:100],  # Limit to 100 reviews
                'title': driver.title
            }
        except Exception as e:
            return {'error': str(e)}
        finally:
            if 'driver' in locals():
                driver.quit()

# --- Analysis Functions ---
def perform_analysis(reviews):
    results = []
    for review in reviews:
        try:
            analysis = sentiment_analyzer(review[:512])[0]
            results.append(analysis)
        except Exception as e:
            continue
    return results

# --- Streamlit UI ---
st.title("National Park Review Analyzer")
url = st.text_input("Enter Review Page URL:", placeholder="https://example.com/park-reviews")

if st.button("Analyze Sentiment"):
    if not check_dependencies():
        st.stop()
        
    with st.spinner("Analyzing reviews..."):
        scraper = ReviewScraper()
        data = scraper.scrape_reviews(url)
        
        if 'error' in data:
            st.error(f"Scraping failed: {data['error']}")
            st.stop()
            
        analysis = perform_analysis(data['reviews'])
        positive = sum(1 for res in analysis if res['label'] == 'POSITIVE')
        negative = len(analysis) - positive
        
        st.success(f"Analysis Complete ({len(analysis)} reviews processed)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Positive Reviews", positive)
        with col2:
            st.metric("Negative Reviews", negative)

# --- Requirements Section ---
st.sidebar.markdown("""
**System Requirements:**
