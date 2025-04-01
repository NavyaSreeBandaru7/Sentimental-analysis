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

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="National Park Review Analyzer",
    page_icon="🏞️",
    layout="wide"
)

# --- System Dependencies Check ---
try:
    import distutils
except ImportError:
    st.error("Missing system dependency: python3-distutils\n\n"
             "Install with:\n"
             "``````")

# --- Enhanced NLP Setup with Permission Handling ---
@st.cache_resource
def load_nlp_models():
    # Check for spaCy model with proper permissions
    try:
        nlp = spacy.load("en_core_web_sm")
    except (OSError, PermissionError) as e:
        st.error(f"Error loading spaCy model: {str(e)}\n\n"
                 "Install with:\n"
                 "```
                 "sudo chown -R $USER:$USER $HOME/.local/lib/python3.12/site-packages\n```")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error loading NLP models: {str(e)}")
        return None, None
    
    # Configure sentiment analyzer
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        return nlp, sentiment_analyzer
    except Exception as e:
        st.error(f"Failed to initialize sentiment analyzer: {str(e)}")
        return None, None

nlp, sentiment_analyzer = load_nlp_models()
if not nlp or not sentiment_analyzer:
    st.stop()

# --- Constants ---
CATEGORIES = {
    'hiking': ['hiking', 'trail', 'hike', 'trek', 'paths', 'walk', 'walking'],
    'fees': ['fee', 'price', 'cost', 'payment', 'dollar', 'money'],
    'equipment': ['equipment', 'gear', 'tent', 'backpack', 'boots'],
    'water': ['water', 'lake', 'river', 'stream', 'swimming'],
    'facilities': ['restroom', 'bathroom', 'shower', 'toilet', 'parking']
}

class ParkScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        
    def validate_url(self, url):
        return True
    
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
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            
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

    def _paginated_review_extraction(self, driver):
        reviews = []
        page = 1
        max_pages = 5  # Reduced for safety
        
        while page <= max_pages:
            current_reviews = self._get_page_reviews(driver)
            if not current_reviews:
                break
            reviews.extend(current_reviews)
            
            next_page = self._find_next_button(driver)
            if not next_page:
                break
                
            try:
                next_page.click()
                WebDriverWait(driver, 10).until(
                    EC.staleness_of(next_page)
                )
                page += 1
            except Exception:
                break
                
        return reviews

    def _get_page_reviews(self, driver):
        selectors = [
            '.review', '.comment', '.testimonial',
            '[itemprop="review"]', '.user-content'
        ]
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    return [e.text.strip() for e in elements if e.text.strip()]
            except NoSuchElementException:
                continue
                
        return []

    def _find_next_button(self, driver):
        next_selectors = [
            'a[rel="next"]', '.pagination-next',
            'button.next', '[aria-label="Next page"]'
        ]
        
        for selector in next_selectors:
            try:
                button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                if button.is_enabled():
                    return button
            except:
                continue
        return None

    def _basic_extractor(self, url):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return {
                'reviews': [p.text.strip() for p in soup.select('p') if len(p.text.strip()) > 50],
                'fees': [],
                'facilities': [],
                'activities': [],
                'title': soup.title.text if soup.title else "Unknown Park"
            }
        except Exception as e:
            return {'error': str(e)}

def analyze_content(url, use_selenium=True):
    scraper = ParkScraper()
    data = scraper.extract_content(url, use_selenium=use_selenium)
    
    if 'error' in data:
        return None, data['error']
    
    try:
        sentiments = []
        for review in data['reviews']:
            result = sentiment_analyzer(review[:512])[0]
            sentiments.append({
                'text': review,
                'label': result['label'],
                'score': result['score']
            })
        
        return {
            'sentiments': sentiments,
            'metrics': {
                'total_reviews': len(sentiments),
                'positive': len([s for s in sentiments if s['label'] == 'POSITIVE']),
                'negative': len([s for s in sentiments if s['label'] == 'NEGATIVE'])
            },
            **data
        }, None
    except Exception as e:
        return None, f"Analysis error: {str(e)}"

# --- Streamlit Interface ---
st.title("🌲 National Park Review Analyzer")
st.write("## Comprehensive Review Analysis Tool")

url_input = st.text_input(
    "Enter Park/Travel Review URL:",
    placeholder="https://www.example.com/park-reviews",
    help="Supports any travel/park review website"
)

use_selenium = st.checkbox(
    "Enable Advanced Scraping (Recommended)",
    value=True,
    help="Uses browser automation for multi-page reviews"
)

if st.button("Analyze Reviews", type="primary"):
    if not url_input:
        st.error("Please enter a valid URL")
    else:
        with st.spinner("Analyzing... This may take 1-2 minutes"):
            report, error = analyze_content(url_input, use_selenium)
            
            if error:
                st.error(f"Analysis Failed: {error}")
            else:
                st.success(f"Analyzed {report['metrics']['total_reviews']} reviews")
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Positive Reviews", report['metrics']['positive'])
                with col2:
                    st.metric("Negative Reviews", report['metrics']['negative'])
                
                # Show word cloud
                st.subheader("Common Themes")
                text = ' '.join([r['text'] for r in report['sentiments']])
                wordcloud = WordCloud(width=800, height=400).generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud)
                plt.axis("off")
                st.pyplot(plt)

st.sidebar.markdown("""

