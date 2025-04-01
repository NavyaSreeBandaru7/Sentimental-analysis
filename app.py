# -*- coding: utf-8 -*-
import sys
import re
import os
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
    except (OSError, PermissionError):
        st.error("spaCy model missing or permission issues. Install with:\n"
                 "``````")
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
        st.error(f"Failed to initialize NLP models: {str(e)}")
        return None, None

nlp, sentiment_analyzer = load_nlp_models()
if not nlp or not sentiment_analyzer:
    st.stop()

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

    def _paginated_review_extraction(self, driver):
        reviews = []
        page = 1
        max_pages = 10
        
        while page <= max_pages:
            current_reviews = self._get_page_reviews(driver)
            reviews.extend(current_reviews)
            
            next_page = self._find_next_button(driver)
            if not next_page:
                break
                
            try:
                next_page.click()
                time.sleep(2.5)
                page += 1
            except Exception:
                break
                
        return reviews

    def _get_page_reviews(self, driver):
        selectors = [
            '.rec-reviews-card', '.review-content',
            '[data-component="review"]', '.review-card',
            '.user-review', '.comment-item'
        ]
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    return [e.text.strip() for e in elements if e.text.strip()]
            except NoSuchElementException:
                continue
                
        paragraphs = driver.find_elements(By.TAG_NAME, 'p')
        return [p.text.strip() for p in paragraphs if len(p.text.strip()) > 50]

    def _find_next_button(self, driver):
        next_selectors = [
            'button[aria-label="Next page"]', 'a.pagination-next',
            'li.next a', 'a.next', 'button.next',
            '[data-testid="pagination-next"]'
        ]
        
        for selector in next_selectors:
            try:
                button = driver.find_element(By.CSS_SELECTOR, selector)
                if button.is_displayed() and button.is_enabled():
                    return button
            except NoSuchElementException:
                continue
        return None

    def _basic_extractor(self, url):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return {
                'reviews': self._get_basic_reviews(soup),
                'fees': self._extract_fees(soup),
                'facilities': self._extract_facilities(soup),
                'activities': self._extract_activities(soup),
                'title': self._get_title(soup)
            }
        except Exception as e:
            return {'error': str(e)}

    # Remaining helper methods (_extract_fees, _extract_facilities, etc.)
    # ... [Keep original helper methods unchanged] ...

def analyze_content(url, use_selenium=True):
    scraper = ParkScraper()
    data = scraper.extract_content(url, use_selenium=use_selenium)
    
    if 'error' in data:
        return None, f"Error: {data['error']}"
    
    # Analysis logic remains unchanged
    # ... [Keep original analysis logic] ...

# Streamlit Interface
st.title("🌲 Advanced Park Review Analyzer")
st.write("""
This tool analyzes reviews from any park/travel website. 
Supports multi-page review scraping using advanced browser automation.
""")

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
            
            if error:
                st.error(error)
            elif report:
                # Display logic remains unchanged
                # ... [Keep original display logic] ...

st.sidebar.header("About")
st.sidebar.write("""
This tool uses natural language processing to analyze reviews from travel websites.
It extracts information about facilities, activities, and visitor sentiments.
""")
