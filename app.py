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

# --- Streamlit Page Config (MUST BE FIRST) ---
st.set_page_config(
    page_title="Advanced Park Review Analyzer",
    page_icon="🏞️",
    layout="wide"
)

# --- NLP Setup ---
@st.cache_resource
def load_nlp_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
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
        """Main extraction method with Selenium pagination support"""
        if use_selenium:
            return self._selenium_extractor(url)
        return self._basic_extractor(url)
    
    def _selenium_extractor(self, url):
        """Selenium-based extractor with pagination handling"""
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
            time.sleep(3)  # Initial load wait
            
            # Extract title
            title = self._get_title(driver)
            
            # Extract reviews with pagination
            reviews = self._paginated_review_extraction(driver)
            
            # Get page source for other elements
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
        """Handle multi-page review extraction"""
        reviews = []
        page = 1
        max_pages = 10  # Safety limit
        
        while page <= max_pages:
            # Extract current page reviews
            current_reviews = self._get_page_reviews(driver)
            reviews.extend(current_reviews)
            
            # Attempt to find next page
            next_page = self._find_next_button(driver)
            if not next_page:
                break
                
            # Click and wait
            try:
                next_page.click()
                time.sleep(2.5)  # Adjust based on site response
                page += 1
            except Exception as e:
                break
                
        return reviews
    
    def _get_page_reviews(self, driver):
        """Extract reviews from current page with multiple selectors"""
        selectors = [
            '.rec-reviews-card', 
            '.review-content',
            '[data-component="review"]',
            '.review-card',
            '.user-review',
            '.comment-item'
        ]
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    return [e.text.strip() for e in elements if e.text.strip()]
            except NoSuchElementException:
                continue
                
        # Fallback to paragraph extraction
        paragraphs = driver.find_elements(By.TAG_NAME, 'p')
        return [p.text.strip() for p in paragraphs if len(p.text.strip()) > 50]
    
    def _find_next_button(self, driver):
        """Find next page button with multiple selectors"""
        next_selectors = [
            'button[aria-label="Next page"]',
            'a.pagination-next',
            'li.next a',
            'a.next',
            'button.next',
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
        """Fallback basic extractor"""
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
    
    def _get_title(self, source):
        """Extract title from Selenium driver or BeautifulSoup"""
        if isinstance(source, webdriver.Chrome):
            try:
                return source.find_element(By.TAG_NAME, 'h1').text.strip()
            except:
                return "Unknown Park"
        else:
            title = source.find('h1')
            return title.text.strip() if title else "Unknown Park"
    
    # Rest of helper methods (_extract_fees, _extract_facilities, etc.) remain same
    # ... [Previous helper methods unchanged] ...

def analyze_content(url, use_selenium=True):
    scraper = ParkScraper()
    data = scraper.extract_content(url, use_selenium=use_selenium)
    
    if 'error' in data:
        return None, f"Error: {data['error']}"
    
    # Rest of analysis logic remains same
    # ... [Previous analysis logic unchanged] ...

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
            
            # Rest of UI logic remains same
            # ... [Previous UI logic unchanged] ...


