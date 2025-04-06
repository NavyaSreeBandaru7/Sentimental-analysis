import re
import requests
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
from collections import defaultdict

# --- Streamlit Page Config (MUST BE FIRST) ---
st.set_page_config(
    page_title="Website Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# --- Constants ---
# Define categories for analysis
CATEGORIES = {
    'hiking': ['hiking', 'trail', 'hike', 'trek', 'trekking', 'paths', 'walk', 'walking'],
    'fees': ['fee', 'price', 'cost', 'payment', 'dollar', 'money', 'expensive', 'cheap', 'affordable'],
    'water': ['water', 'lake', 'river', 'stream', 'pond', 'waterfall', 'creek', 'swimming'],
    'facilities': ['facilities', 'restroom', 'bathroom', 'shower', 'toilet', 'parking']
}

# Define mock sentiment analyzer (for when transformers fails to load)
def mock_sentiment_analysis(text):
    # Simple rule-based sentiment analysis as fallback
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'beautiful', 'love', 'enjoy', 'nice', 'pleasant']
    negative_words = ['bad', 'poor', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'disappointing', 'worst', 'problem']
    
    text = text.lower()
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    if pos_count > neg_count:
        return [{'label': 'POSITIVE', 'score': 0.8}]
    elif neg_count > pos_count:
        return [{'label': 'NEGATIVE', 'score': 0.8}]
    else:
        return [{'label': 'NEUTRAL', 'score': 0.6}]

# --- Set up sentiment analysis function ---
@st.cache_resource
def load_sentiment_analyzer():
    try:
        # Try to import and load the transformer model
        import torch
        from transformers import pipeline
        
        device = 0 if torch.cuda.is_available() else -1
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )
        st.success("✅ Advanced sentiment analysis model loaded successfully")
        return sentiment_analyzer
    except Exception as e:
        st.warning(f"⚠️ Could not load advanced sentiment model: {str(e)}. Using basic analyzer instead.")
        # Return our fallback sentiment analyzer
        return mock_sentiment_analysis

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        })
    
    def extract_content(self, url):
        try:
            response = self.session.get
