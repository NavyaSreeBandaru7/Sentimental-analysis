# app.py
import streamlit as st

# --- Streamlit Page Config (MUST BE FIRST) ---
st.set_page_config(
    page_title="National Park Sentiment Analyzer",
    page_icon="🌲",
    layout="wide"
)

# Rest of imports
import re
import time
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from transformers import pipeline
import spacy
from wordcloud import WordCloud
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Constants ---
CATEGORIES = {
    'hiking': ['trail', 'hike', 'trek', 'path', 'backpack'],
    'fees': ['fee', 'price', 'cost', 'ticket', 'payment'],
    'equipment': ['gear', 'equipment', 'clothing', 'boots', 'tent'],
    'water': ['lake', 'river', 'water', 'swim', 'kayak'],
    'facilities': ['restroom', 'parking', 'campground', 'visitor center', 'picnic']
}

# --- Selenium Setup ---
@st.cache_resource
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

# --- NLP Models ---
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return nlp, sentiment_pipeline

# --- Scraping Functions ---
def scrape_all_comments(url):
    driver = get_driver()
    driver.get(url)
    time.sleep(3)
    
    comments = []
    page_count = 0
    
    while page_count < 5:  # Max 5 pages to prevent infinite loop
        # Extract comments
        elements = driver.find_elements(By.XPATH, '//div[contains(@class, "comment")]')
        comments += [e.text for e in elements if e.text.strip()]
        
        # Try to find next page
        try:
            next_btn = driver.find_element(By.XPATH, '//a[contains(text(), "Next")]')
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(3)
            page_count += 1
        except:
            break
    
    driver.quit()
    return comments

# --- Sentiment Analysis ---
def analyze_sentiment(text, analyzer):
    try:
        return analyzer(text[:512])[0]
    except:
        return {'label': 'NEUTRAL', 'score': 0.5}

# --- Visualization Functions ---
def plot_sentiment_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='sentiment', data=df, palette={'positive':'green', 'neutral':'gray', 'negative':'red'})
    plt.title('Overall Sentiment Distribution')
    st.pyplot(fig)

def plot_category_sentiment(category_data):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='category', y='count', hue='sentiment', data=category_data, 
                palette={'positive':'green', 'neutral':'gray', 'negative':'red'})
    plt.xticks(rotation=45)
    plt.title('Sentiment Distribution by Category')
    st.pyplot(fig)

def generate_wordcloud(texts, sentiment):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(texts))
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'{sentiment.capitalize()} Words Word Cloud')
    plt.axis('off')
    st.pyplot(fig)

# --- Main App ---
def main():
    st.title("National Park Sentiment Analysis 🌄")
    url = st.text_input("Enter National Park Review URL:")
    
    if url:
        with st.spinner("Scraping and Analyzing..."):
            try:
                # Scrape comments
                comments = scrape_all_comments(url)
                if not comments:
                    st.warning("No comments found!")
                    return
                
                # Load models
                nlp, sentiment_analyzer = load_models()
                
                # Process comments
                results = []
                category_data = []
                
                for comment in comments:
                    # Sentiment analysis
                    sentiment = analyze_sentiment(comment, sentiment_analyzer)
                    result = {
                        'text': comment,
                        'sentiment': sentiment['label'].lower(),
                        'score': sentiment['score']
                    }
                    results.append(result)
                    
                    # Category detection
                    doc = nlp(comment.lower())
                    categories = []
                    for cat, keywords in CATEGORIES.items():
                        if any(keyword in token.text for token in doc for keyword in keywords):
                            categories.append(cat)
                    categories = categories or ['general']
                    
                    # Update category data
                    for cat in categories:
                        category_data.append({
                            'category': cat,
                            'sentiment': result['sentiment'],
                            'count': 1
                        })
                
                # Create DataFrames
                df = pd.DataFrame(results)
                cat_df = pd.DataFrame(category_data).groupby(['category', 'sentiment']).sum().reset_index()
                
                # Visualizations
                st.header("Overall Analysis")
                plot_sentiment_distribution(df)
                
                st.header("Category-wise Analysis")
                plot_category_sentiment(cat_df)
                
                st.header("Word Clouds")
                col1, col2, col3 = st.columns(3)
                with col1:
                    generate_wordcloud(df[df['sentiment'] == 'positive']['text'], 'positive')
                with col2:
                    generate_wordcloud(df[df['sentiment'] == 'neutral']['text'], 'neutral')
                with col3:
                    generate_wordcloud(df[df['sentiment'] == 'negative']['text'], 'negative')
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
