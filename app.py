import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from transformers import pipeline
import spacy
from wordcloud import WordCloud
import seaborn as sns
import re
import time

st.set_page_config(
    page_title="Park Sentiment Analyzer",
    page_icon="🌲",
    layout="wide"
)

CATEGORIES = {
    'hiking': ['trail', 'hike', 'trek', 'path', 'backpack'],
    'fees': ['fee', 'price', 'cost', 'ticket', 'payment'],
    'equipment': ['gear', 'equipment', 'clothing', 'boots', 'tent'],
    'water': ['lake', 'river', 'water', 'swim', 'kayak'],
    'facilities': ['restroom', 'parking', 'campground', 'visitor center', 'picnic']
}

@st.cache_resource
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return nlp, analyzer

def scrape_comments(url):
    driver = setup_driver()
    driver.get(url)
    comments = []
    
    for _ in range(3):
        time.sleep(2)
        elements = driver.find_elements(By.XPATH, '//div[contains(@class, "comment")]')
        comments += [e.text for e in elements if e.text.strip()]
        try:
            driver.find_element(By.XPATH, '//a[contains(text(), "Next")]').click()
        except:
            break
    
    driver.quit()
    return comments

def analyze_text(text, analyzer, nlp):
    return {
        'sentiment': analyzer(text[:512])[0]['label'].lower(),
        'categories': list({cat for cat, keys in CATEGORIES.items() if any(k in text.lower() for k in keys)} or ['general'])
    }

def main():
    st.title("National Park Sentiment Analysis 🌄")
    url = st.text_input("Enter Review Page URL:")
    
    if url:
        with st.spinner("Analyzing..."):
            try:
                comments = scrape_comments(url)
                if not comments:
                    st.warning("No comments found")
                    return
                
                nlp, analyzer = load_models()
                results = []
                
                for comment in comments:
                    analysis = analyze_text(comment, analyzer, nlp)
                    results.append({
                        'text': comment,
                        **analysis
                    })
                
                df = pd.DataFrame(results)
                
                # Visualizations
                st.header("Overall Sentiment Distribution")
                fig, ax = plt.subplots(figsize=(10,5))
                sns.countplot(x='sentiment', data=df, palette='viridis')
                st.pyplot(fig)
                
                st.header("Category Analysis")
                category_data = pd.DataFrame([{'category': cat} for item in results for cat in item['categories']])
                if not category_data.empty:
                    fig2, ax2 = plt.subplots(figsize=(12,6))
                    sns.countplot(x='category', data=category_data, palette='Set2')
                    plt.xticks(rotation=45)
                    st.pyplot(fig2)
                
                st.header("Word Clouds")
                col1, col2, col3 = st.columns(3)
                for col, sentiment in zip([col1, col2, col3], ['positive', 'neutral', 'negative']):
                    with col:
                        texts = ' '.join(df[df['sentiment'] == sentiment]['text'])
                        if texts:
                            wc = WordCloud(width=400, height=300).generate(texts)
                            plt.figure(figsize=(6,4))
                            plt.imshow(wc)
                            plt.axis('off')
                            st.pyplot()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
