# app.py
# IMPORTANT: This import and page config must be first
import streamlit as st

# --- Streamlit Page Config (MUST BE FIRST) ---
st.set_page_config(
    page_title="Website Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# Import remaining libraries
import re
import requests
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import time
import warnings
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from wordcloud import WordCloud
import seaborn as sns
import json
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Loading Models ---
@st.cache_resource
def load_nlp_models():
    with st.spinner("Loading models... (this may take a minute on first run)"):
        try:
            import spacy
            # Load spacy model
            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                st.warning(f"Spacy model loading issue: {e}")
                try:
                    # Try downloading it directly
                    st.info("Downloading spaCy model...")
                    spacy.cli.download("en_core_web_sm")
                    nlp = spacy.load("en_core_web_sm")
                except Exception as e2:
                    st.error(f"Failed to download spaCy model: {e2}")
                    nlp = None
            
            # Load transformers only if spaCy loads successfully
            if nlp:
                try:
                    import torch
                    from transformers import pipeline
                    
                    # Get appropriate device
                    device = -1  # Default to CPU
                    if torch.cuda.is_available():
                        device = 0
                    
                    # Use a more recent model that works with updated PyTorch
                    sentiment_analyzer = pipeline(
                        "sentiment-analysis", 
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=device
                    )
                    return nlp, sentiment_analyzer
                except Exception as e:
                    st.error(f"Transformer model error: {e}")
                    st.info("Using simplified sentiment analysis as fallback.")
            
            return nlp, None
        except Exception as e:
            st.error(f"Failed to initialize NLP: {e}")
            return None, None

# Only load models when needed
if 'nlp' not in st.session_state:
    st.session_state.nlp = None
    st.session_state.sentiment_analyzer = None

# --- Constants ---
# Define categories for analysis
CATEGORIES = {
    'hiking': ['hiking', 'trail', 'hike', 'trek', 'trekking', 'paths', 'walk', 'walking', 'hiked', 'treks'],
    'fees': ['fee', 'price', 'cost', 'payment', 'dollar', 'money', 'expensive', 'cheap', 'affordable', 'budget', 'pricing', 'paid', 'free'],
    'equipment': ['equipment', 'gear', 'supplies', 'tent', 'backpack', 'boots', 'poles', 'shoes', 'clothing', 'jacket', 'gloves', 'sunglasses'],
    'water': ['water', 'lake', 'river', 'stream', 'pond', 'waterfall', 'creek', 'swimming', 'boat', 'kayak', 'canoe', 'rafting', 'splash'],
    'facilities': ['facilities', 'restroom', 'bathroom', 'shower', 'toilet', 'visitor center', 'parking', 'campground', 'campsite', 'bench', 'picnic', 'wifi', 'signal']
}

# Selenium WebDriver setup with caching
@st.cache_resource
def get_webdriver():
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        st.error(f"Error initializing WebDriver: {str(e)}")
        return None

class WebScraper:
    def __init__(self, use_selenium=True):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        self.use_selenium = use_selenium
        self.driver = None
        
        if use_selenium:
            self.driver = get_webdriver()
    
    def __del__(self):
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
    
    def extract_content_with_selenium(self, url):
        try:
            if not self.driver:
                st.warning("Selenium WebDriver not available. Falling back to requests.")
                return self.extract_content_with_requests(url)
            
            # Handle URLs without http/https
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            self.driver.get(url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Try to find reviews
            reviews = []
            site_type = self._detect_site_type(url)
            
            # Get page title
            title = self.driver.title
            
            # Different scraping strategies based on site type
            if site_type == "tripadvisor":
                reviews.extend(self._scrape_tripadvisor_reviews())
            elif site_type == "yelp":
                reviews.extend(self._scrape_yelp_reviews())
            elif site_type == "google":
                reviews.extend(self._scrape_google_reviews())
            elif site_type == "nps":
                reviews.extend(self._scrape_nps_reviews())
            else:
                # Generic review scraping
                reviews = self._scrape_generic_reviews()
            
            # If no reviews found through specific methods, try generic approach
            if not reviews:
                reviews = self._scrape_generic_reviews()
                
            # If still no content, get at least page text
            if not reviews:
                page_source = self.driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                all_text = soup.get_text(strip=True)
                if all_text:
                    # Split into manageable chunks
                    chunks = [all_text[i:i+500] for i in range(0, len(all_text), 500)]
                    reviews = chunks
            
            # Extract metadata
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            return {
                'reviews': reviews,
                'fees': self._extract_fees(soup),
                'facilities': self._extract_facilities(soup),
                'activities': self._extract_activities(soup),
                'title': title or self._extract_title(soup)
            }
        except Exception as e:
            st.error(f"Selenium error: {str(e)}")
            # Fall back to requests method
            return self.extract_content_with_requests(url)
        
    def _detect_site_type(self, url):
        if "tripadvisor" in url:
            return "tripadvisor"
        elif "yelp" in url:
            return "yelp"
        elif "google" in url and "maps" in url:
            return "google"
        elif "nps.gov" in url:
            return "nps"
        else:
            return "unknown"
            
    def _scrape_tripadvisor_reviews(self):
        reviews = []
        try:
            # Click "More" buttons to expand reviews
            more_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".moreLink")
            for button in more_buttons[:10]:  # Limit to avoid too many clicks
                try:
                    self.driver.execute_script("arguments[0].click();", button)
                    time.sleep(0.5)
                except:
                    pass
            
            # Extract the reviews
            review_elements = self.driver.find_elements(By.CSS_SELECTOR, ".review-container .reviewSelector")
            
            for element in review_elements:
                try:
                    review_text = element.find_element(By.CSS_SELECTOR, ".partial_entry").text
                    if review_text and len(review_text) > 20:
                        reviews.append(review_text)
                except:
                    continue
                    
            # If there are pagination controls, try to get more reviews
            if len(reviews) < 20:  # Only if we don't have enough reviews yet
                try:
                    # Find the next page button
                    next_button = self.driver.find_element(By.CSS_SELECTOR, ".nav.next")
                    if next_button and next_button.is_enabled():
                        next_button.click()
                        time.sleep(3)
                        
                        # Get reviews from the new page
                        new_review_elements = self.driver.find_elements(By.CSS_SELECTOR, ".review-container .reviewSelector")
                        for element in new_review_elements:
                            try:
                                review_text = element.find_element(By.CSS_SELECTOR, ".partial_entry").text
                                if review_text and len(review_text) > 20:
                                    reviews.append(review_text)
                            except:
                                continue
                except:
                    pass
        except Exception as e:
            st.warning(f"Error scraping TripAdvisor reviews: {str(e)}")
        
        return reviews
        
    def _scrape_yelp_reviews(self):
        reviews = []
        try:
            # Expand review content if needed
            more_links = self.driver.find_elements(By.CSS_SELECTOR, "a.css-jd4g08")
            for link in more_links[:10]:
                try:
                    self.driver.execute_script("arguments[0].click();", link)
                    time.sleep(0.5)
                except:
                    pass
            
            # Extract reviews
            review_elements = self.driver.find_elements(By.CSS_SELECTOR, ".review__09f24__oHr9V")
            for element in review_elements:
                try:
                    review_text = element.find_element(By.CSS_SELECTOR, ".comment__09f24__gu0rG").text
                    if review_text and len(review_text) > 20:
                        reviews.append(review_text)
                except:
                    continue
                    
            # Try to get more reviews by pagination
            if len(reviews) < 20:
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, ".pagination__09f24__VRjN4 .next-link")
                    if next_button:
                        next_button.click()
                        time.sleep(3)
                        
                        # Get reviews from next page
                        new_review_elements = self.driver.find_elements(By.CSS_SELECTOR, ".review__09f24__oHr9V")
                        for element in new_review_elements:
                            try:
                                review_text = element.find_element(By.CSS_SELECTOR, ".comment__09f24__gu0rG").text
                                if review_text and len(review_text) > 20:
                                    reviews.append(review_text)
                            except:
                                continue
                except:
                    pass
        except Exception as e:
            st.warning(f"Error scraping Yelp reviews: {str(e)}")
        
        return reviews
        
    def _scrape_google_reviews(self):
        reviews = []
        try:
            # First, try to click on reviews tab if we're on a Google Maps place page
            try:
                reviews_tab = self.driver.find_element(By.CSS_SELECTOR, "button[data-tab-index='1']")
                reviews_tab.click()
                time.sleep(3)
            except:
                pass
                
            # Scroll to load more reviews
            scroll_attempts = 0
            while scroll_attempts < 5:  # Limit scrolls to avoid infinite loop
                self.driver.execute_script("document.querySelector('div[role=\"feed\"]').scrollTop += 1000")
                time.sleep(1)
                scroll_attempts += 1
            
            # Extract review elements
            review_elements = self.driver.find_elements(By.CSS_SELECTOR, ".jftiEf")
            
            for element in review_elements:
                try:
                    # Try to expand the review if it's collapsed
                    try:
                        more_button = element.find_element(By.CSS_SELECTOR, ".w8nwRe")
                        self.driver.execute_script("arguments[0].click();", more_button)
                        time.sleep(0.5)
                    except:
                        pass
                    
                    # Get the review text
                    review_text = element.find_element(By.CSS_SELECTOR, ".wiI7pd").text
                    if review_text and len(review_text) > 20:
                        reviews.append(review_text)
                except:
                    continue
        except Exception as e:
            st.warning(f"Error scraping Google reviews: {str(e)}")
        
        return reviews
        
    def _scrape_nps_reviews(self):
        reviews = []
        try:
            # NPS.gov doesn't have reviews directly, so get content from different sections
            
            # Get main content
            content_elements = self.driver.find_elements(By.CSS_SELECTOR, ".Main-content p, .Main-content li")
            for element in content_elements:
                try:
                    text = element.text
                    if text and len(text) > 30:  # Slightly longer to avoid navigation text
                        reviews.append(text)
                except:
                    continue
                    
            # Get visitor comments if available
            comment_elements = self.driver.find_elements(By.CSS_SELECTOR, ".comment-content")
            for element in comment_elements:
                try:
                    text = element.text
                    if text and len(text) > 20:
                        reviews.append(text)
                except:
                    continue
        except Exception as e:
            st.warning(f"Error scraping NPS content: {str(e)}")
        
        return reviews
        
    def _scrape_generic_reviews(self):
        reviews = []
        try:
            # Common CSS selectors for reviews across various websites
            review_selectors = [
                '.review', '.comment', '.testimonial', '.feedback', 
                '.rating', '.user-review', '.customer-review', '.product-review',
                '.review-text', '.review-content', '.comment-text', '.comment-content',
                '[class*="review"]', '[class*="comment"]', '[id*="review"]', '[id*="comment"]',
                'p', 'article'
            ]
            
            # Try each selector
            for selector in review_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                
                for element in elements:
                    try:
                        text = element.text
                        if text and len(text) > 20 and text not in reviews:  # Avoid duplicates
                            reviews.append(text)
                    except:
                        continue
                
                if len(reviews) >= 5:  # If we got some content, that's good enough
                    break
        except Exception as e:
            st.warning(f"Error scraping generic content: {str(e)}")
        
        return reviews
    
    def extract_content_with_requests(self, url):
        try:
            # Handle URLs without http/https
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to extract reviews using common review selectors
            reviews = []
            # Common CSS selectors for reviews across various websites
            review_selectors = [
                '.rec-reviews-card', '.review-content', '[data-component="review"]',
                '.review', '.comment', '.feedback', '.testimonial', '.user-review',
                '[class*="review"]', '[class*="comment"]', '[id*="review"]', '[id*="comment"]',
                '.customer-review', '.product-review'
            ]
            
            # Try each selector
            review_elements = []
            for selector in review_selectors:
                elements = soup.select(selector)
                if elements:
                    review_elements = elements
                    break
            
            for review_elem in review_elements:
                review_text = review_elem.get_text(strip=True)
                if review_text and len(review_text) > 20:  # Only add meaningful reviews
                    reviews.append(review_text)
            
            # If no reviews found through specific selectors, fallback to paragraphs
            if not reviews:
                reviews = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 20]
                
            # If still no content, try to get at least some page content
            if not reviews:
                all_text = soup.get_text(strip=True)
                if all_text:
                    # Split into manageable chunks
                    chunks = [all_text[i:i+500] for i in range(0, len(all_text), 500)]
                    reviews = chunks
            
            return {
                'reviews': reviews,
                'fees': self._extract_fees(soup),
                'facilities': self._extract_facilities(soup),
                'activities': self._extract_activities(soup),
                'title': self._extract_title(soup)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def extract_content(self, url):
        """Main extraction method that determines which approach to use"""
        if self.use_selenium and self.driver:
            return self.extract_content_with_selenium(url)
        else:
            return self.extract_content_with_requests(url)
    
    def _extract_title(self, soup):
        try:
            title = soup.find('h1')
            if title:
                return title.get_text(strip=True)
            if soup.title:
                return soup.title.string
            return "Unknown Page"
        except:
            return "Unknown Page"
        
    def _extract_fees(self, soup):
        fees = []
        try:
            fee_patterns = [
                r'\$\d+\.?\d*(?:\s*-\s*\$\d+\.?\d*)?(?:\s*per\s*(?:person|vehicle|night|day|site|entrance|adult|child|senior))?',
                r'(?:Fee|Price|Cost|Admission|Entry):\s*\$\d+\.?\d*',
                r'(?:Fee|Price|Cost|Admission|Entry)(?:\s+is)?\s+\$\d+\.?\d*'
            ]
            
            for pattern in fee_patterns:
                fees.extend(re.findall(pattern, soup.text))
        except:
            pass
        
        return fees[:5]  # Return up to 5 fee matches
    
    def _extract_facilities(self, soup):
        facilities = []
        try:
            facility_keywords = ['restroom', 'shower', 'campsite', 'picnic', 'visitor center', 
                                'parking', 'trailhead', 'lodging', 'camping', 'cabin', 'wifi',
                                'information desk', 'gift shop', 'food service', 'restaurant',
                                'accessibility', 'wheelchair', 'accessible', 'drinking water']
            
            for keyword in facility_keywords:
                if keyword.lower() in soup.text.lower():
                    facilities.append(keyword)
                    
            # Also look for lists that might contain facilities
            for list_item in soup.find_all('li'):
                item_text = list_item.get_text(strip=True).lower()
                if any(keyword in item_text for keyword in facility_keywords):
                    facilities.append(item_text[:50] + "..." if len(item_text) > 50 else item_text)
        except:
            pass
        
        return list(set(facilities))[:5]  # Deduplicate and limit to 5
    
    def _extract_activities(self, soup):
        activities = []
        try:
            activity_keywords = ['hiking', 'swimming', 'fishing', 'boating', 'camping', 
                                'wildlife viewing', 'biking', 'kayaking', 'canoeing', 'photography',
                                'birdwatching', 'stargazing', 'climbing', 'picnicking', 'skiing',
                                'snowshoeing', 'rafting', 'backpacking', 'horseback riding']
            
            for keyword in activity_keywords:
                if keyword.lower() in soup.text.lower():
                    activities.append(keyword)
        except:
            pass
        
        return list(set(activities))  # Deduplicate

def map_sentiment_label(sentiment):
    """Maps sentiment labels to standardized format"""
    if sentiment in ['POSITIVE', 'positive']:
        return 'positive'
    elif sentiment in ['NEGATIVE', 'negative']:
        return 'negative'
    else:
        return 'neutral'

def categorize_text(text):
    """Identify which categories the text belongs to"""
    text_lower = text.lower()
    categories_found = []
    
    for category, keywords in CATEGORIES.items():
        if any(keyword in text_lower for keyword in keywords):
            categories_found.append(category)
    
    # If no categories found, mark as 'general'
    if not categories_found:
        categories_found.append('general')
        
    return categories_found

def analyze_content(url, use_selenium=True):
    """Main analysis function with error handling and fallback options"""
    try:
        # Make sure models are loaded
        if st.session_state.nlp is None or st.session_state.sentiment_analyzer is None:
            nlp, sentiment_analyzer = load_nlp_models()
            st.session_state.nlp = nlp
            st.session_state.sentiment_analyzer = sentiment_analyzer
        
        # Check if models loaded correctly
        if st.session_state.sentiment_analyzer is None:
            st.warning("⚠️ Could not load sentiment analysis model. Using simplified analysis.")
            return simplified_analysis(url, use_selenium)
        
        # Get sentiment analyzer
        sentiment_analyzer = st.session_state.sentiment_analyzer
        
        # Scrape content
        scraper = WebScraper(use_selenium=use_selenium)
        data = scraper.extract_content(url)
        
        if 'error' in data:
            return None, f"Error: {data['error']}"
        
        if not data['reviews']:
            return None, "Error: No content found to analyze on the page."
        
        # Prepare for analysis
        all_sentiments = []
        category_sentiments = defaultdict(list)
        
        # Process each review with progress indication
        progress_bar = st.progress(0)
        total_reviews = len(data['reviews'])
        
        for i, review in enumerate(data['reviews']):
            # Update progress
            progress_bar.progress((i + 1) / total_reviews)
            
            # Split review into sentences for more granular analysis
            try:
                review_sentences = re.split(r'(?<=[.!?])\s+', review)
            except:
                review_sentences = [review]
            
            for sentence in review_sentences:
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                
                # Determine categories this sentence belongs to
                sentence_categories = categorize_text(sentence)
                
                # Break long sentences into chunks for the sentiment analyzer
                try:
                    text_chunks = [sentence[i:i+512] for i in range(0, len(sentence), 512)]
                    
                    for chunk in text_chunks:
                        try:
                            sentiment_result = sentiment_analyzer(chunk)[0]
                            sentiment_label = sentiment_result['label']
                            confidence = sentiment_result['score']
                            
                            # Add neutrality for mid-range confidence scores
                            if 0.55 <= confidence <= 0.70:
                                sentiment_label = 'NEUTRAL'
                                confidence = 0.5 + (confidence - 0.55) * 0.5
                            
                            # Store the sentiment
                            standardized_label = map_sentiment_label(sentiment_label)
                            sentiment_entry = {
                                'text': chunk,
                                'sentiment': standardized_label,
                                'confidence': confidence,
                                'categories': sentence_categories
                            }
                            
                            all_sentiments.append(sentiment_entry)
                            
                            # Categorize by topics
                            for category in sentence_categories:
                                category_sentiments[category].append(sentiment_entry)
                        except Exception as e:
                            pass
                except Exception as e:
                    pass
        
        # Clear progress bar
        progress_bar.empty()
        
        if not all_sentiments:
            return None, "Error: Could not perform sentiment analysis. Try with another URL."
        
        # Create overall sentiment distribution
        sentiment_df = pd.DataFrame([{'sentiment': s['sentiment']} for s in all_sentiments])
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        
        # Add missing sentiment categories if any are absent
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment not in sentiment_counts:
                sentiment_counts[sentiment] = 0
        
        # Create category sentiment distribution
        category_data = []
        for category, sentiments in category_sentiments.items():
            # Count sentiment by category
            cat_sentiment_counts = defaultdict(int)
            for s in sentiments:
                cat_sentiment_counts[s['sentiment']] += 1
            
            # Ensure all sentiments are represented
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment not in cat_sentiment_counts:
                    cat_sentiment_counts[sentiment] = 0
            
            # Add to dataset for plotting
            for sentiment, count in cat_sentiment_counts.items():
                category_data.append({
                    'category': category,
                    'sentiment': sentiment,
                    'count': count
                })
        
        category_df = pd.DataFrame(category_data)
        
        # Create visualizations
        try:
            # Set a consistent style for all plots
            plt.style.use('ggplot')
            
            # 1. Overall Sentiment Distribution
            colors = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
            overall_sentiment_fig = plt.figure(figsize=(10, 6))
            ax = overall_sentiment_fig.add_subplot(111)
            
            # Sort sentiment counts to always show positive, neutral, negative in that order
            ordered_sentiment_counts = pd.Series([
                sentiment_counts.get('positive', 0),
                sentiment_counts.get('neutral', 0),
                sentiment_counts.get('negative', 0)
            ], index=['positive', 'neutral', 'negative'])
            
            bars = ax.bar(
                ordered_sentiment_counts.index, 
                ordered_sentiment_counts.values, 
                color=[colors[s] for s in ordered_sentiment_counts.index]
            )
            
            ax.set_title('Overall Sentiment Distribution', fontsize=16)
            ax.set_ylabel('Number of Text Segments', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add counts as text on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
        except Exception as e:
            st.error(f"Error creating overall sentiment chart: {str(e)}")
            overall_sentiment_fig = None
        
        # 2. Sentiment Distribution by Category
        try:
            cat_fig = None
            if category_sentiments:
                # First check if we have any non-general categories
                non_general_categories = [cat for cat in category_sentiments.keys() if cat != 'general']
                
                # If no categories other than general, no need for this chart
                if non_general_categories:
                    # Pivot the data for easier plotting
                    pivot_df = category_df.pivot_table(index='category', columns='sentiment', values='count', fill_value=0)
                    
                    # Sort categories by total mentions (descending)
                    category_totals = pivot_df.sum(axis=1).sort_values(ascending=False)
                    pivot_df = pivot_df.loc[category_totals.index]
                    
                    cat_fig = plt.figure(figsize=(12, 7))
                    ax = cat_fig.add_subplot(111)
                    
                    # Set width of bars
                    bar_width = 0.25
                    index = np.arange(len(pivot_df.index))
                    
                    # Plot bars for each sentiment
                    for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
                        if sentiment in pivot_df.columns:
                            bars = ax.bar(index + i*bar_width, pivot_df[sentiment], bar_width, 
                                    label=sentiment, color=colors[sentiment])
                            
                            # Ad
