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
import time

# --- Streamlit Page Config (MUST BE FIRST) ---
st.set_page_config(
    page_title="Website Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

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
                    
                    sentiment_analyzer = pipeline(
                        "sentiment-analysis", 
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=device
                    )
                    return nlp, sentiment_analyzer
                except Exception as e:
                    st.error(f"Transformer model error: {e}")
            
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
    'hiking': ['hiking', 'trail', 'hike', 'trek', 'trekking', 'paths', 'walk', 'walking'],
    'fees': ['fee', 'price', 'cost', 'payment', 'dollar', 'money', 'expensive', 'cheap', 'affordable'],
    'equipment': ['equipment', 'gear', 'supplies', 'tent', 'backpack', 'boots', 'poles', 'shoes'],
    'water': ['water', 'lake', 'river', 'stream', 'pond', 'waterfall', 'creek', 'swimming'],
    'facilities': ['facilities', 'restroom', 'bathroom', 'shower', 'toilet', 'visitor center', 'parking']
}

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
    
    def extract_content(self, url):
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
                r'\$\d+\.?\d*(?:\s*-\s*\$\d+\.?\d*)?(?:\s*per\s*(?:person|vehicle|night|day|site|entrance))?',
                r'(?:Fee|Price|Cost):\s*\$\d+\.?\d*'
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
                                'parking', 'trailhead', 'lodging', 'camping', 'cabin']
            
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
                                'wildlife viewing', 'biking', 'kayaking', 'canoeing', 'photography']
            
            for keyword in activity_keywords:
                if keyword.lower() in soup.text.lower():
                    activities.append(keyword)
        except:
            pass
        
        return list(set(activities))  # Deduplicate

def map_sentiment_label(sentiment):
    """Maps sentiment labels to standardized format"""
    if sentiment == 'POSITIVE':
        return 'positive'
    elif sentiment == 'NEGATIVE':
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

def analyze_content(url):
    # Make sure models are loaded
    if st.session_state.nlp is None or st.session_state.sentiment_analyzer is None:
        nlp, sentiment_analyzer = load_nlp_models()
        st.session_state.nlp = nlp
        st.session_state.sentiment_analyzer = sentiment_analyzer
    
    # Check if models loaded correctly
    if st.session_state.sentiment_analyzer is None:
        st.warning("⚠️ Could not load sentiment analysis model. Using simplified analysis.")
        return simplified_analysis(url)
    
    # Get sentiment analyzer
    sentiment_analyzer = st.session_state.sentiment_analyzer
    
    # Scrape content
    scraper = WebScraper()
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
                        st.error(f"Error analyzing chunk: {str(e)}")
            except Exception as e:
                st.error(f"Error processing sentence: {str(e)}")
    
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
        # 1. Overall Sentiment Distribution
        plt.figure(figsize=(10, 6))
        colors = {'positive': 'green', 'neutral': 'gold', 'negative': 'red'}
        overall_sentiment_fig = plt.figure(figsize=(10, 6))
        ax = overall_sentiment_fig.add_subplot(111)
        bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=[colors[s] for s in sentiment_counts.index])
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
            # Pivot the data for easier plotting
            pivot_df = category_df.pivot_table(index='category', columns='sentiment', values='count', fill_value=0)
            
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
                    
                    # Add count labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            # Set plot attributes
            ax.set_title('Sentiment Distribution by Category', fontsize=16)
            ax.set_ylabel('Number of Mentions', fontsize=12)
            ax.set_xticks(index + bar_width)
            ax.set_xticklabels(pivot_df.index, rotation=30, ha='right')
            ax.legend(title='Sentiment')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
    except Exception as e:
        st.error(f"Error creating category sentiment chart: {str(e)}")
        cat_fig = None
    
    # 3. Sentiment Confidence Distribution
    try:
        conf_fig = plt.figure(figsize=(10, 6))
        ax = conf_fig.add_subplot(111)
        
        # Get confidence values for each sentiment
        pos_conf = [s['confidence'] for s in all_sentiments if s['sentiment'] == 'positive']
        neu_conf = [s['confidence'] for s in all_sentiments if s['sentiment'] == 'neutral']
        neg_conf = [s['confidence'] for s in all_sentiments if s['sentiment'] == 'negative']
        
        # Create histogram
        if pos_conf:
            ax.hist(pos_conf, bins=10, alpha=0.7, label='Positive', color='green')
        if neu_conf:
            ax.hist(neu_conf, bins=10, alpha=0.7, label='Neutral', color='gold')
        if neg_conf:
            ax.hist(neg_conf, bins=10, alpha=0.7, label='Negative', color='red')
        
        ax.set_title('Sentiment Confidence Distribution', fontsize=16)
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
    except Exception as e:
        st.error(f"Error creating confidence chart: {str(e)}")
        conf_fig = None
    
    # 4. Word Cloud
    try:
        combined_text = ' '.join(data['reviews'])
        wordcloud_fig = None
        
        if combined_text:
            try:
                from wordcloud import WordCloud
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                    colormap='viridis', max_words=100,
                                    contour_width=1).generate(combined_text)
                wordcloud_fig = plt.figure(figsize=(10, 5))
                ax = wordcloud_fig.add_subplot(111)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title('Most Common Words in Content', fontsize=16)
                ax.axis('off')
                plt.tight_layout()
            except Exception as e:
                st.warning(f"Could not generate word cloud: {str(e)}")
                wordcloud_fig = None
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        wordcloud_fig = None
    
    # 5. Top Positive and Negative Sentences
    try:
        # Sort by confidence
        positive_sentences = sorted(
            [s for s in all_sentiments if s['sentiment'] == 'positive'], 
            key=lambda x: x['confidence'], 
            reverse=True
        )
        
        negative_sentences = sorted(
            [s for s in all_sentiments if s['sentiment'] == 'negative'], 
            key=lambda x: x['confidence'], 
            reverse=True
        )
    except Exception as e:
        st.error(f"Error sorting sentences: {str(e)}")
        positive_sentences = []
        negative_sentences = []
    
    # Prepare report data
    positive_count = sum(1 for s in all_sentiments if s['sentiment'] == 'positive')
    negative_count = sum(1 for s in all_sentiments if s['sentiment'] == 'negative')
    neutral_count = sum(1 for s in all_sentiments if s['sentiment'] == 'neutral')
    total_count = len(all_sentiments)
    
    # Calculate percentages
    if total_count > 0:
        positive_pct = (positive_count / total_count) * 100
        negative_pct = (negative_count / total_count) * 100
        neutral_pct = (neutral_count / total_count) * 100
    else:
        positive_pct = negative_pct = neutral_pct = 0
    
    report = {
        'title': data['title'],
        'url': url,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'total_count': total_count,
        'positive_pct': positive_pct,
        'negative_pct': negative_pct,
        'neutral_pct': neutral_pct,
        'fees': data['fees'],
        'facilities': data['facilities'],
        'activities': data['activities'],
        'overall_sentiment_fig': overall_sentiment_fig,
        'category_sentiment_fig': cat_fig,
        'confidence_fig': conf_fig,
        'wordcloud': wordcloud_fig,
        'top_positive': positive_sentences[:5] if positive_sentences else [],
        'top_negative': negative_sentences[:5] if negative_sentences else [],
        'category_sentiments': category_sentiments
    }
    
    return report, None

def simplified_analysis(url):
    """Fallback if transformer models fail to load"""
    scraper = WebScraper()
    data = scraper.extract_content(url)
    
    if 'error' in data:
        return None, f"Error: {data['error']}"
    
    if not data['reviews']:
        return None, "Error: No content found on the page."
    
    # Simple sentiment analysis based on keyword matching
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'enjoy', 'recommend']
    negative_words = ['bad', 'terrible', 'awful', 'worst', 'poor', 'disappointing', 'avoid', 'waste', 'not worth']
    
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    all_sentiments = []
    category_sentiments = defaultdict(list)
    
    # Process reviews
    for review in data['reviews']:
        # Determine categories
        categories = categorize_text(review)
        
        # Simple sentiment analysis
        positive_matches = sum(1 for word in positive_words if word in review.lower())
        negative_matches = sum(1 for word in negative_words if word in review.lower())
        
        if positive_matches > negative_matches:
            sentiment = 'positive'
            confidence = min(0.5 + (positive_matches * 0.05), 0.95)
        elif negative_matches > positive_matches:
            sentiment = 'negative'
            confidence = min(0.5 + (negative_matches * 0.05), 0.95)
        else:
            sentiment = 'neutral'
            confidence = 0.5
            
        # Create sentiment entry
        sentiment_entry = {
            'text': review[:200] + ("..." if len(review) > 200 else ""),
            'sentiment': sentiment,
            'confidence': confidence,
            'categories': categories
        }
        
        all_sentiments.append(sentiment_entry)
        sentiment_counts[sentiment] += 1
        
        # Categorize
        for category in categories:
            category_sentiments[category].append(sentiment_entry)
    
    # Create simple chart
    try:
        plt.figure(figsize=(10, 6))
        colors = {'positive': 'green', 'neutral': 'gold', 'negative': 'red'}
        overall_sentiment_fig = plt.figure(figsize=(10, 6))
        ax = overall_sentiment_fig.add_subplot(111)
        
        # Convert to Series for easier plotting
        sentiment_series = pd.Series(sentiment_counts)
        
        bars = ax.bar(sentiment_series.index, sentiment_series.values, 
                     color=[colors[s] for s in sentiment_series.index])
        
        ax.set_title('Overall Sentiment (Simplified Analysis)', fontsize=16)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
            
        plt.tight_layout()
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        overall_sentiment_fig = None
    
    # Calculate percentages
    total = sum(sentiment_counts.values())
    if total > 0:
        positive_pct = (sentiment_counts['positive'] / total) * 100
        negative_pct = (sentiment_counts['negative'] / total) * 100
        neutral_pct = (sentiment_counts['neutral'] / total) * 100
    else:
        positive_pct = negative_pct = neutral_pct = 0
    
    report = {
        'title': data['title'],
        'url': url,
        'positive_count': sentiment_counts['positive'],
        'negative_count': sentiment_counts['negative'],
        'neutral_count': sentiment_counts['neutral'],
        'total_count': total,
        'positive_pct': positive_pct,
        'negative_pct': negative_pct,
        'neutral_pct': neutral_pct,
        'fees': data['fees'],
        'facilities': data['facilities'],
        'activities': data['activities'],
        'overall_sentiment_fig': overall_sentiment_fig,
        'category_sentiment_fig': None,
        'confidence_fig': None,
        'wordcloud': None,
        'top_positive': sorted([s for s in all_sentiments if s['sentiment'] == 'positive'], 
                               key=lambda x: x['confidence'], reverse=True)[:3],
        'top_negative': sorted([s for s in all_sentiments if s['sentiment'] == 'negative'], 
                               key=lambda x: x['confidence'], reverse=True)[:3],
        'category_sentiments': category_sentiments
    }
    
    return report, None

# Streamlit Interface
st.title("📊 Website Sentiment Analyzer")
st.write("""
This tool analyzes text content from any website. 
Enter a URL to get started. The tool will extract content and perform sentiment analysis.
""")

url_input = st.text_input(
    "Enter Website URL", 
    placeholder="www.example.com/reviews"
)

if st.button("Analyze", type="primary"):
    if not url_input:
        st.error("Please enter a URL to analyze")
    else:
        with st.spinner("Analyzing... This may take a minute"):
            try:
                report, error = analyze_content(url_input)
                
                if error:
                    st.error(error)
                elif report:
                    # Display report
                    st.header(f"Analysis Report: {report['title']}")
                    
                    # Overall metrics
                    st.subheader("Overall Sentiment")
                    cols = st.columns(3)
                    
                    with cols[0]:
                        st.metric("Positive", f"{report['positive_count']} ({report['positive_pct']:.1f}%)")
                    
                    with cols[1]:
                        st.metric("Neutral", f"{report['neutral_count']} ({report['neutral_pct']:.1f}%)")
                        
                    with cols[2]:
                        st.metric("Negative", f"{report['negative_count']} ({report['negative_pct']:.1f}%)")
                    
                    # Display overall sentiment distribution
                    if report['overall_sentiment_fig']:
                        st.pyplot(report['overall_sentiment_fig'])
                    
                    # Display category sentiment distribution if available
                    if report['category_sentiment_fig']:
                        st.subheader("Sentiment by Category")
                        st.pyplot(report['category_sentiment_fig'])
                        
                        # Show detailed category breakdown
                        st.subheader("Category Details")
                        
                        for category in CATEGORIES.keys():
                            if category in report['category_sentiments']:
                                with st.expander(f"{category.title()} - {len(report['category_sentiments'][category])} mentions"):
                                    cat_sentiments = report['category_sentiments'][category]
                                    pos = sum(1 for s in cat_sentiments if s['sentiment'] == 'positive')
                                    neg = sum(1 for s in cat_sentiments if s['sentiment'] == 'negative')
                                    neu = sum(1 for s in cat_sentiments if s['sentiment'] == 'neutral')
                                    
                                    total = len(cat_sentiments)
                                    if total > 0:
                                        st.write(f"👍 Positive: {pos} ({pos/total*100:.1f}%)")
                                        st.write(f"👎 Negative: {neg} ({neg/total*100:.1f}%)")
                                        st.write(f"😐 Neutral: {neu} ({neu/total*100:.1f}%)")
                                        
                                        # Show top sentence for this category
                                        top_positive = next((s for s in cat_sentiments if s['sentiment'] == 'positive'), None)
                                        top_negative = next((s for s in cat_sentiments if s['sentiment'] == 'negative'), None)
                                        
                                        if top_positive:
                                            st.write("**Sample positive mention:**")
                                
