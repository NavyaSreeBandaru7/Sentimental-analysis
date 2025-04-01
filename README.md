# Sentimental-analysis
National Park Review Analyzer 🏞️
Overview
The National Park Review Analyzer is a powerful tool designed to analyze reviews and extract valuable insights from national park websites. It uses Natural Language Processing (NLP) and Sentiment Analysis to process visitor reviews, categorize sentiments, and visualize the results. The tool is built using Python and Streamlit, making it interactive and easy to use.

Features
Multi-Page Review Scraping: Extracts reviews from multiple pages using Selenium.

Sentiment Analysis: Identifies positive, negative, and neutral sentiments using a pre-trained NLP model.

Category Analysis: Categorizes reviews into topics like hiking, fees, facilities, equipment, water activities, etc.

Visualizations:

Overall sentiment distribution.

Sentiment by category.

Confidence score distribution.

Word cloud of common words in reviews.

Top Mentions: Highlights the most confident positive and negative mentions.

Park Information Extraction: Extracts fees, facilities, and activities mentioned in the reviews.

Technologies Used
Python

Streamlit: For interactive web interface.

Selenium: For scraping multi-page reviews.

BeautifulSoup: For HTML parsing.

Transformers (Hugging Face): For sentiment analysis.

Spacy: For text categorization.

Matplotlib & WordCloud: For data visualization.

Installation
Prerequisites
Ensure you have the following installed:

Python 3.8 or higher

Pip package manager

Steps
Clone the repository:

bash
git clone https://github.com/your-repo/national-park-review-analyzer.git
cd national-park-review-analyzer
Install dependencies:

bash
pip install -r requirements.txt
Run the application:

bash
streamlit run park_analyzer.py
Usage
Open the app in your browser (usually at http://localhost:8501).

Enter a URL from supported websites (e.g., Recreation.gov, NPS.gov).

Click "Analyze" to start processing the reviews.

View detailed insights:

Sentiment distribution charts.

Top positive and negative mentions.

Word cloud visualization.

Expand sections to explore extracted fees, facilities, and activities.

Supported Websites
The tool currently supports scraping and analysis from:

Recreation.gov

NPS.gov

NationalParks.org

Requirements
Save the following dependencies in requirements.txt:

text
requests==2.31.0
beautifulsoup4==4.12.2
matplotlib==3.7.2
numpy==1.24.3
transformers==4.35.0
streamlit==1.28.0
torch==2.0.1
spacy==3.6.1
wordcloud==1.9.2
pandas==2.0.3
selenium==4.12.0
webdriver-manager==4.0.0
Install them with:

bash
pip install -r requirements.txt
Key Files
park_analyzer.py: Main application file containing code for scraping, sentiment analysis, and Streamlit interface.

requirements.txt: List of dependencies required to run the application.

How It Works
Scraping Reviews
The tool uses Selenium for dynamic websites with pagination support to extract reviews across multiple pages.

Sentiment Analysis
It leverages a pre-trained DistilBERT model (distilbert-base-uncased-finetuned-sst-2-english) for sentiment classification into positive, negative, or neutral categories.

Categorization & Visualization
Reviews are categorized into predefined topics (e.g., hiking, fees) based on keywords and visualized using bar charts and word clouds.

Example Output
Metrics Summary:
text
Positive Reviews: 120 (60%)
Neutral Reviews: 50 (25%)
Negative Reviews: 30 (15%)
Total Reviews Analyzed: 200
Visualizations:
Sentiment distribution chart.

Category-wise sentiment breakdown.

Confidence score histogram.

Extracted Information:
text
Fees Mentioned:
- $20 per vehicle entrance fee.
Facilities:
- Restrooms available at trailhead.
Activities:
- Hiking trails, kayaking opportunities.
Troubleshooting
Common Issues:
Selenium Driver Error: Ensure ChromeDriver is installed correctly using webdriver-manager.

No Reviews Found: Check if the URL is valid and contains review content.

Debugging Tips:
Run in verbose mode to see detailed logs:

bash
streamlit run park_analyzer.py --logger.level=debug
Contributing
Feel free to contribute by submitting pull requests or reporting issues on GitHub.

License
This project is licensed under the MIT License.

Let me know if you need further
