import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import re
from datetime import datetime

# Download required NLTK data
nltk.download('vader_lexicon')

# Load environment variables
load_dotenv()

# NewsAPI API key
news_api_key = os.getenv('NEWS_API_KEY')

# Stock-related keyword lists for enhanced sentiment analysis
positive_stock_terms = [
    'growth', 'profit', 'surge', 'beat', 'exceeds', 'up', 'innovation', 'gain',
    'rally', 'record', 'outperform', 'bullish', 'strong', 'higher', 'increase',
    'expansion', 'exceeded expectations', 'upside', 'boost'
]

negative_stock_terms = [
    'drop', 'fall', 'decline', 'loss', 'miss', 'downgrade', 'down', 'bearish',
    'weak', 'lower', 'decrease', 'shrink', 'layoff', 'recall', 'investigation',
    'lawsuit', 'deficit', 'disappoints', 'underperform', 'crash'
]

def get_tesla_articles(days_back=3):
    """Fetch Tesla-related articles from NewsAPI for the past X days"""
    # Calculate date for X days ago in YYYY-MM-DD format
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Format dates for NewsAPI
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    url = 'https://newsapi.org/v2/everything'
    params = {
        'apiKey': news_api_key,
        'q': 'tesla OR "elon musk" OR tsla',
        'language': 'en',
        'from': from_date,
        'to': to_date,
        'sortBy': 'publishedAt',
        'pageSize': 100
    }
    
    try:
        print(f"Connecting to NewsAPI to fetch articles from {from_date} to {to_date}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract articles from response
        articles = data.get('articles', [])
        print(f"Found {len(articles)} articles about Tesla/Elon Musk from the past {days_back} days")
        
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles: {e}")
        return []

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER"""
    sia = SentimentIntensityAnalyzer()
    
    # Get base sentiment
    sentiment = sia.polarity_scores(text)
    
    # Count occurrences of stock-related terms
    positive_count = sum(1 for term in positive_stock_terms if term.lower() in text.lower())
    negative_count = sum(1 for term in negative_stock_terms if term.lower() in text.lower())
    
    # Add to sentiment info
    sentiment['stock_term_positive_count'] = positive_count
    sentiment['stock_term_negative_count'] = negative_count
    
    # Apply a modifier to the compound score based on stock-specific terms
    stock_modifier = (positive_count * 0.05) - (negative_count * 0.05)
    sentiment['compound'] = max(min(sentiment['compound'] + stock_modifier, 1.0), -1.0)
    
    return sentiment

def weigh_by_relevance(text):
    """Calculate article relevance weight for Tesla stock prediction"""
    stock_specific_terms = [
        'stock', 'share', 'market', 'investor', 'earning', 'quarterly',
        'financial', 'revenue', 'profit', 'delivery', 'production'
    ]
    
    # Count occurrences and calculate a relevance score
    count = sum(1 for term in stock_specific_terms if term.lower() in text.lower())
    
    # Normalize to a weight between 0.5 and 1.5
    weight = 0.5 + min(count * 0.1, 1.0)
    
    return weight

def process_articles(articles):
    """Process articles with enhanced stock-focused sentiment analysis"""
    data = []
    print(f"Processing {len(articles)} articles for sentiment analysis...")
    
    for idx, article in enumerate(articles):
        if idx % 20 == 0:
            print(f"Processing article {idx+1}/{len(articles)}...")
            
        # Extract relevant text for sentiment analysis
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        article_text = f"{title} {description} {content}"
        
        # Skip if no substantive content
        if not article_text.strip():
            continue
        
        # Get publication date (as string only)
        published_at = article.get('publishedAt', '')
        
        # Analyze sentiment
        sentiment = analyze_sentiment(article_text)
        
        # Calculate relevance weight
        relevance_weight = weigh_by_relevance(article_text)
        
        # Calculate weighted sentiment
        weighted_compound = sentiment['compound'] * relevance_weight
        
        # Construct article data
        article_data = {
            'title': title,
            'url': article.get('url', ''),
            'published_at': published_at,  # Keep as string
            'source': article.get('source', {}).get('name', ''),
            'compound': sentiment['compound'],
            'weighted_compound': weighted_compound,
            'relevance_weight': relevance_weight,
            'positive_stock_terms': sentiment.get('stock_term_positive_count', 0),
            'negative_stock_terms': sentiment.get('stock_term_negative_count', 0),
            'pos': sentiment['pos'],
            'neg': sentiment['neg'],
            'neu': sentiment['neu']
        }
        data.append(article_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def plot_sentiment_distribution(df, filename='tesla_sentiment_distribution.png'):
    """Plot sentiment distribution"""
    if df.empty:
        print("No data to plot!")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['weighted_compound'], bins=15, color='skyblue', edgecolor='black')
    plt.title('Distribution of Stock-Weighted Sentiment Scores for Tesla News')
    plt.xlabel('Weighted Sentiment Score')
    plt.ylabel('Number of Articles')
    plt.savefig(filename)
    plt.close()

def calculate_metrics(df):
    """Calculate basic metrics from the data"""
    if df.empty:
        return {}
    
    metrics = {
        'total_articles': len(df),
        'avg_sentiment': df['compound'].mean(),
        'avg_weighted_sentiment': df['weighted_compound'].mean(),
        'positive_articles': len(df[df['compound'] > 0.05]),
        'negative_articles': len(df[df['compound'] < -0.05]),
        'neutral_articles': len(df[(df['compound'] >= -0.05) & (df['compound'] <= 0.05)]),
        'stock_term_ratio': df['positive_stock_terms'].sum() / max(1, df['negative_stock_terms'].sum())
    }
    
    return metrics

def main():
    print("=== Tesla Stock Sentiment Analysis Tool ===")
    print("Using NewsAPI to gather Tesla-related news")
    
    # Get articles from the past 7 days (one week)
    days_to_analyze = 7
    articles = get_tesla_articles(days_back=days_to_analyze)
    
    if not articles:
        print("No articles found! Please check your NewsAPI key or try a different date range.")
        return
    
    # Process articles
    df = process_articles(articles)
    
    if df.empty:
        print("No article content to analyze!")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Print summary
    print("\n=== Tesla Sentiment Analysis Results ===")
    print(f"Total articles analyzed: {metrics['total_articles']}")
    print(f"Average sentiment score: {metrics['avg_sentiment']:.3f}")
    print(f"Average weighted sentiment score: {metrics['avg_weighted_sentiment']:.3f}")
    print(f"Positive articles: {metrics['positive_articles']} ({metrics['positive_articles']/metrics['total_articles']*100:.1f}%)")
    print(f"Negative articles: {metrics['negative_articles']} ({metrics['negative_articles']/metrics['total_articles']*100:.1f}%)")
    print(f"Neutral articles: {metrics['neutral_articles']} ({metrics['neutral_articles']/metrics['total_articles']*100:.1f}%)")
    print(f"Positive-to-negative stock term ratio: {metrics['stock_term_ratio']:.2f}")
    
    # Generate plot
    plot_sentiment_distribution(df)
    
    # Save to CSV
    output_file = f"tesla_sentiment_{days_to_analyze}day_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved detailed results to '{output_file}'")
    
    # Save plot with specific timeframe
    plot_filename = f"tesla_sentiment_{days_to_analyze}day_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plot_sentiment_distribution(df, plot_filename)
    print(f"Saved sentiment distribution plot to '{plot_filename}'")
    
    # Simple interpretation
    weighted_score = metrics['avg_weighted_sentiment']
    
    print("\n=== Sentiment Interpretation ===")
    print(f"Analysis of Tesla news from the past {days_to_analyze} days:")
    
    if weighted_score > 0.2:
        print("Overall sentiment: STRONGLY POSITIVE")
    elif weighted_score > 0.05:
        print("Overall sentiment: POSITIVE")
    elif weighted_score > -0.05:
        print("Overall sentiment: NEUTRAL")
    elif weighted_score > -0.2:
        print("Overall sentiment: NEGATIVE")
    else:
        print("Overall sentiment: STRONGLY NEGATIVE")
        
    # Create a simple summary file
    summary_file = f"tesla_summary_{days_to_analyze}day_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"TESLA SENTIMENT ANALYSIS - {days_to_analyze} DAY SUMMARY\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Articles analyzed: {metrics['total_articles']}\n")
        f.write(f"Average weighted sentiment: {metrics['avg_weighted_sentiment']:.3f}\n")
        f.write(f"Positive articles: {metrics['positive_articles']} ({metrics['positive_articles']/metrics['total_articles']*100:.1f}%)\n")
        f.write(f"Negative articles: {metrics['negative_articles']} ({metrics['negative_articles']/metrics['total_articles']*100:.1f}%)\n")
        f.write(f"Stock term ratio: {metrics['stock_term_ratio']:.2f}\n\n")
        
        # Add interpretation
        f.write("SENTIMENT INTERPRETATION:\n")
        if weighted_score > 0.2:
            f.write("Overall sentiment: STRONGLY POSITIVE")
        elif weighted_score > 0.05:
            f.write("Overall sentiment: POSITIVE")
        elif weighted_score > -0.05:
            f.write("Overall sentiment: NEUTRAL")
        elif weighted_score > -0.2:
            f.write("Overall sentiment: NEGATIVE")
        else:
            f.write("Overall sentiment: STRONGLY NEGATIVE")
    
    print(f"Saved summary to '{summary_file}'")

if __name__ == "__main__":
    main()