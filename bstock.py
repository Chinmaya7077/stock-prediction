import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import feedparser
import warnings
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas_ta as ta
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings('ignore')

# Headers for requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Connection": "keep-alive",
}

# Download required NLTK resources
for resource in ["punkt", "stopwords", "wordnet", "vader_lexicon"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

def fetch_company_news(ticker_symbol, max_articles=100):
    """
    Collect top news articles related to a stock ticker using Google News RSS feed.
    
    Parameters:
    ticker_symbol (str): Stock ticker symbol
    max_articles (int): Maximum number of articles to fetch (default: 100)
    
    Returns:
    pandas.DataFrame: DataFrame containing news articles
    """
    try:
        # Get company name from ticker
        company = yf.Ticker(ticker_symbol)
        company_name = company.info.get("shortName", ticker_symbol)
        
        logging.info(f"Searching for top {max_articles} news about {ticker_symbol} ({company_name})")
        
        # URL encode the company name
        encoded_company_name = urllib.parse.quote_plus(company_name)
        
        # Google News RSS feed URL for company stock news
        search_url = f"https://news.google.com/rss/search?q={encoded_company_name}+stock&hl=en-US&gl=US&ceid=US:en"
        
        # Parse the RSS feed
        feed = feedparser.parse(search_url)
        
        # Check if we got any results
        if not feed.entries:
            logging.warning("No news articles found.")
            return pd.DataFrame()
        
        # Process the feed entries
        articles_data = []
        for idx, entry in enumerate(feed.entries[:max_articles], start=1):
            # Parse publication date
            if hasattr(entry, 'published_parsed'):
                pub_date = datetime(*entry.published_parsed[:6])
            else:
                pub_date = datetime.now()  # Use current time if no date available
            
            article = {
                "title": entry.title,
                "description": entry.summary if hasattr(entry, 'summary') else "",
                "publishedAt": pub_date,
                "url": entry.link,
                "source": entry.source.title if hasattr(entry, 'source') and hasattr(entry.source, 'title') else "Google News",
                "ticker": ticker_symbol,
                "query_date": datetime.now().strftime("%Y-%m-%d"),
                "article_rank": idx  # Add ranking based on Google News order
            }
            articles_data.append(article)
        
        # Convert to DataFrame
        articles_df = pd.DataFrame(articles_data)
        
        logging.info(f"Successfully retrieved {len(articles_df)} news articles")
        return articles_df

    except Exception as e:
        logging.error(f"Error occurred in fetch_company_news: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
def preprocess_text(text):
    """Enhanced text cleaning and normalization."""
    if pd.isna(text) or text == "":
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback simple tokenization
            tokens = text.split()
        
        # Remove stopwords - with error handling
        try:
            stop_words = set(stopwords.words('english'))
            # Add finance-specific stopwords
            finance_stopwords = {'stock', 'market', 'shares', 'trading', 'company', 'price', 
                                'prices', 'investor', 'investors', 'share', 'says', 'said', 
                                'report', 'reported', 'quarter', 'quarterly', 'year'}
            stop_words.update(finance_stopwords)
            tokens = [word for word in tokens if word not in stop_words]
        except:
            # Skip stopword removal if it fails
            pass
        
        # Lemmatization - with error handling
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        except:
            # Skip lemmatization if it fails
            pass
        
        # Join tokens back to text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return text  # Return original text if processing fails

def preprocess_news_data(news_df):
    """Prepare news data for sentiment analysis with improved preprocessing."""
    if news_df.empty:
        print("No news data to preprocess")
        return news_df
        
    print(f"Preprocessing {len(news_df)} news articles...")
    
    # Combine title and description for analysis with weighted importance
    # Title often carries more sentiment significance than description
    news_df['combined_text'] = news_df['title'].fillna('') + ' ' + news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')
    
    # Apply preprocessing
    news_df['processed_text'] = news_df['combined_text'].apply(preprocess_text)
    
    # Calculate text length and other features
    news_df['text_length'] = news_df['processed_text'].apply(len)
    news_df['word_count'] = news_df['processed_text'].apply(lambda x: len(x.split()) if x else 0)
    
    # Extract title-only sentiment (titles often have stronger sentiment signals)
    news_df['title_processed'] = news_df['title'].fillna('').apply(preprocess_text)
    
    # Remove empty processed texts
    empty_count = len(news_df[news_df['processed_text'] == ''])
    if empty_count > 0:
        print(f"Found {empty_count} articles with empty processed text")
    
    # Add recency score - newer articles weighted higher
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
    latest_date = news_df['publishedAt'].max()
    news_df['days_old'] = (latest_date - news_df['publishedAt']).dt.total_seconds() / (24 * 3600)
    news_df['recency_weight'] = 1 / (1 + news_df['days_old'] * 0.1)  # Decay factor
    
    print("✅ News preprocessing complete")
    return news_df

def analyze_sentiment(news_df):
    """Analyze sentiment using multiple models with calibration and fusion."""
    if news_df.empty:
        print("No news data for sentiment analysis")
        return news_df
        
    print(f"Analyzing sentiment for {len(news_df)} articles...")
    
    # Initialize sentiment analyzers
    vader = SentimentIntensityAnalyzer()
    
    # Initialize FinBERT (financial sentiment analyzer)
    try:
        print("Loading FinBERT model...")
        finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        finbert = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)
        using_finbert = True
        print("✅ FinBERT model loaded successfully")
    except Exception as e:
        print(f"Error loading FinBERT: {e}")
        print("Continuing without FinBERT...")
        using_finbert = False
    
    # Initialize lists for sentiment scores
    vader_scores = []
    vader_pos = []
    vader_neg = []
    vader_neu = []
    vader_title_scores = []  # Separate analysis for titles
    textblob_scores = []
    textblob_subj = []
    textblob_title_scores = []  # Separate analysis for titles
    finbert_labels = []
    finbert_scores = []
    finbert_title_scores = []  # Separate analysis for titles
    
    # Process each article
    for idx, row in news_df.iterrows():
        text = row['processed_text']
        title_text = row['title_processed']
        
        # Skip empty texts
        if not text or len(text) < 5:
            vader_scores.append(0)
            vader_pos.append(0)
            vader_neg.append(0)
            vader_neu.append(0)
            vader_title_scores.append(0)
            textblob_scores.append(0)
            textblob_subj.append(0)
            textblob_title_scores.append(0)
            finbert_labels.append('neutral')
            finbert_scores.append(0)
            finbert_title_scores.append(0)
            continue
            
        # VADER sentiment
        vader_sentiment = vader.polarity_scores(text)
        vader_scores.append(vader_sentiment['compound'])
        vader_pos.append(vader_sentiment['pos'])
        vader_neg.append(vader_sentiment['neg'])
        vader_neu.append(vader_sentiment['neu'])
        
        # VADER for title only
        if title_text and len(title_text) > 3:
            vader_title_sentiment = vader.polarity_scores(title_text)
            vader_title_scores.append(vader_title_sentiment['compound'])
        else:
            vader_title_scores.append(0)
        
        # TextBlob sentiment
        textblob_sentiment = TextBlob(text).sentiment
        textblob_scores.append(textblob_sentiment.polarity)
        textblob_subj.append(textblob_sentiment.subjectivity)
        
        # TextBlob for title only
        if title_text and len(title_text) > 3:
            textblob_title_sentiment = TextBlob(title_text).sentiment
            textblob_title_scores.append(textblob_title_sentiment.polarity)
        else:
            textblob_title_scores.append(0)
        
        # FinBERT sentiment
        if using_finbert:
            try:
                # Limit text length for FinBERT
                truncated_text = text[:512] if len(text) > 512 else text
                finbert_result = finbert(truncated_text)[0]
                finbert_labels.append(finbert_result['label'])
                
                # Convert label to score: positive=1, neutral=0, negative=-1
                label_score = 1 if finbert_result['label'] == 'positive' else (-1 if finbert_result['label'] == 'negative' else 0)
                # Weight by confidence score
                finbert_scores.append(label_score * finbert_result['score'])
                
                # FinBERT for title
                if title_text and len(title_text) > 3:
                    truncated_title = title_text[:512] if len(title_text) > 512 else title_text
                    finbert_title_result = finbert(truncated_title)[0]
                    title_label_score = 1 if finbert_title_result['label'] == 'positive' else (-1 if finbert_title_result['label'] == 'negative' else 0)
                    finbert_title_scores.append(title_label_score * finbert_title_result['score'])
                else:
                    finbert_title_scores.append(0)
            except Exception as e:
                print(f"FinBERT error on text: {e}")
                finbert_labels.append('neutral')
                finbert_scores.append(0)
                finbert_title_scores.append(0)
        else:
            finbert_labels.append('neutral')
            finbert_scores.append(0)
            finbert_title_scores.append(0)
    
    # Add sentiment scores to dataframe
    news_df['vader_sentiment'] = vader_scores
    news_df['vader_positive'] = vader_pos
    news_df['vader_negative'] = vader_neg
    news_df['vader_neutral'] = vader_neu
    news_df['vader_title_sentiment'] = vader_title_scores
    news_df['textblob_sentiment'] = textblob_scores
    news_df['textblob_subjectivity'] = textblob_subj
    news_df['textblob_title_sentiment'] = textblob_title_scores
    news_df['finbert_label'] = finbert_labels
    news_df['finbert_sentiment'] = finbert_scores
    news_df['finbert_title_sentiment'] = finbert_title_scores
    
    # Calculate combined sentiment score with adjusted weights and calibration
    if using_finbert:
        # Main sentiment with weighted components
        news_df['combined_sentiment'] = (
            news_df['vader_sentiment'] * 0.25 + 
            news_df['textblob_sentiment'] * 0.15 + 
            news_df['finbert_sentiment'] * 0.4 +
            news_df['vader_title_sentiment'] * 0.1 +
            news_df['textblob_title_sentiment'] * 0.05 +
            news_df['finbert_title_sentiment'] * 0.05
        )
    else:
        news_df['combined_sentiment'] = (
            news_df['vader_sentiment'] * 0.45 + 
            news_df['textblob_sentiment'] * 0.3 +
            news_df['vader_title_sentiment'] * 0.15 +
            news_df['textblob_title_sentiment'] * 0.1
        )
    
    # Add sentiment intensity (direction and strength)
    news_df['sentiment_intensity'] = news_df['combined_sentiment'].abs() * news_df['textblob_subjectivity']
    
    # Apply recency and ranking weights
    if 'article_rank' in news_df.columns and 'recency_weight' in news_df.columns:
        max_rank = news_df['article_rank'].max()
        news_df['rank_weight'] = 1 - ((news_df['article_rank'] - 1) / max_rank * 0.5)  # Weight range: 0.5 to 1.0
        
        # Combined weight incorporates both recency and ranking
        news_df['combined_weight'] = (news_df['rank_weight'] * 0.6) + (news_df['recency_weight'] * 0.4)
        news_df['weighted_sentiment'] = news_df['combined_sentiment'] * news_df['combined_weight']
    else:
        news_df['rank_weight'] = 1.0
        news_df['combined_weight'] = 1.0
        news_df['weighted_sentiment'] = news_df['combined_sentiment']
    
    print("✅ Enhanced sentiment analysis complete")
    return news_df

def get_stock_data(ticker_symbol, days_back=365):
    try:
        print(f"Fetching stock data for {ticker_symbol} over the past {days_back} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get stock data
        stock_data = yf.download(ticker_symbol, 
                               start=start_date.strftime('%Y-%m-%d'),
                               end=end_date.strftime('%Y-%m-%d'),
                               auto_adjust=True) # Use adjusted close
        
        if stock_data.empty:
            print(f"No stock data found for {ticker_symbol}")
            return pd.DataFrame()
        
        # Check if enough data is available
        if len(stock_data) < 100:
            print(f"⚠️ Only {len(stock_data)} days of data available. Fetching more data...")
            days_back = 365  # Fetch data for the past year
            start_date = end_date - timedelta(days=days_back)
            stock_data = yf.download(ticker_symbol, 
                                   start=start_date.strftime('%Y-%m-%d'),
                                   end=end_date.strftime('%Y-%m-%d'),
                                   auto_adjust=True)
            if len(stock_data) < 100:
                print(f"⚠️ Only {len(stock_data)} days of data available. At least 100 days are required.")
                return pd.DataFrame()
        
        # Reset index to make date a column
        stock_data = stock_data.reset_index()
        
        # If the dataframe has multi-level columns, flatten them
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stock_data.columns]
        
        # Convert Date to date type for consistency
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.normalize()
        
        # Find the relevant columns
        close_col = [col for col in stock_data.columns if 'Close' in col][0]
        
        # Calculate returns (daily, weekly, monthly)
        stock_data['Return'] = stock_data[close_col].pct_change()
        stock_data['Return_5d'] = stock_data[close_col].pct_change(5)
        stock_data['Return_20d'] = stock_data[close_col].pct_change(20)
        
        # Add ticker column
        stock_data['ticker'] = ticker_symbol
        
        # Standardize column names
        stock_data.columns = [col.replace(f'_{ticker_symbol}', '') for col in stock_data.columns]
        
        # Get market data (S&P 500) for comparison
        try:
            market_data = yf.download('^GSPC', 
                                     start=start_date.strftime('%Y-%m-%d'),
                                     end=end_date.strftime('%Y-%m-%d'),
                                     auto_adjust=True)
            
            market_data = market_data.reset_index()
            market_data['Date'] = pd.to_datetime(market_data['Date']).dt.normalize()
            market_data['Market_Return'] = market_data['Close'].pct_change()
            market_data = market_data[['Date', 'Close', 'Market_Return']]
            market_data.columns = ['Date', 'Market_Close', 'Market_Return']
            
            # Join market data with stock data
            stock_data = pd.merge(stock_data, market_data, on='Date', how='left')
            
            # Calculate excess return (stock return above market return)
            stock_data['Excess_Return'] = stock_data['Return'] - stock_data['Market_Return']
        except Exception as e:
            print(f"Could not get market data: {e}")
            stock_data['Market_Return'] = np.nan
            stock_data['Excess_Return'] = np.nan
        
        print(f"✅ Retrieved {len(stock_data)} days of stock data for {ticker_symbol}")
        return stock_data
    except Exception as e:
        print(f"Error in get_stock_data: {e}")
        return pd.DataFrame()
    
def merge_stock_and_sentiment(stock_data, sentiment_data):
    """Merge stock price data with sentiment analysis results."""
    if stock_data.empty:
        print("Stock data is empty, cannot merge.")
        return pd.DataFrame()
        
    print("Merging stock and sentiment data...")
    
    # Make copies to avoid modifying the original dataframes
    stock_df = stock_data.copy()
    
    # If sentiment data is empty, create a single row of sentiment data
    if sentiment_data.empty:
        print("No sentiment data available, creating dummy sentiment data")
        # Create a single row of sentiment data with all zeros
        sentiment_df = pd.DataFrame({
            'ticker': [stock_df['ticker'].iloc[0]],
            'combined_sentiment': [0],
            'vader_sentiment': [0],
            'textblob_sentiment': [0],
            'finbert_sentiment': [0],
            'article_count': [0],
            'sentiment_std': [0]
        })
    else:
        sentiment_df = sentiment_data.copy()
    
    # Ensure both dataframes have simple column structures
    if isinstance(stock_df.columns, pd.MultiIndex):
        # Create simple column names by joining the levels
        stock_df.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stock_df.columns
        ]
    
    # Broadcast sentiment data to all stock dates
    for col in sentiment_df.columns:
        if col != 'ticker' and col not in stock_df.columns:
            stock_df[col] = sentiment_df[col].iloc[0]
    
    print(f"✅ Successfully added sentiment data to {len(stock_df)} stock records")
    return stock_df

def create_features(data):
    """Create advanced technical and sentiment features."""
    if data.empty:
        print("No data for feature creation")
        return data
        
    print("Creating technical and sentiment features...")
    
    # Make a copy to avoid SettingWithCopyWarning
    df = data.copy()
    
    # Find the correct column names
    close_col = [col for col in df.columns if 'Close' in col][0]
    high_col = [col for col in df.columns if 'High' in col][0]
    low_col = [col for col in df.columns if 'Low' in col][0]
    open_col = [col for col in df.columns if 'Open' in col][0]
    volume_col = [col for col in df.columns if 'Volume' in col][0]

    # Basic price features
    # Moving averages - multiple windows for trend detection
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'MA{window}'] = df[close_col].rolling(window=window).mean()
        # Calculate slope of MA (trend direction and strength)
        df[f'MA{window}_slope'] = df[f'MA{window}'].diff(5) / 5
        # Distance from MA (mean reversion potential)
        df[f'Dist_MA{window}'] = (df[close_col] - df[f'MA{window}']) / df[f'MA{window}']
        
    # Exponential moving averages
    for window in [5, 12, 26, 50]:
        df[f'EMA{window}'] = df[close_col].ewm(span=window, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence) using pandas_ta
    macd = ta.macd(df[close_col], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    df['MACD_Hist_change'] = df['MACD_Hist'].diff()  # Change in MACD histogram (momentum change)
    
    # Bollinger Bands using pandas_ta
    # Bollinger Bands using pandas_ta
    for window in [10, 20, 50]:
        bbands = ta.bbands(df[close_col], length=window, std=2)
        df[f'BB_Upper{window}'] = bbands[f'BBU_{window}_2.0']
        df[f'BB_Lower{window}'] = bbands[f'BBL_{window}_2.0']
        df[f'BB_Width{window}'] = (df[f'BB_Upper{window}'] - df[f'BB_Lower{window}']) / df[f'MA{window}']
        df[f'BB_Pct{window}'] = (df[close_col] - df[f'BB_Lower{window}']) / (df[f'BB_Upper{window}'] - df[f'BB_Lower{window}'])
    
    # Volatility indicators
    # Calculate rolling volatility over multiple windows
    for window in [5, 10, 20, 50]:
        df[f'Volatility_{window}d'] = df['Return'].rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    # ATR (Average True Range) - multiple periods using pandas_ta
    for window in [5, 14, 21]:
        atr = ta.atr(df[high_col], df[low_col], df[close_col], length=window)
        df[f'ATR_{window}'] = atr
        df[f'ATR_{window}_pct'] = df[f'ATR_{window}'] / df[close_col]  # ATR as percentage of price
    
    # RSI (Relative Strength Index) - multiple periods using pandas_ta
    for window in [7, 14, 21]:
        rsi = ta.rsi(df[close_col], length=window)
        df[f'RSI_{window}'] = rsi
        # RSI momentum (change in RSI)
        df[f'RSI_{window}_change'] = df[f'RSI_{window}'].diff(3)
    
    # Stochastic oscillator using pandas_ta
    for window in [14, 21]:
        stoch = ta.stoch(df[high_col], df[low_col], df[close_col], k=window, d=3)
        df[f'%K_{window}'] = stoch[f'STOCHk_{window}_3_3']
        df[f'%D_{window}'] = stoch[f'STOCHd_{window}_3_3']
        df[f'%J_{window}'] = 3 * df[f'%K_{window}'] - 2 * df[f'%D_{window}']  # Williams %J
    
    # Advanced volume indicators
    # Volume moving averages
    for window in [5, 10, 20]:
        df[f'Volume_MA{window}'] = df[volume_col].rolling(window=window).mean()
        df[f'Rel_Volume_{window}'] = df[volume_col] / df[f'Volume_MA{window}']  # Relative volume
    
    # On-Balance Volume (OBV) using pandas_ta
    obv = ta.obv(df[close_col], df[volume_col])
    df['OBV'] = obv
    df['OBV_slope'] = df['OBV'].diff(5) / 5  # OBV momentum
    
    # Money Flow Index (MFI) using pandas_ta
    for window in [14, 21]:
        mfi = ta.mfi(df[high_col], df[low_col], df[close_col], df[volume_col], length=window)
        df[f'MFI_{window}'] = mfi
    
    # Chaikin Money Flow (CMF) using pandas_ta
    for window in [20, 50]:
        cmf = ta.cmf(df[high_col], df[low_col], df[close_col], df[volume_col], length=window)
        df[f'CMF_{window}'] = cmf
    
    # Price patterns and gap analysis
    df['Price_Change'] = df[close_col].pct_change()
    df['Price_Change_5d'] = df[close_col].pct_change(5)
    df['Price_Change_10d'] = df[close_col].pct_change(10)
    
    # Gap analysis
    df['Gap_Up'] = ((df[open_col] > df[close_col].shift(1)) & (df[open_col] > df[open_col].shift(1))).astype(int)
    df['Gap_Down'] = ((df[open_col] < df[close_col].shift(1)) & (df[open_col] < df[open_col].shift(1))).astype(int)
    df['Gap_Size'] = (df[open_col] - df[close_col].shift(1)) / df[close_col].shift(1)
    
    # Candlestick patterns
    df['Body_Size'] = abs(df[close_col] - df[open_col]) / ((df[high_col] + df[low_col]) / 2)
    df['Upper_Shadow'] = (df[high_col] - np.maximum(df[open_col], df[close_col])) / ((df[high_col] + df[low_col]) / 2)
    df['Lower_Shadow'] = (np.minimum(df[open_col], df[close_col]) - df[low_col]) / ((df[high_col] + df[low_col]) / 2)
    
    # Momentum indicators
    # Rate of Change (ROC)
    for window in [5, 10, 20]:
        df[f'ROC_{window}'] = (df[close_col] / df[close_col].shift(window) - 1) * 100
    
    # Williams %R - oversold/overbought indicator
    for window in [14, 21]:
        highest_high = df[high_col].rolling(window=window).max()
        lowest_low = df[low_col].rolling(window=window).min()
        df[f'Williams_%R_{window}'] = -100 * (highest_high - df[close_col]) / (highest_high - lowest_low)
    
    # Price crossovers (trend change signals)
    df['Price_crossover_MA20'] = ((df[close_col] > df['MA20']) & (df[close_col].shift() <= df['MA20'].shift())).astype(int) - \
                                ((df[close_col] < df['MA20']) & (df[close_col].shift() >= df['MA20'].shift())).astype(int)
    df['MA5_crossover_MA20'] = ((df['MA5'] > df['MA20']) & (df['MA5'].shift() <= df['MA20'].shift())).astype(int) - \
                                ((df['MA5'] < df['MA20']) & (df['MA5'].shift() >= df['MA20'].shift())).astype(int)
    
    # Calendar features
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Quarter'] = pd.to_datetime(df['Date']).dt.quarter
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
    df['Is_Month_Start'] = df['Date'].dt.is_month_start.astype(int)
    df['Is_Quarter_End'] = df['Date'].dt.is_quarter_end.astype(int)
    df['Is_Quarter_Start'] = df['Date'].dt.is_quarter_start.astype(int)
    
    # Sentiment features (if available)
    if 'combined_sentiment' in df.columns:
        # Sentiment momentum
        df['Sentiment_Change'] = df['combined_sentiment'].diff()
        df['Sentiment_MA5'] = df['combined_sentiment'].rolling(window=5).mean()
        df['Sentiment_MA10'] = df['combined_sentiment'].rolling(window=10).mean()
        
        # Sentiment volatility
        df['Sentiment_Volatility'] = df['combined_sentiment'].rolling(window=10).std()
        
        # Sentiment extremes
        df['Sentiment_High'] = (df['combined_sentiment'] > df['combined_sentiment'].rolling(window=20).quantile(0.75)).astype(int)
        df['Sentiment_Low'] = (df['combined_sentiment'] < df['combined_sentiment'].rolling(window=20).quantile(0.25)).astype(int)
    
    # Market-related features
    if 'Market_Return' in df.columns:
        # Relative strength
        df['Relative_Strength'] = df['Return'] / df['Market_Return'].replace(0, np.finfo(float).eps)
        
        # Market correlation
        df['Market_Correlation'] = df['Return'].rolling(window=20).corr(df['Market_Return'])
        
        # Beta calculation
        df['Covariance'] = df['Return'].rolling(window=20).cov(df['Market_Return'])
        market_var = df['Market_Return'].rolling(window=20).var()
        df['Beta'] = df['Covariance'] / market_var.replace(0, np.finfo(float).eps)
    
    # Fill NaN values with appropriate methods
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✅ Created {len(df.columns)} features")
    return df

def build_ensemble_model(X_train, y_train):
    """Build an ensemble of models with hyperparameter tuning."""
    print("Building ensemble model...")
    
    # Define base models
    models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gbm', GradientBoostingRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42, objective='reg:squarederror')),
        ('lgbm', LGBMRegressor(random_state=42)),
        ('ridge', Ridge(random_state=42)),
        ('lasso', Lasso(random_state=42))
    ]
    
    # Define parameter grids for tuning
    param_grids = {
        'rf': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'gbm': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'xgb': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'lgbm': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 63]
        },
        'ridge': {
            'alpha': [0.1, 1.0, 10.0]
        },
        'lasso': {
            'alpha': [0.1, 1.0, 10.0]
        }
    }
    
    # Perform grid search for each model
    tuned_models = []
    for name, model in models:
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=TimeSeriesSplit(n_splits=5),
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        tuned_models.append((name, grid_search.best_estimator_))
        print(f"Best params for {name}: {grid_search.best_params_}")
    
    # Create weighted ensemble
    ensemble = VotingRegressor(
        estimators=tuned_models,
        weights=[0.25, 0.25, 0.2, 0.2, 0.05, 0.05]  # Higher weights for tree-based models
    )
    
    print("✅ Ensemble model built")
    return ensemble



def collect_stock_news(ticker_symbol, days_back=30, max_articles=50):
    """Collect news articles for a stock over a specific time period."""
    try:
        # Get company info
        company = yf.Ticker(ticker_symbol)
        company_name = company.info.get("shortName", ticker_symbol)
        
        print(f"Collecting news for {ticker_symbol} ({company_name})...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Fetch news using fetch_company_news function
        news_data = fetch_company_news(ticker_symbol, max_articles)
        
        if news_data.empty:
            return pd.DataFrame()
        
        # Filter by date range
        news_data['publishedAt'] = pd.to_datetime(news_data['publishedAt'])
        filtered_news = news_data[
            (news_data['publishedAt'] >= start_date) & 
            (news_data['publishedAt'] <= end_date)
        ]
        
        print(f"Found {len(filtered_news)} news articles in the specified date range")
        return filtered_news
        
    except Exception as e:
        print(f"Error in collect_stock_news: {e}")
        return pd.DataFrame()

def aggregate_sentiment_by_day(news_df):
    """
    Aggregate sentiment scores by day for time-series analysis.
    """
    if news_df.empty:
        print("No news data to aggregate")
        return pd.DataFrame()
    
    print("Aggregating sentiment by day...")
    
    # Ensure publishedAt is datetime type
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
    
    # Extract date part for grouping
    news_df['date'] = news_df['publishedAt'].dt.date
    
    # Group by date and calculate aggregate metrics
    agg_sentiment = news_df.groupby(['date', 'ticker']).agg({
        'combined_sentiment': 'mean',
        'vader_sentiment': 'mean',
        'textblob_sentiment': 'mean',
        'finbert_sentiment': 'mean',
        'sentiment_intensity': 'mean',
        'textblob_subjectivity': 'mean',
        'article_rank': 'count'  # Count articles per day
    }).reset_index()
    
    # Rename count column
    agg_sentiment.rename(columns={'article_rank': 'article_count'}, inplace=True)
    
    # Calculate standard deviation of sentiment as a volatility measure
    sentiment_std = news_df.groupby(['date', 'ticker'])['combined_sentiment'].std().reset_index()
    sentiment_std.rename(columns={'combined_sentiment': 'sentiment_std'}, inplace=True)
    
    # Merge with standard deviation
    agg_sentiment = pd.merge(agg_sentiment, sentiment_std, on=['date', 'ticker'], how='left')
    
    # Fill NaN values for days with only one article
    agg_sentiment['sentiment_std'].fillna(0, inplace=True)
    
    # Convert date back to datetime for easier merging with stock data
    agg_sentiment['Date'] = pd.to_datetime(agg_sentiment['date'])
    
    print(f"✅ Created daily sentiment aggregation for {len(agg_sentiment)} days")
    return agg_sentiment

def train_and_evaluate_models(data, predict_days=1):
    """Train and evaluate models with enhanced feature selection and validation."""
    if len(data) < 100:  # Increased minimum data requirement
        print("⚠️ Insufficient data for model training. Skipping prediction.")
        return None
    
    print(f"Building prediction models for {predict_days}-day ahead returns...")
    
    # Prepare features
    enhanced_data = create_features(data)
    
    # Prepare target variable (n-day return)
    enhanced_data['target_return'] = enhanced_data['Return'].shift(-predict_days)
    
    # Drop rows with NaN values
    model_data = enhanced_data.dropna()
    
    if len(model_data) < 100:
        print("⚠️ After preparing features, insufficient data for model training.")
        return None
    
    # Feature selection
    exclude_cols = ['Date', 'ticker', 'target_return', 'Return']
    feature_cols = [col for col in model_data.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(model_data[col])]
    
    # Remove highly correlated features
    corr_matrix = model_data[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    if to_drop:
        print(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
        feature_cols = [col for col in feature_cols if col not in to_drop]
    
    # Prepare data
    X = model_data[feature_cols]
    y = model_data['target_return']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-series split
    tscv = TimeSeriesSplit(n_splits=5)
    split = list(tscv.split(X_scaled))[-1]  # Use the last fold
    train_idx, test_idx = split
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train ensemble model
    ensemble_model = build_ensemble_model(X_train, y_train)
    ensemble_model.fit(X_train, y_train)
    
    # Make predictions
    ensemble_pred = ensemble_model.predict(X_test)
    
    # Evaluate model
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(ensemble_mse)  # Calculate RMSE
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print(f"Ensemble Model - RMSE: {ensemble_rmse:.6f}, MAE: {ensemble_mae:.6f}, R²: {ensemble_r2:.6f}")
    
    return {
        'model': ensemble_model,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'metrics': {
            'rmse': ensemble_rmse,
            'mae': ensemble_mae,
            'r2': ensemble_r2
        },
        'models': {
            'ensemble_model': ensemble_model 
        }
    }
def predict_future_returns(data, model_package, days_ahead=1):
    """Predict future returns using the ensemble model."""
    if model_package is None:
        print("No valid model package available.")
        return None
    
    # Prepare features
    enhanced_data = create_features(data)
    latest_data = enhanced_data.tail(10)  # Use recent data for prediction
    
    # Prepare input features
    X_pred = latest_data[model_package['feature_cols']].iloc[-1:].copy()
    X_pred_scaled = model_package['scaler'].transform(X_pred)
    
    # Make prediction
    prediction = model_package['model'].predict(X_pred_scaled)[0]
    
    return prediction


def complete_analysis_pipeline(ticker_symbol, days_back=365, predict_days=1):
    """Run the complete analysis pipeline with enhanced features and models."""
    print(f"🔍 Starting analysis for {ticker_symbol}")
    
    # Step 1: Get stock data
    print("Retrieving stock price data...")
    stock_data = get_stock_data(ticker_symbol, days_back)
    if stock_data.empty:
        print("❌ No stock data found. Exiting.")
        return None
    
    # Step 2: Collect and process news
    print("Collecting and processing news articles...")
    news_data = collect_stock_news(ticker_symbol, days_back)
    if not news_data.empty:
        news_data = preprocess_news_data(news_data)
        news_data = analyze_sentiment(news_data)
        sentiment_by_day = aggregate_sentiment_by_day(news_data)
        stock_data = merge_stock_and_sentiment(stock_data, sentiment_by_day)
    
    # Step 3: Train models
    print("Training prediction models...")
    model_package = train_and_evaluate_models(stock_data, predict_days)
    
    if model_package:
        # Step 4: Make prediction
        print(f"Predicting {predict_days}-day ahead returns...")
        future_return = predict_future_returns(stock_data, model_package, predict_days)
        
        if future_return is not None:
            # Get last known price
            close_col = [col for col in stock_data.columns if 'Close' in col][0]
            last_price = stock_data[close_col].iloc[-1]
            predicted_price = last_price * (1 + future_return)
            
            # Determine direction based on predicted return
            direction = "up ⬆️" if future_return > 0 else "down ⬇️"
            
            print(f"\n📊 Prediction Results:")
            print(f"Ticker: {ticker_symbol}")
            print(f"Last Price: ${last_price:.2f}")
            print(f"Predicted {predict_days}-day Return: {future_return*100:.2f}%")
            print(f"Predicted Price: ${predicted_price:.2f}")
            print(f"Prediction Direction: {direction}")
            print(f"Model R²: {model_package['metrics']['r2']:.4f}")
            
            return {
                'ticker': ticker_symbol,
                'last_price': last_price,
                'predicted_return': future_return,
                'predicted_price': predicted_price,
                'direction': direction,
                'model_metrics': model_package['metrics'],
                'model_used': 'ensemble_model',  # Add this line
                'prediction_days': predict_days
            }
    
    print("❌ Unable to make prediction.")
    return None
def main():
    # Ticker symbol and parameters
    ticker_symbol = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    days_back = int(input("Enter number of days to analyze (min 100): ") or 365)
    if days_back < 100:
       logging.warning("Days to analyze should be at least 100. Setting to 100.")
       days_back = 100
    predict_days = int(input("Enter number of days ahead to predict (1-5): ") or 1)
    
    # Validate inputs
    if not ticker_symbol:
        logging.error("Ticker symbol is required.")
        return
    
    if days_back < 100:
        logging.warning("Days to analyze should be at least 100 for reliable analysis.")
        days_back = 100
    
    if predict_days < 1 or predict_days > 5:
        logging.warning("Prediction days should be between 1 and 5. Setting to 1.")
        predict_days = 1
    
    # Run the pipeline
    result = complete_analysis_pipeline(ticker_symbol, days_back, predict_days)
    
    if result:
        logging.info("\nAnalysis complete!")
    else:
        logging.error("\nAnalysis failed. Please check the inputs and try again.")

if __name__ == "__main__":
    main()