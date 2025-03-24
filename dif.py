import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time
import concurrent.futures
import logging
import warnings
from functools import partial
from tqdm import tqdm
import requests
import json

# Suppress warnings to clean up output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Import all functions from bstock module
from bstock import *

# Set page config
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# App title
st.title("📊 Stock Prediction Dashboard")
st.write("Analyze stock performance and predict future prices based on technical indicators and news sentiment.")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", 
    ["Stock Price Prediction", "Market Overview", "All Nifty 50 Prediction"])

# Function to get current price with caching and refresh time
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_current_price(ticker):
    """
    Fetches the current/latest price for a given ticker with timestamp
    Returns: (price, timestamp)
    """
    try:
        ticker_data = yf.Ticker(ticker)
        current_data = ticker_data.history(period="1d", interval="1m")
        
        if not current_data.empty:
            timestamp = current_data.index[-1]
            return float(current_data['Close'].iloc[-1]), timestamp
        else:
            info = ticker_data.info
            return (float(info.get('previousClose', 0)), datetime.now()) if 'previousClose' in info else (None, datetime.now())
    except Exception as e:
        st.error(f"Error fetching current price: {str(e)}")
        return None, datetime.now()

# Enhanced function to plot stock with prediction
def plot_stock_with_prediction(data, prediction_result, current_price=None):
    close_col = [col for col in data.columns if 'Close' in col][0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data[close_col],
        mode='lines',
        name='Historical Price',
        line=dict(color='royalblue')
    ))
    
    last_date = data['Date'].iloc[-1]
    prediction_date = last_date + timedelta(days=prediction_result['prediction_days'])
    
    if current_price is not None:
        current_time = datetime.now()
        fig.add_trace(go.Scatter(
            x=[current_time],
            y=[current_price],
            mode='markers',
            name='Current Price',
            marker=dict(color='gold', size=10, symbol='star')
        ))
    
    fig.add_trace(go.Scatter(
        x=[last_date, prediction_date],
        y=[prediction_result['last_price'], prediction_result['predicted_price']],
        mode='lines+markers',
        name='Prediction',
        line=dict(color='firebrick', dash='dash'),
        marker=dict(size=8)
    ))
    
    if 'confidence_interval' in prediction_result:
        lower = prediction_result['predicted_price'] - prediction_result['confidence_interval']
        upper = prediction_result['predicted_price'] + prediction_result['confidence_interval']
        
        fig.add_trace(go.Scatter(
            x=[prediction_date, prediction_date],
            y=[lower, upper],
            mode='lines',
            name='95% Confidence',
            line=dict(color='rgba(220, 20, 60, 0.5)', width=10)
        ))
    
    if 'Volume' in data.columns:
        fig.add_trace(go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name='Volume',
            marker_color='rgba(200, 200, 200, 0.5)',
            yaxis='y2',
            opacity=0.5
        ))
    
    fig.update_layout(
        title=f"{ticker_symbol} Stock Price with {prediction_result['prediction_days']}-Day Prediction",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        shapes=[
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=data['Date'].iloc[0],
                x1=prediction_date,
                y0=prediction_result['last_price'],
                y1=prediction_result['last_price'],
                line=dict(color="green", width=1, dash="dot"),
            )
        ],
        annotations=[
            dict(
                x=prediction_date,
                y=prediction_result['predicted_price'],
                xref="x",
                yref="y",
                text=f"${prediction_result['predicted_price']:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=0
            )
        ]
    )
    
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

# Function to set auto-refresh as enabled by default
def create_auto_refresh_component():
    # Set auto-refresh to be enabled by default
    auto_refresh = st.checkbox("Enable auto-refresh", value=True)
    # Set default refresh interval to 5 seconds
    refresh_interval = 5  # Fixed value for auto-refresh interval
    
    return auto_refresh, refresh_interval

# Function to get company info
@st.cache_data(ttl=3600)
def get_company_info(ticker):
    """Get company information for a given ticker"""
    try:
        company = yf.Ticker(ticker)
        info = company.info
        return info
    except Exception as e:
        logging.error(f"Error fetching company info for {ticker}: {e}")
        return None
    

def get_indian_market_tickers():
    """
    Attempts to dynamically fetch Indian market tickers (Nifty 50)
    Falls back to hardcoded lists if dynamic fetching fails
    
    Returns:
        list: List of Indian ticker symbols with .NS suffix
    """
    # Define fallback tickers (short list) - used only if all dynamic methods fail
    fallback_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
        "ITC.NS", "SBIN.NS", "BAJFINANCE.NS", "AXISBANK.NS", "KOTAKBANK.NS"
    ]
    
    # Method 2: Try direct API request to NSE
    try:
        import requests
        import json
        
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = json.loads(response.text)
            # Extract symbols from the response
            indian_symbols = [item['symbol'] for item in data['data']]
            return [symbol + ".NS" for symbol in indian_symbols]
        else:
            print(f"NSE API request failed with status code: {response.status_code}")
            raise Exception("NSE API request failed")
    except Exception as e:
        print(f"Direct API method failed: {str(e)}. Using fallback list.")
    
    # If we reach here, all dynamic methods failed
    print("Using fallback ticker list for Indian market")
    return fallback_tickers


# Function to get Nifty 50 stocks
def get_nifty50_stocks():
    """Get the list of Nifty 50 stocks with their Yahoo Finance tickers"""
    return get_indian_market_tickers()

# Simplified analysis pipeline for batch processing
def simplified_analysis_pipeline(ticker_symbol, days_back=365, predict_days=1, use_news=True):
    """
    A simplified version of the analysis pipeline for batch processing.
    This version skips some expensive operations to improve performance.
    """
    try:
        # Step 1: Get stock data
        stock_data = get_stock_data(ticker_symbol, days_back)
        if stock_data.empty:
            return None
        
        # Step 2: Collect and process news (optional, can be turned off for speed)
        if use_news:
            news_data = collect_stock_news(ticker_symbol, min(30, days_back))
            if not news_data.empty:
                news_data = preprocess_news_data(news_data)
                news_data = analyze_sentiment(news_data)
                sentiment_by_day = aggregate_sentiment_by_day(news_data)
                stock_data = merge_stock_and_sentiment(stock_data, sentiment_by_day)
        
        # Step 3: Train models
        model_package = train_and_evaluate_models(stock_data, predict_days)
        
        if model_package:
            # Step 4: Make prediction
            future_return = predict_future_returns(stock_data, model_package, predict_days)
            
            if future_return is not None:
                # Get last known price
                close_col = [col for col in stock_data.columns if 'Close' in col][0]
                last_price = stock_data[close_col].iloc[-1]
                predicted_price = last_price * (1 + future_return)
                
                # Determine direction based on predicted return
                direction = "UP" if future_return > 0 else "DOWN"
                
                # Get current live price using yfinance
                try:
                    live_data = yf.Ticker(ticker_symbol).history(period='1d')
                    live_price = live_data['Close'].iloc[-1] if not live_data.empty else last_price
                except:
                    live_price = last_price
                
                # Calculate difference between predicted and live price
                price_diff = predicted_price - live_price
                
                return {
                    'ticker': ticker_symbol.replace('.NS', ''),  # Remove .NS suffix for display
                    'previous_close_price': last_price,
                    'live_price': live_price,
                    'predicted_return': future_return,
                    'predicted_close_price': predicted_price,
                    'price_diff': price_diff,
                    'direction': direction,
                    'confidence': abs(model_package['metrics']['r2']),  # Use R² as confidence
                    'model_r2': model_package['metrics']['r2'],
                    'model_rmse': model_package['metrics']['rmse'],
                    'prediction_date': (datetime.now() + timedelta(days=predict_days)).strftime('%Y-%m-%d')
                }
        
        return None
    except Exception as e:
        logging.error(f"Error analyzing {ticker_symbol}: {e}")
        return None

# Function to process Nifty 50 stocks in parallel
def process_nifty50_stocks(days_back=365, predict_days=1, use_news=True, max_workers=1):
    """
    Process all Nifty 50 stocks in parallel and return a dataframe of predictions
    """
    nifty50_stocks = get_nifty50_stocks()
    
    # Create a results list
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with fixed parameters
        analysis_func = partial(simplified_analysis_pipeline, 
                               days_back=days_back, 
                               predict_days=predict_days, 
                               use_news=use_news)
        
        # Submit all tasks
        future_to_ticker = {executor.submit(analysis_func, ticker): ticker 
                           for ticker in nifty50_stocks}
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                completed += 1
                # Update progress bar
                progress_bar.progress(completed / len(nifty50_stocks))
                status_text.text(f"Processed {completed}/{len(nifty50_stocks)} stocks")
            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")
    
    # Convert results to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        # Sort by predicted return (descending)
        results_df = results_df.sort_values('predicted_return', ascending=False)
        # Add index column for display starting from 1
        results_df.insert(0, 'rank', range(1, len(results_df) + 1))
        return results_df
    else:
        return pd.DataFrame()


def get_top_movers(market="US", period="1d"):
    """
    Fetches top gainers and losers for specified market and time period
    Args:
        market: 'US' or 'INDIA'
        period: '1d', '1w', '1m' for day, week, month
    Returns:
        tuple of (gainers_df, losers_df)
    """
    # Map period to yfinance period parameter
    period_mapping = {
        "1d": "2d",       # Need 2 days of data to compute 1-day change
        "1w": "7d",       # Get 7 days data for weekly change
        "1m": "1mo"       # Get 1 month data for monthly change
    }
    
    # Set market-specific tickers
    if market == "US":
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "WMT", 
                "JPM", "BAC", "GS", "V", "MA", "JNJ", "PFE", "KO", "PEP", "DIS", "NFLX", 
                "HD", "INTC", "VZ", "CSCO", "MRK", "CVX", "XOM", "BA", "MMM",
                "ADBE", "PYPL", "CRM", "ABT", "COST", "TMO", "NKE", "ORCL", "IBM", "MDT"]

        market_name = "US"
    else:  # Indian market
        try:
            # Use our refactored function to get Indian tickers
            tickers = get_indian_market_tickers()
        except Exception as e:
            st.warning(f"Error fetching Indian market tickers: {str(e)}. Using default list.")
            # Fallback to the minimal list if everything fails
            tickers = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                "ITC.NS", "SBIN.NS", "BAJFINANCE.NS", "AXISBANK.NS", "KOTAKBANK.NS"
            ]
        
        market_name = "India"
    
    # Download data for all tickers
    data = yf.download(tickers, period=period_mapping[period], auto_adjust=True)['Close']
    
    # Calculate percentage change based on period
    if period == "1d":
        # For 1-day period, use last 2 days
        if len(data) >= 2:
            pct_change = ((data.iloc[-1] - data.iloc[-2]) / data.iloc[-2] * 100).round(2)
        else:
            # Fallback if not enough data
            pct_change = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100).round(2)
    else:
        # For week or month, use first and last available values
        pct_change = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100).round(2)
    
    # Clean up NaN values
    pct_change = pct_change.dropna()
    
    # Ensure we have valid price data
    valid_tickers = pct_change.index
    last_prices = data.iloc[-1][valid_tickers].round(2)
    
    # Create dataframe with percentage change
    movers_df = pd.DataFrame({
        'Symbol': valid_tickers,
        'Last Price': last_prices.values,
        'Change %': pct_change.values
    })
    
    # Sort for gainers and losers
    gainers = movers_df.sort_values('Change %', ascending=False).head(10).reset_index(drop=True)
    gainers.index = gainers.index + 1
    losers = movers_df.sort_values('Change %', ascending=True).head(10).reset_index(drop=True)
    losers.index = losers.index + 1  # Start index from 1
    
    # Add market and period labels
    gainers['Market'] = market_name
    losers['Market'] = market_name
    
    period_labels = {"1d": "1 Day", "1w": "1 Week", "1m": "1 Month"}
    gainers['Period'] = period_labels[period]
    losers['Period'] = period_labels[period]
    
    return gainers, losers
# Main app flow based on selected mode
if app_mode == "Stock Price Prediction":
    # Sidebar for inputs
    with st.sidebar:
        st.header("Analysis Parameters")
        
        # Input for ticker symbol
        ticker_symbol = st.text_input("Enter Ticker Symbol", "AAPL").upper()
        
        # Input for days to analyze
        days_back = st.slider("Days to Analyze", 
                            min_value=30, 
                            max_value=90, 
                            value=30, 
                            help="Number of past days to include in analysis")
        
        # Input for prediction days
        predict_days = st.slider("Days to Predict Ahead", 
                                min_value=1, 
                                max_value=5, 
                                value=1, 
                                help="Number of days ahead to predict")
        
        # Add auto-refresh options
        st.divider()
        st.subheader("Real-time Updates")
        auto_refresh, refresh_interval = create_auto_refresh_component()  # Now refresh_interval is fixed at 5 seconds
        
        # Run analysis button
        run_analysis = st.button("Run Analysis", type="primary")
        
        st.divider()
        st.markdown("### About")
        st.info("This app uses machine learning to predict stock price movements based on technical indicators and news sentiment analysis.")
    
    # Initialize session state for storing results
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'current_price' not in st.session_state:
        st.session_state.current_price = None
    
    if 'price_timestamp' not in st.session_state:
        st.session_state.price_timestamp = None
    
    # Add auto-refresh functionality with st.rerun()
    if auto_refresh and st.session_state.analysis_result is not None:
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = datetime.now()
            time.sleep(0.1)  # Small delay to prevent browser issues
            st.rerun()  # This will rerun the entire app
    
    # Determine if we should run analysis (button click or first load)
    should_run_analysis = run_analysis
    
    # Run analysis automatically on first load if ticker is provided
    if st.session_state.analysis_result is None and ticker_symbol:
        should_run_analysis = True
    
    if should_run_analysis:
        with st.spinner(f"Analyzing {ticker_symbol}..."):
            # Display company info while loading
            company_info = get_company_info(ticker_symbol)
            if company_info:
                col1, col2 = st.columns([1, 2])
                with col1:
                    if 'logo_url' in company_info and company_info['logo_url']:
                        st.image(company_info['logo_url'], width=100)
                    st.subheader(company_info.get('shortName', ticker_symbol))
                    st.write(f"Sector: {company_info.get('sector', 'N/A')}")
                    st.write(f"Industry: {company_info.get('industry', 'N/A')}")
                with col2:
                    st.write(company_info.get('longBusinessSummary', ''))
            
            # Run the analysis pipeline
            result = complete_analysis_pipeline(ticker_symbol, days_back, predict_days)
                        
            # Store result in session state
            if result:
                st.session_state.analysis_result = result
    
    # If we have analysis results, display them
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Prediction", "Recent News Articles"])
        
        with tab1:
            # Fetch the current price (this will update even with auto-refresh)
            current_price, price_timestamp = get_current_price(ticker_symbol)
            
            # Store in session state
            st.session_state.current_price = current_price
            st.session_state.price_timestamp = price_timestamp
            
            # Display a refresh button for manual updates
            col_refresh, _ = st.columns([1, 3])
            with col_refresh:
                if st.button("🔄 Refresh Price"):
                    # Clear cache to force refresh
                    get_current_price.clear()
                    current_price, price_timestamp = get_current_price(ticker_symbol)
                    st.session_state.current_price = current_price
                    st.session_state.price_timestamp = price_timestamp
            
            # Show last refresh time
            time_format = "%Y-%m-%d %H:%M:%S"
            if auto_refresh:
                st.caption(f"Auto-refreshing every {refresh_interval} seconds. Last refresh: {st.session_state.last_refresh.strftime(time_format)}")
            
            # Display results in a nice format
            col1, col2, col3, col4 = st.columns(4)
            
            currency_symbol = "₹" if ticker_symbol.endswith('.NS') else "$"
            
            with col1:
                st.metric(
                    label="Previous Closed Price", 
                    value=f"{currency_symbol}{result['last_price']:.2f}"
                )
            
            with col2:
                if current_price:
                    change = ((current_price - result['last_price']) / result['last_price']) * 100
                    st.metric(
                        label=f"Current Price ({price_timestamp.strftime('%H:%M:%S')})", 
                        value=f"{currency_symbol}{current_price:.2f}",
                        delta=f"{change:.2f}%"
                    )
                else:
                    st.metric(
                        label="Current Price", 
                        value="Unavailable"
                    )
            
            with col3:
                st.metric(
                    label=f"Predicted Price ({predict_days} days)", 
                    value=f"{currency_symbol}{result['predicted_price']:.2f}",
                    delta=f"{result['predicted_return']*100:.2f}%"
                )
            
            with col4:
                direction_icon = "⬆️ UP" if result['direction'] == "up ⬆️" else "⬇️ DOWN"
                direction_color = "green" if result['direction'] == "up ⬆️" else "red"
                st.markdown(f"<h2 style='text-align: center; color: {direction_color};'>{direction_icon}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Prediction Direction</p>", unsafe_allow_html=True)
            
            # Display additional model information
            st.write(f"**Model Used:** {result['model_used'].replace('_', ' ').title()}")
            
            with st.expander("Model Metrics Details"):
                st.write(f"- RMSE: {result['model_metrics']['rmse']:.6f}")
                st.write(f"- R²: {result['model_metrics']['r2']:.6f}")
                st.write(f"- MAE: {result['model_metrics']['mae']:.6f}")
            
            # Get combined data for visualization
            stock_data = get_stock_data(ticker_symbol, days_back + 10)
            sentiment_data = aggregate_sentiment_by_day(analyze_sentiment(preprocess_news_data(collect_stock_news(ticker_symbol, days_back))))
            combined_data = merge_stock_and_sentiment(stock_data, sentiment_data)
            
            # Plot stock price with prediction (including current price)
            st.plotly_chart(plot_stock_with_prediction(combined_data, result, current_price), use_container_width=True)
        
        with tab2:
            if not sentiment_data.empty:
                st.subheader("LATEST 5 NEWS ARTICLES")
                news_data = collect_stock_news(ticker_symbol, days_back)
                news_data = analyze_sentiment(preprocess_news_data(news_data))
                
                sorted_news = news_data.sort_values(by='publishedAt', ascending=False).head(5)
                
                if len(sentiment_data) > 3:
                    st.subheader("Sentiment Trend Analysis")
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=sentiment_data.index,
                        y=sentiment_data['combined_sentiment'],
                        mode='lines+markers',
                        name='Sentiment Score',
                        line=dict(color='purple', width=2),
                        marker=dict(size=6)
                    ))
                    
                    fig.add_shape(
                        type="line",
                        x0=sentiment_data.index.min(),
                        x1=sentiment_data.index.max(),
                        y0=0,
                        y1=0,
                        line=dict(color="gray", width=1, dash="dot")
                    )
                    
                    fig.add_hrect(
                        y0=0.3, y1=1.0, 
                        fillcolor="green", opacity=0.1, 
                        line_width=0,
                        annotation_text="Positive",
                        annotation_position="top right"
                    )
                    
                    fig.add_hrect(
                        y0=-0.3, y1=0.3, 
                        fillcolor="gray", opacity=0.1, 
                        line_width=0,
                        annotation_text="Neutral",
                        annotation_position="top right"
                    )
                    
                    fig.add_hrect(
                        y0=-1.0, y1=-0.3, 
                        fillcolor="red", opacity=0.1, 
                        line_width=0,
                        annotation_text="Negative",
                        annotation_position="bottom right"
                    )
                    
                    fig.update_layout(
                        title="News Sentiment Trend Analysis",
                        xaxis_title="Date",
                        yaxis_title="Sentiment Score",
                        yaxis=dict(range=[-1, 1]),
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                for i, (_, article) in enumerate(sorted_news.iterrows()):
                    sentiment = article['combined_sentiment']
                    sentiment_color = "green" if sentiment > 0 else "red" if sentiment < 0 else "gray"
                    sentiment_icon = "✅" if sentiment > 0 else "❌" if sentiment < 0 else "⚪"
                    
                    st.markdown(f"""
                    <div style="border-left: 5px solid {sentiment_color}; padding-left: 10px; margin-bottom: 20px;">
                        <h5>{sentiment_icon} {article['title']}</h5>
                        <p><em>{article['description']}</em></p>
                        <p>Sentiment Score: <span style="color: {sentiment_color}; font-weight: bold;">{sentiment:.3f}</span></p>
                        <p>Published: {pd.to_datetime(article['publishedAt']).strftime('%Y-%m-%d %H:%M')}</p>
                        <a href="{article['url']}" target="_blank">Read more</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent news articles found for this stock.")
    else:
        st.info("Enter a ticker symbol and click 'Run Analysis' to start.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("How it works")
            st.write("""
            This dashboard performs comprehensive stock analysis by:
            1. Collecting recent news articles about the stock
            2. Analyzing the sentiment of these articles
            3. Retrieving historical stock data
            4. Training machine learning models
            5. Predicting future price movements
            """)
        
        with col2:
            st.subheader("Available visualizations")
            st.write("""
            After running the analysis, you'll see:
            - Stock price chart with prediction
            - Sentiment analysis from news articles
            """)
        
        st.image("https://static.seekingalpha.com/uploads/2018/1/7/48200183-15154003071692858_origin.jpg", 
                caption="Sample stock analysis chart (not real data)",
                use_container_width=True)
                
elif app_mode == "Market Overview":
    st.header("Market Overview")
    
    selected_period = st.radio(
        "Select Time Period",
        ["1d", "1w", "1m"],
        format_func=lambda x: {"1d": "1 Day", "1w": "1 Week", "1m": "1 Month"}[x],
        horizontal=True
    )
    
    tab1, tab2 = st.tabs(["US Market", "Indian Market"])
    
    with tab1:
        st.subheader("US Market Top Movers")
        
        try:
            us_gainers, us_losers = get_top_movers(market="US", period=selected_period)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"Last updated: {timestamp}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Top Gainers")
                if not us_gainers.empty:
                    st.dataframe(us_gainers[['Symbol', 'Last Price', 'Change %']], 
                                  use_container_width=True, 
                                  column_config={
                                      "Change %": st.column_config.NumberColumn(
                                          format="%.2f%%",
                                          help="Percentage change",
                                      )
                                  })
                else:
                    st.info("No gainers data available at the moment.")
            
            with col2:
                st.write("### Top Losers")
                if not us_losers.empty:
                    st.dataframe(us_losers[['Symbol', 'Last Price', 'Change %']], 
                                  use_container_width=True,
                                  column_config={
                                      "Change %": st.column_config.NumberColumn(
                                          format="%.2f%%",
                                          help="Percentage change"
                                      )
                                  })
                else:
                    st.info("No losers data available at the moment.")
            
            # Add market indices with period-specific changes
            st.write("### Market Indices")
            indices_tickers = ['^DJI', '^GSPC', '^IXIC']
            indices_names = ['Dow Jones', 'S&P 500', 'NASDAQ']
            
            period_mapping = {"1d": "2d", "1w": "7d", "1m": "1mo"}
            yf_period = period_mapping.get(selected_period, "2d")
            
            try:
                indices_data = yf.download(indices_tickers, period=yf_period, auto_adjust=True)['Close']
                
                if not indices_data.empty:
                    if selected_period == "1d" and len(indices_data) >= 2:
                        indices_pct_change = ((indices_data.iloc[-1] - indices_data.iloc[-2]) / indices_data.iloc[-2] * 100).round(2)
                    else:
                        if len(indices_data) >= 2:
                            indices_pct_change = ((indices_data.iloc[-1] - indices_data.iloc[0]) / indices_data.iloc[0] * 100).round(2)
                        else:
                            indices_pct_change = pd.Series([0.0, 0.0, 0.0], index=indices_data.columns)
                    
                    indices_pct_change = indices_pct_change.fillna(0)
                    
                    indices_df = pd.DataFrame({
                        'Index': indices_names,
                        'Value': indices_data.iloc[-1].values.round(2),
                        'Change %': indices_pct_change.values
                    })
                    
                    st.dataframe(indices_df, use_container_width=True,
                                  column_config={
                                      "Change %": st.column_config.NumberColumn(
                                          format="%.2f%%",
                                          help="Percentage change"
                                      )
                                  })
                else:
                    st.warning("Unable to fetch indices data. Please try again later.")
            except Exception as e:
                st.error(f"Error fetching indices data: {str(e)}")
            
            # Add visualization for top movers
            st.write("### Performance Visualization")
            
            if not us_gainers.empty and not us_losers.empty:
                gainers_top = us_gainers[['Symbol', 'Change %']].head(3)
                losers_top = us_losers[['Symbol', 'Change %']].head(3)
                
                if not gainers_top.empty or not losers_top.empty:
                    all_movers = pd.concat([gainers_top, losers_top])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=all_movers['Symbol'],
                        y=all_movers['Change %'],
                        marker_color=['green' if x >= 0 else 'red' for x in all_movers['Change %']]
                    ))
                    
                    fig.update_layout(
                        title=f"Top Movers ({selected_period})",
                        xaxis_title="Symbol",
                        yaxis_title="Change %",
                        showlegend=False,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data available to visualize movers.")
            else:
                st.info("No movers data available to visualize.")
                
        except Exception as e:
            st.error(f"Error processing US market data: {str(e)}")
            st.info("Please try again later or check your internet connection.")
    
    with tab2:
        st.subheader("Indian Market Top Movers")
        
        try:
            india_gainers, india_losers = get_top_movers(market="INDIA", period=selected_period)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"Last updated: {timestamp}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Top Gainers")
                if not india_gainers.empty:
                    st.dataframe(india_gainers[['Symbol', 'Last Price', 'Change %']], 
                                  use_container_width=True,
                                  column_config={
                                      "Change %": st.column_config.NumberColumn(
                                          format="%.2f%%",
                                          help="Percentage change"
                                      )
                                  })
                else:
                    st.info("No gainers data available at the moment.")
            
            with col2:
                st.write("### Top Losers")
                if not india_losers.empty:
                    st.dataframe(india_losers[['Symbol', 'Last Price', 'Change %']], 
                                  use_container_width=True,
                                  column_config={
                                      "Change %": st.column_config.NumberColumn(
                                          format="%.2f%%",
                                          help="Percentage change"
                                      )
                                  })
                else:
                    st.info("No losers data available at the moment.")
            
            # Add Indian market indices with period-specific changes
            st.write("### Market Indices")
            indices_tickers = ['^NSEI', '^BSESN']
            indices_names = ['NIFTY 50', 'BSE SENSEX']
            
            period_mapping = {"1d": "2d", "1w": "7d", "1m": "1mo"}
            yf_period = period_mapping.get(selected_period, "2d")
            
            try:
                indices_data = yf.download(indices_tickers, period=yf_period, auto_adjust=True)['Close']
                
                if not indices_data.empty:
                    if selected_period == "1d" and len(indices_data) >= 2:
                        indices_pct_change = ((indices_data.iloc[-1] - indices_data.iloc[-2]) / indices_data.iloc[-2] * 100).round(2)
                    else:
                        if len(indices_data) >= 2:
                            indices_pct_change = ((indices_data.iloc[-1] - indices_data.iloc[0]) / indices_data.iloc[0] * 100).round(2)
                        else:
                            indices_pct_change = pd.Series([0.0, 0.0], index=indices_data.columns)
                    
                    indices_pct_change = indices_pct_change.fillna(0)
                    
                    indices_df = pd.DataFrame({
                        'Index': indices_names,
                        'Value': indices_data.iloc[-1].values.round(2),
                        'Change %': indices_pct_change.values
                    })
                    
                    st.dataframe(indices_df, use_container_width=True,
                                  column_config={
                                      "Change %": st.column_config.NumberColumn(
                                          format="%.2f%%",
                                          help="Percentage change"
                                      )
                                  })
                else:
                    st.warning("Unable to fetch Indian indices data. Please try again later.")
            except Exception as e:
                st.error(f"Error fetching Indian indices data: {str(e)}")
            
            # Add visualization for top movers
            st.write("### Performance Visualization")
            
            if not india_gainers.empty and not india_losers.empty:
                gainers_top = india_gainers[['Symbol', 'Change %']].head(3)
                losers_top = india_losers[['Symbol', 'Change %']].head(3)
                
                if not gainers_top.empty or not losers_top.empty:
                    all_movers = pd.concat([gainers_top, losers_top])
                    
                    all_movers['Display Symbol'] = all_movers['Symbol'].apply(lambda x: x.replace('.NS', '') if isinstance(x, str) else x)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=all_movers['Display Symbol'],
                        y=all_movers['Change %'],
                        marker_color=['green' if x >= 0 else 'red' for x in all_movers['Change %']]
                    ))
                    
                    fig.update_layout(
                        title=f"Top Movers ({selected_period})",
                        xaxis_title="Symbol",
                        yaxis_title="Change %",
                        showlegend=False,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data available to visualize Indian market movers.")
            else:
                st.info("No Indian market movers data available to visualize.")
            
            # Additional information
            with st.expander("About Indian Market Data"):
                st.write("""
                - Data is sourced from Yahoo Finance API
                - All stock symbols have '.NS' extension for NSE listings
                - Market timing is 9:15 AM to 3:30 PM IST (Monday to Friday)
                - The NIFTY 50 consists of 50 of the largest Indian companies listed on the National Stock Exchange
                - The BSE SENSEX (S&P BSE SENSEX) comprises 30 established and financially sound companies listed on the Bombay Stock Exchange
                """)
        
        except Exception as e:
            st.error(f"Error processing Indian market data: {str(e)}")
            st.info("Please try again later or check your internet connection.")
    
    # Add footer
    st.markdown("---")
    st.caption("Disclaimer: This app is for informational purposes only and should not be construed as investment advice.")
    st.caption("Data source: Yahoo Finance API")
    st.caption("© 2025 Stock Prediction Dashboard")


elif app_mode == "All Nifty 50 Prediction":
    st.header("📊 Nifty 50 Stock Price Predictor")
    st.write("This app predicts the future price movements of all Nifty 50 stocks using machine learning.")

    # Fixed/default values
    days_back = 365
    predict_days = 1
    use_news = True
    max_workers = 1
    
    # Run button in sidebar
    if st.sidebar.button("Run Analysis", type="primary"):
        start_time = time.time()
        
        with st.spinner("Analyzing Nifty 50 stocks... This may take several minutes."):
            # Process all stocks with fixed parameters
            results_df = process_nifty50_stocks(
                days_back=days_back,
                predict_days=predict_days,
                use_news=use_news,
                max_workers=max_workers
            )
            
            if not results_df.empty:
                # Display results
                elapsed_time = time.time() - start_time
                st.success(f"✅ Analysis completed in {elapsed_time:.2f} seconds")
                
                # Display prediction date
                prediction_date = results_df['prediction_date'].iloc[0]
                st.subheader(f"Price Predictions for today")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📈 Top Gainers", "📉 Top Losers", "📊 All Stocks"])
                
                # Add trend indicators based on direction
                results_df['predict_price_indicator'] = results_df.apply(
                    lambda x: f"↑ {x['predicted_return']*100:.2f}%" if x['direction'] == 'UP' 
                    else f"↓ {x['predicted_return']*100:.2f}%", 
                    axis=1
                )
                
                # Calculate difference between predicted and live price
                results_df['price_diff'] = results_df['predicted_close_price'] - results_df['live_price']
                
                # Calculate price difference percentage correctly (percentage change from live price to predicted price)
                results_df['price_diff_percent'] = (results_df['price_diff'] / results_df['live_price']) * 100
                
                # Format the dataframe for display
                display_df = results_df.copy()
                display_df['previous_close_price'] = display_df['previous_close_price'].apply(lambda x: f"₹{x:.2f}")
                display_df['live_price'] = display_df['live_price'].apply(lambda x: f"₹{x:.2f}")
                display_df['predicted_close_price'] = display_df['predicted_close_price'].apply(lambda x: f"₹{x:.2f}")
                display_df['price_diff'] = display_df['price_diff'].apply(lambda x: f"₹{x:.2f}")
                
                # Format the price difference percentage without arrows but keep the sign
                display_df['price_diff_percent'] = results_df['price_diff_percent'].apply(
                    lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
                )
                
                # Format the price_change_indicator column for live price
                display_df['live_price_indicator'] = results_df.apply(
                    lambda x: f"↑ +{(x['live_price']/x['previous_close_price']-1)*100:.2f}%" 
                    if x['live_price'] > x['previous_close_price'] 
                    else f"↓ {(x['live_price']/x['previous_close_price']-1)*100:.2f}%", 
                    axis=1
                )
                
                # Columns to display in the UI - added price_diff_percent
                display_cols = ['rank', 'ticker', 'previous_close_price', 'live_price', 
                               'live_price_indicator', 'predicted_close_price','predict_price_indicator', 'price_diff', 'price_diff_percent']
                               
                
                # Apply styling to the dataframe using Streamlit's styling functionality
                def color_indicators(val):
                    # For trend indicators and price change indicators
                    if isinstance(val, str):
                        if val.startswith('↑') or val.startswith('+'):
                            return 'color: green; font-weight: bold'
                        elif val.startswith('↓') or val.startswith('-'):
                            return 'color: red; font-weight: bold'
                    return ''
                
                # Top gainers
                with tab1:
                    gainers = display_df[display_df['direction'] == 'UP'].sort_values('rank')
                    if not gainers.empty:
                        st.write(f"### All Gainers ({len(gainers)} stocks)")
                        
                        # Apply styling to dataframe
                        styled_gainers = gainers[display_cols].style.applymap(
                            color_indicators, 
                            subset=['live_price_indicator', 'predict_price_indicator', 'price_diff_percent']
                        )
                        
                        st.dataframe(
                            styled_gainers,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Create a bar chart for top gainers
                        num_to_show = min(15, len(gainers))  # Show up to 15 gainers in chart
                        top_gainers = gainers.head(num_to_show)
                        
                        # Convert percentage string back to float for plotting
                        plot_data = results_df[results_df['direction'] == 'UP'].head(num_to_show)
                        
                        fig = px.bar(
                            plot_data,
                            x='ticker',
                            y='predicted_return',
                            color='predicted_return',
                            color_continuous_scale='Greens',
                            title=f"Top {num_to_show} Stocks - Expected Gains",
                            labels={'ticker': 'Stock', 'predicted_return': 'Expected Return'},
                            text=plot_data['predicted_return'].apply(lambda x: f"{x*100:.2f}%")
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No stocks predicted to gain.")
                
                # Top losers
                with tab2:
                    # Reset rank for losers to start from 1
                    losers = display_df[display_df['direction'] == 'DOWN'].sort_values('predicted_return').reset_index(drop=True)
                    losers['rank'] = losers.index + 1  # Start rank from 1
                    
                    if not losers.empty:
                        st.write(f"### All Losers ({len(losers)} stocks)")
                        
                        # Apply styling to dataframe
                        styled_losers = losers[display_cols].style.applymap(
                            color_indicators, 
                            subset=['live_price_indicator', 'predict_price_indicator', 'price_diff_percent']
                        )
                        
                        st.dataframe(
                            styled_losers,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Create a bar chart for top losers
                        num_to_show = min(15, len(losers))  # Show up to 15 losers in chart
                        
                        # Convert percentage string back to float for plotting
                        plot_data = results_df[results_df['direction'] == 'DOWN'].sort_values('predicted_return').head(num_to_show)
                        
                        fig = px.bar(
                            plot_data,
                            x='ticker',
                            y='predicted_return',
                            color='predicted_return',
                            color_continuous_scale='Reds_r',
                            title=f"Top {num_to_show} Stocks - Expected Losses",
                            labels={'ticker': 'Stock', 'predicted_return': 'Expected Return'},
                            text=plot_data['predicted_return'].apply(lambda x: f"{x*100:.2f}%")
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No stocks predicted to lose.")
                
                # All stocks
                with tab3:
                    st.write("### All Nifty 50 Predictions")
                    
                    # Apply styling to the all stocks dataframe
                    styled_all = display_df[display_cols].style.applymap(
                        color_indicators, 
                        subset=['live_price_indicator', 'predict_price_indicator', 'price_diff_percent']
                    )
                    
                    # Create a custom dataframe for display with only the specified columns
                    st.dataframe(
                        styled_all,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Prepare CSV data with requested columns including the new percentage column
                    csv_df = results_df.copy()
                    # Keep only the columns needed for the CSV download
                    csv_columns = ['rank', 'ticker', 'previous_close_price', 'live_price', 
                                  'predicted_close_price', 'price_diff', 'price_diff_percent', 
                                  'predicted_return', 'direction']
                    csv_df = csv_df[csv_columns]
                    
                    # Download button
                    csv = csv_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"nifty50_predictions_{datetime.now().strftime('%Y-%m-%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Create a visualization of all stocks
                    fig = go.Figure()
                    
                    # Add trace for each stock
                    for direction, color in [('UP', 'green'), ('DOWN', 'red')]:
                        df_subset = results_df[results_df['direction'] == direction]
                        fig.add_trace(go.Bar(
                            x=df_subset['ticker'],
                            y=df_subset['predicted_return'],
                            name=direction,
                            marker_color=color,
                            text=df_subset['predicted_return'].apply(lambda x: f"{x*100:.2f}%"),
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title="All Nifty 50 Stocks - Predicted Returns",
                        xaxis_tickangle=-45,
                        xaxis_title="Stock",
                        yaxis_title="Predicted Return",
                        barmode='group',
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                st.subheader("Model Performance Metrics")
                cols = st.columns(3)
                avg_r2 = results_df['model_r2'].mean()
                avg_rmse = results_df['model_rmse'].mean()
                processed_count = len(results_df)
                total_count = len(get_nifty50_stocks())
                
                cols[0].metric("Average R²", f"{avg_r2:.4f}")
                cols[1].metric("Average RMSE", f"{avg_rmse:.6f}")
                cols[2].metric("Stocks Processed", f"{processed_count}/{total_count-1}")
                
            else:
                st.error("No predictions could be generated. Please try again.")
    
    # Information section
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This app uses machine learning to predict the future price movements of 
        Nifty 50 stocks. The predictions are based on:
        
        • Historical data (365 days)
        • Technical indicators
        • News sentiment analysis
        • 1-day ahead predictions
        
        **Note**: These predictions are for educational purposes only and should 
        not be used as financial advice.
        """
    )