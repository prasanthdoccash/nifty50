from flask import Flask, render_template
import pandas as pd
from jugaad_data.nse import stock_df
import datetime as dt

app = Flask(__name__)

def fetch_intraday_data(symbols, start_date, end_date):
    all_data = {}
    for symbol in symbols:
        df = stock_df(symbol=symbol, from_date=start_date, to_date=end_date, series="EQ")
        if not df.empty:
            all_data[symbol] = df
    return all_data

def calculate_vwap(df):
    if 'CLOSE' in df.columns and 'VOLUME' in df.columns:
        df['cumulative_volume'] = df['VOLUME'].cumsum()
        df['cumulative_price_volume'] = (df['CLOSE'] * df['VOLUME']).cumsum()
        df['VWAP'] = df['cumulative_price_volume'] / df['cumulative_volume']
        df.drop(['cumulative_volume', 'cumulative_price_volume'], axis=1, inplace=True)
    return df

def calculate_rsi(df, period=14):
    if 'CLOSE' in df.columns:
        delta = df['CLOSE'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_sma(df, period=20):
    if 'CLOSE' in df.columns:
        df['SMA'] = df['CLOSE'].rolling(window=period).mean()
    return df

def calculate_ema(df, span=20):
    if 'CLOSE' in df.columns:
        df['EMA'] = df['CLOSE'].ewm(span=span, min_periods=span).mean()
    return df

def calculate_bollinger_bands(df, window=20, std_dev=2):
    if 'CLOSE' in df.columns:
        df['SMA'] = df['CLOSE'].rolling(window=window).mean()
        df['Upper Band'] = df['SMA'] + std_dev * df['CLOSE'].rolling(window=window).std()
        df['Lower Band'] = df['SMA'] - std_dev * df['CLOSE'].rolling(window=window).std()
    return df

def calculate_bid_ask_spread(df):
    if 'HIGH' in df.columns and 'LOW' in df.columns:
        df['Bid-Ask Spread'] = df['HIGH'] - df['LOW']
    return df

def calculate_turnover_ratio(df):
    if 'VOLUME' in df.columns and 'NO OF TRADES' in df.columns:
        df['Turnover Ratio'] = df['VOLUME'] / df['NO OF TRADES']
    return df

def determine_sentiment(df):
    sentiment = {}
    # Calculate VWAP sentiment
    if 'VWAP' in df.columns and 'CLOSE' in df.columns:
        vwap = df['VWAP'].iloc[-1]
        current_price = df['CLOSE'].iloc[-1]
        if current_price > vwap:
            sentiment['VWAP'] = 'Bullish'
        elif current_price < vwap:
            sentiment['VWAP'] = 'Bearish'
        else:
            sentiment['VWAP'] = 'Neutral'
    # Calculate RSI sentiment
    if 'RSI' in df.columns:
        rsi = df['RSI'].iloc[-1]
        if rsi < 30:
            sentiment['RSI'] = 'Undervalued'
        elif rsi > 70:
            sentiment['RSI'] = 'Overvalued'
        else:
            sentiment['RSI'] = 'Neutral'
    
    if 'SMA' in df.columns and 'CLOSE' in df.columns:
        if df['CLOSE'].iloc[-1] < df['SMA'].iloc[-1]:
            sentiment['SMA'] = 'Bearish'
        else:
            sentiment['SMA'] = 'Bullish'
    
    if 'EMA' in df.columns and 'CLOSE' in df.columns:
        if df['CLOSE'].iloc[-1] < df['EMA'].iloc[-1]:
            sentiment['EMA'] = 'Bearish'
        else:
            sentiment['EMA'] = 'Bullish'
    
    if 'Upper Band' in df.columns and 'Lower Band' in df.columns and 'CLOSE' in df.columns:
        upper_band = df['Upper Band'].iloc[-1]
        lower_band = df['Lower Band'].iloc[-1]
        current_price = df['CLOSE'].iloc[-1]
        
        if current_price < lower_band:
            sentiment['Bollinger Bands'] = 'Oversold'
        elif current_price > upper_band:
            sentiment['Bollinger Bands'] = 'Overbought'
        else:
            sentiment['Bollinger Bands'] = 'Neutral'
    
    if 'Bid-Ask Spread' in df.columns:
        sentiment['Bid-Ask Spread'] = 'Positive'
    
    if 'Turnover Ratio' in df.columns:
        turnover_ratio = df['Turnover Ratio'].iloc[-1]
        if turnover_ratio > 1:
            sentiment['Turnover Ratio'] = 'Positive'
        elif turnover_ratio < 1:
            sentiment['Turnover Ratio'] = 'Negative'
        else:
            sentiment['Turnover Ratio'] = 'Neutral'
    
    return sentiment

def final_decision(sentiment):
    # Decision logic based on combined sentiments
    if (sentiment.get('RSI') == 'Undervalued' and 
        sentiment.get('SMA') != 'Bearish' and 
        sentiment.get('EMA') != 'Bearish'):
        return "Buy"
    elif (sentiment.get('VWAP') == 'Bearish' and 
          sentiment.get('SMA') == 'Bearish' and 
          sentiment.get('EMA') == 'Bearish' and 
          sentiment.get('RSI') != 'Undervalued'):
        return "Sell"
    elif (sentiment.get('Bollinger Bands') == 'Oversold' and 
          sentiment.get('RSI') == 'Undervalued'):
        return "Buy"
    elif (sentiment.get('Bollinger Bands') == 'Overbought' and 
          sentiment.get('RSI') != 'Undervalued'):
        return "Sell"
    elif sentiment.get('Bid-Ask Spread') == 'Positive' and sentiment.get('Turnover Ratio') == 'Positive':
        return "Buy"
    elif sentiment.get('Bid-Ask Spread') == 'Negative' and sentiment.get('Turnover Ratio') == 'Negative':
        return "Sell"
    else:
        return "Hold"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/intraday')
def intraday():
    symbols = [
    "ADANIPORTS"]
    #start_date = dt.date(2023, 6, 1)
    #end_date = dt.date(2023, 6, 28)

    today = dt.date.today()
    end_date = today - dt.timedelta(days=1)  # Yesterday's date
    start_date = end_date - dt.timedelta(days=60)  # 60 days before yesterday
    
    data = fetch_intraday_data(symbols, start_date, end_date)
    
    results = []
    for symbol, df in data.items():
        df = calculate_vwap(df)
        df = calculate_rsi(df)
        df = calculate_sma(df)
        df = calculate_ema(df)
        df = calculate_bollinger_bands(df)
        df = calculate_bid_ask_spread(df)
        df = calculate_turnover_ratio(df)
        
        # Extract current price and company name
        current_price = df['CLOSE'].iloc[-1] if 'CLOSE' in df.columns else None
        company_name = df['SYMBOL'].iloc[0] if 'SYMBOL' in df.columns else None
        
        # Prepare indicators data
        indicators_data = {}
        if 'VWAP' in df.columns:
            indicators_data['VWAP'] = round(df['VWAP'].iloc[-1], 2)
        if 'RSI' in df.columns:
            indicators_data['RSI'] = round(df['RSI'].iloc[-1],2)
        if 'SMA' in df.columns:
            indicators_data['SMA'] = round(df['SMA'].iloc[-1],2)
        if 'EMA' in df.columns:
            indicators_data['EMA'] = round(df['EMA'].iloc[-1],2)
        if 'Upper Band' in df.columns and 'Lower Band' in df.columns:
            indicators_data['Bollinger Bands'] = f"{round(df['Upper Band'].iloc[-1],2)}, {round(df['Lower Band'].iloc[-1],2)}"
        if 'Bid-Ask Spread' in df.columns:
            indicators_data['Bid-Ask Spread'] = round(df['Bid-Ask Spread'].iloc[-1],2)
        if 'Turnover Ratio' in df.columns:
            indicators_data['Turnover Ratio'] = round(df['Turnover Ratio'].iloc[-1],2)
        
        # Determine sentiment for each indicator
        sentiment = determine_sentiment(df)
        
        # Determine final decision based on sentiment
        decision = final_decision(sentiment)
        
        # Prepare data for each symbol
        symbol_data = {
            'symbol': symbol,
            'company_name': current_price,
            #'LTP': current_price,
            'indicators_data': indicators_data,
            'sentiment': sentiment,
            'decision': decision
        }
        
        results.append(symbol_data)
    
    return render_template('intraday_analysis.html', results=results)
if __name__ == "__main__":
    app.run(debug=True)
