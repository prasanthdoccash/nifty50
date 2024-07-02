from flask import Flask, render_template, request, redirect, url_for
import datetime as dt
from jugaad_data.nse import NSELive
from jugaad_data.nse import stock_df
import pandas as pd
from live import final_decision
import os
import shutil
cache_dir = '/opt/render/.cache/nsehistory-stock'
# Check if the directory exists
if os.path.exists(cache_dir):
    # Remove the directory and all its contents
    shutil.rmtree(cache_dir)
    print(f"Deleted existing directory '{cache_dir}'.")

# Now create the directory
os.makedirs(cache_dir)

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
# Function to calculate EMA
def calculate_ema9(df, span=9):
    if 'CLOSE' in df.columns:
        
        #ema_label = f'EMA_{span}'
        df['EMA_9'] = df['CLOSE'].ewm(span=span, min_periods=span).mean()
        
    return df
def calculate_ema50(df, span=50):
    if 'CLOSE' in df.columns:
        
        #ema_label = f'EMA_{span}'
        df['EMA_50'] = df['CLOSE'].ewm(span=span, min_periods=span).mean()
        
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

# Function to calculate MACD and identify crossovers
def calculate_macd(df, n_fast=12, n_slow=26, n_signal=9):
    # Calculate EMAs
    df['EMA_fast'] = df['CLOSE'].ewm(span=n_fast, min_periods=1, adjust=False).mean()
    df['EMA_slow'] = df['CLOSE'].ewm(span=n_slow, min_periods=1, adjust=False).mean()
    
    # Calculate MACD line
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    
    # Calculate Signal line
    df['Signal'] = df['MACD'].ewm(span=n_signal, min_periods=1, adjust=False).mean()
    
    # Identify crossovers
    
    df['MACD_Crossover'] = 'Hold'
    df.loc[(df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1)), 'MACD_Crossover'] = 'Bullish'
    df.loc[(df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1)), 'MACD_Crossover'] = 'Bearish'
    
    return df

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
# Dummy symbols data for demonstration
predefined_symbols = ["ADANIENT","ADANIPORTS", "APOLLOHOSP","ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE", "BHARTIARTL", "BPCL", "CIPLA", "COALINDIA", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH","HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "INDUSINDBK", "INFY", "IOC", "ITC", "JSWSTEEL", "KOTAKBANK", "LTIM","LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHREECEM", "SUNPHARMA", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO"]
predefined_symbols1 = [
    "ADANIENT","ADANIPORTS", "APOLLOHOSP","ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE", "BHARTIARTL", "BPCL", "CIPLA", "COALINDIA", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH","HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "INDUSINDBK", "INFY", "IOC", "ITC", "JSWSTEEL", "KOTAKBANK", "LTIM","LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHREECEM", "SUNPHARMA", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO"
]

nse = NSELive()

def fetch_live_data(symbols):
    data = {}
    for symbol in symbols:
        try:
            q = nse.stock_quote(symbol)
            data[symbol] = q['priceInfo']
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            data[symbol] = {'lastPrice': 'N/A'}  # Handle error case
    return data


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/last_prices')
def last_prices():
    
    symbols = request.args.get('symbols')
    symbols_list = symbols.split(',') if symbols else predefined_symbols1
    #symbols_list = [res_ltp1]
    try:
        data = fetch_live_data(symbols_list)
    except Exception as e:
        print(f"Error fetching live data: {str(e)}")
        data = {}
    
    last_refreshed = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #global res_ltp
    #res_ltp1 = []
    results = []
    for symbol, price_info in data.items():
        # Prepare data for each symbol
        symbol_data = {
            'Symbol': symbol,
            'Last Price': price_info.get('lastPrice', 'N/A'),
            #'Decision': final_decision(price_info)  # Using final_decision function
        }
         
        results.append(symbol_data)
        #res_ltp1 = symbol_data['Last Price']
        
        #return res_ltp1
    return render_template('last_prices.html', data=results, last_refreshed=last_refreshed)


def last_prices1(res_ltp1):
    
    #symbols = request.args.get('symbols')
    #symbols_list = symbols.split(',') if symbols else predefined_symbols
    symbols_list = [res_ltp1]
    try:
        data = fetch_live_data(symbols_list)
    except Exception as e:
        print(f"Error fetching live data: {str(e)}")
        data = {}
    
    #last_refreshed = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    #res_ltp1 = []
    results = []
    for symbol, price_info in data.items():
        # Prepare data for each symbol
        symbol_data = {
            'Symbol': symbol,
            'Last Price': price_info.get('lastPrice', 'N/A'),
            #'Decision': final_decision(price_info)  # Using final_decision function
        }
         
        results.append(symbol_data)
        res_ltp1 = symbol_data['Last Price']
        
        return res_ltp1
    #return render_template('last_prices.html', data=results, last_refreshed=last_refreshed)

@app.route('/input_symbol', methods=['GET', 'POST'])
def input_symbol():
    if request.method == 'POST':
        symbols = request.form.get('symbols')
        if symbols:
            return redirect(url_for('delivery', symbols=symbols))
    
    return render_template('input_symbol.html')

def delivery_longdate(symbol):
    symbols = [symbol]
    
    today = dt.date.today()
    end_date = today #- dt.timedelta(days=1)  # Yesterday's date
    start_date = end_date - dt.timedelta(days=80)  # 50 days before yesterday
    
    data = fetch_intraday_data(symbols, start_date, end_date)
    results = []
    for symbol, df in data.items():
        
        df = calculate_ema9(df)
        df = calculate_ema50(df)

        ema_9 = df['EMA_9'].iloc[-1] if 'EMA_9' in df.columns else 0
        ema_50 = df['EMA_50'].iloc[-1] if 'EMA_50' in df.columns else 0
        current_price = last_prices1(symbol)
        
        # Determine pullback buy condition
        if current_price <= ema_9 and ema_9 > ema_50:
            
            df['pullback_buy_action'] = 'Buy'
        else:
            df['pullback_buy_action'] = 'Hold'
        
        pullback = df['pullback_buy_action'].iloc[-1]
        # Calculate MACD and identify crossovers
        calculate_macd(df)
        macd_crossover = df['MACD_Crossover'].iloc[-1]  # Get the last crossover signal
        
    #return df['pullback_buy_action'],df['MACD_Crossover']
    return pullback, macd_crossover
@app.route('/delivery')
def delivery():
    symbols1 = request.args.get('symbols')
    symbols = symbols1.split(',') if symbols1 else predefined_symbols
    
    today = dt.date.today()
    end_date = today #- dt.timedelta(days=1)  # Yesterday's date
    start_date = end_date - dt.timedelta(days=50)  # 50 days before yesterday
    last_refreshed = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #start_date = dt.date(2023, 6, 1)
    #end_date = dt.date(2023, 6, 28)
    #last_prices()
    data = fetch_intraday_data(symbols, start_date, end_date)
    
    results = []
    for symbol, df in data.items():
        try:
            df = calculate_vwap(df)
            df = calculate_rsi(df)
            df = calculate_sma(df)
            df = calculate_ema(df)
            #df = calculate_ema9(df)
            #df = calculate_ema50(df)
            df = calculate_bollinger_bands(df)
            df = calculate_bid_ask_spread(df)
            df = calculate_turnover_ratio(df)
            
            current_price = last_prices1(symbol)
            
            #current_price = df['CH_LAST_TRADED_PRICE'].iloc[-1] if 'CH_LAST_TRADED_PRICE' in df.columns else 'N/A'
            #company_name = df['CH_SYMBOL'].iloc[0] if 'CH_SYMBOL' in df.columns else 'N/A'
            
            df['pullback_buy_action'],df['MACD_Crossover'] = delivery_longdate(symbol)
            
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
            if 'MACD_Crossover' in df.columns:
                indicators_data['MACD_Crossover'] = df['MACD_Crossover'].iloc[-1]
            if 'Upper Band' in df.columns and 'Lower Band' in df.columns:
                indicators_data['Bollinger Bands'] = f"{round(df['Upper Band'].iloc[-1],2)}, {round(df['Lower Band'].iloc[-1],2)}"
            if 'Bid-Ask Spread' in df.columns:
                indicators_data['Bid-Ask Spread'] = round(df['Bid-Ask Spread'].iloc[-1],2)
            if 'Turnover Ratio' in df.columns:
                indicators_data['Turnover Ratio'] = round(df['Turnover Ratio'].iloc[-1],2)
            if 'pullback_buy_action' in df.columns:
                indicators_data['pullback_buy_action'] = df['pullback_buy_action'].iloc[-1]
            # Determine sentiment for each indicator
            sentiment = determine_sentiment(df)
            
            # Determine final decision based on sentiment
            decision = final_decision(sentiment)
            
            # Prepare data for each symbol
            symbol_data = {
                'symbol': symbol,
                #'company_name': company_name,
                'LTP': current_price,
                'indicators_data': indicators_data,
                'sentiment': sentiment,
                'decision': decision
            }
            
            results.append(symbol_data)  
        
        except KeyError as e:
            print(f"KeyError: {str(e)}. Skipping symbol {symbol}.")
            continue
    return render_template('delivery_analysis.html', results=results, last_refreshed=last_refreshed)

# Global variable to store today's high at a specific time
todays_high_at_specific_time = 0
# Function to determine market sentiment (Bullish or Bearish)
def market_sentiment(open_price, previous_close):
    if open_price > previous_close:
        return 'Bullish'
    else:
        return 'Bearish'

# Function to identify intraday high and low and determine intra trade action
def intraday_high_low(intra_day_data, last_price,open_price):
    
    intraday_high = intra_day_data.get('max', 0)
    
    intraday_low = intra_day_data.get('min', 0)
    
    # Define the threshold for "near" as a percentage of the price range
    threshold = 0.005  # 1% for example

    # Calculate thresholds for high and low
    high_threshold = intraday_high - (intraday_high * threshold)
    
    low_threshold = intraday_low + (intraday_low * threshold)
    

    if last_price >= high_threshold:
        
        
        intra_trade_action = 'Buy'
    
    elif last_price < low_threshold:
        intra_trade_action = 'Sell'
    else: intra_trade_action = 'Hold'
    
    global todays_high_at_specific_time
    if todays_high_at_specific_time == 0 or last_price > todays_high_at_specific_time:
        todays_high_at_specific_time = intraday_high
        
     # Check for Break Buy condition
    break_buy_action = 'Buy' if last_price > todays_high_at_specific_time else 'Hold'
    # Calculate stop-loss level (below intraday low)
    stop_loss = open_price * 0.99  # Example: 1% below intraday low
    # Calculate target level 
    target = open_price * 1.01  # Example: 1% below intraday low
    return intraday_high, intraday_low, intra_trade_action, break_buy_action, stop_loss,target

# Route to fetch stock data and render HTML template
@app.route('/intraday')
def stock_analysis():
    global todays_high_at_specific_time  # Access the global variable
    #symbols = predefined_symbols  # Example symbol for demonstration
    symbols1 = request.args.get('symbols')
    symbols = symbols1.split(',') if symbols1 else predefined_symbols
    
    stock_analysis_data = []
    
    for symbol in symbols:
        
        try:
            q = nse.stock_quote(symbol)
            stock_data = q['priceInfo']

            # Extract open price and previous close from priceInfo block
            open_price = float(stock_data.get('open', 0))
            previous_close = float(stock_data.get('previousClose', 0))
            last_price = float(stock_data.get('lastPrice', 0))

            # Calculate market sentiment
            sentiment = market_sentiment(open_price, previous_close)

            # Get intraday high, low, and intra trade action
            intra_day_data = stock_data.get('intraDayHighLow', {})
            
            intraday_high, intraday_low, intra_trade_action, break_buy_action, stop_loss,target = intraday_high_low(intra_day_data, last_price,open_price)
            
            price_data = pd.DataFrame({
                'CLOSE': [open_price, last_price, intraday_high, intraday_low]  # Example data points
            })
            
            #pullback_buy_action,macd_crossover = delivery_longdate(symbol)
            price_data['pullback_buy_action'],price_data['MACD_Crossover'] = delivery_longdate(symbol)
            pullback_buy_action =price_data['pullback_buy_action'].iloc[-1]
            macd_crossover = price_data['MACD_Crossover'].iloc[-1]

            symbol_data = {'symbol': symbol,
                'last_price': last_price,
                'change': round(float(stock_data.get('change', 0)), 2),
                'p_change': round(float(stock_data.get('pChange', 0)), 2),
                'previous_close': previous_close,
                'open_price': open_price,
                'intraday_high': intraday_high,
                'intraday_low': intraday_low,
                'market_sentiment': sentiment,
                'intra_trade_action': intra_trade_action,
                'break_buy_action': break_buy_action,
                'stop_loss': round(stop_loss, 2),
                'target': round(target,2),
                'macd_crossover': macd_crossover,
                'pullback_buy_action': pullback_buy_action
                           }
            stock_analysis_data.append(symbol_data)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            stock_data = {'lastPrice': 'N/A'}
            sentiment = 'N/A'
            intraday_high = 'N/A'
            intraday_low = 'N/A'
            intra_trade_action = 'N/A'
            break_buy_action = 'N/A'
            stop_loss = 'N/A'
            target = 'N/A'
            macd_crossover = 'N/A'
            pullback_buy_action = 'N/A'
        
    return render_template('intraday_analysis.html', stock_analysis_data=stock_analysis_data)
    


if __name__ == "__main__":
    #app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=80)
