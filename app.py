from flask import Flask, render_template, request, redirect, url_for
import datetime as dt
from jugaad_data.nse import NSELive,stock_df
import pandas as pd
import ta
#from live import final_decision
import combine
from datetime import date, timedelta
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
import json

'''import os
import shutil
cache_dir = '/opt/render/.cache/nsehistory-stock'
# Check if the directory exists
if os.path.exists(cache_dir):
    # Remove the directory and all its contents
    shutil.rmtree(cache_dir)
    print(f"Deleted existing directory '{cache_dir}'.")

# Now create the directory
os.makedirs(cache_dir)'''

app = Flask(__name__)

# Function to fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d", interval="1m")  # Changed to 5 days to get enough data for indicators
    return hist

def calculate_indicators(df):
    df['rsi'] = RSIIndicator(df['Close'], window=14).rsi()
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df['ema'] = EMAIndicator(df['Close'], window=14).ema_indicator()
    df['sma'] = SMAIndicator(df['Close'], window=14).sma_indicator()
    bollinger = BollingerBands(df['Close'], window=20, window_dev=2)
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()
    df['adx'] = ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['stochastic'] = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14).stoch()
    #df['vwap'] = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'], window=14).vwap()
    df['vwap'] = 0
    return df

# Function to make trading decisions
def final_decision(df,vix):
    last_row = df.iloc[-1]
    indicators = {
        'rsi': last_row['rsi'],
        'macd': last_row['macd'],
        'ema': last_row['ema'],
        'sma': last_row['sma'],
        'bollinger_high': last_row['bollinger_high'],
        'bollinger_low': last_row['bollinger_low'],
        'adx': last_row['adx'],
        'stochastic': last_row['stochastic'],
        'vwap': last_row['vwap'],
        
    }

    buy_signals = 0
    sell_signals = 0
    hold_signal = 0
    
    buy = []
    sell = []
    hold = []
    
    

    # RSI
    if indicators['rsi'] <= 30:
        buy.append('RSI')
        buy_signals += 1
    elif indicators['rsi'] >= 70:
        sell_signals += 1
        sell.append('RSI')
    elif indicators['rsi'] > 30 and indicators['rsi'] < 70:
        hold_signal +=1
        hold.append('RSI')

    # MACD
    if indicators['macd'] > 0:
        buy_signals += 1
        buy.append('MACD')
    elif indicators['macd'] < 0:
        sell_signals += 1
        sell.append('MACD')

    # EMA and SMA
    if indicators['ema'] > indicators['sma']:
        buy_signals += 1
        buy.append('EMA')
    else:
        sell_signals += 1
        sell.append('EMA')
    
    # Bollinger Bands
    
    if last_row['Close'] < indicators['bollinger_low']:
        buy_signals += 1
        buy.append('BOLLINGER')
    elif last_row['Close'] > indicators['bollinger_high']:
        sell_signals += 1
        sell.append('BOLLINGER')
    elif last_row['Close'] > indicators['bollinger_low'] and last_row['Close'] < indicators['bollinger_high']:
        hold_signal +=1
        hold.append('BOLLINGER')

    # ADX
    
    if indicators['adx'] > 25:
        if last_row['Close'] > indicators['ema']:
            buy_signals += 1
            buy.append('ADX')
        else:
            sell_signals += 1
            sell.append('ADX')
    else:
        hold_signal +=1
        hold.append('ADX')

    # Stochastic Oscillator
    if indicators['stochastic'] < 20:
        buy_signals += 1
        buy.append('STOCHASTIC')
    elif indicators['stochastic'] > 80:
        sell_signals += 1
        sell.append('STOCHASTIC')
    elif indicators['stochastic'] >20 and indicators['stochastic'] <80:
        hold_signal +=1
        hold.append('STOCHASTIC')
    

    # VWAP
    if last_row['Close'] > indicators['vwap']:
        buy_signals += 1
        buy.append('VWAP')
    else:
        sell_signals += 1
        sell.append('VWAP')
    
    if buy_signals > sell_signals and buy_signals > (sell_signals +hold_signal) and sell_signals ==0:

        return 'Buy',buy_signals,sell_signals,hold_signal, buy,sell,hold
    elif buy_signals > sell_signals and buy_signals > (sell_signals +hold_signal):

        return 'Super Buy',buy_signals,sell_signals,hold_signal, buy,sell,hold
    elif sell_signals > buy_signals:
        return 'Sell',buy_signals,sell_signals,hold_signal, buy,sell,hold
    else:
        return 'Hold',buy_signals,sell_signals,hold_signal, buy,sell,hold
   
    #if result.decision == 'Buy' and (result.buy_signals >0 and result.sell_signals ==0)

###########################################################################
nse = NSELive()
def fetch_delivery_data(symbols):
    
    all_data = {}
    for symbol in symbols:
        
        stock = yf.Ticker(symbol)
        
        df = stock.history(period="30d", interval="1h") # 30d 1h identifies Stocks to Buys
        #print(df)
        
        if not df.empty:
            all_data[symbol] = df
    
    return all_data

def fetch_price_data(symbol):
    dq = {}
    
    q = nse.stock_quote(symbol)
    
    dq[symbol] =  q['priceInfo']
    
    lastPrice =float(dq[symbol].get('lastPrice', 0))
    
        
    
    return lastPrice

def fetch_intraday_data(symbols):
    
    all_data = {}
    for symbol in symbols:
        #df = stock_df(symbol=symbol, from_date=start_date, to_date=end_date, series="EQ")
        stock = yf.Ticker(symbol)
        df = stock.history(period="5d", interval="1m")  # Changed to 5 days to get enough data for indicators
        if not df.empty:
            all_data[symbol] = df
    
    return all_data

def calculate_roc(df, window=12):
    df['ROC'] = ta.momentum.roc(df['CLOSE'], window=window)
    return df

def calculate_volume_trend(df, window=20):
    if 'VOLUME' in df.columns:
        df['Volume_Trend'] = df['VOLUME'].rolling(window=window).mean()
    return df

def calculate_vix(symb):
    #df['VIX'] = ta.volatility.vix(df['HIGH'], df['LOW'], df['CLOSE'])
    stock = yf.Ticker(symb)
    df = stock.history(period="1d", interval="1m") # Changed to 5 days to get enough data for indicators
    
    df['VIX'] = df['High'].iloc[-1]
    vix = df['High'].iloc[-1]
    vix = round(vix,2)
    df['VIX_Level'] = df['VIX'].apply(lambda x: 'Low' if x < 15 else 'High')
    if vix < 15:
        vix_senti = 'Neutral'
        
    elif vix >=15:
        vix_senti = 'Volatile'
    return df,vix,vix_senti

def calculate_call_put_ratio(df):
    _, _,call_put_ratio,_ = combine.predict_nifty()
    df['CallPutRatio'] = call_put_ratio
    df['CallPutSentiment'] = df['CallPutRatio'].apply(lambda x: 'Bullish' if x > 1 else 'Bearish')
    
    return df

def determine_sentiment(df):
    sentiment = {}
    
    if 'VWAP' in df.columns and 'CLOSE' in df.columns:
        vwap = df['VWAP'].iloc[-1]
        current_price = df['CLOSE'].iloc[-1]
        if current_price > vwap:
            sentiment['VWAP'] = 'Bullish'
        elif current_price < vwap:
            sentiment['VWAP'] = 'Bearish'
        else:
            sentiment['VWAP'] = 'Neutral'
    
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

    if 'Williams_R' in df.columns:
        williams_r = df['Williams_R'].iloc[-1]
        if williams_r < -80:
            sentiment['Williams_R'] = 'Overbought'
        elif williams_r > -20:
            sentiment['Williams_R'] = 'Oversold'
        else:
            sentiment['Williams_R'] = 'Neutral'
    if '%K' in df.columns:
        stochastic_k = df['%K'].iloc[-1]
        if stochastic_k < 20:
            sentiment['Stochastic'] = 'Buy Signal'
        elif stochastic_k > 80:
            sentiment['Stochastic'] = 'Sell Signal'
        else:
            sentiment['Stochastic'] = 'Neutral'
    if 'ROC' in df.columns:
        roc = df['ROC'].iloc[-1]
        if roc > 0:
            sentiment['ROC'] = 'Increasing'
        elif roc < 0:
            sentiment['ROC'] = 'Decreasing'
        else:
            sentiment['ROC'] = 'Neutral'

    if 'Volume_Trend' in df.columns:
        volume_trend = df['Volume_Trend'].iloc[-1]
        if volume_trend > df['Volume_Trend'].shift(1).iloc[-1]:
            sentiment['Volume_Trend'] = 'Increasing'
        elif volume_trend < df['Volume_Trend'].shift(1).iloc[-1]:
            sentiment['Volume_Trend'] = 'Decreasing'
        else:
            sentiment['Volume_Trend'] = 'Neutral'
    if 'MACD_Crossover' in df.columns:
        macd_crossover = df['MACD_Crossover'].iloc[-1]
        if macd_crossover == 'Bullish':
            sentiment['MACD'] = 'Buy Signal'
        elif macd_crossover == 'Bearish':
            sentiment['MACD'] = 'Sell Signal'
        else:
            sentiment['MACD'] = 'Neutral'
    
    return sentiment



# Dummy symbols data for demonstration
predefined_symbols123 = [ "ADANIENT","ADANIPORTS", "APOLLOHOSP","ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE","BRITANNIA", "BHARTIARTL", "BPCL", "CIPLA", "COALINDIA", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH","HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "INDUSINDBK", "INFY", "IOC", "ITC", "JSWSTEEL", "KOTAKBANK", "LTIM","LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHREECEM", "SUNPHARMA", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO"
   ]

predefined_symbols = [f"{symbol}.NS" for symbol in predefined_symbols123]
predefined_symbols1 = [
    "ADANIENT","ADANIPORTS", "APOLLOHOSP","ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE","BRITANNIA", "BHARTIARTL", "BPCL", "CIPLA", "COALINDIA", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH","HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "INDUSINDBK", "INFY", "IOC", "ITC", "JSWSTEEL", "KOTAKBANK", "LTIM","LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHREECEM", "SUNPHARMA", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO"
]
'''predefined_symbols = [
"ABB", "ACC", "AUBANK", "AARTIIND", "ABBOTINDIA", "ADANIENT", "ADANIPORTS", 
                       "ABCAPITAL", "ABFRL", "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE", 
                       "ASHOKLEY", "ASIANPAINT", "ASTRAL", "ATUL", "AUROPHARMA", "AXISBANK", "BSOFT", 
                       "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BALKRISIND", "BALRAMCHIN", 
                       "BANDHANBNK", "BANKBARODA", "BATAINDIA", "BERGEPAINT", "BEL", "BHARATFORG", 
                       "BHEL", "BPCL", "BHARTIARTL", "BIOCON", "BOSCHLTD", "BRITANNIA", "CANFINHOME", 
                       "CANBK", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "CUB", "COALINDIA", "COFORGE", 
                       "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON", "CUMMINSIND", "DLF", "DABUR", 
                       "DALBHARAT", "DEEPAKNTR", "DIVISLAB", "DIXON", "LALPATHLAB", "DRREDDY", 
                       "EICHERMOT", "ESCORTS", "EXIDEIND", "GAIL", "GMRINFRA", "GLENMARK", "GODREJCP", 
                       "GODREJPROP", "GRANULES", "GRASIM", "GUJGASLTD", "GNFC", "HCLTECH", "HDFCAMC", 
                       "HDFCBANK", "HDFCLIFE", "HAVELLS", "HEROMOTOCO", "HINDALCO", "HAL", "HINDCOPPER", 
                       "HINDPETRO", "HINDUNILVR", "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDFCFIRSTB", 
                       "IDFC", "IPCALAB", "ITC", "INDIAMART", "IEX", "IOC", "IRCTC", "IGL", "INDUSTOWER", 
                       "INDUSINDBK", "NAUKRI", "INFY", "INDIGO", "JKCEMENT", "JSWSTEEL", "JINDALSTEL", 
                       "JUBLFOOD", "KOTAKBANK", "LTF", "LTTS", "LICHSGFIN", "LTIM", "LT", "LAURUSLABS", 
                       "LUPIN", "MRF", "MGL", "M&MFIN", "M&M", "MANAPPURAM", "MARICO", "MARUTI", "MFSL", 
                       "METROPOLIS", "MPHASIS", "MCX", "MUTHOOTFIN", "NMDC", "NTPC", "NATIONALUM", 
                       "NAVINFLUOR", "NESTLEIND", "OBEROIRLTY", "ONGC", "OFSS", "PIIND", "PVRINOX", 
                       "PAGEIND", "PERSISTENT", "PETRONET", "PIDILITIND", "PEL", "POLYCAB", "PFC", 
                       "POWERGRID", "PNB", "RBLBANK", "RECLTD", "RELIANCE", "SBICARD", "SBILIFE", 
                       "SHREECEM", "SRF", "MOTHERSON", "SHRIRAMFIN", "SIEMENS", "SBIN", "SAIL", 
                       "SUNPHARMA", "SUNTV", "SYNGENE", "TATACONSUM", "TVSMOTOR", "TATACHEM", "TATACOMM", 
                       "TCS", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TECHM", "FEDERALBNK", "INDIACEM", 
                       "INDHOTEL", "RAMCOCEM", "TITAN", "TORNTPHARM", "TRENT", "UPL", "ULTRACEMCO", "UBL",
                         "UNITDSPR", "VEDL", "IDEA", "VOLTAS", "WIPRO", "ZYDUSLIFE"
                         ]'''

nse = NSELive()

def fetch_live_data(symbols):
    data = {}
    for symbol in symbols:
        try:
            nse = NSELive()
            q = nse.stock_quote(symbol)
            data[symbol] = q['priceInfo']
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            data[symbol] = {'lastPrice': 'N/A'}  # Handle error case
    return data


@app.route('/')
def home():
    _,vix,vix_senti = calculate_vix('^INDIAVIX')
    return render_template('index.html',vix=vix,vix_senti=vix_senti)
    #return render_template('index.html',senti=senti,vix=vix,call_put_ratio=ncall_put_ratio,volume=nvolume)


@app.route('/predict')
def final():
    if request.method == 'POST':
        symbols = request.form['symbols'].split()
        if not symbols:
            symbols = predefined_symbols
    else:
        symbols = predefined_symbols

    intraday_decisions = {}
    for stock in symbols:
        df = fetch_live_data(stock)
        df = calculate_indicators(df)
        decision = final_decision(df)
        intraday_decisions[stock] = decision

    delivery_decisions = {}
    for stock in symbols:
        df = fetch_stock_data(stock)
        df = calculate_indicators(df)
        decision = final_decision(df)
        delivery_decisions[stock] = decision

    combined_decisions = [(stock, intraday_decisions[stock], delivery_decisions[stock]) for stock in symbols]

    return render_template('combine.html', stocks=combined_decisions)
    


@app.route('/input_symbol', methods=['GET', 'POST'])
def input_symbol():
    if request.method == 'POST':
        symbols = request.form.get('symbols')
        if symbols:
            
            return redirect(url_for('delivery', symbols=symbols))
    
    return render_template('input_symbol.html')


#@app.route('/delivery')
def delivery(symbols_get):

    
    if symbols_get != "":
        print("1",symbols_get)
        # Get symbols from predefined or request
        symbols_list = symbols_get
        symbols = symbols_list if symbols_list else predefined_symbols
    else:
        symbols1 = request.args.get('symbols')
        print("2",symbols1)
        symbols = symbols1.split(',') if symbols1 else predefined_symbols
    print("2pritn",symbols)
        #today = dt.date.today()
    #end_date = today #- dt.timedelta(days=1)  # Yesterday's date
    #start_date = end_date - dt.timedelta(days=50)  # 50 days before yesterday
    last_refreshed = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #start_date = dt.date(2023, 6, 1)
    #end_date = dt.date(2023, 6, 28)
    #last_prices()
    
    #symbols =  "{}.NS".format(symbolsk)
    #calculate_vix()
    data = fetch_delivery_data(symbols)
    _,vix,vix_senti = calculate_vix('^INDIAVIX')
    results = []
    
    for symbol, df in data.items():
        try:
            # Calculate indicators
            df = calculate_indicators(df)
            
            #current_price = df['CH_LAST_TRADED_PRICE'].iloc[-1] if 'CH_LAST_TRADED_PRICE' in df.columns else 'N/A'
            #company_name = df['CH_SYMBOL'].iloc[0] if 'CH_SYMBOL' in df.columns else 'N/A'
            
            #df['pullback_buy_action'],df['MACD_Crossover'] = delivery_longdate(symbol)
            '''price_targets = combine.get_analyst_recommendations(symbol)
            
                    
            if last_price <= price_targets:
                target = price_targets
            else:
                target = target'''
                     
            
            # Determine final decision based on sentiment
            decision,buy_signals,sell_signals,hold_signal, buy,sell,hold = final_decision(df,vix)
            symbols_NS = symbol[:-3]
            last_Price = fetch_price_data(symbols_NS)
            
            # Prepare data for each symbol
            symbol_data = {
                'symbol': symbol,
                #'company_name': company_name,
                'LTP': last_Price,
                #'indicators_data': indicators,
                #'sentiment': sentiment,
                'decision': decision,
                #'price_target': price_targets
                'buy_signals':buy_signals,
                'sell_signals':sell_signals,
                'hold_signal':hold_signal,
                'buy':buy,
                'sell':sell,
                'hold': hold
                
            }
            
            results.append(symbol_data)  
    
        except KeyError as e:
            print(f"KeyError: {str(e)}. Skipping symbol {symbol}.")
            continue
    
    return results,last_refreshed,vix,vix_senti
    
    
@app.route('/delivery')
def delivery1():
    symb = ''
    results,last_refreshed,vix,vix_senti = delivery(symb)
    
    
    #print(vix)
    return render_template('delivery_analysis.html', results=results,last_refreshed=last_refreshed,vix=vix,vix_senti=vix_senti)
    #return render_template('delivery_analysis.html', results=results, last_refreshed=last_refreshed,vix=vix, call_put_ratio=call_put_ratio,volume=volume,senti=senti)

# Global variable to store today's high at a specific time
todays_high_at_specific_time = 0

@app.route('/add_to_watchlist', methods=['POST'])
def add_to_watchlist():
    data = request.get_json()
    symbols = data.get('symbols', [])
    
    # Save the symbols to a watchlist (e.g., a file or database)
    with open('watchlist.json', 'w') as f:
        json.dump(symbols, f)
    
    return jsonify({'message': 'Symbols added to watchlist successfully!'})

@app.route('/get_watchlist', methods=['GET'])
def get_watchlist():
    try:
        with open('watchlist.json', 'r') as f:
            symbols = json.load(f)
    except FileNotFoundError:
        symbols = []

    return jsonify({'symbols': symbols})

# Route to show delivery analysis for watchlist symbols
@app.route('/show_watchlist')
def show_watchlist():
    symbols = get_watchlist_symbols()
    print("getwatchlist",symbols)
    results,last_refreshed,vix,vix_senti = delivery(symbols)
    return render_template('delivery_analysis.html', results=results,last_refreshed=last_refreshed,vix=vix, vix_senti=vix_senti)

# Function to get symbols from watchlist
def get_watchlist_symbols():
    try:
        with open('watchlist.json', 'r') as f:
            symbols = json.load(f)
    except FileNotFoundError:
        symbols = []
    
    return symbols

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
    #break_buy_action = 'Buy' if last_price > todays_high_at_specific_time else 'Hold'
    if last_price >=(intraday_low *1.01) or last_price > todays_high_at_specific_time: #opportunity to buy
        break_buy_action= 'Buy'
    elif last_price <=(intraday_high *0.98): #opportunity to buy
        break_buy_action= 'sell' 
    else:
        break_buy_action= 'Hold'
    # Calculate stop-loss level (below intraday low)
    stop_loss = open_price * 0.99  # Example: 1% below intraday low
    # Calculate target level 
    target = open_price * 1.01  # Example: 1% below intraday low
    return intraday_high, intraday_low, intra_trade_action, break_buy_action, stop_loss,target


# Route to fetch stock data and render HTML template
#@app.route('/intraday')
def stock_analysis():
    global todays_high_at_specific_time  # Access the global variable
    #symbols = predefined_symbols  # Example symbol for demonstration
    symbols1 = request.args.get('symbols')
    symbols = symbols1.split(',') if symbols1 else predefined_symbols
    
    stock_analysis_data = []
    
    for symbol in symbols:
        
        try:
            #q = nse.stock_quote(symbol)
            stock_data = fetch_intraday_data(symbol)

            # Extract open price and previous close from priceInfo block
            open_price = float(stock_data.get('open', 0))
            previous_close = float(stock_data.get('previousClose', 0))
            last_price = float(stock_data.get('lastPrice', 0))
            
            # Calculate market sentiment
            #sentiment = market_sentiment(open_price, previous_close)

            # Get intraday high, low, and intra trade action
            intra_day_data = stock_data.get('intraDayHighLow', {})
            
            intraday_high, intraday_low, intra_trade_action, break_buy_action, stop_loss,target = intraday_high_low(intra_day_data, last_price,open_price)
            
            price_data = pd.DataFrame({
                'CLOSE': [open_price, last_price, intraday_high, intraday_low]  # Example data points
            })
            price_targets = combine.get_analyst_recommendations(symbol)
            
                    
            if last_price <= price_targets:
                target = price_targets
            else:
                target = target
            
            #pullback_buy_action,macd_crossover = delivery_longdate(symbol)
            #price_data['pullback_buy_action'],price_data['MACD_Crossover'] = delivery_longdate(symbol)
            #pullback_buy_action =price_data['pullback_buy_action'].iloc[-1]
            #macd_crossover = price_data['MACD_Crossover'].iloc[-1]
            
            symbol_data = {'symbol': symbol,
                'last_price': last_price,
                'change': round(float(stock_data.get('change', 0)), 2),
                'p_change': round(float(stock_data.get('pChange', 0)), 2),
                'previous_close': previous_close,
                'open_price': open_price,
                'intraday_high': intraday_high,
                'intraday_low': intraday_low,
                #'market_sentiment': sentiment,
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
    
    return stock_analysis_data
    #return render_template('intraday_analysis.html', stock_analysis_data=stock_analysis_data)

@app.route('/intraday')
def intraday_stock():
    stock_analysis_dat = stock_analysis()
    try:
        senti, vix, call_put_ratio,volume = combine.predict_nifty()
    except:
        senti, vix, call_put_ratio,volume = "None",1,1,1
    
    return render_template('intraday_analysis.html', data=stock_analysis_dat,vix=vix, call_put_ratio=call_put_ratio,volume=volume,senti=senti)
    #return render_template('intraday_analysis.html', stock_analysis_data=stock_analysis_data)
    


if __name__ == "__main__":
    app.run(debug=True)
    
    #app.run(debug=True, host='0.0.0.0', port=80)
