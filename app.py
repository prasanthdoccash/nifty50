from flask import Flask, render_template, request, redirect, url_for, jsonify
import datetime as dt
from jugaad_data.nse import NSELive,stock_df
import pandas as pd
import ta
from datetime import date, timedelta
import numpy as np
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
#Start for deployment
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
#stop for deployment

app = Flask(__name__)


def calculate_indicators(df,num2=5):
    
    if num2 == 5: #delivery
    #     df =  df[(df.index.hour == 0) & (df.index.minute == 0)] 
    # else:
        df = df[df.index.minute % 5 == 0]
     
    # Calculate historical P/E ratios using price and EPS
    df['historical_pe'] = df['Close'] / df['eps']

    # Calculate average historical P/E ratio
    df['average_pe'] = np.mean(df['historical_pe'])
    #print("df['average_pe']",df['average_pe'])
    

    
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
    #df['adx'] = ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    # Calculate ADX, +DI, and -DI using ta library
    df['adx1'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    df['+DI'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=14)
    df['-DI'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=14)

    df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['support'] = (2 * df['pivot_point']) - df['High']
    df['resistance'] = (2 * df['pivot_point']) - df['Low']
    #df['support_2'] = df['pivot_point'] - (df['High'] - df['Low'])
    #df['resistance_2'] = df['pivot_point'] + (df['High'] - df['Low'])

    
    # Determine buy signals
    df['adx'] = (df['adx1'] > 20) & (df['+DI'] > df['-DI']) & (df['+DI'].shift(1) <= df['-DI'].shift(1))
    
    df['stochastic'] = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14).stoch()
    
    vwap_indicator = ta.volume.VolumeWeightedAveragePrice(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume'],
        window=14,  # the window parameter, you can adjust this based on your needs
        fillna=True  # handle NaN values
    )
    df['vwap'] = vwap_indicator.vwap
    df['ROC'] = ta.momentum.roc(df['Close'], window=12)
    df['Volume_Trend'] = df['Volume'].rolling(window=20).mean()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        lbp=14,  # look-back period, commonly set to 14
        fillna=True  # handle NaN values
    ).williams_r()
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
        'Volume_Trend' :last_row['Volume_Trend'],
        'ROC':last_row['ROC'],
        'williams_r': last_row['williams_r']
    }

    

    condition_1 = df['Close'].shift(1) > df['vwap'].shift(1)
    condition_2 = df['Open'].shift(1) < df['vwap'].shift(1)
    condition_3 = df['Close'] > df['High'].shift(1)
    
    # Combine conditions to create a signal column (e.g., True if all conditions are met)
    df['signal'] = condition_1.iloc[-1] & condition_2.iloc[-1] & condition_3.iloc[-1]
    
    buy_signals = 0
    sell_signals = 0
    hold_signal = 0
    
    buy = []
    sell = []
    hold = []
    print("df['pe']",df['pe'].iloc[-1])
    #print("df['eps']",df['eps'].iloc[-1])
    #print("average",df['average_pe'].iloc[-1])
    #print("symf",df['symbol'])
    if df['pe'].iloc[-1] != 0:

        if df['pe'].iloc[-1] <= df['average_pe'].iloc[-1]:
            buy_signals += 1
            buy.append('PE')
        else:
            sell_signals += 1
            sell.append('PE')

    if df['signal'].iloc[-1] == True:
        buy_signals += 1
        buy.append('VWAP')
    else:
        sell_signals += 1
        sell.append('VWAP')

    if indicators['williams_r'] > -20: #overbought
        sell_signals += 1 
        sell.append('williamsR')
    elif indicators['williams_r'] < -80: #oversold
        buy.append('williamsR') 
        buy_signals += 1
    else:
        hold_signal +=1
        hold.append('williamsR')
    
    if indicators['ROC'] > 0: #increasing
        buy.append('ROC')
        buy_signals += 1
    elif indicators['ROC'] < 0: #decreasing
        sell_signals += 1
        sell.append('ROC')
    else:
        hold_signal +=1
        hold.append('ROC')
    
    
    if indicators['Volume_Trend'] > df['Volume_Trend'].shift(1).iloc[-1]: #increasing
        buy.append('Volume_Trend')
        buy_signals += 1  
    elif indicators['Volume_Trend'] < df['Volume_Trend'].shift(1).iloc[-1]: #decreasing
        sell_signals += 1
        sell.append('Volume_Trend')
    else:
        hold_signal +=1
        hold.append('Volume_Trend')
   
    rsi2 = df['rsi'].iloc[-1] - df['rsi'].iloc[-2]
    roc2 = df['ROC'].iloc[-1] - df['ROC'].iloc[-2]
    stochastic2 = df['stochastic'].iloc[-1] - df['stochastic'].iloc[-2]
    
    # RSI , ROC, Stoch
    if rsi2 > 0 and roc2 > 0 and stochastic2 > 0: #moving up
        buy.append('Super Buy')
        
    elif (rsi2 > 0 and roc2 > 0) or (rsi2>0 and stochastic2>0) or(roc2>0 and stochastic2>0) : #either
        
        buy.append('Buy')
    elif (rsi2 > 0 or roc2 > 0 or stochastic2>0):
        
        buy.append('Hold')
    else:
        buy.append('Sell')
    

    if indicators['rsi'] <= 30: #oversold
        buy.append('RSI')
        buy_signals += 1
    elif indicators['rsi'] >= 80: #overbought
        sell_signals += 1 
        sell.append('RSI')
    elif indicators['rsi'] > 30 and indicators['rsi'] < 80:
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
    
    if indicators['adx'] == True:
        
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
    '''if last_row['Close'] > indicators['vwap']:
        buy_signals += 1
        buy.append('VWAP')
    else:
        sell_signals += 1
        sell.append('VWAP')'''
            
    decision = ''
    
    if 'Super Buy' in buy and 'ADX' in buy and 'VWAP' in buy:
        decision = 'Intra Buy' #orange
    elif ('VWAP' in buy and 'MACD' in buy and 'EMA' in buy and 'ROC' in buy ):
        if ('Super Buy' in buy  and 'williamsR' not in sell and 'STOCHASTIC' not in sell and 'BOLLINGER' not in sell) or ('Super Buy' in buy  and 'Volume_Trend' in buy):
            decision = 'Buy' #Blue
        elif ('Buy' in buy or 'Super Buy' in buy)  or ('Buy' in buy  and 'Volume_Trend' in buy):
            decision = 'Watch'#Green
        elif  ('Hold' in buy and 'williamsR' not in sell and 'STOCHASTIC' not in sell and 'BOLLINGER' not in sell):
            decision = 'Hold' #Green
         
        else:
            decision= 'Sell' #white
    elif ('Buy' in buy or 'Super Buy' in buy) and 'VWAP' in buy and 'ROC' in buy and ( 'EMA' in buy or 'MACD' in buy):
        decision = 'Buy'#Green
    else:
        decision= 'Sell'

    if decision == 'Sell':
        if ('Buy' in buy or 'Super Buy' in buy and  'PE' in buy):
            decision = 'Watch'#Green
   

    '''elif 'Super Buy' in buy and 'MACD' in buy and 'STOCHASTIC' not in sell and 'williamsR' not in sell:
        decision= 'Super Buy'
    
    elif 'Buy' in buy and 'MACD' in buy and 'STOCHASTIC' not in sell and 'williamsR' not in sell: 
        decision= 'Buy'
    
    elif 'Super Buy' in buy or 'Buy' in buy and 'Volume_Trend' in buy:
            
        decision = 'Buy
        
    else:
        decision= 'Sell'
        '''
    return decision,buy_signals,sell_signals,hold_signal, buy,sell,hold
    '''if buy_signals > sell_signals and buy_signals > (sell_signals +hold_signal) and sell_signals ==0:

        return 'Buy',buy_signals,sell_signals,hold_signal, buy,sell,hold
    elif buy_signals > sell_signals and buy_signals > (sell_signals +hold_signal):

        return 'Super Buy',buy_signals,sell_signals,hold_signal, buy,sell,hold
    elif sell_signals > buy_signals:
        return 'Sell',buy_signals,sell_signals,hold_signal, buy,sell,hold
    else:
        return 'Hold',buy_signals,sell_signals,hold_signal, buy,sell,hold
   
    #if result.decision == 'Buy' and (result.buy_signals >0 and result.sell_signals ==0)'''

###########################################################################
nse = NSELive()
def fetch_delivery_data(symbols, num1):
   
    all_data = {}
    for symbol in symbols:
        print(symbol)
        stock = yf.Ticker(symbol)
        if num1 == 0:
            df = stock.history(period="1y", interval="1d") # 30d 1h identifies Stocks for delivery
        else:
            df = stock.history(period="30d", interval="5m") # 7d 1m identifies Stocks for intraday
        
        df['pe'] = stock.info.get('trailingPE')
        df['eps'] = stock.info.get('trailingEps')

        checkpe = df['pe'].iloc[-1]
        checkeps = df['eps'].iloc[-1]
        if checkpe is None:
            df['pe'] =0
        if checkeps is None:
            df['eps'] = 0
        
        if not df.empty:
            all_data[symbol] = df
    
    return all_data

def fetch_price_data(symbol):
    dq = {}
    
    q = nse.stock_quote(symbol)
    
    dq[symbol] =  q['priceInfo']
    
    lastPrice =float(dq[symbol].get('lastPrice', 0))
    
    pChange =float(dq[symbol].get('pChange', 0))
    pChange =round(pChange,2)
    return lastPrice ,pChange

def calculate_roc(df, window=12):
    df['ROC'] = ta.momentum.roc(df['CLOSE'], window=12)
    
    return df

def calculate_volume_trend(df, window=20):
    if 'VOLUME' in df.columns:
        df['Volume_Trend'] = df['VOLUME'].rolling(window=20).mean()
    return df

def calculate_vix(symb):
    #df['VIX'] = ta.volatility.vix(df['HIGH'], df['LOW'], df['CLOSE'])
    stock = yf.Ticker(symb)
    df = stock.history(period="1d", interval="1m") # Changed to 5 days to get enough data for indicators
    
    df['VIX'] = df['High'].iloc[-1]
    vix = df['High'].iloc[-1]
    vix2 = df['High'].iloc[-2]
    vix3 = vix - vix2
    
    vix = round(vix,2)
    df['VIX_Level'] = df['VIX'].apply(lambda x: 'Low' if x < 15 else 'High')
    if vix3 > 0:
        vix_senti = 'Bearish'
        
    elif vix < 0:
        vix_senti = 'Bullish'
    else: 
        vix_senti = 'Neutral'
    return df,vix,vix_senti



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
     "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "KOTAKBANK", "SBIN", "BAJFINANCE",
    "BHARTIARTL",  "ASIANPAINT", "MARUTI", "DMART", "AXISBANK", "LT", "SUNPHARMA", "ADANIGREEN", "TITAN",
    "WIPRO", "ONGC", "M&M", "DIVISLAB", "HCLTECH", "ADANIENT", "BAJAJFINSV", "NTPC", "ULTRACEMCO", "TATACONSUM",
    "SBILIFE", "JSWSTEEL", "HEROMOTOCO", "INDUSINDBK", "COALINDIA", "TECHM", "BPCL", "POWERGRID", "TATAMOTORS",
    "SHREECEM", "ADANIPORTS", "BAJAJ-AUTO", "EICHERMOT", "HDFCLIFE", "CIPLA", "NESTLEIND", "GRASIM", "VEDL",
    "BRITANNIA", "GAIL", "DABUR",  "SIEMENS", "HINDALCO", "ICICIPRULI", "SBICARD", "ABBOTINDIA",
    "PIDILITIND", "HAVELLS", "BANKBARODA", "LUPIN", "BEL", "BANDHANBNK", "TATASTEEL", "INDIGO", "MUTHOOTFIN",
    "BATAINDIA", "BIOCON", "GLENMARK", "MANAPPURAM", "ESCORTS", "LICHSGFIN", "TORNTPHARM", "MCDOWELL-N", "AMBUJACEM",
    "IOC", "IDFCFIRSTB", "JUBLFOOD", "ACC", "COLPAL",  "APOLLOHOSP", "GODREJCP", "CHOLAFIN", "PAGEIND",
    "NAUKRI", "CONCOR", "ICICIGI", "NMDC", "MFSL", "IIFL", "TATACHEM", "ASTRAL", "AUBANK", "JINDALSTEL", "BHEL",
    "GODREJPROP", "ASHOKLEY", "PNB", "CANBK", "TRENT", "SUNTV", "VOLTAS", "DALBHARAT", "METROPOLIS", "POLYCAB",
    "PETRONET",  "BAJAJELEC", "MGL", "IGL", "MRF", "BERGEPAINT", "PIIND", "DLF", "INDHOTEL", 
    "SRF", "SUNDARMFIN", "ZEEL", "IDBI", "INDIANB", "EXIDEIND", "AIAENG", "ALEMBICLTD", "ALKEM", "ALOKINDS",
     "APOLLOTYRE", "ARVIND", "ASHOKA", "BALKRISIND", "BEML", "BHARATFORG", "BIOCON", "BLISSGVS",
    "BLUEBLENDS", "BOSCHLTD",  "CANFINHOME", "CARBORUNIV", "CASTROLIND", "CESC", "CIPLA", "COROMANDEL",
    "CROMPTON", "CUMMINSIND", "CYIENT", "DEEPAKNTR", "DHANUKA", "DRREDDY", "ECLERX", "ENDURANCE", "FDC", "FINEORG",
    "GILLETTE", "GLAXO", "GLENMARK", "GRINDWELL", "HATSUN", "HINDCOPPER", "HINDPETRO", "HINDZINC", "HUDCO",
    "IBULHSGFIN", "IDEA", "IGL", "IRCTC", "IRCON", "JAMNAAUTO", "JBCHEPHARM", "JKCEMENT", "JSWENERGY", "JUSTDIAL",
    "JYOTHYLAB",  "KARURVYSYA", "KEC", "KNRCON", "KOTAKBANK", "KRBL",  "LALPATHLAB", "LICHSGFIN",
    "LTTS", "LUXIND",  "MAHSCOOTER", "MARICO", "MCX", "METROPOLIS", "MGL",  "MPHASIS", "MRPL",
    "NAM-INDIA", "NATCOPHARM", "NCC", "NESCO", "NETWORK18", "NIACL", "NMDC", "NOCIL", "OBEROIRLTY", "OFSS", "ONGC",
    "ORIENTELEC", "PAGEIND", "PERSISTENT", "PFIZER", "PIDILITIND", "PIIND", "PNBHOUSING", "POLYCAB", "PRAJIND",
    "PRESTIGE",  "RADICO", "RAIN", "RALLIS", "RAMCOCEM", "RAYMOND", "RBLBANK", "RECLTD", "REDINGTON", "RELAXO",
    "RELIANCE", "ROUTE", "SANOFI", "SBILIFE", "SHILPAMED", "SHOPERSTOP", "SIS", "SJVN", "SKFINDIA", "SRF", "STARCEMENT",
    "STLTECH", "SUNPHARMA", "SUNTV", "SUPRAJIT", "SYMPHONY", "TATACHEM", "TATAELXSI",  "TATACOMM",
    "TATAMOTORS", "TATASTEEL", "TCI", "TCNSBRANDS", "TECHM", "THERMAX", "TITAN", "TORNTPOWER", "TRENT", "TVSMOTOR",
    "UBL", "UFLEX", "ULTRACEMCO", "UNIONBANK", "UPL", "VARROC", "VBL", "VEDL", "VINATIORGA", "VOLTAS", 
    "WELCORP",  "WHIRLPOOL", "WIPRO", "WOCKPHARMA", "YESBANK", "ZEEL", "ZENSARTECH"
]'''

@app.route('/')
def home():
    _,vix,vix_senti = calculate_vix('^INDIAVIX')
    return render_template('index.html',vix=vix,vix_senti=vix_senti)
   

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
        
        # Get symbols from predefined or request
        symbols_list = symbols_get
        symbols = symbols_list if symbols_list else predefined_symbols
    else:
        symbols1 = request.args.get('symbols')
        
        symbols = symbols1.split(',') if symbols1 else predefined_symbols
    
        
    last_refreshed = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = fetch_delivery_data(symbols,0)
    
    _,vix,vix_senti = calculate_vix('^INDIAVIX')
    results = []
    now = 0
    for symbol, df in data.items():
        try:
            # Calculate indicators
            
            df = calculate_indicators(df,now)

            target = df['resistance'].iloc[-1]
            target=round(target,2)
            stoploss = df['support'].iloc[-1]
            stoploss=round(stoploss,2)
                                 
            
            # Determine final decision based on sentiment
            decision,buy_signals,sell_signals,hold_signal, buy,sell,hold = final_decision(df,vix)

            symbols_NS = symbol[:-3]
            last_Price,pChange = fetch_price_data(symbols_NS)
            
            
            # Prepare data for each symbol
            symbol_data = {
                'symbol': symbol,
                #'company_name': company_name,
                'LTP': last_Price,
                'pChange':pChange,
                #'indicators_data': indicators,
                #'sentiment': sentiment,
                'decision': decision,
                'price_target': target,
                'stoploss':stoploss,
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

@app.route('/analysis/<symbol>', methods=['GET'])
def view_stock_analysis(symbol):
    # Remove the '.NS' from the symbol if it exists
    trimmed_symbol = symbol.split('.')[0]  # This will keep everything before the '.' (e.g., RELIANCE)
    
    # Pass the trimmed symbol to index1.html
    return render_template('index1.html', stock_symbol=trimmed_symbol)
    
@app.route('/delivery')
def delivery1():
    symb = ''
    results,last_refreshed,vix,vix_senti = delivery(symb)
    
    return render_template('delivery_analysis.html', results=results,last_refreshed=last_refreshed,vix=vix,vix_senti=vix_senti)
   
def intraday(symbols_get):

    
    if symbols_get != "":
        
        # Get symbols from predefined or request
        symbols_list = symbols_get
        symbols = symbols_list if symbols_list else predefined_symbols
    else:
        symbols1 = request.args.get('symbols')
        
        symbols = symbols1.split(',') if symbols1 else predefined_symbols
    
        
    last_refreshed = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = fetch_delivery_data(symbols,1)
    _,vix,vix_senti = calculate_vix('^INDIAVIX')
    results = []
    new=5
    for symbol, df in data.items():
        try:
            # Calculate indicators
            
            df = calculate_indicators(df,new)
            
                                
            
            # Determine final decision based on sentiment
            decision,buy_signals,sell_signals,hold_signal, buy,sell,hold = final_decision(df,vix)
            symbols_NS = symbol[:-3]
            last_Price ,pChange= fetch_price_data(symbols_NS)
            
            

            # Prepare data for each symbol
            symbol_data = {
                'symbol': symbol,
                #'company_name': company_name,
                'LTP': last_Price,
                'pChange':pChange,
                #'indicators_data': indicators,
                #'sentiment': sentiment,
                'decision': decision,
                #'price_target': target,
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
    
    
@app.route('/intraday')
def intraday1():
    symb = ''
    results,last_refreshed,vix,vix_senti = intraday(symb)
    
    return render_template('intraday_analysis.html', results=results,last_refreshed=last_refreshed,vix=vix,vix_senti=vix_senti)
   
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
    
    results,last_refreshed,vix,vix_senti = delivery(symbols)
    return render_template('show_delivery_analysis.html', results=results,last_refreshed=last_refreshed,vix=vix, vix_senti=vix_senti)

# Set pandas option to avoid future warnings
pd.set_option('future.no_silent_downcasting', True)

@app.route('/show_watchlist_int')
def show_watchlist_intra():
    symbols = get_watchlist_symbols()
    
    results,last_refreshed,vix,vix_senti = intraday(symbols)
    return render_template('show_intraday_analysis.html', results=results,last_refreshed=last_refreshed,vix=vix, vix_senti=vix_senti)


# Function to get symbols from watchlist
def get_watchlist_symbols():
    try:
        with open('watchlist.json', 'r') as f:
            symbols = json.load(f)
    except FileNotFoundError:
        symbols = []
    
    return symbols


if __name__ == "__main__":
    #app.run(debug=True)
    
    app.run(debug=True, host='0.0.0.0', port=80)
