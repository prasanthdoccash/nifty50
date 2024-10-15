from flask import Flask, render_template, request, redirect, url_for, jsonify
import datetime as dt
from jugaad_data.nse import NSELive,stock_df
import pandas as pd
import ta
from datetime import date, timedelta
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
#from news import merger
#final_decision_news = merger()
csv_file = 'merged_output.csv'
# Read the CSV file
df_csv = pd.read_csv(csv_file)
df_csv.columns = df_csv.columns.str.strip()
df_csv = df_csv.dropna(subset=[df_csv.columns[0]], how='all')
final_decision_news = df_csv

csv_file_old = 'merged_output1.csv'
# Read the CSV file
df_csv_old = pd.read_csv(csv_file_old)
df_csv_old.columns = df_csv_old.columns.str.strip()
df_csv_old = df_csv_old.dropna(subset=[df_csv_old.columns[0]], how='all')
final_decision_news_old = df_csv_old
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
def calculate_supertrend(df, atr_period, multiplier):
# Calculate ATR
    
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_period).average_true_range()
    
    # Calculate basic upper and lower bands
    df['Upper Basic Band'] = ((df['High'] + df['Low']) / 2) + (multiplier * df['ATR'])
    df['Lower Basic Band'] = ((df['High'] + df['Low']) / 2) - (multiplier * df['ATR'])
    
    # Initialize Supertrend columns
    df['Supertrend'] = 0.0
    df['Final Upper Band'] = df['Upper Basic Band']
    df['Final Lower Band'] = df['Lower Basic Band']
    
    for i in range(1, len(df)):
        # Adjust final upper band
        if df['Close'][i-1] > df['Final Upper Band'][i-1]:
            df['Final Upper Band'][i] = max(df['Upper Basic Band'][i], df['Final Upper Band'][i-1])
        else:
            df['Final Upper Band'][i] = df['Upper Basic Band'][i]
        
        # Adjust final lower band
        if df['Close'][i-1] < df['Final Lower Band'][i-1]:
            df['Final Lower Band'][i] = min(df['Lower Basic Band'][i], df['Final Lower Band'][i-1])
        else:
            df['Final Lower Band'][i] = df['Lower Basic Band'][i]
        
        # Determine Supertrend value
        if df['Close'][i] > df['Final Upper Band'][i-1]:
            df['Supertrend'][i] = df['Final Lower Band'][i]
        elif df['Close'][i] < df['Final Lower Band'][i-1]:
            df['Supertrend'][i] = df['Final Upper Band'][i]
        else:
            df['Supertrend'][i] = df['Supertrend'][i-1]
    
    return df['Supertrend']

def apply_supertrend_strategy(df):
    # Create 3 Supertrend indicators with different settings
    df['Supertrend_1'] = calculate_supertrend(df.copy(), atr_period=12, multiplier=3)
    df['Supertrend_2'] = calculate_supertrend(df.copy(), atr_period=10, multiplier=1)
    df['Supertrend_3'] = calculate_supertrend(df.copy(), atr_period=11, multiplier=2)
  
    # Initialize final signal column
    df['Final Signal'] = ''
    
    # Generate buy/sell signals based on conditions
    for i in range(1, len(df)):
        buy_signals = 0
        sell_signals = 0
        
        # Check buy/sell for each Supertrend
        if df['Close'][-1] > df['Supertrend_1'][-1]:
            buy_signals += 1
        else:
            sell_signals += 1
            
        if df['Close'][-1] > df['Supertrend_2'][-1]:
            buy_signals += 1
        else:
            sell_signals += 1
            
        if df['Close'][-1] > df['Supertrend_3'][-1]:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Determine final signal based on buy/sell counts
        if buy_signals == 3:
            df['Final Signal'][-1] = 'Buy'
        elif buy_signals == 2:
            df['Final Signal'][-1] = 'Watch'
        else:
            df['Final Signal'][-1] = 'Sell'
    
    return df['Final Signal'][-1]
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
    df['DI+'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=14)
    
    df['DI-'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=14)
   
    # Calculate the divergence between EMA and SMA
    df['ema_sma_divergence'] = df['ema'] - df['sma']

    df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['support'] = (2 * df['pivot_point']) - df['High']
    df['resistance'] = (2 * df['pivot_point']) - df['Low']
    #df['support_2'] = df['pivot_point'] - (df['High'] - df['Low'])
    #df['resistance_2'] = df['pivot_point'] + (df['High'] - df['Low'])

    
    
    # Determine buy signals
    df['adx'] = (df['adx1'] > 20) & (df['DI+'] > df['DI-']) & (df['DI+'].shift(1) <= df['DI-'].shift(1))
    
    # Calculate the lowest low and highest high over the look-back period (k_period)
    df['Lowest_Low'] = df['Low'].rolling(window=14).min()
    df['Highest_High'] = df['High'].rolling(window=14).max()
    
    
    # Calculate %K
    
    df['stochastic'] = 100 * ((df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low']))
    
    # Calculate %D (3-period moving average of %K)
    df['stochastic_d'] = df['stochastic'].rolling(window=3).mean()
    
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
def final_decision(df,vix,news_tech,news_pcr,pChange =1):
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
    hold_signals = 0
    
    buy = []
    sell = []
    hold = []
    
    supertrend = apply_supertrend_strategy(df)
    if supertrend == 'Buy':
        buy_signals += 1
        buy.append('supertrend')
    elif supertrend == 'Sell':
        sell_signals += 1
        sell.append('supertrend')
    else:
        hold_signals += 1
        hold.append('supertrend')


    if df['pe'].iloc[-1] != 0:

        if df['pe'].iloc[-1] <= df['average_pe'].iloc[-1]:
            buy_signals += 1
            buy.append('PE')
        elif abs(df['pe'].iloc[-1] - df['average_pe'].iloc[-1]) < (0.05 * df['average_pe'].iloc[-1]):
            hold_signals += 1
            hold.append('PE')
        else:
            sell_signals += 1
            sell.append('PE')

    vwap_deviation = (last_row['Close'] - last_row['vwap']) / last_row['vwap']
    if vwap_deviation > 0.02:  # Close is more than 2% above VWAP
        buy_signals += 1
        buy.append('VWAP Strong Buy')
    elif vwap_deviation < -0.02:  # Close is more than 2% below VWAP
        sell_signals += 1
        sell.append('VWAP Strong Sell')
    # Check if VWAP is trending up or down
    if last_row['vwap'] > df['vwap'].iloc[-2]:
        buy_signals += 1
        buy.append('VWAP Trend Up')
    elif last_row['vwap'] < df['vwap'].iloc[-2]:
        sell_signals += 1
        sell.append('VWAP Trend Down')
    # VWAP Crossover Logic
    if (df['Close'].iloc[-2] < df['vwap'].iloc[-2]) and (last_row['Close'] > last_row['vwap']):  # Crossed above VWAP
        buy_signals += 1
        buy.append('VWAP Cross Above')
    elif (df['Close'].iloc[-2] > df['vwap'].iloc[-2]) and (last_row['Close'] < last_row['vwap']):  # Crossed below VWAP
        sell_signals += 1
        sell.append('VWAP Cross Below')
    

    if news_tech == 'SuperBuy':
        buy_signals += 2  # Adding a strong influence to buy signals
        buy.append('Tech SuperBuy')
    elif news_tech == 'IntraBuy':
        buy_signals += 1
        buy.append('Tech IntraBuy')
    elif news_tech == 'Sell':
        sell_signals += 2
        sell.append('Tech Sell')
    else:
        hold_signals += 1
        hold.append('Tech Watch')
    if news_pcr == 'SuperBuy':
        buy_signals += 2
        buy.append('PCR SuperBuy')
    
    elif news_pcr == 'Sell':
        sell_signals += 2
        sell.append('PCR Sell')
    else:
        buy_signals += 1
        buy.append('PCR Watch')


    if indicators['williams_r'] > -20 and df['williams_r'].iloc[-2] <= -20:  # just crossed into overbought zone
        sell_signals += 1
        sell.append('Williams R')
    elif indicators['williams_r'] < -80 and df['williams_r'].iloc[-2] >= -80:  # just crossed into oversold zone
        buy_signals += 1
        buy.append('Williams R')
    else:
        hold_signals +=1
        hold.append('williamsR')
    
     # Improved ROC Logic
    roc_strength = indicators['ROC']  # Current ROC value
    roc_previous = df['ROC'].iloc[-2]
    if roc_strength > 5:
        buy.append('Strong ROC Buy')
        buy_signals += 1
    elif roc_strength < -5:
        sell_signals += 1
        sell.append('Strong ROC Sell')

    if roc_strength > roc_previous:
        buy_signals += 1
        buy.append('ROC Momentum Increase')
    elif roc_strength < roc_previous:
        sell_signals += 1
        sell.append('ROC Momentum Decrease')
    # Divergence detection: If price is increasing but ROC is decreasing, indicate weakening trend
    if (last_row['Close'] > df['Close'].iloc[-2]) and (roc_strength < roc_previous):
        sell_signals += 1
        sell.append('ROC Price Divergence')
    
    volume_change = (indicators['Volume_Trend'] - df['Volume_Trend'].iloc[-2]) / df['Volume_Trend'].iloc[-2]
    avg_volume = df['Volume_Trend'].rolling(window=20).mean().iloc[-1]  # Average volume over the past 20 periods
    if indicators['Volume_Trend'] > avg_volume and volume_change > 0.1:  # If volume is 10% higher than the average volume
        buy_signals += 1
        buy.append('High Volume Buy')
        
    elif indicators['Volume_Trend'] < avg_volume and volume_change < -0.1:  # If volume is 10% lower than the average volume
        sell_signals += 1
        sell.append('Low Volume Sell')
    # Check for volume trend direction
    if volume_change > 0.05:  # If volume trend is increasing more than 5%
        buy_signals += 1
        buy.append('Volume Trend Increase')
    elif volume_change < -0.05:  # If volume trend is decreasing more than 5%
        sell_signals += 1
        sell.append('Volume Trend Decrease')
   
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
    

    if indicators['rsi'] < 30 and df['rsi'].iloc[-2] >= 30:  # RSI just moved into oversold
        buy.append('RSI')
        buy_signals += 1
    elif indicators['rsi'] > 80 and df['rsi'].iloc[-2] <= 80:  # RSI just moved into overbought
        sell_signals += 1 
        sell.append('RSI')
    elif indicators['rsi'] > 30 and indicators['rsi'] < 80:
        hold_signals +=1
        hold.append('RSI')
    

    # MACD
    if indicators['macd'] > df['macd_signal'].iloc[-1]:  # if MACD line is above the signal line
        buy_signals += 1
        buy.append('MACD Cross')
    elif indicators['macd'] < 0:
        sell_signals += 1
        sell.append('MACD')

    # EMA and SMA
    if indicators['ema'] > indicators['sma'] and (indicators['ema'] - indicators['sma']) > df['ema_sma_divergence'].iloc[-1]:
        buy_signals += 1
        buy.append('EMA Crossover Strengthening')
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
        hold_signals +=1
        hold.append('BOLLINGER')

    # ADX
    
    if 'DI+' in df.columns and 'DI-' in df.columns:
        adx_value = indicators['adx']
        adx_prev = df['adx'].iloc[-2]
        # Check ADX thresholds
        if adx_value > 25:
            buy_signals += 1
            buy.append('Strong ADX Buy')
        elif adx_value < 20:
            sell_signals += 1
            sell.append('Weak Trend Sell')
        # Detect trend change: if ADX is increasing after being below 20
        if adx_value > adx_prev and adx_prev < 20:
            buy_signals += 1
            buy.append('ADX Trend Change Buy')
        # Check directional movement (assuming DI+ and DI- are in the dataframe)
    
        if df['DI+'].iloc[-1] > df['DI-'].iloc[-1] and adx_value > 25:  # Strong bullish trend
            buy_signals += 1
            buy.append('Bullish DI+')
        elif df['DI+'].iloc[-1] < df['DI-'].iloc[-1] and adx_value > 25:  # Strong bearish trend
            sell_signals += 1
            sell.append('Bearish DI-')
        
      
    
    # Stochastic Oscillator
    stochastic_k = indicators['stochastic']  # Assuming %K value
    stochastic_d = df['stochastic_d'].iloc[-1]  # Assuming %D value is in the dataframe
    # Crossover strategy
    if stochastic_k < 20 and stochastic_k > stochastic_d:  # %K crossing above %D below 20 (oversold)
        buy_signals += 1
        buy.append('Stochastic Cross Buy')
    elif stochastic_k > 80 and stochastic_k < stochastic_d:  # %K crossing below %D above 80 (overbought)
        sell_signals += 1
        sell.append('Stochastic Cross Sell')
    # Detect divergence: e.g., price making lower lows but Stochastic making higher lows
    if (last_row['Close'] < df['Close'].min()) and (stochastic_k > stochastic_d):
        buy_signals += 1
        buy.append('Stochastic Divergence Buy')

            
    decision = ''
    
    # Apply priority rules for Buy, Hold, and Sell decisions
    if buy_signals >= 4 and sell_signals < 2:
        decision = 'Super Buy'
    elif sell_signals >= 4 and buy_signals < 2:
        decision = 'Sell'
    elif buy_signals >= 3 and hold_signals >= 2 and buy_signals >sell_signals:
        decision = 'Buy'
    elif hold_signals > buy_signals and hold_signals > sell_signals:
        decision = 'Hold'
    else:
        decision = 'Sell'
    
    if 'Sell' in buy and 'supertrend' in sell:
        decision = 'Sell'
    
    if 'Hold' in buy and 'supertrend' in hold:
        decision = 'Watch'

    if 'supertrend' in buy and 'VWAP Strong Buy' in buy and 'RSI' in buy and 'MACD Cross' in buy and  'Strong ROC Buy' in buy and 'Super Buy' in buy and (decision == 'Buy' or decision == 'Super Buy'):
        decision = 'Super Buy'
    elif 'supertrend' in buy and 'PCR SuperBuy' in buy and ('ROC Momentum Increase' in buy or 'Volume Trend Increase' in buy or 'ADX Trend Change Buy' in buy or 'Stochastic Divergence Buy' in buy or 'EMA Crossover Strengthening' in buy) and 'Super Buy' in buy and (decision == 'Buy' or decision == 'Super Buy'):
        decision = 'Super Buy'
    elif 'supertrend' in buy and ('ROC Momentum Increase' in buy or 'Volume Trend Increase' in buy or 'ADX Trend Change Buy' in buy or 'Stochastic Divergence Buy' in buy or 'EMA Crossover Strengthening' in buy) and 'Super Buy' in buy and (decision == 'Buy' or decision == 'Super Buy'):
        decision = 'Buy'
    
    if ('supertrend' in buy or 'supertrend' in hold) and 'Tech SuperBuy' in buy and ('Buy' in buy or 'Super Buy' in buy):
        decision = 'Buy'

    if ('supertrend' in buy or 'supertrend' in hold) and 'Tech SuperBuy' in buy and ('Buy' in buy or 'Super Buy' in buy) and (decision == 'Buy' or decision == 'Super Buy'):
        decision = 'Super Buy'
    elif 'supertrend' in hold and 'Tech SuperBuy' in buy and 'Hold' in buy and (decision == 'Buy' or decision == 'Super Buy'):
        decision = 'Buy'
    elif ('supertrend' in buy or 'supertrend' in hold) and 'Tech SuperBuy' in buy and ('Sell' in buy or 'PE' in buy) and (decision == 'Buy' or decision == 'Super Buy'):
        decision = 'Watch'
     
    if 'supertrend' in buy and 'VWAP Strong Buy' in buy and 'VWAP Trend Up' in buy and 'Tech SuperBuy' in buy and 'PCR SuperBuy' in buy and 'Strong ROC Buy' in buy and  'MACD Cross' in buy and 'Sell' not in buy and (decision == 'Buy' or decision == 'Super Buy'):
        decision = 'Super Buy'
    
    if decision == 'Super Buy' and ('Volume Trend Decrease' in sell or pChange <0) :
        decision = 'Buy'
    return decision,buy_signals,sell_signals,hold_signals, buy,sell,hold
    

###########################################################################
nse = NSELive()
# Caching stock data fetching
#@lru_cache(maxsize=1000)
def fetch_delivery_data(symbols, num1):
   
    all_data = {}
    for symbol in symbols:
        
        stock = yf.Ticker(symbol)
        if num1 == 0:
            df = stock.history(period="1y", interval="1d") # 30d 1h identifies Stocks for delivery
        else:
            df = stock.history(period="5d", interval="5m") # 7d 1m identifies Stocks for intraday
        
        df['pe'] = stock.info.get('trailingPE')
        df['eps'] = stock.info.get('trailingEps')

        if not df.empty:
            checkpe = df['pe'].iloc[-1]
            checkeps = df['eps'].iloc[-1]
            if checkpe is None:
                df['pe'] =0
            if checkeps is None:
                df['eps'] = 0
        
        if not df.empty:
            all_data[symbol] = df
    
    return all_data

# New function to handle parallel processing
def fetch_delivery_data_parallel(symbols, num1):
    all_data = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Create a future for each symbol
        future_to_symbol = {executor.submit(fetch_single_symbol_data, symbol, num1): symbol for symbol in symbols}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if data is not None:
                    all_data[symbol] = data
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
    
    return all_data

# Helper function to fetch data for a single symbol
def fetch_single_symbol_data(symbol, num1):
    stock = yf.Ticker(symbol)
    if num1 == 0:
        df = stock.history(period="1y", interval="1d")
    else:
        df = stock.history(period="5d", interval="5m")

    df['pe'] = stock.info.get('trailingPE')
    df['eps'] = stock.info.get('trailingEps')

    if not df.empty:
        checkpe = df['pe'].iloc[-1]
        checkeps = df['eps'].iloc[-1]
        if checkpe is None:
            df['pe'] = 0
        if checkeps is None:
            df['eps'] = 0

    return df if not df.empty else None

# def fetch_price_data(symbol):
#     dq = {}
    
#     q = nse.stock_quote(symbol)
    
#     dq[symbol] =  q['priceInfo']
    
#     lastPrice =float(dq[symbol].get('lastPrice', 0))
    
#     pChange =float(dq[symbol].get('pChange', 0))
#     pChange =round(pChange,2)
#     return lastPrice ,pChange

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
predefined_symbols123 = [  "ADANIENT","ADANIPORTS", "APOLLOHOSP","ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE","BRITANNIA", "BHARTIARTL", "BPCL", "CIPLA", "COALINDIA", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH","HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "INDUSINDBK", "INFY", "IOC", "ITC", "JSWSTEEL", "KOTAKBANK", "LTIM","LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHREECEM", "SUNPHARMA", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO"
   ]
predefined_symbols_s = ["360ONE.NS", "AARTIIND.NS", "AAVAS.NS", "ACE.NS", "AETHER.NS", "AFFLE.NS",     "APLLTD.NS", "ALKYLAMINE.NS", "ALLCARGO.NS", "ALOKINDS.NS", "ARE&M.NS", "AMBER.NS",     "ANANDRATHI.NS", "ANGELONE.NS", "ANURAS.NS", "APARINDS.NS", "APTUS.NS", "ACI.NS",     "ASAHIINDIA.NS", "ASTERDM.NS", "ASTRAZEN.NS", "AVANTIFEED.NS", "BEML.NS", "BLS.NS",     "BALAMINES.NS", "BALRAMCHIN.NS", "BIKAJI.NS", "BIRLACORPN.NS", "BSOFT.NS", "BLUEDART.NS",     "BLUESTARCO.NS", "BBTC.NS", "BORORENEW.NS", "BRIGADE.NS", "MAPMYINDIA.NS", "CCL.NS",     "CESC.NS", "CIEINDIA.NS", "CSBBANK.NS", "CAMPUS.NS", "CANFINHOME.NS", "CAPLIPOINT.NS",     "CGCL.NS", "CASTROLIND.NS", "CEATLTD.NS", "CELLO.NS", "CENTRALBK.NS", "CDSL.NS",     "CENTURYPLY.NS", "CENTURYTEX.NS", "CERA.NS", "CHALET.NS", "CHAMBLFERT.NS", "CHEMPLASTS.NS",     "CHENNPETRO.NS", "CHOLAHLDNG.NS", "CUB.NS", "CLEAN.NS", "COCHINSHIP.NS", "CAMS.NS",     "CONCORDBIO.NS", "CRAFTSMAN.NS", "CREDITACC.NS", "CROMPTON.NS", "CYIENT.NS", "DCMSHRIRAM.NS",     "DOMS.NS", "DATAPATTNS.NS", "DEEPAKFERT.NS", "EIDPARRY.NS", "EIHOTEL.NS", "EPL.NS",     "EASEMYTRIP.NS", "ELECON.NS", "ELGIEQUIP.NS", "ENGINERSIN.NS", "EQUITASBNK.NS", "ERIS.NS",     "EXIDEIND.NS", "FDC.NS", "FINEORG.NS", "FINCABLES.NS", "FINPIPE.NS", "FSL.NS", "FIVESTAR.NS",     "GMMPFAUDLR.NS", "GRSE.NS", "GILLETTE.NS", "GLS.NS", "GLENMARK.NS", "MEDANTA.NS", "GPIL.NS",     "GODFRYPHLP.NS", "GRANULES.NS", "GRAPHITE.NS", "GESHIP.NS", "GAEL.NS", "GMDCLTD.NS",     "GNFC.NS", "GPPL.NS", "GSFC.NS", "GSPL.NS", "HEG.NS", "HFCL.NS",     "HAPPSTMNDS.NS", "HAPPYFORGE.NS", "HSCL.NS", "HINDCOPPER.NS", "POWERINDIA.NS", "HOMEFIRST.NS",     "HONASA.NS", "HUDCO.NS", "IDFC.NS", "IIFL.NS", "IRB.NS", "IRCON.NS", "ITI.NS",     "INDIACEM.NS", "INDIAMART.NS", "IEX.NS", "IOB.NS", "INDIGOPNTS.NS", "INOXWIND.NS", "INTELLECT.NS",     "JBCHEPHARM.NS", "JBMA.NS", "JKLAKSHMI.NS", "JKPAPER.NS", "JMFINANCIL.NS", "JAIBALAJI.NS",     "J&KBANK.NS", "JINDALSAW.NS", "JUBLINGREA.NS", "JUBLPHARMA.NS", "JWL.NS", "JUSTDIAL.NS",     "JYOTHYLAB.NS", "KNRCON.NS", "KRBL.NS", "KSB.NS", "KPIL.NS", "KARURVYSYA.NS", "KAYNES.NS",     "KEC.NS", "KFINTECH.NS", "KIMS.NS", "LATENTVIEW.NS", "LXCHEM.NS", "LEMONTREE.NS", "MMTC.NS",     "MTARTECH.NS", "MGL.NS", "MAHSEAMLES.NS", "MHRIL.NS", "MAHLIFE.NS", "MANAPPURAM.NS",     "MRPL.NS", "MASTEK.NS", "MEDPLUS.NS", "METROPOLIS.NS", "MINDACORP.NS", "MOTILALOFS.NS",     "MCX.NS", "NATCOPHARM.NS", "NBCC.NS", "NCC.NS", "NLCINDIA.NS", "NSLNISP.NS", "NH.NS",     "NATIONALUM.NS", "NAVINFLUOR.NS", "NETWORK18.NS", "NAM-INDIA.NS", "NUVAMA.NS", "NUVOCO.NS",     "OLECTRA.NS", "PCBL.NS", "PNBHOUSING.NS", "PNCINFRA.NS", "PVRINOX.NS", "PPLPHARMA.NS",     "POLYMED.NS", "PRAJIND.NS", "PRINCEPIPE.NS", "PRSMJOHNSN.NS", "QUESS.NS", "RRKABEL.NS",     "RBLBANK.NS", "RHIM.NS", "RITES.NS", "RADICO.NS", "RAILTEL.NS", "RAINBOW.NS",     "RAJESHEXPO.NS", "RKFORGE.NS", "RCF.NS", "RATNAMANI.NS", "RTNINDIA.NS", "RAYMOND.NS",     "REDINGTON.NS", "RBA.NS", "ROUTE.NS", "SBFC.NS", "SAFARI.NS", "SAMMAANCAP.NS",     "SANOFI.NS", "SAPPHIRE.NS", "SAREGAMA.NS", "SCHNEIDER.NS", "RENUKA.NS", "SHYAMMETL.NS",     "SIGNATURE.NS", "SOBHA.NS", "SONATSOFTW.NS", "SWSOLAR.NS", "STLTECH.NS", "SPARC.NS",     "SUNTECK.NS", "SUVENPHAR.NS", "SWANENERGY.NS", "SYRMA.NS", "TV18BRDCST.NS", "TVSSCS.NS",     "TMB.NS", "TANLA.NS", "TATAINVEST.NS", "TTML.NS", "TEJASNET.NS", "TITAGARH.NS",     "TRIDENT.NS", "TRIVENI.NS", "TRITURBINE.NS", "UCOBANK.NS", "UTIAMC.NS", "UJJIVANSFB.NS",     "USHAMART.NS", "VGUARD.NS", "VIPIND.NS", "VAIBHAVGBL.NS", "VTL.NS", "VARROC.NS",     "VIJAYA.NS", "WELCORP.NS", "WELSPUNLIV.NS", "WESTLIFE.NS", "WHIRLPOOL.NS", "ZENSARTECH.NS",     "ECLERX.NS"]

predefined_symbols = [f"{symbol}.NS" for symbol in predefined_symbols123]

predefined_symbols_5 = [
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
]
predefined_symbols_500 = [f"{symbol}.NS" for symbol in predefined_symbols_5]
predefined_symbols_m =["ACC.NS", "APLAPOLLO.NS", "AUBANK.NS", "ABCAPITAL.NS", "ABFRL.NS", "ALKEM.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASTRAL.NS", "AUROPHARMA.NS", "BSE.NS", "BALKRISIND.NS", "BANDHANBNK.NS", "BANKINDIA.NS", "MAHABANK.NS", "BDL.NS", "BHARATFORG.NS", "BHARTIHEXA.NS", "BIOCON.NS", "CGPOWER.NS", "COCHINSHIP.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "CUMMINSIND.NS", "DELHIVERY.NS", "DIXON.NS", "ESCORTS.NS", "EXIDEIND.NS", "NYKAA.NS", "FEDERALBNK.NS", "FACT.NS", "GMRINFRA.NS", "GODREJPROP.NS", "HDFCAMC.NS", "HINDPETRO.NS", "HINDZINC.NS", "HUDCO.NS", "IDBI.NS", "IDFCFIRSTB.NS", "IRB.NS", "INDIANB.NS", "INDHOTEL.NS", "IOB.NS", "IREDA.NS", "IGL.NS", "INDUSTOWER.NS", "JSWINFRA.NS", "JUBLFOOD.NS", "KPITTECH.NS", "KALYANKJIL.NS", "LTF.NS", "LICHSGFIN.NS", "LUPIN.NS", "MRF.NS", "M&MFIN.NS", "MRPL.NS", "MANKIND.NS", "MARICO.NS", "MFSL.NS", "MAXHEALTH.NS", "MAZDOCK.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NLCINDIA.NS", "NMDC.NS", "OBEROIRLTY.NS", "OIL.NS", "PAYTM.NS", "OFSS.NS", "POLICYBZR.NS", "PIIND.NS", "PAGEIND.NS", "PATANJALI.NS", "PERSISTENT.NS", "PETRONET.NS", "PHOENIXLTD.NS", "POLYCAB.NS", "POONAWALLA.NS", "PRESTIGE.NS", "RVNL.NS", "SBICARD.NS", "SJVN.NS", "SRF.NS", "SOLARINDS.NS", "SONACOMS.NS", "SAIL.NS", "SUNDARMFIN.NS", "SUPREMEIND.NS", "SUZLON.NS", "TATACHEM.NS", "TATACOMM.NS", "TATAELXSI.NS", "TATATECH.NS", "TORNTPOWER.NS", "TIINDIA.NS", "UPL.NS", "IDEA.NS", "VOLTAS.NS", "YESBANK.NS"]

#tech superbuy symbols
def tech_superbuy(intrabuy):
    #df_tech = pd.read_excel('auto_updated_with_decisions.xlsx')  # Adjust sheet name if necessary
    if intrabuy == 1:
        final_decision_news1 = final_decision_news_old
    else:
        final_decision_news1 = final_decision_news

    results1 = []
    for index, row in final_decision_news1.iterrows():
        #tech_stock_symbol = row['Stock Symbol']
        #results1.append(tech_stock_symbol) 

        tech_stock_symbol =row['Stock Symbol']
        news_symb1 =tech_stock_symbol +".NS"
        
        news_tech = row['Decision']
         
        
        
        
        if news_tech =='SuperBuy' or news_tech =='IntraBuy':
            results1.append(news_symb1)
    return results1

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
        
        if symbols1 == 'small':
            symbols =predefined_symbols_s
            #symbols = symbols1 if symbols1 else predefined_symbols
        elif symbols1 == 'nifty50':
            symbols =predefined_symbols
        elif symbols1 == '500':
            symbols = predefined_symbols_500
        elif symbols1 == 'mid':
            symbols = predefined_symbols_m
        elif symbols1 == 'superbuy':
            
            symbols = tech_superbuy(0)
        elif symbols1 == 'intrabuy':
            
            symbols = tech_superbuy(1)
        else:
            symbols = symbols1.split(',') if symbols1 else predefined_symbols
    
    
    last_refreshed = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = fetch_delivery_data(symbols,0)
    
    _,vix,vix_senti = calculate_vix('^INDIAVIX')
    results = []
    now = 0
    i=1
    for symbol, df in data.items():
        try:
            # Calculate indicators
            for index, row in final_decision_news.iterrows():
                    
                news_symb =row['Stock Symbol']
                news_symb1 =news_symb +".NS"
                if symbol == news_symb1:
                    news_decision_t = row['Decision']
                    #news_decision_pcr = row['Final Decision']
                    break
                else:
                    news_decision_t = 'Hold'  
                    #news_decision_pcr = 'Hold' 
            news_tech = news_decision_t
            #news_pcr = news_decision_pcr
            news_pcr = 'Hold'  
            #if news_tech =='SuperBuy':

            df = calculate_indicators(df,now)
            
            last_Price = round(df['Close'].iloc[-1],2)
            
            open_price = df['Open'].iloc[-1]
            open_price=round(open_price,2)
            stoploss = open_price - (open_price*0.01)
            stoploss=round(stoploss,2)
            pChange = last_Price - open_price
            pChange=round((pChange/open_price)*100,2)   

            target = open_price + (open_price*0.012)
            target = round(target,2)

            # for index, row in final_decision_news.iterrows():
                
            #     news_symb1 =row['Stock Symbol']
            #     news_symb =news_symb1 +".NS"
            #     if symbol == news_symb:
            #         news_decision_t = row['Decision']
            #         news_decision_pcr = row['Final Decision']
            #         break
            #     else:
            #         news_decision_t = 'Hold'  
            #         news_decision_pcr = 'Hold' 
            # news_tech = news_decision_t
            # news_pcr = news_decision_pcr
            # Determine final decision based on sentiment
            decision,buy_signals,sell_signals,hold_signals, buy,sell,hold = final_decision(df,vix,news_tech,news_pcr,pChange)

            symbols_NS = symbol[:-3]
            #last_Price,pChange = fetch_price_data(symbols_NS)
        
        
           # decision = "Buy"
            print(i,symbol)
            i=i+1
            # Prepare data for each symbol
            symbol_data = {
                'symbol': symbol,
                #'company_name': company_name,
                'LTP': last_Price, #close price
                'pChange':pChange,
                #'indicators_data': indicators,
                #'sentiment': sentiment,
                'decision': decision,
                'target': target,
                'stoploss':stoploss,
                
                
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
        
        if symbols1 == 'small':
            symbols =predefined_symbols_s
            #symbols = symbols1 if symbols1 else predefined_symbols
        elif symbols1 == 'nifty50':
            symbols =predefined_symbols
        elif symbols1 == '500':
            symbols = predefined_symbols_500
        elif symbols1 == 'mid':
            symbols = predefined_symbols_m
        elif symbols1 == 'superbuy':
            
            symbols = tech_superbuy(0)
        elif symbols1 == 'intrabuy':
            
            symbols = tech_superbuy(1)
        else:
            symbols = symbols1.split(',') if symbols1 else predefined_symbols
    
    
    
    last_refreshed = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = fetch_delivery_data(symbols,1)
    _,vix,vix_senti = calculate_vix('^INDIAVIX')
    results = []
    new=5
    
    i=1
    for symbol, df in data.items():

        try:
            #supertrend = apply_supertrend_strategy(df)
            
            
            # Calculate indicators
            
            df = calculate_indicators(df,new)
            last_Price = round(df['Close'].iloc[-1],2)
            
            
            
             

            
            #supertrend = apply_supertrend_strategy(df)

            # target = df['resistance'].iloc[-1]
            # target=round(target,2)
            # stoploss = df['support'].iloc[-1]
            # stoploss=round(stoploss,2)
                                 
            for index, row in final_decision_news.iterrows():
                
                news_symb1 =row['Stock Symbol']
                news_symb =news_symb1 +".NS"
                if symbol == news_symb:
                    news_decision_t = row['Decision']
                    #news_decision_pcr = row['Final Decision']
                    break
                else:
                    news_decision_t = 'Hold'  
                   # news_decision_pcr = 'Hold' 
            news_tech = news_decision_t
           # news_pcr = news_decision_pcr    
            news_pcr = 'Hold'                
            
            # Determine final decision based on sentiment
            decision,buy_signals,sell_signals,hold_signals, buy,sell,hold = final_decision(df,vix,news_tech,news_pcr)

            symbols_NS = symbol[:-3]
            #last_Price ,pChange= fetch_price_data(symbols_NS)
            
            print(i,symbol, "i")
            i=i+1
            # Prepare data for each symbol
            symbol_data = {
                'symbol': symbol,
                #'company_name': company_name,
                'LTP': last_Price,
                #'pChange':pChange,
                #'indicators_data': indicators,
                #'sentiment': sentiment,
                'decision': decision,
                #'supertrend': supertrend,
                #'stoploss':stoploss,
                # 'sell_signals':sell_signals,
                # 'hold_signal':hold_signals,
                'buy':buy,
                'sell':sell,
                'hold': hold
                
            }
            
            results.append(symbol_data) 
    
        except KeyError as e:
            print(f"KeyError: {str(e)}. Skipping symbol {symbol}.")
            continue
    
    return results,last_refreshed,vix,vix_senti
    
@app.route('/both')
def both(): 
    symb = ''  
    results_i,last_refreshed,vix,vix_senti = intraday(symb)
    
    results,last_refreshed,vix,vix_senti = delivery(symb)
    zipped_results = zip(results, results_i)
    return render_template('both.html', zipped_results=zipped_results, last_refreshed=last_refreshed,vix=vix,vix_senti=vix_senti)
   
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