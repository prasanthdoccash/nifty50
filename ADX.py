from flask import Flask, render_template
import pandas as pd
import yfinance as yf
from ta.trend import ADXIndicator

app = Flask(__name__)

# Function to fetch and process data
def fetch_process_data(ticker_symbol,start_date, end_date):
    # Fetch historical market data for the ticker symbol
    ticker_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Calculate True Range (TR)
    ticker_data['TR'] = pd.DataFrame({
        'TR': ticker_data['High'].combine(ticker_data['Close'].shift(1), lambda x, y: x - y).combine(ticker_data['Close'].shift(1), lambda x, y: abs(x - y)).combine(ticker_data['Low'].shift(1), lambda x, y: abs(x - y))
    })

    # Calculate Directional Movement (DM)
    ticker_data['+DM'] = pd.DataFrame({
        '+DM': (ticker_data['High'] - ticker_data['High'].shift(1)).where((ticker_data['High'] - ticker_data['High'].shift(1)) > (ticker_data['Low'].shift(1) - ticker_data['Low']), 0)
    })

    ticker_data['-DM'] = pd.DataFrame({
        '-DM': (ticker_data['Low'].shift(1) - ticker_data['Low']).where((ticker_data['Low'].shift(1) - ticker_data['Low']) > (ticker_data['High'] - ticker_data['High'].shift(1)), 0)
    })

    # Calculate ATR (6-day smoothed for ADX calculation)
    ticker_data['ATR'] = ticker_data['TR'].rolling(window=6).mean()

    # Calculate +DI and -DI
    ticker_data['+DI'] = (ticker_data['+DM'].rolling(window=14).mean() / ticker_data['ATR']) * 100
    ticker_data['-DI'] = (ticker_data['-DM'].rolling(window=14).mean() / ticker_data['ATR']) * 100

    # Calculate ADX (with a reduced window size)
    adx_indicator = ADXIndicator(high=ticker_data['High'], low=ticker_data['Low'], close=ticker_data['Close'], window=6)
    ticker_data['ADX'] = adx_indicator.adx()

    # Function to interpret trend
    def interpret_trend(row, prev_row):
        if prev_row is None:
            prev_row = row.copy()
            return 'Indeterminate'  # No trend comparison for the first row

        if row['+DI'] > row['-DI'] and row['ADX'] > prev_row['ADX']:
            trend = 'Strong uptrend'
        elif row['-DI'] > row['+DI'] and row['ADX'] > prev_row['ADX']:
            trend = 'Strong downtrend'
        elif row['ADX'] < 25:
            trend = 'Weak trend or no trend'
        else:
            trend = 'Indeterminate'

        prev_row = row.copy()  # Update prev_row for next iteration
        return trend

    # Initialize prev_row to None
    prev_row = None

    # Apply the interpretation function row-wise
    ticker_data['Trend'] = ticker_data.apply(lambda row: interpret_trend(row, prev_row), axis=1)

    return ticker_data

# Route to display interpreted trends
@app.route('/')
def display_trends(ticker_symbols,start_date, end_date):
    # Get the ticker symbol from another source (could be passed as a parameter, hardcoded, etc.)
    #ticker_symbol = 'ALKEM.NS'  # Example: Hardcoded for demonstration
    
    ticker_symbol = f"{ticker_symbols}.NS"
    
    # Fetch and process data for the specified ticker symbol
    ticker_data = fetch_process_data(ticker_symbol,start_date, end_date)
    
    latest_trend = ticker_data['Trend'].iloc[-1]

    # Pass data to the HTML template for rendering
    return latest_trend
    #return render_template('index.html', ticker_symbol=ticker_symbol, latest_trend=latest_trend, data=ticker_data[['+DI', '-DI', 'ADX', 'Trend']].to_html())

if __name__ == '__main__':
    app.run(debug=True)
