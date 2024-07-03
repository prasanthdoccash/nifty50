# nse.py

# Existing imports
from flask import Flask, render_template, request
import yfinance as yf
from datetime import date, timedelta
import requests
from lxml import html

app = Flask(__name__)

def fetch_vix_data():
    end = date.today()
    start = end - timedelta(days=30)
    vix_data = yf.download("^VIX", start=start, end=end)
    return vix_data['Close'].iloc[-1]

def fetch_call_put_ratio():
    # Fetch PCR ratio and volume from the website using XPath
    url = 'https://www.indiainfoline.com/markets/derivatives/put-call-ratio'
    response = requests.get(url)
    tree = html.fromstring(response.content)

    # Define XPaths for PCR ratio and volume
    xpath_ratio = '/html/body/app-root/div/app-putcall-ratio/div/div[1]/div[1]/div/app-putcall-ratio-detail/div[1]/span/div/div[2]/div/app-fund-overview-put-call-ratio-details-table/div/app-scrollbar/div[2]/table/tbody/tr/td[4]'
    xpath_volume = '/html/body/app-root/div/app-putcall-ratio/div/div[1]/div[1]/div/app-putcall-ratio-detail/div[1]/span/div/div[2]/div/app-fund-overview-put-call-ratio-details-table/div/app-scrollbar/div[2]/table/tbody/tr/td[7]'

    # Extract PCR ratio and volume
    ratio_element = tree.xpath(xpath_ratio)
    volume_element = tree.xpath(xpath_volume)

    if ratio_element and volume_element:
        pcr_ratio = float(ratio_element[0].text.strip())
        pcr_volume = float(volume_element[0].text.strip())
        return pcr_ratio, pcr_volume
    else:
        return None, None

def predict_market_sentiment(vix, call_put_ratio, volume):
    if call_put_ratio < 1 and volume < 1:
        return "Bullish"
    elif vix > 20 and call_put_ratio < 1:
        if volume >1.5:
            return "Strong Bearish"
        else:      # Example condition, adjust as per your prediction logic
            return "Bearish"
    elif vix < 20 and call_put_ratio > 1:  # Example condition, adjust as per your prediction logic
        return "Bullish"
    else:
        return "Neutral"

def predict_nifty():
    # Nifty prediction
    nsymbol ="NIFTY"
    ncall_put_ratio, nvolume = fetch_call_put_ratio()  # Only fetch PCR ratio, ignore volume for now
    vix = fetch_vix_data()
    vix = round(vix,2)
    nsentiment = predict_market_sentiment(vix, ncall_put_ratio,nvolume)
    
    # For GET request or initial page load
    return nsentiment


@app.route('/', methods=['GET', 'POST'])
def predict_nse():
    # Nifty prediction
    nsymbol ="NIFTY"
    ncall_put_ratio, nvolume = fetch_call_put_ratio()  # Only fetch PCR ratio, ignore volume for now
    vix = fetch_vix_data()
    vix = round(vix,2)
    nsentiment = predict_market_sentiment(vix, ncall_put_ratio,nvolume)
    

    if request.method == 'POST':
        # Get form data
        symbol = request.form['symbol']
        call_put_ratio = float(request.form['call_put_ratio'])  # Assuming call_put_ratio is a float input
        volume = float(request.form['volume'])  # Assuming volume is a float input

        

        # Fetch VIX data (assuming it's needed for prediction)
        vix = fetch_vix_data()
        vix = round(vix,2)
        # Perform market sentiment prediction
        sentiment = predict_market_sentiment(vix, call_put_ratio, volume)
        nsymbol ="NIFTY"
        ncall_put_ratio, nvolume = fetch_call_put_ratio()  # Only fetch PCR ratio, ignore volume for now
        nsentiment = predict_market_sentiment(vix, ncall_put_ratio,nvolume)
        # Render template with results
        return render_template('nse.html', vix=vix, call_put_ratio=call_put_ratio, sentiment=sentiment, symbol=symbol, volume=volume,ncall_put_ratio=ncall_put_ratio, nsentiment=nsentiment, nsymbol=nsymbol, nvolume=nvolume)

    # For GET request or initial page load
    return render_template('nse.html',vix=vix,ncall_put_ratio=ncall_put_ratio, nsentiment=nsentiment, nsymbol=nsymbol, nvolume=nvolume)

if __name__ == '__main__':
    app.run(debug=True)
