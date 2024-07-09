from flask import Flask, render_template
import requests
from lxml import html

app = Flask(__name__)

def get_analyst_recommendations(ticker_symbol):
    url = f'https://finance.yahoo.com/quote/{ticker_symbol}/analysis'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    tree = html.fromstring(response.content)

    
    price_targets = []

    

    # Extracting analyst price targets
    price_target_xpath = '//*[@id="nimbus-app"]/section/section/section/article/div[1]/section/div[1]/section[1]/div/div/div[3]/div[1]/span[1]'
    price_target_elements = tree.xpath(price_target_xpath)
    for elem in price_target_elements:
        '''price_targets.append({
            'label': 'Current Price Target',
            'value': elem.text_content().strip()
        })'''
        price_targets =elem.text_content().strip()
    
    return  price_targets

@app.route('/')
def index():
    # Replace 'BPCL.NS' with the stock ticker symbol you want to analyze
    stock_ticker = 'TCS.NS'
    price_targets = get_analyst_recommendations(stock_ticker)
    
    return render_template('index.html', price_targets=price_targets, ticker=stock_ticker)

if __name__ == '__main__':
    app.run(debug=True)
