import yfinance as yf
import pandas as pd

def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

def preprocess_data(data):
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return data
