import yfinance as yf
import pandas as pd

AVAILABLE_STOCKS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "SBIN": "SBIN.NS"
}

def fetch_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance for NSE tickers."""
    ticker = AVAILABLE_STOCKS.get(symbol.upper())
    if not ticker:
        raise ValueError(f"Stock {symbol} not found. Available: {list(AVAILABLE_STOCKS.keys())}")

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df.reset_index(inplace=True)
    return df
