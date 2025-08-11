import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame):
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])

    df['lag1'] = df['Close'].shift(1)
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['vol5'] = df['Close'].rolling(window=5).std()
    df['dow'] = df['Date'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

    df = df.dropna()
    feature_cols = ['lag1', 'ma5', 'vol5', 'dow_sin', 'dow_cos']
    return df, feature_cols
