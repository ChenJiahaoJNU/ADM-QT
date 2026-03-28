import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'transaction_cost': 0.0003,
    'ma_short': 5,
    'ma_long': 20,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
}

def preprocess_data(data):
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2021-01-01') & (df['date'] <= '2022-12-31')]
    df = df.dropna(subset=['open']).reset_index(drop=True)
    return df

def calculate_portfolio(df, signal_col='signal'):
    cash = 100000
    holdings = 0
    portfolio_value = []

    for _, row in df.iterrows():
        price = row['open']
        signal = row[signal_col]

        if signal == 1 and holdings == 0:
            shares = int(cash / (price * (1 + CONFIG['transaction_cost'])))
            if shares > 0:
                cash -= shares * price * (1 + CONFIG['transaction_cost'])
                holdings = shares
        elif signal == -1 and holdings > 0:
            cash += holdings * price * (1 - CONFIG['transaction_cost'])
            holdings = 0

        portfolio_value.append(cash + holdings * price)

    if holdings > 0:
        cash += holdings * df['open'].iloc[-1] * (1 - CONFIG['transaction_cost'])

    df['portfolio_value'] = portfolio_value
    ret = (cash - 100000) / 100000
    return df, ret

def run_buy_hold_strategy(data):
    df = preprocess_data(data)
    df['signal'] = 1
    start = time.time()
    df, ret = calculate_portfolio(df)
    return df, ret, round(time.time() - start, 2)

def run_ma_strategy(data):
    df = preprocess_data(data)
    df['ma_short'] = df['open'].rolling(CONFIG['ma_short']).mean()
    df['ma_long'] = df['open'].rolling(CONFIG['ma_long']).mean()
    df['signal'] = 0
    df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
    df.loc[df['ma_short'] < df['ma_long'], 'signal'] = -1
    df = df.dropna()
    start = time.time()
    df, ret = calculate_portfolio(df)
    return df, ret, round(time.time() - start, 2)

def run_macd_strategy(data):
    df = preprocess_data(data)
    df['ema12'] = df['open'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['open'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['signal'] = 0
    df.loc[df['macd'] > df['signal_line'], 'signal'] = 1
    df.loc[df['macd'] < df['signal_line'], 'signal'] = -1
    df = df.dropna()
    start = time.time()
    df, ret = calculate_portfolio(df)
    return df, ret, round(time.time() - start, 2)

def run_rsi_strategy(data):
    df = preprocess_data(data)
    delta = df['open'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(CONFIG['rsi_period']).mean()
    avg_loss = loss.rolling(CONFIG['rsi_period']).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi'] = 100 - 100 / (1 + rs)
    df['signal'] = 0
    df.loc[df['rsi'] < CONFIG['rsi_oversold'], 'signal'] = 1
    df.loc[df['rsi'] > CONFIG['rsi_overbought'], 'signal'] = -1
    df = df.dropna()
    start = time.time()
    df, ret = calculate_portfolio(df)
    return df, ret, round(time.time() - start, 2)