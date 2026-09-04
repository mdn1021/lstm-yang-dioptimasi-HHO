"""
data_utils.py
Preprocessing bersama untuk notebook baseline (01) dan HHO (02).

Kedua model HARUS memakai pipeline yang identik agar perbandingan adil
(BAB 3.6.2): split walk-forward 70/15/15 sebelum scaling, lalu Min-Max
Normalization langsung pada OHLCV mentah (BUKAN percentage-change/return),
sesuai keputusan final di README & BAB 3.4.2-3.4.3.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TARGET_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
N_FEATURES  = len(TARGET_COLS)
CLOSE_IDX   = TARGET_COLS.index('Close')

SOURCE_FILES = {
    'nasdaq':   'nvda_d_nasdaq.csv',
    'stooq':    'nvda_d_stooq.csv',
    'yfinance': 'nvda_day_yfinance.csv',
}


def load_source(source, data_dir='../data'):
    filename = SOURCE_FILES[source]
    df = pd.read_csv(f'{data_dir}/{filename}')

    if df.isnull().values.any():
        df = df.dropna()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    return df[TARGET_COLS].copy()


def create_dataset(dataset, n_input, n_forecast):
    """Strategi MIMO: X = window n_input hari x 5 fitur, y = Close n_forecast hari ke depan."""
    X, Y = [], []
    for i in range(n_input, len(dataset) - n_forecast):
        X.append(dataset[i - n_input:i, :])
        Y.append(dataset[i:i + n_forecast, CLOSE_IDX])
    return np.array(X), np.array(Y)


def inverse_close(scaler, scaled_values):
    """Inverse-transform kolom Close dari skala [0,1] -> harga USD asli."""
    samples, horizon = scaled_values.shape
    dummy = np.zeros((samples * horizon, N_FEATURES))
    dummy[:, CLOSE_IDX] = scaled_values.reshape(-1)
    inv = scaler.inverse_transform(dummy)
    return inv[:, CLOSE_IDX].reshape(samples, horizon)


def get_anchor_prices(df, n_input, n_forecast):
    """
    Harga Close aktual (USD) pada hari terakhir sebelum tiap window forecast
    dimulai - dipakai sebagai basis arah pergerakan untuk Directional
    Accuracy (DA): apakah prediksi & aktual sama-sama naik/turun relatif
    terhadap harga terakhir yang diketahui.
    """
    close = df['Close'].values
    anchors = [close[i - 1] for i in range(n_input, len(df) - n_forecast)]
    return np.array(anchors)


def prepare_source(source, n_input=60, n_forecast=5, data_dir='../data'):
    """
    Load + walk-forward split (70/15/15) + Min-Max scaling (fit hanya pada
    train) + pembuatan dataset MIMO, untuk satu sumber data (nasdaq/stooq/
    yfinance). Dipakai baik oleh notebook baseline maupun HHO agar kedua
    model dievaluasi pada representasi data yang identik.
    """
    data = load_source(source, data_dir)

    test_size = int(len(data) * 0.15)
    val_size  = int(len(data) * 0.15)

    test_df  = data.iloc[-test_size:]
    val_df   = data.iloc[-(test_size + val_size):-test_size]
    train_df = data.iloc[:-(test_size + val_size)]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df)

    train_scaled = scaler.transform(train_df)
    val_scaled   = scaler.transform(val_df)
    test_scaled  = scaler.transform(test_df)

    X_train, y_train = create_dataset(train_scaled, n_input, n_forecast)
    X_val,   y_val   = create_dataset(val_scaled,   n_input, n_forecast)
    X_test,  y_test  = create_dataset(test_scaled,  n_input, n_forecast)

    y_train_abs = inverse_close(scaler, y_train)
    y_val_abs   = inverse_close(scaler, y_val)
    y_test_abs  = inverse_close(scaler, y_test)

    val_dates  = val_df.index[n_input: n_input + len(X_val)]
    test_dates = test_df.index[n_input: n_input + len(X_test)]

    anchor_train = get_anchor_prices(train_df, n_input, n_forecast)
    anchor_val   = get_anchor_prices(val_df,   n_input, n_forecast)
    anchor_test  = get_anchor_prices(test_df,  n_input, n_forecast)

    return {
        'source': source,
        'data': data,
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'scaler': scaler,
        'train_scaled': train_scaled, 'val_scaled': val_scaled, 'test_scaled': test_scaled,
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'y_train_abs': y_train_abs,
        'y_val_abs': y_val_abs,
        'y_test_abs': y_test_abs,
        'anchor_train': anchor_train,
        'anchor_val': anchor_val,
        'anchor_test': anchor_test,
        'val_dates': val_dates,
        'test_dates': test_dates,
        'n_input': n_input,
        'n_forecast': n_forecast,
    }
