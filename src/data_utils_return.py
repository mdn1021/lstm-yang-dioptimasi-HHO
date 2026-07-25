"""
data_utils_return.py
Preprocessing bersama berbasis percentage-return, dipakai oleh baseline
(_return) dan HHO untuk KEDUA arsitektur, agar preprocessing identik.

Kenapa return, bukan harga absolut langsung (lihat data_utils.py)?
NVDA mengalami price jump besar (rally AI 2024-2025): test set berada
~60% di atas rentang harga yang dilihat MinMaxScaler saat fit pada train
(train Close max ~$135, test Close sampai ~$217). MinMaxScaler pada harga
absolut tidak bisa mengekstrapolasi rentang sejauh itu (input test ter-scale
sampai ~1.65, padahal model hanya pernah melihat [0,1] saat training).
Percentage change (return harian) tidak punya masalah ini karena skalanya
konsisten terlepas dari level harga absolut.

Rekonstruksi harga dari return memakai ANCHOR TETAP (harga Close aktual
hari terakhir sebelum window forecast) untuk semua horizon t+1..t+n -
BUKAN 'prev = harga hasil prediksi horizon sebelumnya'. Anchor tetap
mencegah error horizon awal ikut membiaskan basis horizon berikutnya
(compounding error antar horizon dalam satu window).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TARGET_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
PRICE_COLS  = ['Open', 'High', 'Low', 'Close']
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


def compute_returns(data):
    """
    OHLC -> percentage change harian (%); Volume -> log-return (skalanya
    jauh berbeda dari harga). Baris pertama (NaN dari pct_change) dibuang.

    Returns
    -------
    data_ret  : DataFrame return per fitur, index bergeser +1 dari `data`
    close_abs : Series harga Close absolut, index selaras dengan data_ret
    """
    data_ret = pd.DataFrame(index=data.index)
    for col in PRICE_COLS:
        data_ret[col] = data[col].pct_change() * 100
    data_ret['Volume'] = np.log1p(data['Volume'].pct_change() * 100 + 100)

    data_ret = data_ret.dropna()
    data_ret = data_ret.replace([np.inf, -np.inf], np.nan).dropna()

    close_abs = data['Close'].loc[data_ret.index].copy()
    return data_ret[TARGET_COLS], close_abs


def walk_forward_split_ret(data_ret, close_abs):
    """Split time-ordered 70/15/15 pada data return + harga absolut selaras."""
    test_size = int(len(data_ret) * 0.15)
    val_size  = int(len(data_ret) * 0.15)

    test_ret  = data_ret.iloc[-test_size:]
    val_ret   = data_ret.iloc[-(test_size + val_size):-test_size]
    train_ret = data_ret.iloc[:-(test_size + val_size)]

    test_close  = close_abs.iloc[-test_size:]
    val_close   = close_abs.iloc[-(test_size + val_size):-test_size]
    train_close = close_abs.iloc[:-(test_size + val_size)]

    return (train_ret, val_ret, test_ret), (train_close, val_close, test_close)


def create_dataset(dataset, n_input, n_forecast):
    """Strategi MIMO: X = window n_input hari x 5 fitur return, y = Close return n_forecast hari ke depan."""
    X, Y = [], []
    for i in range(n_input, len(dataset) - n_forecast):
        X.append(dataset[i - n_input:i, :])
        Y.append(dataset[i:i + n_forecast, CLOSE_IDX])
    return np.array(X), np.array(Y)


def get_anchor_prices(close_abs_series, ret_index, n_input, n_forecast):
    """Harga Close absolut pada hari terakhir sebelum tiap window forecast dimulai."""
    anchors = []
    for i in range(n_input, len(ret_index) - n_forecast):
        last_date = ret_index[i - 1]
        anchors.append(close_abs_series.loc[last_date])
    return np.array(anchors)


def get_true_prices(close_abs_series, ret_index, n_input, n_forecast):
    """Harga Close absolut aktual untuk n_forecast hari setelah tiap window."""
    prices = []
    for i in range(n_input, len(ret_index) - n_forecast):
        future_dates = ret_index[i:i + n_forecast]
        prices.append(close_abs_series.loc[future_dates].values)
    return np.array(prices)


def inverse_return_to_price(scaler, pred_scaled, base_prices, n_forecast):
    """
    Konversi prediksi return (scaled) -> harga absolut (USD).

    1. Inverse scaler -> return % per hari
    2. Rekonstruksi harga via cumulative log-return dari ANCHOR TETAP:
       price[t+k] = base * exp(sum(log_return[0..k]))
       (base_prices tidak pernah diganti dengan hasil prediksi horizon
       sebelumnya, agar error tidak berantai / compounding antar horizon)
    """
    samples = pred_scaled.shape[0]

    dummy = np.zeros((samples * n_forecast, N_FEATURES))
    dummy[:, CLOSE_IDX] = pred_scaled.reshape(-1)
    inv = scaler.inverse_transform(dummy)
    returns_pct = inv[:, CLOSE_IDX].reshape(samples, n_forecast)

    prices = np.zeros_like(returns_pct)
    for s in range(samples):
        base = base_prices[s]
        log_returns = np.log1p(returns_pct[s] / 100)
        for k in range(n_forecast):
            cum_log_return = np.sum(log_returns[:k + 1])
            prices[s, k] = base * np.exp(cum_log_return)
    return prices


def prepare_source_return(source, n_input=60, n_forecast=5, data_dir='../data'):
    """
    Load + transformasi ke return + walk-forward split (70/15/15) + Min-Max
    scaling (fit hanya pada train) + pembuatan dataset MIMO, untuk satu
    sumber data. Dipakai baik oleh notebook baseline (_return) maupun HHO
    agar kedua model dievaluasi pada representasi data yang identik.
    """
    data = load_source(source, data_dir)
    data_ret, close_abs = compute_returns(data)

    (train_ret, val_ret, test_ret), (train_close, val_close, test_close) = \
        walk_forward_split_ret(data_ret, close_abs)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_ret)

    train_scaled = scaler.transform(train_ret)
    val_scaled   = scaler.transform(val_ret)
    test_scaled  = scaler.transform(test_ret)

    X_train, y_train = create_dataset(train_scaled, n_input, n_forecast)
    X_val,   y_val   = create_dataset(val_scaled,   n_input, n_forecast)
    X_test,  y_test  = create_dataset(test_scaled,  n_input, n_forecast)

    base_train = get_anchor_prices(close_abs, train_ret.index, n_input, n_forecast)
    base_val   = get_anchor_prices(close_abs, val_ret.index,   n_input, n_forecast)
    base_test  = get_anchor_prices(close_abs, test_ret.index,  n_input, n_forecast)

    y_train_abs = get_true_prices(close_abs, train_ret.index, n_input, n_forecast)
    y_val_abs   = get_true_prices(close_abs, val_ret.index,   n_input, n_forecast)
    y_test_abs  = get_true_prices(close_abs, test_ret.index,  n_input, n_forecast)

    val_dates  = val_ret.index[n_input: n_input + len(X_val)]
    test_dates = test_ret.index[n_input: n_input + len(X_test)]

    train_df = data.loc[train_ret.index]
    val_df   = data.loc[val_ret.index]
    test_df  = data.loc[test_ret.index]

    return {
        'source': source,
        'data': data,
        'data_ret': data_ret,
        'close_abs': close_abs,
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'train_ret': train_ret, 'val_ret': val_ret, 'test_ret': test_ret,
        'scaler': scaler,
        'train_scaled': train_scaled, 'val_scaled': val_scaled, 'test_scaled': test_scaled,
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'y_train_abs': y_train_abs,
        'y_val_abs': y_val_abs,
        'y_test_abs': y_test_abs,
        'base_train': base_train,
        'base_val': base_val,
        'base_test': base_test,
        'val_dates': val_dates,
        'test_dates': test_dates,
        'n_input': n_input,
        'n_forecast': n_forecast,
    }
