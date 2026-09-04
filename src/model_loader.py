"""
model_loader.py
Fungsi loading model (.h5) + metadata (.json) dan komputasi prediksi/metrik
untuk keenam kombinasi (3 sumber x 2 arsitektur), dipakai oleh app.py.

PENTING: sejak baseline diretrain ulang, KEDUA arsitektur (baseline & hho)
memakai pipeline preprocessing yang SAMA -- return (percentage change) +
MinMaxScaler + rekonstruksi harga anchor-based (data_utils_return.py).
Perbedaan baseline vs HHO sekarang murni di hyperparameter (default vs hasil
optimasi HHO), bukan di preprocessing. Jangan campur dengan inverse_close/
data_utils.py (versi lama, harga absolut) -- itu sudah tidak dipakai lagi
untuk kedua arsitektur ini.
"""

import os
import json

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

from data_utils_return import prepare_source_return, inverse_return_to_price, load_source, compute_returns
from evaluation import evaluate, directional_accuracy
from live_data import get_live_or_fallback, merge_with_history

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

SOURCES = ['yfinance', 'nasdaq', 'stooq']
SOURCE_LABELS = {'yfinance': 'Yahoo Finance', 'nasdaq': 'Nasdaq', 'stooq': 'Stooq'}
ARCHITECTURES = ['baseline', 'hho']
ARCH_LABELS = {'baseline': 'LSTM MIMO Standar', 'hho': 'LSTM MIMO + HHO'}

MODEL_FILENAME = {
    'baseline': 'lstm_mimo_baseline_{source}.h5',
    'hho': 'lstm_mimo_hho_{source}.h5',
}
META_FILENAME = {
    'baseline': 'lstm_mimo_baseline_{source}_meta.json',
    'hho': 'lstm_mimo_hho_{source}_meta.json',
}


@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load keenam model (.h5) + metadata (.json) ke memori, sekali saja."""
    registry = {}
    for source in SOURCES:
        for arch in ARCHITECTURES:
            model_path = os.path.join(MODELS_DIR, MODEL_FILENAME[arch].format(source=source))
            meta_path = os.path.join(MODELS_DIR, META_FILENAME[arch].format(source=source))
            entry = {"model": None, "meta": None, "error": None}
            try:
                entry["model"] = load_model(model_path, compile=False)
                with open(meta_path) as f:
                    entry["meta"] = json.load(f)
            except Exception as e:
                entry["error"] = str(e)
            registry[(source, arch)] = entry
    return registry


@st.cache_data(show_spinner=False)
def load_prepared_source(source: str, arch: str, n_input: int = 60, n_forecast: int = 5):
    """Load & preprocess data untuk satu sumber. `arch` dipertahankan sebagai
    parameter (bagian cache key & API compatibility dengan app.py), tapi kedua
    arch sekarang pakai pipeline return-based yang sama persis."""
    return prepare_source_return(source, n_input=n_input, n_forecast=n_forecast,
                                  data_dir=DATA_DIR)


def predict_all_splits(model, ds, arch: str):
    """Prediksi train/val/test dan inverse-transform (anchor-based) ke harga asli.
    `arch` tidak lagi mengubah cara inverse -- disimpan untuk API compatibility."""
    n_forecast = ds['n_forecast']
    scaler = ds['scaler']

    train_pred = inverse_return_to_price(scaler, model.predict(ds['X_train'], verbose=0),
                                          ds['base_train'], n_forecast)
    val_pred = inverse_return_to_price(scaler, model.predict(ds['X_val'], verbose=0),
                                        ds['base_val'], n_forecast)
    test_pred = inverse_return_to_price(scaler, model.predict(ds['X_test'], verbose=0),
                                         ds['base_test'], n_forecast)

    return {
        'TRAIN': (ds['y_train_abs'], train_pred, ds['base_train']),
        'VAL': (ds['y_val_abs'], val_pred, ds['base_val']),
        'TEST': (ds['y_test_abs'], test_pred, ds['base_test']),
    }


def metrics_table(splits: dict, n_forecast: int):
    """Hitung RMSE/MAE/MAPE/Directional Accuracy per horizon + rata-rata
    untuk tiap subset. Return list of rows.

    `splits` : dict {"TRAIN": (y_true_abs, y_pred_abs, base_prices), ...}
    """
    rows = []
    for split_label, split_val in splits.items():
        y_true_abs, y_pred_abs, base = split_val
        for h in range(n_forecast):
            _, rmse, mae, mape = evaluate(y_true_abs[:, h], y_pred_abs[:, h])
            da = directional_accuracy(y_true_abs[:, h], y_pred_abs[:, h], base)
            rows.append({
                'Split': split_label, 'Horizon': f't+{h+1}',
                'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'DA (%)': da,
            })
        _, rmse_a, mae_a, mape_a = evaluate(y_true_abs.flatten(), y_pred_abs.flatten())
        base_rep = np.repeat(base, n_forecast)
        da_a = directional_accuracy(y_true_abs.flatten(), y_pred_abs.flatten(), base_rep)
        rows.append({
            'Split': split_label, 'Horizon': 'Average',
            'RMSE': rmse_a, 'MAE': mae_a, 'MAPE (%)': mape_a, 'DA (%)': da_a,
        })
    return rows


def forecast_next(model, ds, arch: str):
    """Prediksi n_forecast hari ke depan dari titik data terakhir di TEST SET
    (statis, dari CSV historis -- bukan live). Dipertahankan untuk kompatibilitas;
    Tab 1 dashboard sekarang memakai forecast_next_live di bawah.
    `arch` tidak lagi mengubah cara inverse -- disimpan untuk API compatibility."""
    n_input = ds['n_input']
    n_forecast = ds['n_forecast']
    scaler = ds['scaler']

    test_scaled = ds['test_scaled']  # sudah di-scale saat prepare_source_return
    last_seq = test_scaled[-n_input:].reshape(1, n_input, test_scaled.shape[1])
    pred_scaled = model.predict(last_seq, verbose=0)
    last_price = float(ds['close_abs'].iloc[-1])
    base_future = np.array([last_price])
    pred_abs = inverse_return_to_price(scaler, pred_scaled, base_future, n_forecast)[0]
    return pred_abs


@st.cache_data(ttl=300, show_spinner=False)
def get_live_source_data(source: str):
    """Live-or-fallback OHLCV untuk satu sumber, digabung dengan histori CSV
    (perlu histori supaya pct_change tetap kontinu jauh ke belakang, bukan
    cuma dari beberapa hari live saja -- lihat live_data.merge_with_history).

    Cache 5 menit: cukup segar untuk dashboard "live", tapi tidak membebani
    endpoint tidak resmi (stooq/nasdaq) di setiap rerun Streamlit.

    Returns dict: {"data": DataFrame OHLCV gabungan, "is_live": bool,
                   "fetched_at": datetime|None, "error": str|None}
    """
    historical_df = load_source(source, DATA_DIR)
    live_result = get_live_or_fallback(source, historical_df)
    if live_result['is_live']:
        combined = merge_with_history(historical_df, live_result['data'])
    else:
        combined = historical_df
    return {
        'data': combined,
        'is_live': live_result['is_live'],
        'fetched_at': live_result['fetched_at'],
        'error': live_result['error'],
    }


def forecast_next_live(model, ds, source: str):
    """Prediksi n_forecast hari ke depan dari data LIVE (fallback ke histori
    CSV kalau live gagal/diblokir -- lihat get_live_source_data).

    `ds` (hasil load_prepared_source) dipakai untuk `scaler` -- scaler TIDAK
    di-refit di sini, tetap scaler yang sama yang di-fit hanya pada train saat
    training (menghindari data leakage) -- hanya di-transform ke data return
    terbaru. `ds['data']` TIDAK dipakai di sini (itu histori CSV statis);
    histori dimuat ulang di dalam get_live_source_data supaya cache-nya per-
    source, independen dari cache load_prepared_source.

    Returns dict: {"pred": np.ndarray|None, "last_price": float|None,
                   "last_date": Timestamp|None, "is_live": bool,
                   "fetched_at": datetime|None, "error": str|None}
    """
    n_input = ds['n_input']
    n_forecast = ds['n_forecast']
    scaler = ds['scaler']

    live = get_live_source_data(source)
    data_ret, close_abs = compute_returns(live['data'])

    if len(data_ret) < n_input:
        return {
            'pred': None, 'last_price': None, 'last_date': None,
            'is_live': live['is_live'], 'fetched_at': live['fetched_at'],
            'error': (f"Data return tersedia hanya {len(data_ret)} baris, "
                      f"butuh minimal {n_input} untuk window input."),
        }

    scaled = scaler.transform(data_ret)
    last_seq = scaled[-n_input:].reshape(1, n_input, scaled.shape[1])
    pred_scaled = model.predict(last_seq, verbose=0)

    last_price = float(close_abs.iloc[-1])
    last_date = close_abs.index[-1]
    pred_abs = inverse_return_to_price(scaler, pred_scaled, np.array([last_price]), n_forecast)[0]

    return {
        'pred': pred_abs,
        'last_price': last_price,
        'last_date': last_date,
        'is_live': live['is_live'],
        'fetched_at': live['fetched_at'],
        'error': live['error'],
    }
