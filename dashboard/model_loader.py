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
import sys

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_utils_return import prepare_source_return, inverse_return_to_price  # noqa: E402
from evaluation import evaluate, directional_accuracy  # noqa: E402

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
    """Prediksi n_forecast hari ke depan dari titik data terakhir (anchor-based).
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
