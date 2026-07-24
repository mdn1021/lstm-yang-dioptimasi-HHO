"""
model_loader.py
Fungsi loading model (.h5) + metadata (.json) dan komputasi prediksi/metrik
untuk keenam kombinasi (3 sumber x 2 arsitektur), dipakai oleh app.py.
"""

import os
import json
import sys

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_utils import prepare_source, inverse_close  # noqa: E402
from evaluation import evaluate  # noqa: E402

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
    """Load keenam model (.h5) + metadata (.json) ke memori, sekali saja.

    Returns
    -------
    dict[(source, arch)] -> {"model": keras.Model, "meta": dict, "error": str|None}
    """
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
def load_prepared_source(source: str, n_input: int = 60, n_forecast: int = 5):
    """Load & preprocess data untuk satu sumber (split, scaler, windowing)."""
    return prepare_source(source, n_input=n_input, n_forecast=n_forecast, data_dir=DATA_DIR)


def predict_all_splits(model, ds):
    """Prediksi train/val/test dan inverse-transform ke skala harga asli."""
    scaler = ds['scaler']
    train_pred = inverse_close(scaler, model.predict(ds['X_train'], verbose=0))
    val_pred = inverse_close(scaler, model.predict(ds['X_val'], verbose=0))
    test_pred = inverse_close(scaler, model.predict(ds['X_test'], verbose=0))
    return {
        'TRAIN': (ds['y_train_abs'], train_pred),
        'VAL': (ds['y_val_abs'], val_pred),
        'TEST': (ds['y_test_abs'], test_pred),
    }


def metrics_table(splits: dict, n_forecast: int):
    """Hitung RMSE/MAE/MAPE per horizon + rata-rata untuk tiap subset. Return list of rows."""
    rows = []
    for split_label, (y_true_abs, y_pred_abs) in splits.items():
        for h in range(n_forecast):
            _, rmse, mae, mape = evaluate(y_true_abs[:, h], y_pred_abs[:, h])
            rows.append({
                'Split': split_label, 'Horizon': f't+{h+1}',
                'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape,
            })
        _, rmse_a, mae_a, mape_a = evaluate(y_true_abs.flatten(), y_pred_abs.flatten())
        rows.append({
            'Split': split_label, 'Horizon': 'Average',
            'RMSE': rmse_a, 'MAE': mae_a, 'MAPE (%)': mape_a,
        })
    return rows


def forecast_next(model, ds):
    """Prediksi n_forecast hari ke depan dari titik data terakhir."""
    n_input = ds['n_input']
    scaler = ds['scaler']
    last_seq = None
    # Gunakan data test terakhir (sudah di-scale) sebagai basis window terakhir
    test_scaled = scaler.transform(ds['test_df'])
    last_seq = test_scaled[-n_input:].reshape(1, n_input, test_scaled.shape[1])
    pred_scaled = model.predict(last_seq, verbose=0)
    pred_abs = inverse_close(scaler, pred_scaled)[0]
    return pred_abs
