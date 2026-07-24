"""
model_utils.py
Arsitektur LSTM MIMO bersama, dipakai baik oleh model baseline (hyperparameter
default) maupun model HHO (hyperparameter dioptimasi) - hanya nilai
hyperparameter yang berbeda, arsitektur & training regime identik.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

DEFAULT_EPOCHS   = 100
DEFAULT_PATIENCE = 10


def decode_hyperparameters(hp):
    """Decode vektor kontinu HHO (6 dimensi) -> hyperparameter LSTM MIMO."""
    units1        = int(np.clip(round(hp[0]), 32, 256))
    units2        = int(np.clip(round(hp[1]), 32, 256))
    dropout_rate1 = float(np.clip(hp[2], 0.1, 0.5))
    dropout_rate2 = float(np.clip(hp[3], 0.1, 0.5))
    learning_rate = float(np.clip(hp[4], 1e-4, 1e-2))
    batch_size    = int(np.clip(round(hp[5]), 16, 128))
    return units1, units2, dropout_rate1, dropout_rate2, learning_rate, batch_size


def build_lstm_mimo(n_input, n_features, n_forecast, units1, units2, drop1, drop2, learning_rate):
    model = Sequential([
        LSTM(units1, return_sequences=True, input_shape=(n_input, n_features)),
        Dropout(drop1),
        LSTM(units2, return_sequences=False),
        Dropout(drop2),
        Dense(n_forecast)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model
