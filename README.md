# lstm-yang-dioptimasi-HHO

# NVDA Stock Prediction: LSTM MIMO Standar vs LSTM MIMO + HHO

Skripsi project: dashboard web untuk membandingkan performa prediksi harga saham
NVIDIA (NVDA) multi-hari (Multi-Step MIMO) antara model LSTM standar (baseline)
dan LSTM yang dioptimasi dengan Harris Hawks Optimization (HHO).

## Struktur Folder

```
nvda-lstm-hho-dashboard/
├── data/                     # Dataset mentah & hasil integrasi (nvda_gabungan.csv)
├── notebooks/
│   ├── 01_lstm_mimo_baseline.ipynb   # Model 1: LSTM MIMO Standar (default hyperparameter)
│   └── 02_lstm_mimo_hho.ipynb        # Model 2: LSTM MIMO + HHO (hyperparameter dioptimasi)
├── models/                   # Model terlatih (.h5) + metadata JSON (n_input, n_forecast, dll)
├── src/                      # Kode Python reusable (preprocessing, evaluasi, HHO) untuk dashboard
├── dashboard/                # Aplikasi Streamlit (sesuai PRD v1.1)
└── docs/                     # BAB 3, PRD, dan dokumen pendukung lainnya
```

## Pipeline Ringkas (BAB 3 & PRD v1.1)

1. **Data**: OHLCV NVDA gabungan dari Yahoo Finance, Nasdaq, Stooq -> `data/nvda_gabungan.csv`
2. **Preprocessing**: Split time-ordered 70/15/15 (SEBELUM scaling) -> Min-Max Normalization
   per fitur (fit hanya pada train) -> Sliding window MIMO (n_input=60, n_forecast=5)
3. **Model 1 (Baseline)**: `notebooks/01_lstm_mimo_baseline.ipynb`
   - Hyperparameter default: LSTM(128) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2) -> Dense(5)
4. **Model 2 (LSTM+HHO)**: `notebooks/02_lstm_mimo_hho.ipynb`
   - Hyperparameter dioptimasi HHO: units, dropout, learning rate, batch size (6 dimensi)
5. **Evaluasi**: RMSE, MAE, MAPE per horizon (t+1..t+5) dan rata-rata per subset (train/val/test)
6. **Dashboard**: `dashboard/` (Streamlit) menampilkan proyeksi, perbandingan performa, dan riwayat data

## Kriteria Keberhasilan (BAB 3.6.3)
- MAPE test < 10% untuk kedua model, dengan LSTM+HHO lebih rendah
- RMSE & MAE LSTM+HHO lebih kecil dari baseline pada seluruh horizon
- Rasio generalisasi (test/train) LSTM+HHO lebih mendekati 1 dibanding baseline

## Catatan Penting
- **Normalisasi**: kedua notebook menggunakan Min-Max Normalization langsung pada
  OHLCV mentah (BUKAN percentage-change/return), sesuai keputusan final & BAB 3.4.2-3.4.3.
- **n_input & n_forecast**: ditetapkan tetap (60 hari / 5 hari) di kedua model untuk
  menjaga fair comparison dan stabilitas evaluasi. HHO difokuskan mengoptimasi
  hyperparameter arsitektur (units, dropout, learning rate, batch size) — bukan window/horizon.
