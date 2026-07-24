# lstm-yang-dioptimasi-HHO

# NVDA Stock Prediction: LSTM MIMO Standar vs LSTM MIMO + HHO

Skripsi project: dashboard web untuk membandingkan performa prediksi harga saham
NVIDIA (NVDA) multi-hari (Multi-Step MIMO) antara model LSTM standar (baseline)
dan LSTM yang dioptimasi dengan Harris Hawks Optimization (HHO).

## Struktur Folder

```
lstm-yang-dioptimasi-HHO/
├── data/                      # Dataset mentah per sumber + hasil integrasi
│   ├── nvda_d_nasdaq.csv
│   ├── nvda_d_stooq.csv
│   ├── nvda_day_yfinance.csv
│   └── nvda_gabungan.csv
├── notebooks/
│   ├── 01_lstm_mimo_baseline_nas.ipynb     # Model 1 (Nasdaq)  - hyperparameter default
│   ├── 01_lstm_mimo_baseline _stooq.ipynb  # Model 1 (Stooq)   - hyperparameter default
│   ├── 01_lstm_mimo_baseline_yfi.ipynb     # Model 1 (Yahoo Finance) - hyperparameter default
│   ├── 02_lstm_mimo_hho_nasdaq new.ipynb   # Model 2 (Nasdaq)  - hyperparameter dioptimasi HHO
│   ├── 02_lstm_mimo_hho_stooq new.ipynb    # Model 2 (Stooq)   - hyperparameter dioptimasi HHO
│   ├── 02_lstm_mimo_hho_yfinance new.ipynb # Model 2 (Yahoo Finance) - hyperparameter dioptimasi HHO
│   └── (varian lain / eksperimen return-normalization: DEPRECATED, lihat Catatan Penting)
├── src/                       # Modul Python bersama (dipakai oleh notebook baseline & HHO)
│   ├── data_utils.py          # load + walk-forward split + Min-Max scaling + dataset MIMO
│   ├── model_utils.py         # arsitektur LSTM MIMO + decode hyperparameter HHO
│   ├── hho_utils.py           # algoritma Harris Hawks Optimization + objective function
│   ├── evaluation.py          # RMSE/MAE/MAPE per horizon + tabel ringkasan
│   └── solution.py            # kelas pelacak hasil HHO (best individual, convergence, dll)
├── models/                    # Model terlatih (.h5) + metadata JSON (n_input, n_forecast, dll)
├── dashboard/                 # Aplikasi Streamlit (sesuai PRD v1.1)
└── docs/                      # BAB 3, PRD, dan dokumen pendukung lainnya
```

## Pipeline Ringkas (BAB 3 & PRD v1.1)

1. **Data**: OHLCV NVDA per sumber (Yahoo Finance, Nasdaq, Stooq), dievaluasi terpisah
   -> `data/nvda_d_nasdaq.csv`, `data/nvda_d_stooq.csv`, `data/nvda_day_yfinance.csv`
2. **Preprocessing** (`src/data_utils.py`, dipakai bersama oleh kedua model): Split
   time-ordered 70/15/15 (SEBELUM scaling) -> Min-Max Normalization per fitur langsung
   pada OHLCV (fit hanya pada train) -> Sliding window MIMO (n_input=60, n_forecast=5)
3. **Model 1 (Baseline)**: `notebooks/01_lstm_mimo_baseline_{nas,stooq,yfi}.ipynb`
   - Hyperparameter default: LSTM(128) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2) -> Dense(5)
4. **Model 2 (LSTM+HHO)**: `notebooks/02_lstm_mimo_hho_{nasdaq,stooq,yfinance} new.ipynb`
   - Hyperparameter dioptimasi HHO: units, dropout, learning rate, batch size (6 dimensi)
   - Memakai `src/data_utils.py` yang SAMA dengan baseline, agar fair comparison
5. **Evaluasi** (`src/evaluation.py`): RMSE, MAE, MAPE, **dan DA (Directional Accuracy)**
   per horizon (t+1..t+5) dan rata-rata per subset (train/val/test)
6. **Dashboard**: `dashboard/` (Streamlit) menampilkan proyeksi, perbandingan performa, dan riwayat data

## Kriteria Keberhasilan (BAB 3.6.3)
- MAPE test < 10% untuk kedua model, dengan LSTM+HHO lebih rendah
- RMSE & MAE LSTM+HHO lebih kecil dari baseline pada seluruh horizon
- DA (Directional Accuracy) LSTM+HHO lebih tinggi dari baseline
- Rasio generalisasi (test/train) LSTM+HHO lebih mendekati 1 dibanding baseline

## Catatan Penting

- **Normalisasi**: kedua notebook menggunakan Min-Max Normalization langsung pada
  OHLCV mentah (BUKAN percentage-change/return), sesuai keputusan final & BAB 3.4.2-3.4.3.
  Split & scaling dilakukan lewat `src/data_utils.py` yang SAMA persis untuk baseline
  maupun HHO, per sumber data — bukan diimplementasikan ulang secara terpisah di tiap
  notebook — agar tidak ada celah preprocessing berbeda di antara kedua model.
- **n_input & n_forecast**: ditetapkan tetap (60 hari / 5 hari) di kedua model untuk
  menjaga fair comparison dan stabilitas evaluasi. HHO difokuskan mengoptimasi
  hyperparameter arsitektur (units, dropout, learning rate, batch size) — bukan window/horizon.
- **DA (Directional Accuracy)**: dihitung per horizon sebagai persentase sampel di mana
  arah pergerakan prediksi (naik/turun relatif terhadap harga Close terakhir yang
  diketahui sebelum window forecast) sama dengan arah pergerakan aktual
  (`src/evaluation.py::directional_accuracy`).
- **Riwayat bug (sudah diperbaiki)**: notebook HHO versi lama (`02_lstm_mimo_hho_nasdaq.ipynb`,
  `02_lstm_mimo_hho_stooq.ipynb`, `02_lstm_mimo_hho_yfinance.ipynb`, serta seluruh varian
  `*_return.ipynb`) memakai normalisasi **percentage-change/return** yang direkonstruksi
  ke harga lewat cumulative compounding, SEDANGKAN baseline memakai normalisasi harga
  mentah langsung. Ini bukan perbandingan yang adil — return harian jauh lebih sulit
  diprediksi (mendekati random walk) dan compounding error-nya membesar tiap horizon,
  sehingga model HHO tampak jauh lebih buruk daripada baseline padahal bukan itu
  penyebab sebenarnya. Notebook-notebook tersebut **DEPRECATED**, dipertahankan hanya
  untuk riwayat eksperimen. Pipeline yang berlaku sekarang: `01_lstm_mimo_baseline_*.ipynb`
  + `02_lstm_mimo_hho_*_new.ipynb`, keduanya memakai `src/data_utils.py` yang identik.
