# lstm-yang-dioptimasi-HHO

# NVDA Stock Prediction: LSTM MIMO Standar vs LSTM MIMO + HHO

Skripsi project: dashboard web untuk membandingkan performa prediksi harga saham
NVIDIA (NVDA) multi-hari (Multi-Step MIMO) antara model LSTM standar (baseline)
dan LSTM yang dioptimasi dengan Harris Hawks Optimization (HHO).

## Struktur Folder

```
lstm-yang-dioptimasi-HHO/
├── README.md
├── requirements.txt            # streamlit, tensorflow, scikit-learn, pandas, numpy, plotly
├── .gitignore
├── .streamlit/
│   └── config.toml             # tema dashboard (dark, aksen NVIDIA green)
├── app.py                      # CANONICAL entry point -> `streamlit run app.py` (dari root)
├── src/                        # Modul Python bersama (dipakai notebook training DAN app.py)
│   ├── data_utils_return.py    # CANONICAL: load + return transform + split + scale + dataset MIMO
│   ├── model_loader.py         # load model/metadata + prediksi live + metrik, dipakai app.py
│   ├── model_utils.py          # arsitektur LSTM MIMO + decode hyperparameter HHO
│   ├── hho_utils.py            # algoritma Harris Hawks Optimization + objective function
│   ├── evaluation.py           # RMSE/MAE/MAPE/DA per horizon + tabel ringkasan
│   ├── solution.py             # kelas pelacak hasil HHO (best individual, convergence, dll)
│   ├── live_data.py            # fetch OHLCV live (yfinance/nasdaq/stooq) + fallback ke CSV
│   └── deprecated/             # data_utils.py (versi raw-price) — lihat Catatan Penting
├── notebooks/
│   ├── 01_lstm_mimo_baseline_nasdaq_return .ipynb   # Model 1 (Nasdaq)  - CANONICAL
│   ├── 01_lstm_mimo_baseline_stooq_return.ipynb     # Model 1 (Stooq)   - CANONICAL
│   ├── 01_lstm_mimo_baseline_yfinance_return.ipynb  # Model 1 (Yahoo Finance) - CANONICAL
│   ├── 02_lstm_mimo_hho_nasdaq_return.ipynb         # Model 2 (Nasdaq)  - CANONICAL
│   ├── 02_lstm_mimo_hho_stooq_return.ipynb          # Model 2 (Stooq)   - CANONICAL
│   ├── 02_lstm_mimo_hho_yfinance_return.ipynb       # Model 2 (Yahoo Finance) - CANONICAL
│   └── deprecated/            # raw-price & eksperimen awal — lihat Catatan Penting
├── models/                     # Model terlatih (.h5) + metadata JSON (n_input, n_forecast, dll)
├── data/                       # Dataset mentah per sumber + hasil integrasi
│   ├── nvda_d_nasdaq.csv
│   ├── nvda_d_stooq.csv
│   ├── nvda_day_yfinance.csv
│   └── nvda_gabungan.csv
├── results/                    # Grafik & tabel metrik hasil evaluasi (untuk laporan/sidang)
└── archive/                    # Versi lama dashboard yang sudah tidak dipakai — lihat Catatan Penting
    ├── dashboard_app.py
    ├── dashboard_app2.py
    ├── model_loader_old/
    └── model_loader_return_wip/
```

## Pipeline Ringkas (BAB 3 & PRD v1.1)

1. **Data**: OHLCV NVDA per sumber (Yahoo Finance, Nasdaq, Stooq), dievaluasi terpisah
   -> `data/nvda_d_nasdaq.csv`, `data/nvda_d_stooq.csv`, `data/nvda_day_yfinance.csv`
2. **Preprocessing** (`src/data_utils_return.py`, dipakai bersama oleh baseline, HHO, DAN
   dashboard): Transformasi OHLC -> percentage change harian (Volume -> log-return) ->
   split time-ordered 70/15/15 (SEBELUM scaling) -> Min-Max Normalization per fitur pada
   return (fit hanya pada train) -> Sliding window MIMO (n_input=60, n_forecast=5) ->
   rekonstruksi harga USD lewat `inverse_return_to_price` (anchor tetap = harga Close
   terakhir sebelum window forecast, TANPA compounding antar horizon).
3. **Model 1 (Baseline)**: `notebooks/01_lstm_mimo_baseline_{nasdaq,stooq,yfinance}_return.ipynb`
   - Hyperparameter default: LSTM(128) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2) -> Dense(5)
4. **Model 2 (LSTM+HHO)**: `notebooks/02_lstm_mimo_hho_{nasdaq,stooq,yfinance}_return.ipynb`
   - Hyperparameter dioptimasi HHO: units, dropout, learning rate, batch size (6 dimensi)
   - Memakai `src/data_utils_return.py` yang SAMA dengan baseline, agar fair comparison
   - Fitness HHO = **RMSE validasi pada skala harga hasil rekonstruksi (USD)**, bukan
     MSE return ternormalisasi — lihat Catatan Penting
5. **Evaluasi** (`src/evaluation.py`): RMSE, MAE, MAPE, **dan DA (Directional Accuracy)**
   per horizon (t+1..t+5) dan rata-rata per subset (train/val/test)
6. **Dashboard**: `app.py` (Streamlit, jalankan dari root) menampilkan proyeksi, perbandingan
   performa (termasuk DA), dan riwayat data, membaca model dari `models/` lewat
   `src/model_loader.py`. Tab "Proyeksi Multi-Hari" mengambil harga **live**
   (`src/live_data.py`) sebagai basis prediksi — bukan cuma titik terakhir test set
   statis — dengan badge status 🟢 (live berhasil) / 🟡 (fallback ke CSV historis)
   dan tombol refresh manual di sidebar.

## Kriteria Keberhasilan (BAB 3.6.3)
- MAPE test < 10% untuk kedua model, dengan LSTM+HHO lebih rendah
- RMSE & MAE LSTM+HHO lebih kecil dari baseline pada seluruh horizon
- DA (Directional Accuracy) LSTM+HHO lebih tinggi dari baseline
- Rasio generalisasi (test/train) LSTM+HHO lebih mendekati 1 dibanding baseline

## Catatan Penting

- **Kenapa normalisasi return, bukan harga absolut langsung?** NVDA mengalami rally
  harga sangat besar (2024-2025): test set berada **~60% di atas** rentang harga yang
  dilihat scaler saat fit pada train (train Close max ~$135, test Close sampai ~$217).
  MinMaxScaler pada harga absolut TIDAK BISA mengekstrapolasi rentang sejauh itu (input
  test ter-scale sampai ~1.65, padahal model hanya pernah melihat [0,1] saat training).
  Percentage change (return harian) tidak punya masalah ini karena skalanya konsisten
  terlepas dari level harga absolut. Split & scaling dilakukan lewat
  `src/data_utils_return.py` yang SAMA persis untuk baseline maupun HHO, per sumber data
  — bukan diimplementasikan ulang secara terpisah di tiap notebook — agar tidak ada
  celah preprocessing berbeda di antara kedua model.
- **n_input & n_forecast**: ditetapkan tetap (60 hari / 5 hari) di kedua model untuk
  menjaga fair comparison dan stabilitas evaluasi. HHO difokuskan mengoptimasi
  hyperparameter arsitektur (units, dropout, learning rate, batch size) — bukan window/horizon.
- **Fitness HHO = RMSE harga hasil rekonstruksi, bukan val_loss return**: val_loss
  (MSE pada skala return ternormalisasi) TIDAK otomatis selaras dengan RMSE harga hasil
  rekonstruksi, karena `inverse_return_to_price` mengakumulasi return lewat cumulative
  log-return — kandidat hyperparameter bisa punya val_loss bagus di skala return tapi
  merekonstruksi buruk di skala harga. `src/hho_utils.py::make_objective_function`
  mendukung mode `reconstruct_price` yang menghitung fitness langsung di skala harga USD,
  dipakai oleh seluruh notebook `02_lstm_mimo_hho_*_return.ipynb` — supaya HHO benar-benar
  mengoptimasi kriteria yang dipakai untuk membandingkan model (BAB 3.6.3), bukan proxy
  yang bisa tidak selaras.
- **DA (Directional Accuracy)**: dihitung per horizon sebagai persentase sampel di mana
  arah pergerakan prediksi (naik/turun relatif terhadap harga Close terakhir yang
  diketahui sebelum window forecast / anchor) sama dengan arah pergerakan aktual
  (`src/evaluation.py::directional_accuracy`).
- **Riwayat bug (root cause, sudah diperbaiki)**: dashboard sempat menampilkan LSTM+HHO
  dengan RMSE/MAE/MAPE jauh lebih tinggi dari baseline di semua sumber data — kebalikan
  dari kriteria keberhasilan. Penyebabnya bertingkat:
  1. Baseline & HHO memakai preprocessing yang BERBEDA (baseline: harga absolut langsung;
     HHO: return) — bukan perbandingan yang adil.
  2. Bahkan setelah preprocessing disamakan ke return, HHO mengoptimasi val_loss pada
     skala return, bukan RMSE pada skala harga hasil rekonstruksi — proxy yang bisa
     menyesatkan pencarian hyperparameter (lihat poin fitness HHO di atas).
  3. Modul `src/data_utils_return.py` yang dirujuk `src/model_loader.py` sempat
     tidak ada di repo sama sekali, sehingga dashboard tidak bisa jalan.

  Ketiganya sudah diperbaiki di pipeline `*_return.ipynb` + `src/data_utils_return.py`
  saat ini. Seluruh notebook raw-price & eksperimen awal (baseline non-`_return`, HHO
  non-`_return`, notebook generik `02_lstm_mimo_hho.ipynb` / `... new.ipynb`) sudah
  dipindahkan ke `notebooks/deprecated/` supaya tidak membingungkan saat membuka folder
  `notebooks/` — TIDAK dipakai dashboard, dan punya keterbatasan ekstrapolasi harga di
  luar rentang training seperti dijelaskan di atas. `src/deprecated/data_utils.py`
  (versi raw-price) juga deprecated dengan alasan yang sama, begitu juga
  `archive/dashboard_app.py` dan `archive/dashboard_app2.py` (versi lama dashboard,
  sebelum entry point disatukan jadi `app.py` di root — tidak kompatibel dengan
  `src/model_loader.py` saat ini).
- **Status retrain**: keenam model (baseline DAN HHO, 3 sumber data) SUDAH diretrain
  dengan pipeline yang sudah diperbaiki ini — `.h5` + `_meta.json` seluruhnya sudah
  ter-commit di `models/`. Jalankan `streamlit run app.py` dari root repo untuk
  melihat hasilnya di dashboard.
- **Live data & fallback (`src/live_data.py`)**: dashboard mencoba fetch harga terbaru
  saat prediksi (`get_live_source_data`/`forecast_next_live` di `model_loader.py`), lalu
  fallback ke CSV historis kalau gagal. `yfinance` (library resmi) paling stabil; `nasdaq`
  (endpoint publik tidak resmi `api.nasdaq.com`) dan `stooq` (endpoint CSV publik tidak
  resmi) rawan diblokir/berubah sewaktu-waktu. **Stooq khususnya sering gagal karena
  proteksi anti-bot** — badge 🟡 (fallback) untuk Stooq adalah perilaku yang DIHARAPKAN,
  bukan bug; kalau perlu data Stooq lebih baru, update manual `data/nvda_d_stooq.csv`
  (download dari `stooq.com/q/d/l/?s=nvda.us&i=d` lewat browser, bukan lewat kode —
  JANGAN mencoba bypass proteksi JavaScript-nya). Live fetch di-cache 5 menit per sumber
  (`get_live_source_data`), ada tombol "🔄 Refresh Data Live" di sidebar untuk memaksa
  fetch ulang.
