"""
app.py
Dashboard Streamlit: Perbandingan Prediksi Multi-Hari Saham NVDA
LSTM MIMO Standar vs LSTM MIMO + HHO, per sumber data (Yahoo Finance, Nasdaq, Stooq).

Jalankan dengan:
    streamlit run app.py
(dari dalam folder dashboard/, dengan model & data sudah tersedia di ../models & ../data)
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_loader import (
    SOURCES, SOURCE_LABELS, ARCHITECTURES, ARCH_LABELS,
    MODEL_FILENAME, META_FILENAME,
    load_all_models, load_prepared_source, predict_all_splits,
    metrics_table, forecast_next,
)


st.set_page_config(
    page_title="NVDA LSTM MIMO vs HHO Dashboard",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# THEME - Nvidia Tech Dark Mode (lihat .streamlit/config.toml)
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_BASELINE = "#63B3ED"
COLOR_HHO = "#76B900"   # NVIDIA green
COLOR_ACTUAL = "#E2E8F0"

N_INPUT = 60
N_FORECAST = 5

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Kontrol Dashboard")

registry = load_all_models()

st.sidebar.markdown("**Status Model (6 total)**")
for source in SOURCES:
    for arch in ARCHITECTURES:
        entry = registry[(source, arch)]
        label = f"{ARCH_LABELS[arch]} · {SOURCE_LABELS[source]}"
        if entry["error"] is None:
            st.sidebar.success(f"✅ {label}", icon="✅")
        else:
            st.sidebar.error(f"❌ {label}", icon="❌")
            with st.sidebar.expander(f"Detail error: {label}"):
                st.code(entry["error"], language=None)
                model_fn = MODEL_FILENAME[arch].format(source=source)
                meta_fn = META_FILENAME[arch].format(source=source)
                st.caption(f"Dicari di: models/{model_fn}\n\nmodels/{meta_fn}")


st.sidebar.markdown("---")
selected_source = st.sidebar.selectbox(
    "Pilih Sumber Data",
    options=SOURCES,
    format_func=lambda s: SOURCE_LABELS[s],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Konfigurasi Model (fixed, tidak dioptimasi HHO)**")
st.sidebar.info(f"n_input = {N_INPUT} hari\n\nn_forecast = {N_FORECAST} hari")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Window (n_input) dan horizon (n_forecast) ditetapkan tetap di seluruh "
    "6 model. HHO hanya mengoptimasi hyperparameter arsitektur (units, dropout, "
    "learning rate, batch size)."
)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("📈 Dashboard Prediksi Multi-Hari Saham NVDA")
st.caption(
    "Perbandingan LSTM MIMO Standar vs LSTM MIMO + HHO, dievaluasi terpisah "
    "pada 3 sumber data historis: Yahoo Finance, Nasdaq, dan Stooq."
)

tab1, tab2, tab3 = st.tabs([
    "🔮 Proyeksi Multi-Hari",
    "📊 Perbandingan Performa",
    "🗃️ Riwayat Data",
])

# ---------------------------------------------------------------------------
# TAB 1 — PROYEKSI MULTI-HARI
# ---------------------------------------------------------------------------
with tab1:
    st.subheader(f"Proyeksi {N_FORECAST} Hari ke Depan — {SOURCE_LABELS[selected_source]}")

    baseline_entry = registry[(selected_source, 'baseline')]
    hho_entry = registry[(selected_source, 'hho')]

    if baseline_entry["error"] or hho_entry["error"]:
        st.warning(
            "Salah satu atau kedua model untuk sumber ini belum berhasil dimuat. "
            "Pastikan file model (.h5) dan metadata (.json) ada di folder `models/` "
            "dengan nama sesuai konvensi (mis. `lstm_mimo_baseline_yfinance.h5`)."
        )
    else:
        # Dua pipeline berbeda: baseline (harga absolut) vs hho (return-normalization)
        ds_baseline = load_prepared_source(selected_source, 'baseline', N_INPUT, N_FORECAST)
        ds_hho = load_prepared_source(selected_source, 'hho', N_INPUT, N_FORECAST)
        last_price = float(ds_baseline['data']['Close'].iloc[-1])
        last_date = ds_baseline['data'].index[-1]

        pred_baseline = forecast_next(baseline_entry["model"], ds_baseline, 'baseline')
        pred_hho = forecast_next(hho_entry["model"], ds_hho, 'hho')

        future_dates = pd.bdate_range(start=last_date, periods=N_FORECAST + 1)[1:]

        # --- Kartu Metrik ---
        st.markdown("**Estimasi Harga Close (USD) per Horizon**")
        cols = st.columns(N_FORECAST)
        for h in range(N_FORECAST):
            with cols[h]:
                delta_b = pred_baseline[h] - last_price
                delta_h = pred_hho[h] - last_price
                st.metric(
                    label=f"H+{h+1} ({future_dates[h].strftime('%d %b')})",
                    value=f"${pred_hho[h]:,.2f}",
                    delta=f"{delta_h:+.2f} (HHO)",
                )
                st.caption(f"Standar: ${pred_baseline[h]:,.2f} ({delta_b:+.2f})")

        # --- Grafik Interaktif ---
        st.markdown("**Grafik Historis + Proyeksi (Standar vs +HHO)**")
        lookback = 90
        hist = ds_baseline['data']['Close'].iloc[-lookback:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist.values, mode='lines',
            name='Aktual (90 hari terakhir)', line=dict(color=COLOR_ACTUAL, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[last_date] + list(future_dates),
            y=[last_price] + list(pred_baseline),
            mode='lines+markers', name='Proyeksi — Standar',
            line=dict(color=COLOR_BASELINE, width=2, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=[last_date] + list(future_dates),
            y=[last_price] + list(pred_hho),
            mode='lines+markers', name='Proyeksi — HHO',
            line=dict(color=COLOR_HHO, width=2, dash='dash'),
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=450,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Tanggal", yaxis_title="Harga (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, width='stretch')

        st.caption(
            f"Basis proyeksi: ${last_price:,.2f} pada {last_date.strftime('%d %b %Y')}. "
            "Kedua model memprediksi 5 hari sekaligus dalam satu forward pass (strategi "
            "MIMO), sehingga tidak ada akumulasi error antar hari."
        )

# ---------------------------------------------------------------------------
# TAB 2 — PERBANDINGAN PERFORMA
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Perbandingan Performa — 6 Model (3 Sumber x 2 Arsitektur)")

    all_rows = []
    with st.spinner("Menghitung metrik evaluasi untuk seluruh model..."):
        for source in SOURCES:
            for arch in ARCHITECTURES:
                entry = registry[(source, arch)]
                if entry["error"] is not None:
                    continue
                ds = load_prepared_source(source, arch, N_INPUT, N_FORECAST)
                splits = predict_all_splits(entry["model"], ds, arch)
                rows = metrics_table(splits, N_FORECAST)
                for r in rows:
                    r["Sumber"] = SOURCE_LABELS[source]
                    r["Model"] = ARCH_LABELS[arch]
                all_rows.extend(rows)

    if not all_rows:
        st.warning("Belum ada model yang berhasil dimuat untuk dievaluasi.")
    else:
        df_metrics = pd.DataFrame(all_rows)
        df_metrics = df_metrics[["Sumber", "Model", "Split", "Horizon", "RMSE", "MAE", "MAPE (%)", "DA (%)"]]

        st.markdown("**A. Tabel Perbandingan Lengkap (Test Set, per Horizon)**")
        st.caption(
            "DA (Directional Accuracy) = persentase sampel di mana arah pergerakan "
            "harga (naik/turun dari harga terakhir) diprediksi benar -- pelengkap "
            "MAPE/RMSE/MAE, penting karena model bisa punya MAPE rendah hanya dengan "
            "memprediksi hampir flat tanpa menangkap arah tren yang sebenarnya."
        )
        df_test = df_metrics[df_metrics["Split"] == "TEST"]
        st.dataframe(
            df_test.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE (%)": "{:.2f}", "DA (%)": "{:.2f}"}),
            width='stretch', hide_index=True,
        )

        st.markdown("**B. Ringkasan Rata-Rata (Average, Test Set) — 6 Model**")
        df_avg = df_test[df_test["Horizon"] == "Average"].sort_values(["Sumber", "Model"])
        st.dataframe(
            df_avg[["Sumber", "Model", "RMSE", "MAE", "MAPE (%)", "DA (%)"]]
            .style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE (%)": "{:.2f}", "DA (%)": "{:.2f}"}),
            width='stretch', hide_index=True,
        )

        st.markdown("**C. Kriteria Keberhasilan (BAB 3.6.3) — per Sumber Data**")
        for source in SOURCES:
            sub = df_avg[df_avg["Sumber"] == SOURCE_LABELS[source]]
            if len(sub) < 2:
                continue
            base_row = sub[sub["Model"] == ARCH_LABELS["baseline"]].iloc[0]
            hho_row = sub[sub["Model"] == ARCH_LABELS["hho"]].iloc[0]
            mape_ok = hho_row["MAPE (%)"] < 10 and base_row["MAPE (%)"] < 10 and hho_row["MAPE (%)"] < base_row["MAPE (%)"]
            rmse_ok = hho_row["RMSE"] < base_row["RMSE"]
            mae_ok = hho_row["MAE"] < base_row["MAE"]
            st.write(f"**{SOURCE_LABELS[source]}**: "
                     f"{'✅' if mape_ok else '⚠️'} MAPE<10% & HHO lebih rendah  |  "
                     f"{'✅' if rmse_ok else '⚠️'} RMSE HHO < Standar  |  "
                     f"{'✅' if mae_ok else '⚠️'} MAE HHO < Standar")

        st.markdown("**D. Grafik Perbandingan Metrik Agregat**")
        fig2 = go.Figure()
        for arch in ARCHITECTURES:
            sub = df_avg[df_avg["Model"] == ARCH_LABELS[arch]]
            fig2.add_trace(go.Bar(
                x=sub["Sumber"], y=sub["MAPE (%)"], name=ARCH_LABELS[arch],
                marker_color=COLOR_BASELINE if arch == 'baseline' else COLOR_HHO,
            ))
        fig2.update_layout(
            template=PLOTLY_TEMPLATE, barmode='group', height=400,
            yaxis_title="MAPE (%)", margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig2, width='stretch')

# ---------------------------------------------------------------------------
# TAB 3 — RIWAYAT DATA
# ---------------------------------------------------------------------------
with tab3:
    st.subheader(f"Riwayat Data Mentah — {SOURCE_LABELS[selected_source]}")

    ds = load_prepared_source(selected_source, 'baseline', N_INPUT, N_FORECAST)
    data = ds['data']
    train_df, val_df, test_df = ds['train_df'], ds['val_df'], ds['test_df']

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Baris", f"{len(data):,}")
    c2.metric("Periode", f"{data.index[0].date()} → {data.index[-1].date()}")
    c3.metric("Split", "70 / 15 / 15")

    st.markdown("**Indikator Split Train / Validasi / Test**")
    split_info = pd.DataFrame({
        "Subset": ["Train (70%)", "Validasi (15%)", "Test (15%)"],
        "Jumlah Baris": [len(train_df), len(val_df), len(test_df)],
        "Mulai": [train_df.index[0].date(), val_df.index[0].date(), test_df.index[0].date()],
        "Selesai": [train_df.index[-1].date(), val_df.index[-1].date(), test_df.index[-1].date()],
    })
    st.dataframe(split_info, width='stretch', hide_index=True)

    st.markdown("**Data Mentah (OHLCV)**")
    display_df = data.copy()
    display_df["Subset"] = "Train"
    display_df.loc[val_df.index, "Subset"] = "Validasi"
    display_df.loc[test_df.index, "Subset"] = "Test"
    st.dataframe(display_df.sort_index(ascending=False), width='stretch', height=400)
