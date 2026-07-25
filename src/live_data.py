"""
live_data.py
Fetch data OHLCV "realtime" (harga terbaru yang tersedia) untuk NVDA dari
tiga sumber: Yahoo Finance, Nasdaq, Stooq. Digabung dengan data historis
CSV yang sudah ada supaya perhitungan return (pct_change) tetap kontinu
dari histori lama ke harga terbaru.

CATATAN PENTING SOAL KEANDALAN SUMBER:
- yfinance   : library resmi, dipakai luas, PALING STABIL.
- stooq      : TIDAK ADA API resmi. Dipakai endpoint CSV publik
               (stooq.com/q/d/l/) yang umum dipakai komunitas, tapi sewaktu-
               waktu bisa berubah/diblokir tanpa pemberitahuan.
- nasdaq     : TIDAK ADA API gratis resmi untuk realtime (Nasdaq Data Link
               berbayar). Dipakai endpoint publik tidak resmi
               (api.nasdaq.com/api/quote/...) yang sering dipakai tapi paling
               rawan diblokir / rate-limited / berubah format.

Karena stooq & nasdaq tidak resmi, SEMUA fungsi fetch di sini punya fallback:
kalau live fetch gagal (timeout, diblokir, format berubah), kembalikan data
historis CSV terakhir yang ada (tidak crash), dan beri tahu pemanggilnya
lewat flag `is_live=False`.
"""

import io
import datetime as dt

import numpy as np
import pandas as pd
import requests

TARGET_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
TICKER_YFINANCE = 'NVDA'
TICKER_STOOQ = 'nvda.us'
TICKER_NASDAQ = 'NVDA'

REQUEST_TIMEOUT = 8  # detik, jangan sampai dashboard nge-hang lama kalau sumber down


def _empty_result(reason: str):
    return {'data': None, 'is_live': False, 'error': reason, 'fetched_at': None}


def fetch_live_yfinance(period='6mo'):
    """Fetch OHLCV terbaru via library yfinance (paling stabil)."""
    try:
        import yfinance as yf
        df = yf.download(TICKER_YFINANCE, period=period, interval='1d',
                          progress=False, auto_adjust=False)
        if df.empty:
            return _empty_result("yfinance mengembalikan data kosong")

        # yfinance versi baru bisa mengembalikan kolom MultiIndex (Price, Ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[TARGET_COLS].copy()
        df.index.name = 'Date'
        df = df.sort_index()
        return {'data': df, 'is_live': True, 'error': None,
                'fetched_at': dt.datetime.now()}
    except Exception as e:
        return _empty_result(f"yfinance error: {e}")


def fetch_live_stooq(days_back=200):
    """Fetch OHLCV terbaru via endpoint CSV publik Stooq (tidak resmi).

    Stooq menolak request tanpa header browser (mengembalikan "Access denied"
    / HTTP 403) -- disamakan dengan header User-Agent + Referer seperti
    browser sungguhan.
    """
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=days_back)
        url = (f"https://stooq.com/q/d/l/?s={TICKER_STOOQ}"
               f"&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}&i=d")
        headers = {
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/124.0.0.0 Safari/537.36'),
            'Accept': 'text/csv,application/csv,text/plain,*/*',
            'Referer': 'https://stooq.com/q/d/?s=nvda.us',
        }
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        # Stooq kadang mengembalikan HTTP 200 OK tapi isinya cuma teks
        # "Access denied" (bukan error HTTP resmi) -- deteksi eksplisit.
        body_preview = resp.text.strip()[:200].lower()
        if 'access denied' in body_preview or 'exceeded' in body_preview:
            return _empty_result(f"Stooq menolak request: '{resp.text.strip()[:100]}'")

        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty or 'Date' not in df.columns:
            return _empty_result("Stooq mengembalikan data kosong/format tidak dikenal")

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df = df[TARGET_COLS].copy()
        return {'data': df, 'is_live': True, 'error': None,
                'fetched_at': dt.datetime.now()}
    except Exception as e:
        return _empty_result(f"Stooq error: {e}")


def fetch_live_nasdaq(days_back=200):
    """Fetch OHLCV terbaru via endpoint publik tidak resmi api.nasdaq.com.

    PALING RAWAN gagal di antara ketiganya -- endpoint ini tidak
    didokumentasikan resmi dan bisa berubah/diblokir kapan saja.
    """
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=days_back)
        url = f"https://api.nasdaq.com/api/quote/{TICKER_NASDAQ}/historical"
        params = {
            'assetclass': 'stocks',
            'fromdate': start.strftime('%Y-%m-%d'),
            'todate': end.strftime('%Y-%m-%d'),
            'limit': 9999,
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'application/json',
        }
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()

        rows = payload.get('data', {}).get('tradesTable', {}).get('rows')
        if not rows:
            return _empty_result("Nasdaq API tidak mengembalikan baris data (mungkin diblokir)")

        df = pd.DataFrame(rows)
        df['Date'] = pd.to_datetime(df['date'])
        for col_src, col_dst in [('open', 'Open'), ('high', 'High'),
                                  ('low', 'Low'), ('close', 'Close'),
                                  ('volume', 'Volume')]:
            df[col_dst] = (df[col_src].astype(str)
                           .str.replace('$', '', regex=False)
                           .str.replace(',', '', regex=False)
                           .astype(float))
        df = df.set_index('Date').sort_index()
        df = df[TARGET_COLS].copy()
        return {'data': df, 'is_live': True, 'error': None,
                'fetched_at': dt.datetime.now()}
    except Exception as e:
        return _empty_result(f"Nasdaq error: {e}")


FETCHERS = {
    'yfinance': fetch_live_yfinance,
    'stooq': fetch_live_stooq,
    'nasdaq': fetch_live_nasdaq,
}


def get_live_or_fallback(source: str, historical_df: pd.DataFrame):
    """Coba fetch live untuk `source`; kalau gagal, fallback ke
    `historical_df` (data CSV lama yang sudah di-load sebelumnya).

    Returns dict: {"data": DataFrame, "is_live": bool, "error": str|None,
                   "fetched_at": datetime|None}
    """
    fetcher = FETCHERS.get(source)
    if fetcher is None:
        raise ValueError(f"Sumber tidak dikenal: {source}")

    result = fetcher()
    if result['data'] is not None and len(result['data']) > 0:
        return result

    # Fallback: pakai data historis CSV yang sudah ada
    return {
        'data': historical_df,
        'is_live': False,
        'error': result['error'],
        'fetched_at': None,
    }


def merge_with_history(historical_df: pd.DataFrame, live_df: pd.DataFrame):
    """Gabungkan histori CSV lama dengan hasil fetch live, dedup by date
    (baris live menang kalau tanggalnya bentrok, karena lebih update),
    lalu urutkan ascending. Perlu histori supaya perhitungan return
    (pct_change) tetap kontinu dari jauh ke belakang, bukan cuma dari
    beberapa hari live saja.
    """
    combined = pd.concat([historical_df, live_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    return combined
