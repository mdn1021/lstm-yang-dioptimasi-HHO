"""
evaluation.py
Perhitungan & pelaporan metrik evaluasi (RMSE, MAE, MAPE, DA) per horizon,
dipakai bersama oleh notebook baseline dan HHO agar format perbandingan
konsisten.

DA (Directional Accuracy): persentase sampel di mana arah pergerakan harga
yang diprediksi (naik/turun relatif terhadap harga Close terakhir yang
diketahui sebelum window forecast / "anchor") sama dengan arah pergerakan
aktual. Dihitung per horizon (t+1..t+n) memakai anchor yang sama
(get_anchor_prices di data_utils.py), bukan membandingkan horizon berurutan
satu sama lain.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mse, rmse, mae, mape


def directional_accuracy(y_true, y_pred, anchor):
    """
    y_true, y_pred : array (samples,) - harga aktual/prediksi pada satu horizon
    anchor         : array (samples,) - harga Close terakhir sebelum window forecast
    """
    true_dir = np.sign(y_true - anchor)
    pred_dir = np.sign(y_pred - anchor)
    return float(np.mean(true_dir == pred_dir) * 100)


def evaluate_full(y_true, y_pred, anchor):
    """RMSE, MAE, MAPE, DA sekaligus untuk satu horizon (atau flatten semua horizon)."""
    mse, rmse, mae, mape = evaluate(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred, anchor)
    return mse, rmse, mae, mape, da


def print_evaluation_report(title, n_forecast, splits):
    """splits: dict {label: (y_true_abs, y_pred_abs, anchor)}"""
    for label, (y_true_abs, y_pred_abs, anchor) in splits.items():
        print("\n" + "=" * 65)
        print(f"  EVALUASI {label} SET - {title}")
        print("=" * 65)
        for i in range(n_forecast):
            _, rmse, mae, mape, da = evaluate_full(y_true_abs[:, i], y_pred_abs[:, i], anchor)
            print(f"  Horizon t+{i + 1} (Hari ke-{i + 1}) | "
                  f"RMSE={rmse:8.4f}  MAE={mae:8.4f}  MAPE={mape:.2f}%  DA={da:.2f}%")
        anchor_flat = np.repeat(anchor, n_forecast)
        _, rmse_a, mae_a, mape_a, da_a = evaluate_full(
            y_true_abs.flatten(), y_pred_abs.flatten(), anchor_flat)
        print(f"\n  Average (all horizons)   | "
              f"RMSE={rmse_a:8.4f}  MAE={mae_a:8.4f}  MAPE={mape_a:.2f}%  DA={da_a:.2f}%")


def summary_table(title, n_forecast, splits):
    """splits: dict {label: (y_true_abs, y_pred_abs, anchor)}"""
    print("\n" + "=" * 85)
    print(f"  RINGKASAN EVALUASI - {title}")
    print("=" * 85)
    print(f"  {'Set':<8} {'Horizon':<14} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'DA':>8}")
    print(f"  {'-' * 64}")

    for label, (y_true_abs, y_pred_abs, anchor) in splits.items():
        for i in range(n_forecast):
            _, rmse, mae, mape, da = evaluate_full(y_true_abs[:, i], y_pred_abs[:, i], anchor)
            h = f"t+{i + 1} (Hari ke-{i + 1})"
            print(f"  {label:<8} {h:<14} {rmse:>10.4f} {mae:>10.4f} {mape:>9.2f}% {da:>7.2f}%")
        anchor_flat = np.repeat(anchor, n_forecast)
        _, rmse_a, mae_a, mape_a, da_a = evaluate_full(
            y_true_abs.flatten(), y_pred_abs.flatten(), anchor_flat)
        print(f"  {label:<8} {'Average':<14} {rmse_a:>10.4f} {mae_a:>10.4f} {mape_a:>9.2f}% {da_a:>7.2f}%")
        print(f"  {'-' * 64}")
