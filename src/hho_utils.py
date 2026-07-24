"""
hho_utils.py
Harris Hawks Optimization untuk mencari hyperparameter LSTM MIMO terbaik
(units, dropout, learning rate, batch size - 6 dimensi).

Fitness function punya dua mode:

1. Default (reconstruct_price=None) - fitness = val_loss pada skala data
   yang dipakai untuk training (MSE). Cocok kalau skala training == skala
   evaluasi akhir (mis. data_utils.py, harga absolut langsung).

2. Reconstruction-aware (reconstruct_price diberikan) - fitness = RMSE
   validasi pada SKALA HARGA HASIL REKONSTRUKSI (USD), bukan MSE return
   ternormalisasi. Wajib dipakai kalau training memakai data_utils_return.py
   (return/percentage-change): val_loss di situ mengukur MSE pada return
   harian, TIDAK otomatis selaras dengan RMSE harga hasil rekonstruksi
   (inverse_return_to_price mengakumulasi return via cumulative log-return,
   jadi kandidat dengan val_loss bagus di skala return bisa saja
   merekonstruksi buruk di skala harga). Tanpa mode ini, HHO mengoptimasi
   proxy yang tidak sama dengan kriteria keberhasilan sebenarnya (BAB 3.6.3:
   RMSE/MAE/DA pada skala harga), sehingga hasil pencarian tidak bisa
   diandalkan mengalahkan baseline.
"""

import math
import random
import time

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from model_utils import build_lstm_mimo, decode_hyperparameters, DEFAULT_EPOCHS, DEFAULT_PATIENCE
from solution import solution


def make_objective_function(X_train, y_train, X_val, y_val, n_input, n_features, n_forecast,
                             reconstruct_price=None, scaler=None, base_val=None, y_val_abs=None):
    """
    reconstruct_price : callable(scaler, pred_scaled, base_prices, n_forecast) -> harga USD
                         (mis. data_utils_return.inverse_return_to_price). Kalau None,
                         fitness = val_loss (skala training), bukan RMSE harga.
    scaler, base_val, y_val_abs : wajib diisi kalau reconstruct_price diisi.
    """
    if reconstruct_price is not None:
        assert scaler is not None and base_val is not None and y_val_abs is not None, \
            "scaler, base_val, y_val_abs wajib diisi saat reconstruct_price dipakai"

    def objective_function(hyperparameters):
        units1, units2, drop1, drop2, lr, bs = decode_hyperparameters(hyperparameters)
        model = build_lstm_mimo(n_input, n_features, n_forecast, units1, units2, drop1, drop2, lr)

        es = EarlyStopping(monitor='val_loss', patience=DEFAULT_PATIENCE,
                            restore_best_weights=True, verbose=0)
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=DEFAULT_EPOCHS, batch_size=bs,
                shuffle=False, callbacks=[es], verbose=0
            )
            if reconstruct_price is not None:
                val_pred_scaled = model.predict(X_val, verbose=0)
                val_pred_price = reconstruct_price(scaler, val_pred_scaled, base_val, n_forecast)
                val_loss = float(np.sqrt(np.mean((y_val_abs - val_pred_price) ** 2)))
            else:
                val_loss = min(history.history['val_loss'])
        except Exception as e:
            print(f"Error: {e}")
            val_loss = np.inf
        return val_loss
    return objective_function


def HHO(objf, lb, ub, dim, search_agents_no, max_iter, solution_obj):
    Rabbit_Location   = np.zeros(dim)
    Rabbit_Energy     = float('inf')
    X                 = np.random.uniform(0, 1, (search_agents_no, dim)) * (ub - lb) + lb
    convergence_curve = np.zeros(max_iter)

    print(f'HHO mengoptimasi "{objf.__name__}" (dim={dim})')
    t = 0
    while t < max_iter:
        for i in range(search_agents_no):
            X[i, :] = np.clip(X[i, :], lb, ub)
            solution_obj.fitness_evaluations += 1
            fitness = objf(X[i, :])
            if fitness < Rabbit_Energy:
                Rabbit_Energy   = fitness
                Rabbit_Location = X[i, :].copy()

        E1 = 2 * (1 - t / max_iter)
        for i in range(search_agents_no):
            E0              = 2 * random.random() - 1
            Escaping_Energy = E1 * E0

            if abs(Escaping_Energy) >= 1:
                q = random.random()
                rand_hawk_index = math.floor(search_agents_no * random.random())
                X_rand = X[rand_hawk_index, :]
                if q < 0.5:
                    X[i, :] = X_rand - random.random() * abs(
                        X_rand - 2 * random.random() * X[i, :])
                    solution_obj.mutation_count += 1
                else:
                    X[i, :] = (Rabbit_Location - X.mean(0)
                               - random.random() * ((ub - lb) * random.random() + lb))
                    solution_obj.crossover_count += 1
            else:
                r = random.random()
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    X[i, :] = Rabbit_Location - Escaping_Energy * abs(
                        Rabbit_Location - X[i, :])
                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :])
                    solution_obj.fitness_evaluations += 1
                    if objf(X1) < fitness:
                        X[i, :] = X1.copy()
                        solution_obj.crossover_count += 1

        convergence_curve[t] = Rabbit_Energy
        u1, u2, d1, d2, lr, bs = decode_hyperparameters(Rabbit_Location)
        print(f"  Iter {t+1:2d}/{max_iter} | val_loss={Rabbit_Energy:.6f} | "
              f"units=({u1},{u2}) dropout=({d1:.2f},{d2:.2f}) "
              f"lr={lr:.5f} bs={bs}")
        t += 1

    solution_obj.best           = Rabbit_Energy
    solution_obj.bestIndividual = Rabbit_Location
    solution_obj.convergence    = convergence_curve
    return solution_obj


def run_hho_search(objective_function, lb, ub, dim, search_agents_no, max_iter, num_runs):
    """
    Jalankan HHO sebanyak num_runs kali dan kembalikan individu terbaik di
    ANTARA SEMUA run (bukan hanya run terakhir - lihat solusi lama yang
    overwrite `sol` tiap iterasi loop).
    """
    best_overall_fitness    = np.inf
    best_overall_individual = None
    best_overall_solution   = None
    all_solutions           = []

    for run in range(num_runs):
        print(f"\n{'='*65}\nRun {run + 1}/{num_runs}\n{'='*65}")
        start_time = time.time()
        sol = solution()
        sol = HHO(objective_function, lb, ub, dim, search_agents_no, max_iter, sol)
        sol.executionTime = time.time() - start_time
        all_solutions.append(sol)

        if sol.best < best_overall_fitness:
            best_overall_fitness    = sol.best
            best_overall_individual = sol.bestIndividual.copy()
            best_overall_solution   = sol
            print(f"  -> New best ditemukan di Run {run + 1}: val_loss={best_overall_fitness:.6f}")

    return best_overall_individual, best_overall_fitness, best_overall_solution, all_solutions
