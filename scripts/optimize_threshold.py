#!/usr/bin/env python3
"""
Optimización de umbral (threshold) para maximizar F1
----------------------------------------------------
Entrada: CSV con columnas y_true (0/1) y y_score (score continuo en [0,1]).
Salida: mejor threshold y métricas asociadas (F1, precision, recall), guardado en JSON y mostrado por consola.
Opcional: guarda una curva F1 vs threshold en PNG.

Uso:
  python scripts/optimize_threshold.py --csv path/a/scores.csv --out results/ --steps 1000
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


def compute_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    y_pred = (y_score >= thr).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return f1, prec, rec


def grid_search_threshold(y_true: np.ndarray, y_score: np.ndarray, steps: int = 1000):
    # Usar thresholds sobre el rango real de scores
    lo = float(np.min(y_score))
    hi = float(np.max(y_score))
    if lo == hi:
        return lo, {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    thrs = np.linspace(lo, hi, steps)
    best = (-1.0, None)
    f1_list, thr_list = [], []
    for t in thrs:
        f1, prec, rec = compute_metrics_at_threshold(y_true, y_score, t)
        f1_list.append(f1)
        thr_list.append(t)
        if f1 > best[0]:
            best = (f1, (t, prec, rec))
    best_thr, best_prec, best_rec = best[1]
    return best_thr, {'f1': best[0], 'precision': best_prec, 'recall': best_rec, 'curve': (thr_list, f1_list)}


def refine_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float, window: float = 0.05, steps: int = 500):
    lo = max(0.0, thr - window)
    hi = min(1.0, thr + window)
    thrs = np.linspace(lo, hi, steps)
    best = (-1.0, None)
    for t in thrs:
        f1, prec, rec = compute_metrics_at_threshold(y_true, y_score, t)
        if f1 > best[0]:
            best = (f1, (t, prec, rec))
    return best[1][0], {'f1': best[0], 'precision': best[1][1], 'recall': best[1][2]}


def main():
    ap = argparse.ArgumentParser(description='Optimizar threshold para maximizar F1')
    ap.add_argument('--csv', required=True, help='CSV con columnas y_true,y_score (y opcionalmente y_id)')
    ap.add_argument('--out', default='validated_results', help='Carpeta de salida para JSON/PNG')
    ap.add_argument('--steps', type=int, default=1000, help='Pasos en búsqueda gruesa')
    ap.add_argument('--refine-steps', type=int, default=500, help='Pasos en búsqueda fina alrededor del mejor')
    ap.add_argument('--no-plot', action='store_true', help='No guardar PNG de F1 vs threshold')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if not {'y_true', 'y_score'} <= set(df.columns):
        raise ValueError('El CSV debe contener columnas y_true e y_score')
    y_true = df['y_true'].astype(int).to_numpy()
    y_score = df['y_score'].astype(float).to_numpy()

    # Búsqueda gruesa
    thr0, res0 = grid_search_threshold(y_true, y_score, steps=args.steps)
    # Búsqueda fina
    thr1, res1 = refine_threshold(y_true, y_score, thr0, window=0.05, steps=args.refine_steps)

    best_thr = float(thr1)
    best_f1 = float(res1['f1'])
    best_prec = float(res1['precision'])
    best_rec = float(res1['recall'])

    print(f"Mejor threshold: {best_thr:.4f} | F1={best_f1:.4f} | P/R={best_prec:.4f}/{best_rec:.4f}")

    # Guardar JSON
    out_json = out_dir / 'best_threshold.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'best_threshold': best_thr,
            'f1': best_f1,
            'precision': best_prec,
            'recall': best_rec,
            'coarse_steps': args.steps,
            'refine_steps': args.refine_steps,
            'source_csv': str(Path(args.csv).resolve()),
        }, f, indent=2)
    print(f"Guardado: {out_json}")

    # Curva (si se dispone de ella y no se desactiva)
    if not args.no_plot:
        thr_list, f1_list = res0.get('curve', (None, None))
        if thr_list is not None:
            plt.figure(figsize=(6,4))
            plt.plot(thr_list, f1_list, label='F1 (búsqueda gruesa)')
            plt.axvline(best_thr, color='red', linestyle='--', label=f'Mejor thr={best_thr:.4f}')
            plt.xlabel('Threshold')
            plt.ylabel('F1')
            plt.title('F1 vs Threshold')
            plt.legend()
            out_png = out_dir / 'best_threshold_curve.png'
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            print(f"Guardado: {out_png}")


if __name__ == '__main__':
    main()
