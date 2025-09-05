#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PRED = BASE / 'predictions'

def load_top(label: str, k: int = 100):
    df = pd.read_csv(PRED / f'predicted_links_{label}_by_best_model.csv')
    top = df.sort_values('score', ascending=False).head(k)
    # clave par ordenada para comparar
    key = top.apply(lambda r: tuple(sorted((str(r['u']), str(r['v'])))), axis=1)
    return list(key), top

def main():
    for k in (20, 100):
        a_keys, a_top = load_top('2024_2025', k)
        b_keys, b_top = load_top('2025_2026', k)
        inter = len(set(a_keys).intersection(b_keys))
        same_order = a_keys == b_keys
        print(f"Top-{k}: intersección={inter}/{k} | mismo orden={same_order}")

if __name__ == '__main__':
    main()
