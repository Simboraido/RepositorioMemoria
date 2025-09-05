#!/usr/bin/env python3
"""
Reconstruye matriz de confusión desde CSV (y_true,y_score) con un threshold dado.
Genera JSON compatible con validated_results/confusion_matrices/*_validated_confusion_matrix.json
y una imagen PNG de la matriz de confusión.

Uso:
  python scripts/rebuild_confusion_from_scores.py \
    --csv validated_results/evolvegcn-h_validation_scores.csv \
    --model-name EvolveGCN-H \
    --threshold 0.91 \
    --out validated_results/confusion_matrices
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    ap = argparse.ArgumentParser(description='Reconstruir matriz de confusión desde scores y threshold')
    ap.add_argument('--csv', required=True, help='Ruta al CSV con columnas y_true,y_score (opcional u,v)')
    ap.add_argument('--model-name', required=True, help='Nombre del modelo (para nombrar archivos)')
    ap.add_argument('--threshold', type=float, required=True, help='Threshold a aplicar')
    ap.add_argument('--out', default='validated_results/confusion_matrices', help='Carpeta de salida de JSON/PNG')
    args = ap.parse_args()

    out_dir = Path(args.out)
    plot_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if not {'y_true', 'y_score'} <= set(df.columns):
        raise ValueError('El CSV debe contener columnas y_true e y_score')

    y_true = df['y_true'].astype(int).to_numpy()
    y_score = df['y_score'].astype(float).to_numpy()

    if len(np.unique(y_true)) < 2:
        raise ValueError('y_true debe contener ambas clases 0 y 1')

    y_pred = (y_score > args.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc = roc_auc_score(y_true, y_score)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    out_json = out_dir / f"{args.model_name.lower()}_validated_confusion_matrix.json"
    payload = {
        'model': args.model_name,
        'threshold': float(args.threshold),
        'auc_roc': float(auc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"💾 Guardado JSON: {out_json}")

    # Plot matriz de confusión
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(4.2, 3.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negativo','Positivo'], yticklabels=['Negativo','Positivo'])
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title(f"{args.model_name} | thr={args.threshold:.3f} | F1={f1:.3f}")
    out_png = plot_dir / f"{args.model_name.replace(' ', '_')}_validated_cm.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"🖼️  Guardado PNG: {out_png}")


if __name__ == '__main__':
    main()
