#!/usr/bin/env python3
"""
Recomputa métricas reales (Precision, Recall, F1) desde matrices de confusión
para modelos validados y baselines, y genera un reporte Markdown corregido.
"""
import json
import os
import glob
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
VALIDATED_DIR = BASE_DIR / 'validated_results'
BASELINE_DIR = BASE_DIR / 'baseline_predictions' / 'evaluations' / 'confusion_matrices'
OUTPUT_MD = VALIDATED_DIR / 'REPORTE_F1_CORREGIDO.md'


def prf_from_cm(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def load_validated_models_confusions():
    cm_dir = VALIDATED_DIR / 'confusion_matrices'
    models = {}
    models_diss = {}
    for path in cm_dir.glob('*_validated_confusion_matrix.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cm = data.get('confusion_matrix', {})
        tn, fp, fn, tp = cm.get('tn',0), cm.get('fp',0), cm.get('fn',0), cm.get('tp',0)
        precision, recall, f1 = prf_from_cm(tp, fp, fn)
        models[data['model']] = {
            'threshold': data.get('threshold', 0.5),
            'auc_roc': data.get('auc_roc'),
            'f1_original': data.get('f1_score'),
            'precision': precision,
            'recall': recall,
            'f1_real': f1,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        }
    # Disolución
    for path in cm_dir.glob('*_validated_confusion_matrix_dissolution.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cm = data.get('confusion_matrix', {})
        tn, fp, fn, tp = cm.get('tn',0), cm.get('fp',0), cm.get('fn',0), cm.get('tp',0)
        precision, recall, f1 = prf_from_cm(tp, fp, fn)
        models_diss[data['model']] = {
            'threshold': data.get('threshold', 0.5),
            'auc_roc': data.get('auc_roc'),
            'f1_original': data.get('f1_score'),
            'precision': precision,
            'recall': recall,
            'f1_real': f1,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        }
    return models, models_diss


def plot_validated_confusions(validated_models: dict):
    """Genera PNGs de las matrices de confusión de modelos validados."""
    if not validated_models:
        return []
    plots_dir = VALIDATED_DIR / 'confusion_matrices' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for model, m in validated_models.items():
        tn, fp, fn, tp = m.get('tn',0), m.get('fp',0), m.get('fn',0), m.get('tp',0)
        mat = np.array([[tn, fp],[fn, tp]])
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Pred 0','Pred 1'], yticklabels=['Real 0','Real 1'])
        ax.set_title(f"{model}\nCM (validados)")
        fig.tight_layout()
        out = plots_dir / f"{model}_validated_cm.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        created.append(out)
    return created

def plot_validated_confusions_diss(validated_models: dict):
    if not validated_models:
        return []
    plots_dir = VALIDATED_DIR / 'confusion_matrices' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for model, m in validated_models.items():
        tn, fp, fn, tp = m.get('tn',0), m.get('fp',0), m.get('fn',0), m.get('tp',0)
        mat = np.array([[tn, fp],[fn, tp]])
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        sns.heatmap(mat, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax,
                    xticklabels=['Pred 0','Pred 1'], yticklabels=['Real 0','Real 1'])
        ax.set_title(f"{model}\nCM (validados - disolución)")
        fig.tight_layout()
        out = plots_dir / f"{model}_validated_cm_dissolution.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        created.append(out)
    return created


def load_baseline_aggregate():
    agg_csv = BASELINE_DIR / 'baseline_confusion_aggregate.csv'
    if not agg_csv.exists():
        return None
    df = pd.read_csv(agg_csv)
    return df

def load_baseline_aggregate_dissolution():
    agg_csv = BASELINE_DIR / 'baseline_confusion_aggregate_dissolution.csv'
    if not agg_csv.exists():
        return None
    df = pd.read_csv(agg_csv)
    return df


def main():
    validated, validated_diss = load_validated_models_confusions()
    baseline = load_baseline_aggregate()
    baseline_diss = load_baseline_aggregate_dissolution()
    validated_pngs = plot_validated_confusions(validated)
    validated_pngs_d = plot_validated_confusions_diss(validated_diss)

    lines = []
    lines.append('# Resultados corregidos desde matrices de confusión')
    lines.append('')
    lines.append('## Modelos validados (F1 real)')
    lines.append('')
    if validated:
        lines.append('| Modelo | AUC-ROC | Threshold | Precision | Recall | F1 real | F1 original | TN | FP | FN | TP |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for model, m in sorted(validated.items(), key=lambda kv: kv[1].get('f1_real',0), reverse=True):
            lines.append(f"| {model} | {m.get('auc_roc'):.4f} | {m.get('threshold'):.2f} | {m.get('precision'):.4f} | {m.get('recall'):.4f} | {m.get('f1_real'):.4f} | {m.get('f1_original'):.4f} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} |")
    else:
        lines.append('No se encontraron matrices de confusión de modelos validados.')

    # Enlaces a imágenes de matrices de confusión (modelos validados) si existen
    if validated_pngs:
        lines.append('')
        lines.append('### Matrices de confusión (modelos validados)')
        for p in validated_pngs:
            rel = p.relative_to(VALIDATED_DIR.parent)
            lines.append(f"![{p.stem}]({rel.as_posix()})")

    # Validated disolución
    lines.append('')
    lines.append('## Modelos validados - Disolución (F1 real)')
    lines.append('')
    if validated_diss:
        lines.append('| Modelo | AUC-ROC | Threshold | Precision | Recall | F1 real | TN | FP | FN | TP |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for model, m in sorted(validated_diss.items(), key=lambda kv: kv[1].get('f1_real',0), reverse=True):
            lines.append(f"| {model} | {m.get('auc_roc'):.4f} | {m.get('threshold'):.2f} | {m.get('precision'):.4f} | {m.get('recall'):.4f} | {m.get('f1_real'):.4f} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} |")
    else:
        lines.append('No se encontraron matrices de confusión de disolución de modelos validados.')

    if validated_pngs_d:
        lines.append('')
        lines.append('### Matrices de confusión (modelos validados - disolución)')
        for p in validated_pngs_d:
            rel = p.relative_to(VALIDATED_DIR.parent)
            lines.append(f"![{p.stem}]({rel.as_posix()})")

    lines.append('')
    lines.append('## Baselines (agregado por algoritmo)')
    lines.append('')
    if baseline is not None and not baseline.empty:
        lines.append('| Algoritmo | TN | FP | FN | TP | Precision | Recall | F1 real |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        for _, r in baseline.iterrows():
            lines.append(f"| {r['algorithm']} | {int(r['tn'])} | {int(r['fp'])} | {int(r['fn'])} | {int(r['tp'])} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_real']:.4f} |")
    else:
        lines.append('No se encontraron agregados de baselines.')

    # Baselines disolución
    lines.append('')
    lines.append('## Baselines - Disolución (agregado por algoritmo)')
    lines.append('')
    if baseline_diss is not None and not baseline_diss.empty:
        lines.append('| Algoritmo | TN | FP | FN | TP | Precision | Recall | F1 real |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        for _, r in baseline_diss.iterrows():
            lines.append(f"| {r['algorithm']} | {int(r['tn'])} | {int(r['fp'])} | {int(r['fn'])} | {int(r['tp'])} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_real']:.4f} |")
    else:
        lines.append('No se encontraron agregados de disolución.')

    # Enlaces a imágenes de matrices de confusión (baselines) si existen
    plots_dir = BASELINE_DIR / 'plots'
    if plots_dir.exists():
        lines.append('')
        lines.append('## Matrices de confusión (baselines - apariciones)')
        sample_pngs = sorted(plots_dir.glob('*_cm.png'))[:6]
        for p in sample_pngs:
            rel = p.relative_to(VALIDATED_DIR.parent)
            lines.append(f"![{p.stem}]({rel.as_posix()})")

        # Disolución
        lines.append('')
        lines.append('## Matrices de confusión (baselines - disolución)')
        sample_pngs_d = sorted(plots_dir.glob('*_dissolution_cm.png'))[:6]
        for p in sample_pngs_d:
            rel = p.relative_to(VALIDATED_DIR.parent)
            lines.append(f"![{p.stem}]({rel.as_posix()})")

    OUTPUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Reporte generado: {OUTPUT_MD}")


if __name__ == '__main__':
    main()
