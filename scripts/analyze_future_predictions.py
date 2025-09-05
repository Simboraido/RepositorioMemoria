#!/usr/bin/env python3
"""
Análisis de predicciones futuras (2024-2025, 2025-2026, 2026-2027)
------------------------------------------------------------------
Lee CSV/JSON en predictions/ generados por los predictores y produce:
- Estadísticas descriptivas de score por periodo.
- Top-20 pares más probables con nombres.
- Chequeos estructurales: cuántos son de dos saltos vs más lejanos (en la red base).
- Resumen en Markdown (`predictions/ANALISIS_PREDICCIONES.md`).
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import networkx as nx
import glob
import os

BASE_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = BASE_DIR / 'predictions'
NETWORKS_DIR = BASE_DIR / 'temporal_networks' / 'networks'


def load_base_graph():
    files = sorted(glob.glob(str(NETWORKS_DIR / 'semantic_network_*.graphml')))
    if not files:
        raise FileNotFoundError(f"No se encontraron GraphML en {NETWORKS_DIR}")
    last = files[-1]
    G = nx.read_graphml(last)
    period = os.path.basename(last).replace('semantic_network_', '').replace('.graphml', '')
    return period, G


def describe_scores(df: pd.DataFrame) -> str:
    s = df['score']
    q = s.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    return (
        f"n={len(s)} | mean={s.mean():.4f} | std={s.std(ddof=0):.4f} | min={q.loc[0.0]:.4f} | "
        f"p25={q.loc[0.25]:.4f} | p50={q.loc[0.5]:.4f} | p75={q.loc[0.75]:.4f} | p90={q.loc[0.9]:.4f} | p99={q.loc[0.99]:.4f} | max={q.loc[1.0]:.4f}"
    )


def pair_distance(G: nx.Graph, u: str, v: str) -> int:
    if u not in G or v not in G:
        return -1
    if G.has_edge(u, v):
        return 1
    try:
        return nx.shortest_path_length(G, u, v)
    except Exception:
        return 1_000_000


def analyze_period(csv_path: Path, G: nx.Graph, topk: int = 20) -> dict:
    df = pd.read_csv(csv_path)
    desc = describe_scores(df)
    # distancias aproximadas en la red base
    sample = df.head(500)
    dists = [pair_distance(G, str(r.u), str(r.v)) for r in sample.itertuples(index=False)]
    dist_counts = pd.Series(dists).value_counts().to_dict()
    top = df.sort_values('score', ascending=False).head(topk)
    top_records = top[['u','v','u_name','v_name','score']].to_dict(orient='records')
    return {
        'describe': desc,
        'dist_counts_head500': dist_counts,
        'top': top_records,
    }


def main():
    period, G = load_base_graph()
    print(f"Base: {period} | N={G.number_of_nodes()} | E={G.number_of_edges()}")

    outputs = {}
    for label in ['2024_2025', '2025_2026']:
        path = PRED_DIR / f'predicted_links_{label}_by_best_model.csv'
        if path.exists():
            outputs[label] = analyze_period(path, G)

    md = [
        f"# Análisis de Predicciones (base: {period})",
        "",
    ]
    for label, info in outputs.items():
        md += [
            f"## {label}",
            f"- Estadística de score: {info['describe']}",
            f"- Distancias (head 500): {info['dist_counts_head500']}",
            "- Top-20 predicciones:",
        ]
        for i, r in enumerate(info['top'], 1):
            md.append(f"  {i:02d}. {r['u_name']} — {r['v_name']} (score={r['score']:.5f})")
        md.append("")

    out_md = PRED_DIR / 'ANALISIS_PREDICCIONES.md'
    out_md.write_text("\n".join(md), encoding='utf-8')
    print(f"Reporte escrito en {out_md}")


if __name__ == '__main__':
    main()
