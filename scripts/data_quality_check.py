#!/usr/bin/env python3
import os, json
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]

problems = []

print("=== Data Quality Check ===")
print(f"Base: {BASE}")

# 1) Revisar CSVs (baselines)
csv_paths = list(BASE.glob('baseline_predictions/**/*.csv'))
print(f"CSV encontrados: {len(csv_paths)}")
for p in csv_paths:
    try:
        df = pd.read_csv(p)
        nulls = df.isnull().sum()
        total_nulls = int(nulls.sum())
        if total_nulls > 0:
            problems.append((str(p), 'csv_nulls', total_nulls, {k:int(v) for k,v in nulls[nulls>0].items()}))
            print(f"[NULLS] {p}: total={total_nulls} por columna={nulls[nulls>0].to_dict()}")
    except Exception as e:
        problems.append((str(p), 'csv_read_error', str(e)))
        print(f"[ERROR CSV] {p}: {e}")

# 2) Revisar JSONs de matrices de confusión (baseline y validados)
json_paths = list(BASE.glob('baseline_predictions/evaluations/confusion_matrices/*.json'))
json_paths += list(BASE.glob('validated_results/confusion_matrices/*_validated_confusion_matrix.json'))
print(f"JSON CMs encontrados: {len(json_paths)}")
for p in json_paths:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'confusion_matrix' in data:
            cm = data['confusion_matrix']
            keys = {'tn','fp','fn','tp'}
            if not keys.issubset(cm.keys()):
                problems.append((str(p), 'cm_missing_keys', list(keys - set(cm.keys()))))
            else:
                for k in keys:
                    v = cm[k]
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        problems.append((str(p), 'cm_null', k))
        # métricas opcionales
        for k in ['auc_roc','f1','threshold','f1_score']:
            if k in data and data[k] is None:
                problems.append((str(p), 'metric_null', k))
    except Exception as e:
        problems.append((str(p), 'json_read_error', str(e)))
        print(f"[ERROR JSON] {p}: {e}")

# 3) Revisar GraphML (estructura básica)
try:
    import networkx as nx
    net_paths = list(BASE.glob('temporal_networks/networks/semantic_network_*.graphml'))
    print(f"GraphML encontrados: {len(net_paths)}")
    for p in net_paths:
        try:
            G = nx.read_graphml(p)
            node_ids = list(G.nodes())
            empty_nodes = [n for n in node_ids if n is None or (isinstance(n, str) and n.strip()=="")]
            if empty_nodes:
                problems.append((str(p), 'empty_node_ids', len(empty_nodes)))
            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                problems.append((str(p), 'empty_graph', (G.number_of_nodes(), G.number_of_edges())))
        except Exception as e:
            problems.append((str(p), 'graph_read_error', str(e)))
            print(f"[ERROR GRAPHML] {p}: {e}")
except ImportError:
    print("networkx no disponible: omitiendo revisión de GraphML")

print("\n=== Resumen de problemas ===")
if not problems:
    print("SIN NULOS NI ERRORES DETECTADOS")
else:
    for item in problems:
        print("-", item)

# Escribir un reporte sencillo si hay problemas
out = BASE / 'validated_results' / 'DATA_QUALITY_REPORT.txt'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    if not problems:
        f.write('SIN NULOS NI ERRORES DETECTADOS\n')
    else:
        for item in problems:
            f.write(str(item) + '\n')
print(f"\nReporte: {out}")
