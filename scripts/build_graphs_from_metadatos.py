#!/usr/bin/env python3
"""
Construcción de redes temáticas por periodo desde Metadatos (OpenAlex)
----------------------------------------------------------------------
Lee archivos Metadatos/batch_XXX_metadata.json y genera, por cada
ventana bienal (p.ej. 2000-2001, 2002-2003, ..., 2022-2023):
 - semantic_network_<periodo>.graphml
 - semantic_network_<periodo>_nodes.csv
 - semantic_network_<periodo>_edges.csv

Nodos: topics (OpenAlex T...), con atributos: name, frequency, field, subfield, domain
Aristas: co-ocurrencia de topics dentro del mismo trabajo; atributo: weight (conteo)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict, Iterable, List, Tuple
import networkx as nx
import pandas as pd


def year_to_period(year: int, min_year: int, max_year: int) -> str | None:
    if year is None:
        return None
    if year < min_year or year > max_year:
        return None
    base = year if year % 2 == 0 else year - 1
    return f"{base}-{base+1}"


def iter_metadata_files(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.glob('batch_*_metadata.json')):
        yield p


def extract_topics(work: dict) -> List[dict]:
    md = work.get('metadata') or {}
    topics = md.get('topics') or []
    out = []
    for t in topics:
        tid = t.get('id')
        if not tid:
            continue
        out.append({
            'id': str(tid),
            'name': t.get('display_name') or '',
            'field': (t.get('field') or {}).get('display_name'),
            'subfield': (t.get('subfield') or {}).get('display_name'),
            'domain': (t.get('domain') or {}).get('display_name'),
        })
    return out


def build_graphs(input_dir: Path, out_dir: Path, min_year: int, max_year: int,
                 min_edge_weight: int = 1, limit_batches: int | None = None) -> None:
    # Acumuladores por periodo
    node_freq: Dict[str, Counter] = defaultdict(Counter)  # period -> topic_id -> freq
    node_attrs: Dict[str, Dict[str, dict]] = defaultdict(dict)  # period -> topic_id -> attrs
    edge_weights: Dict[str, Counter] = defaultdict(Counter)  # period -> (u,v) -> weight

    batches = list(iter_metadata_files(input_dir))
    if limit_batches is not None:
        batches = batches[:limit_batches]
    print(f"Procesando {len(batches)} archivos de metadatos desde {input_dir}")

    for i, path in enumerate(batches, 1):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"[WARN] No se pudo leer {path.name}: {e}")
            continue
        works = data.get('works_data') or []
        for w in works:
            md = w.get('metadata') or {}
            year = md.get('publication_year')
            period = year_to_period(year, min_year, max_year)
            if period is None:
                continue
            topics = extract_topics(w)
            if not topics:
                continue
            # Actualizar frecuencias y atributos de nodos
            for t in topics:
                tid = t['id']
                node_freq[period][tid] += 1
                if tid not in node_attrs[period]:
                    node_attrs[period][tid] = t
            # Actualizar aristas por co-ocurrencia (pares únicos)
            # Evitar pares repetidos si un topic aparece duplicado
            unique_ids = sorted({t['id'] for t in topics})
            for u, v in combinations(unique_ids, 2):
                edge = (u, v)
                edge_weights[period][edge] += 1
        if i % 5 == 0:
            print(f"  ... {i}/{len(batches)} archivos procesados")

    out_dir.mkdir(parents=True, exist_ok=True)
    periods = sorted(edge_weights.keys() | node_freq.keys())
    for period in periods:
        # Construir grafo
        G = nx.Graph()
        # Nodos
        for tid, freq in node_freq[period].items():
            attrs = node_attrs[period].get(tid, {})
            G.add_node(tid, name=attrs.get('name', ''), frequency=int(freq),
                       field=attrs.get('field'), subfield=attrs.get('subfield'), domain=attrs.get('domain'))
        # Aristas
        for (u, v), w in edge_weights[period].items():
            if w >= min_edge_weight:
                if u not in G:
                    G.add_node(u)
                if v not in G:
                    G.add_node(v)
                G.add_edge(u, v, weight=int(w))

        # Escribir GraphML y CSV
        gml = out_dir / f"semantic_network_{period}.graphml"
        nodes_csv = out_dir / f"semantic_network_{period}_nodes.csv"
        edges_csv = out_dir / f"semantic_network_{period}_edges.csv"
        try:
            nx.write_graphml(G, gml)
        except Exception as e:
            print(
                f"[ERROR] No se pudo escribir {gml.name}: {e}. "
                "Sugerencia: actualiza networkx o verifica caracteres en atributos."
            )
        # CSVs
        if len(G) > 0:
            nd = []
            for n, a in G.nodes(data=True):
                nd.append({
                    'id': n,
                    'name': a.get('name'),
                    'frequency': a.get('frequency'),
                    'field': a.get('field'),
                    'subfield': a.get('subfield'),
                    'domain': a.get('domain'),
                })
            pd.DataFrame(nd).to_csv(nodes_csv, index=False, encoding='utf-8')

            ed = []
            for u, v, a in G.edges(data=True):
                ed.append({'source': u, 'target': v, 'weight': a.get('weight', 1)})
            pd.DataFrame(ed).to_csv(edges_csv, index=False, encoding='utf-8')
        print(f"✔️ {period}: N={G.number_of_nodes()} E={G.number_of_edges()} → {gml.name}")


def parse_args():
    ap = argparse.ArgumentParser(description='Construir redes GraphML desde Metadatos OpenAlex')
    ap.add_argument('--input-dir', type=str, default='Metadatos', help='Carpeta con batch_*_metadata.json')
    ap.add_argument('--out-dir', type=str, default='temporal_networks/networks', help='Salida de GraphML/CSV')
    ap.add_argument('--min-year', type=int, default=2000)
    ap.add_argument('--max-year', type=int, default=2023)
    ap.add_argument('--min-edge-weight', type=int, default=1, help='Umbral mínimo de co-ocurrencia para crear arista')
    ap.add_argument('--limit-batches', type=int, default=None, help='Procesar solo N archivos (prueba)')
    return ap.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    build_graphs(input_dir, out_dir, args.min_year, args.max_year,
                 min_edge_weight=args.min_edge_weight, limit_batches=args.limit_batches)


if __name__ == '__main__':
    main()
