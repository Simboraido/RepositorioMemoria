#!/usr/bin/env python3
"""
Predicción futura (2024-2025 y 2026-2027) con el mejor modelo validado (mayor F1)
--------------------------------------------------------------------------------

Flujo:
1) Lee validated_results/validated_ai_training_results.json y selecciona el modelo con mayor F1.
2) Importa dinámicamente la clase del modelo desde scripts/train_validated_ai_models.py.
3) Carga el checkpoint correspondiente en models/<model>_validated_ai.pth.
4) Carga la última red real (GraphML) y genera candidatos de dos saltos (pares no conectados con ≥1 vecino común).
5) Puntúa los candidatos con el modelo y guarda rankings para 2024-2025 y 2026-2027 en predictions/.
"""

from pathlib import Path
import json
import os
import glob
import time
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
VALIDATED_DIR = BASE_DIR / 'validated_results'
MODELS_DIR = BASE_DIR / 'models'
NETWORKS_DIR = BASE_DIR / 'temporal_networks' / 'networks'
PRED_DIR = BASE_DIR / 'predictions'
MAPPING_FILE = PRED_DIR / 'topic_id_to_name_mapping.json'


def load_best_model_info() -> Tuple[str, Dict]:
    results_file = VALIDATED_DIR / 'validated_ai_training_results.json'
    if not results_file.exists():
        raise FileNotFoundError(f"No existe {results_file}")
    data = json.loads(results_file.read_text(encoding='utf-8'))
    results = data.get('results', {})
    if not results:
        raise RuntimeError('Sin resultados en validated_ai_training_results.json')
    # Elegir por F1
    best_name = None
    best_cfg = None
    best_f1 = -1.0
    for name, info in results.items():
        cfg = info.get('best_config') or {}
        f1 = cfg.get('f1_score')
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_cfg = cfg
    if best_name is None:
        raise RuntimeError('No se pudo determinar el mejor modelo por F1')
    return best_name, best_cfg


def import_model_class(model_name: str):
    import importlib.util
    src = BASE_DIR / 'scripts' / 'train_validated_ai_models.py'
    spec = importlib.util.spec_from_file_location('validated_models', src)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    mapping = {
        'A3TGCN': 'A3TGCNModel',
        'EvolveGCN-H': 'EvolveGCNHModel',
        'TDGNN': 'TDGNNModel',
        'TGN-Simple': 'TGNSimpleModel',
    }
    cls_name = mapping.get(model_name)
    if not cls_name:
        raise ValueError(f"Modelo no soportado: {model_name}")
    return getattr(module, cls_name)


def load_checkpoint_path(model_name: str) -> Path:
    fname = f"{model_name.lower()}_validated_ai.pth"
    return MODELS_DIR / fname


def load_topic_mapping() -> Dict:
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"No existe {MAPPING_FILE}")
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def originalid_to_index_and_name(topic_mapping: Dict) -> Tuple[Dict[str, int], Dict[str, str]]:
    id2idx, id2name = {}, {}
    for _, info in topic_mapping.items():
        orig = info.get('original_id')
        idx = info.get('index')
        name = info.get('display_name')
        if orig is not None and idx is not None:
            id2idx[orig] = int(idx)
            id2name[orig] = name
    return id2idx, id2name


def load_last_graph() -> Tuple[str, nx.Graph]:
    files = sorted(glob.glob(str(NETWORKS_DIR / 'semantic_network_*.graphml')))
    if not files:
        raise FileNotFoundError(f"No se encontraron GraphML en {NETWORKS_DIR}")
    last = files[-1]
    period = os.path.basename(last).replace('semantic_network_', '').replace('.graphml', '')
    G = nx.read_graphml(last)
    return period, G


def candidate_pairs_two_hops(G: nx.Graph, max_per_node: int = 5000) -> List[Tuple[str, str]]:
    seen = set()
    pairs = []
    for u in G.nodes():
        nbrs = set(G.neighbors(u))
        two_hops = set()
        for v in nbrs:
            two_hops.update(G.neighbors(v))
        two_hops.discard(u)
        cand = [w for w in two_hops if w not in nbrs and not G.has_edge(u, w)]
        if max_per_node:
            cand = cand[:max_per_node]
        for w in cand:
            a, b = (u, w) if str(u) < str(w) else (w, u)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def predict_with_model(model, device, pairs_idx: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs_idx), batch_size):
            batch = torch.from_numpy(pairs_idx[i:i+batch_size]).long().to(device)
            out = model(batch)
            scores.append(out.detach().cpu().numpy())
    if not scores:
        return np.array([])
    return np.concatenate(scores)


def main():
    model_name, best_cfg = load_best_model_info()
    print(f"🏆 Mejor modelo por F1: {model_name} | F1={best_cfg.get('f1_score'):.4f} | AUC={best_cfg.get('auc_roc'):.4f}")

    # Importar clase y cargar checkpoint
    ModelClass = import_model_class(model_name)
    ckpt_path = load_checkpoint_path(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")
    # PyTorch 2.6+: fallback a weights_only=False si es necesario
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    num_nodes = int(ckpt.get('num_nodes', 0))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Si num_nodes no está en el checkpoint, inferir desde el mapeo de tópicos
    if num_nodes <= 0:
        try:
            topic_mapping = load_topic_mapping()
            num_nodes = len([1 for v in topic_mapping.values() if v.get('index') is not None])
            print(f"ℹ️ num_nodes inferido desde mapping: {num_nodes}")
        except Exception:
            pass
    model = ModelClass(num_nodes=num_nodes).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✅ Modelo cargado desde {ckpt_path.name} (num_nodes={num_nodes})")

    # Cargar datos y candidatos
    period, G = load_last_graph()
    id2idx, id2name = originalid_to_index_and_name(load_topic_mapping())
    print(f"📅 Último período: {period} | Nodos={G.number_of_nodes()} | Aristas={G.number_of_edges()}")

    pairs = candidate_pairs_two_hops(G)
    print(f"🔎 Candidatos generados: {len(pairs)}")
    if not pairs:
        print("❌ No hay candidatos; nada que predecir.")
        return

    # Mapear a índices; filtrar pares que no estén en el mapeo
    mapped = []
    for u, v in pairs:
        if u in id2idx and v in id2idx:
            mapped.append([id2idx[u], id2idx[v]])
    if not mapped:
        print("❌ Ningún candidato mapeado a índices; revisar mapeo.")
        return

    pairs_idx = np.array(mapped, dtype=np.int64)

    # Predecir probabilidad de aparición (score de existencia)
    scores = predict_with_model(model, device, pairs_idx)
    print(f"📈 Puntajes calculados: {len(scores)}")

    # Construir DataFrame resultados
    df = pd.DataFrame({
        'u_idx': pairs_idx[:, 0],
        'v_idx': pairs_idx[:, 1],
        'score': scores,
    })

    # Añadir IDs originales y nombres (reversa de id2idx)
    # Como pairs venían en IDs originales, reutilizamos esa lista en el mismo orden que mapped
    kept_pairs = [(u, v) for (u, v) in pairs if u in id2idx and v in id2idx]
    df['u'] = [u for u, _ in kept_pairs]
    df['v'] = [v for _, v in kept_pairs]
    df['u_name'] = df['u'].map(lambda x: id2name.get(x, str(x)))
    df['v_name'] = df['v'].map(lambda x: id2name.get(x, str(x)))

    # Guardar JSON/CSV para ambos periodos
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        'model': model_name,
        'f1_score': best_cfg.get('f1_score'),
        'auc_roc': best_cfg.get('auc_roc'),
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'base_period': period,
        'num_candidates': int(len(df)),
    }

    def save_for(label: str):
        out_json = PRED_DIR / f'predicted_links_{label}_by_best_model.json'
        out_csv = PRED_DIR / f'predicted_links_{label}_by_best_model.csv'
        top = df.sort_values('score', ascending=False).head(1000)
        records = top[['u','v','u_name','v_name','score']].to_dict(orient='records')
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({**meta, 'target_period': label, 'predictions': records}, f, indent=2, ensure_ascii=False)
        top[['u','v','u_name','v_name','score']].to_csv(out_csv, index=False, encoding='utf-8')
        print(f"✅ Guardado: {out_json}")
        print(f"✅ Guardado: {out_csv}")

    save_for('2024_2025')
    save_for('2025_2026')
    save_for('2026_2027')
    print("🎉 Predicción con mejor modelo completada.")


if __name__ == '__main__':
    main()
