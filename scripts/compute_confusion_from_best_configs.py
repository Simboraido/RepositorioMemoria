#!/usr/bin/env python3
"""
Calcula matrices de confusión para modelos validados usando las mejores configuraciones
preexistentes en validated_results/validated_ai_training_results.json, evitando re-optimización.
Genera archivos JSON en validated_results/confusion_matrices/*_validated_confusion_matrix.json.
"""
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from train_validated_ai_models import ValidatedAITrainer
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / 'validated_results'
CM_DIR = RESULTS_DIR / 'confusion_matrices'
CM_DIR.mkdir(parents=True, exist_ok=True)


def tensorize(pairs, labels, device):
    return (
        torch.tensor(pairs, dtype=torch.long).to(device),
        torch.tensor(labels, dtype=torch.float32).to(device)
    )


def train_one_config(model: torch.nn.Module,
                     train_data: Tuple[list, list],
                     config: dict,
                     device: torch.device) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.BCELoss()

    train_pairs, train_labels = train_data
    pairs_tensor, labels_tensor = tensorize(train_pairs, train_labels, device)

    best_loss = float('inf')
    patience_counter = 0
    num_batches = len(train_pairs) // config['batch_size'] + 1

    for epoch in range(config['max_epochs']):
        total_loss = 0.0
        for b in range(num_batches):
            s, e = b * config['batch_size'], min((b + 1) * config['batch_size'], len(train_pairs))
            if s >= len(train_pairs):
                break
            optimizer.zero_grad()
            outputs = model(pairs_tensor[s:e])
            loss = criterion(outputs, labels_tensor[s:e])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(num_batches, 1)
        if avg < best_loss:
            best_loss = avg
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config['patience']:
            break


def main():
    results_file = RESULTS_DIR / 'validated_ai_training_results.json'
    if not results_file.exists():
        print('No existe validated_ai_training_results.json')
        return

    data = json.loads(results_file.read_text(encoding='utf-8'))
    results = data.get('results', {})

    trainer = ValidatedAITrainer(str(BASE_DIR))

    # Preparar datos y split
    topic_mapping = trainer.load_topic_mapping()
    valid_ai_topics = trainer.identify_valid_ai_topics(topic_mapping)
    all_pairs, all_labels = trainer.create_synthetic_ai_data(valid_ai_topics, topic_mapping)

    # Split estratificado idéntico al entrenamiento validado
    import random
    random.seed(42)
    pos_idx = [i for i, y in enumerate(all_labels) if y == 1]
    neg_idx = [i for i, y in enumerate(all_labels) if y == 0]
    split_pos = int(0.8 * len(pos_idx))
    split_neg = int(0.8 * len(neg_idx))
    train_idx = pos_idx[:split_pos] + neg_idx[:split_neg]
    valid_idx = pos_idx[split_pos:] + neg_idx[split_neg:]
    random.shuffle(train_idx)
    random.shuffle(valid_idx)

    train_pairs = [all_pairs[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    valid_pairs = [all_pairs[i] for i in valid_idx]
    valid_labels = [all_labels[i] for i in valid_idx]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name, info in results.items():
        cfg = info.get('best_config')
        if not cfg:
            print(f'{model_name}: sin best_config; omitido')
            continue

        print(f'Calculando matriz de confusión para {model_name}...')
        num_nodes = max(max(u, v) for u, v in all_pairs) + 1
        model = trainer.model_classes[model_name](num_nodes=num_nodes).to(device)

        # Entrenar con la mejor config en el split de entrenamiento
        train_one_config(model, (train_pairs, train_labels), cfg, device)

        # Evaluación en validación
        model.eval()
        with torch.no_grad():
            v_pairs_tensor, _ = tensorize(valid_pairs, valid_labels, device)
            y_scores = model(v_pairs_tensor).cpu().numpy()
            y_true = np.array(valid_labels)
            auc = roc_auc_score(y_true, y_scores)
            y_pred = (y_scores > cfg.get('threshold', 0.5)).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        out = {
            'model': model_name,
            'threshold': cfg.get('threshold', 0.5),
            'auc_roc': float(auc),
            'f1_score': float(f1),
            'precision': float(prec),
            'recall': float(rec),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        }
        out_path = CM_DIR / f"{model_name.lower()}_validated_confusion_matrix.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Guardado: {out_path}')

        # Exportar CSV con y_true e y_score para optimización de threshold
        try:
            df = pd.DataFrame({
                'u': [u for u, _ in valid_pairs],
                'v': [v for _, v in valid_pairs],
                'y_true': y_true.astype(int),
                'y_score': y_scores.astype(float)
            })
            csv_path = RESULTS_DIR / f"{model_name.lower()}_validation_scores.csv"
            df.to_csv(csv_path, index=False)
            print(f'Guardado CSV: {csv_path}')
        except Exception as e:
            print(f'⚠️ No se pudo guardar CSV de scores para {model_name}: {e}')

        # -------- Disolución: generar y_true/y_score (reales t->t+1) --------
        try:
            dissolved_pairs_eval, persisted_pairs_eval = trainer.create_real_dissolution_pairs(topic_mapping)
            if len(dissolved_pairs_eval) == 0 or len(persisted_pairs_eval) == 0:
                print(f'⚠️ Disolución omitida para {model_name}: no hay pares reales (disueltos o persistentes)')
            else:
                diss_arr = np.array(dissolved_pairs_eval, dtype=np.int64).reshape(-1, 2)
                pers_arr = np.array(persisted_pairs_eval, dtype=np.int64).reshape(-1, 2)
                v_diss_tensor, _ = tensorize(diss_arr, np.ones(len(diss_arr)), device)
                v_pers_tensor, _ = tensorize(pers_arr, np.zeros(len(pers_arr)), device)
                model.eval()
                with torch.no_grad():
                    diss_scores_exist = np.asarray(model(v_diss_tensor).cpu().numpy()).reshape(-1)
                    pers_scores_exist = np.asarray(model(v_pers_tensor).cpu().numpy()).reshape(-1)
                # score de disolución = 1 - score de existencia
                y_scores_diss = np.concatenate([1 - diss_scores_exist, 1 - pers_scores_exist]).reshape(-1)
                y_true_diss = np.concatenate([
                    np.ones(diss_scores_exist.shape[0]),
                    np.zeros(pers_scores_exist.shape[0])
                ]).astype(int).reshape(-1)

                # Construir DataFrame (incluye pares si están disponibles)
                u_diss = [int(u) for u, _ in dissolved_pairs_eval]
                v_diss = [int(v) for _, v in dissolved_pairs_eval]
                u_pers = [int(u) for u, _ in persisted_pairs_eval]
                v_pers = [int(v) for _, v in persisted_pairs_eval]
                u_all = u_diss + u_pers
                v_all = v_diss + v_pers
                df_diss = pd.DataFrame({
                    'u': u_all,
                    'v': v_all,
                    'y_true': y_true_diss,
                    'y_score': y_scores_diss.astype(float)
                })
                csv_diss_path = RESULTS_DIR / f"{model_name.lower()}_dissolution_scores.csv"
                df_diss.to_csv(csv_diss_path, index=False)
                print(f'Guardado CSV (Disolución): {csv_diss_path}')
        except Exception as e:
            print(f'⚠️ No se pudo preparar scores de disolución para {model_name}: {e}')

    print('Listo.')


if __name__ == '__main__':
    main()
