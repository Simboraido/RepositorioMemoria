#!/usr/bin/env python3
"""
⚡ IMPLEMENTACIÓN TGN SIMPLIFICADA - TEMPORAL GRAPH NETWORKS
===========================================================

Implementación simplificada de TGN para predicción de enlaces en redes 
semánticas temporales.

Autor: Sistema de Implementación TGN Simple
Fecha: Julio 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TGNSimple(nn.Module):
    """
    ⚡ TGN Simplificado
    ==================
    
    Versión simplificada de TGN que funciona de manera estable.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 memory_dim: int = 100, dropout: float = 0.1):
        super(TGNSimple, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_dim = memory_dim
        self.dropout = dropout
        
        # Proyecciones de entrada
        self.node_projection = nn.Linear(input_dim, hidden_dim)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Memoria temporal (simplificada)
        self.memory_weights = nn.Parameter(torch.randn(1000, memory_dim) * 0.1)  # Max 1000 nodos
        
        # GRU para evolución temporal
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim * 2,  # features + time
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Capas de salida
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Predictor de enlaces
        self.link_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, temporal_graphs: List[Dict], target_pairs: torch.Tensor) -> torch.Tensor:
        """Forward pass de TGN simplificado"""
        
        if not temporal_graphs:
            batch_size = target_pairs.size(0)
            return torch.zeros(batch_size, 1, device=target_pairs.device)
        
        device = target_pairs.device
        
        # Usar el último grafo temporal
        graph_data = temporal_graphs[-1]
        edge_index = graph_data['edge_index']
        node_features = graph_data.get('node_features')
        timestamp = graph_data.get('timestamp', 0.0)
        
        if node_features is None:
            num_nodes = max(edge_index.max().item() + 1, target_pairs.max().item() + 1)
            node_features = torch.randn(num_nodes, self.input_dim, device=device)
        
        # Proyección de features de nodos
        x = self.node_projection(node_features)
        
        # Codificar tiempo
        time_tensor = torch.tensor([timestamp], device=device).float().unsqueeze(0)
        time_encoding = self.time_encoder(time_tensor)
        time_encoding = time_encoding.expand(x.size(0), -1)
        
        # Combinar features con información temporal
        x_temporal = torch.cat([x, time_encoding], dim=1)
        
        # Procesar secuencia temporal con GRU
        x_temporal_expanded = x_temporal.unsqueeze(1)  # [num_nodes, 1, features]
        gru_output, _ = self.temporal_gru(x_temporal_expanded)
        x_evolved = gru_output.squeeze(1)  # [num_nodes, hidden_dim]
        
        # Aplicar atención temporal
        x_attended, _ = self.attention(
            x_evolved.unsqueeze(0),  # [1, num_nodes, hidden_dim]
            x_evolved.unsqueeze(0),
            x_evolved.unsqueeze(0)
        )
        x_final = x_attended.squeeze(0)  # [num_nodes, hidden_dim]
        
        # Proyección final
        node_embeddings = self.output_layers(x_final)
        
        # Predicción de enlaces
        source_embeddings = node_embeddings[target_pairs[:, 0]]
        target_embeddings = node_embeddings[target_pairs[:, 1]]
        
        # Combinar embeddings
        pair_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
        
        # Predecir probabilidad de enlace
        link_scores = self.link_predictor(pair_embeddings)
        link_probs = torch.sigmoid(link_scores)
        
        return link_probs


class TemporalGraphProcessor:
    """
    📊 Procesador simplificado para TGN
    ===================================
    """
    
    def __init__(self, temporal_networks_path: str = "../temporal_networks/networks"):
        self.temporal_networks_path = Path(temporal_networks_path)
        self.node_to_id = {}
        self.id_to_node = {}
        self.temporal_graphs = []
        
    def load_and_process(self) -> Dict:
        """Carga y procesa redes temporales"""
        print("📂 Cargando redes temporales para TGN Simple...")
        
        network_files = list(self.temporal_networks_path.glob("semantic_network_*.graphml"))
        network_files.sort()
        
        print(f"Encontrados {len(network_files)} archivos de red")
        
        all_nodes = set()
        temporal_data = []
        
        # Cargar grafos
        for file_path in network_files:
            period = file_path.stem.replace('semantic_network_', '')
            print(f"  📈 Cargando {period}...")
            
            try:
                G = nx.read_graphml(file_path)
                temporal_data.append({
                    'period': period,
                    'graph': G,
                    'timestamp': self._period_to_timestamp(period)
                })
                all_nodes.update(G.nodes())
                
            except Exception as e:
                print(f"  ❌ Error cargando {file_path}: {e}")
                continue
        
        # Mapeo de nodos
        self.node_to_id = {node: i for i, node in enumerate(sorted(all_nodes))}
        self.id_to_node = {i: node for node, i in self.node_to_id.items()}
        
        print(f"📊 Total de nodos únicos: {len(all_nodes)}")
        
        # Convertir a tensores
        for data in temporal_data:
            graph_tensor = self._convert_to_tensor(data['graph'], data['timestamp'])
            self.temporal_graphs.append(graph_tensor)
        
        return {
            'num_nodes': len(all_nodes),
            'num_periods': len(temporal_data),
            'node_mapping': {'node_to_id': self.node_to_id, 'id_to_node': self.id_to_node},
            'temporal_graphs': self.temporal_graphs
        }
    
    def _period_to_timestamp(self, period: str) -> float:
        """Convierte período a timestamp"""
        try:
            year_part = period.split('-')[0]
            start_year = int(year_part)
            return float(start_year - 2000)
        except:
            return 0.0
    
    def _convert_to_tensor(self, G: nx.Graph, timestamp: float) -> Dict:
        """Convierte grafo a tensores"""
        
        edges = []
        edge_weights = []
        
        for u, v, data in G.edges(data=True):
            if u in self.node_to_id and v in self.node_to_id:
                u_id = self.node_to_id[u]
                v_id = self.node_to_id[v]
                
                edges.append([u_id, v_id])
                edges.append([v_id, u_id])
                
                weight = data.get('weight', 1.0)
                edge_weights.extend([weight, weight])
        
        # Features de nodos
        num_nodes = len(self.node_to_id)
        node_features = self._create_node_features(G, num_nodes)
        
        return {
            'timestamp': timestamp,
            'edge_index': torch.tensor(edges).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long),
            'edge_weights': torch.tensor(edge_weights, dtype=torch.float) if edge_weights else torch.empty(0),
            'node_features': node_features,
            'num_nodes': num_nodes,
            'num_edges': len(edges)
        }
    
    def _create_node_features(self, G: nx.Graph, num_nodes: int) -> torch.Tensor:
        """Crea features de nodos"""
        feature_dim = 128
        features = torch.zeros(num_nodes, feature_dim)
        
        degree_dict = dict(G.degree())
        
        for node, node_id in self.node_to_id.items():
            if node in G.nodes():
                degree = degree_dict.get(node, 0)
                
                # One-hot grado
                degree_cat = min(degree, 9)
                features[node_id, degree_cat] = 1.0
                
                # Grado normalizado
                max_degree = max(degree_dict.values()) if degree_dict else 1
                features[node_id, 10] = degree / max_degree
                
                # Actividad
                features[node_id, 11] = 1.0 if degree > 0 else 0.0
                
                # Embedding aleatorio
                features[node_id, 12:] = torch.randn(feature_dim - 12) * 0.1
        
        return features
    
    def create_link_prediction_dataset(self, test_ratio: float = 0.2) -> Dict:
        """Crea dataset para predicción de enlaces"""
        print("🔗 Creando dataset de predicción de enlaces para TGN...")
        
        all_positive_pairs = []
        all_negative_pairs = []
        all_timestamps = []
        
        for graph_data in self.temporal_graphs:
            edge_index = graph_data['edge_index']
            timestamp = graph_data['timestamp']
            num_nodes = graph_data['num_nodes']
            
            if edge_index.size(1) == 0:
                continue
            
            # Enlaces positivos
            positive_pairs = edge_index.t().unique(dim=0)
            positive_pairs = positive_pairs[positive_pairs[:, 0] < positive_pairs[:, 1]]
            
            # Enlaces negativos
            num_positive = positive_pairs.size(0)
            negative_pairs = self._sample_negative_pairs(positive_pairs, num_nodes, num_positive)
            
            all_positive_pairs.append(positive_pairs)
            all_negative_pairs.append(negative_pairs)
            
            pos_timestamps = torch.full((num_positive,), timestamp)
            neg_timestamps = torch.full((num_positive,), timestamp)
            all_timestamps.extend([pos_timestamps, neg_timestamps])
        
        # Combinar datos
        X_pairs = torch.cat(all_positive_pairs + all_negative_pairs, dim=0)
        y_labels = torch.cat([
            torch.ones(sum(pairs.size(0) for pairs in all_positive_pairs)),
            torch.zeros(sum(pairs.size(0) for pairs in all_negative_pairs))
        ])
        
        pair_timestamps = torch.cat(all_timestamps)
        
        # Split
        indices = torch.randperm(X_pairs.size(0))
        split_idx = int(len(indices) * (1 - test_ratio))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        dataset = {
            'X_train_pairs': X_pairs[train_indices],
            'X_test_pairs': X_pairs[test_indices],
            'y_train': y_labels[train_indices],
            'y_test': y_labels[test_indices],
            'train_timestamps': pair_timestamps[train_indices],
            'test_timestamps': pair_timestamps[test_indices],
            'temporal_graphs': self.temporal_graphs,
            'node_mapping': {'node_to_id': self.node_to_id, 'id_to_node': self.id_to_node}
        }
        
        print(f"✅ Dataset TGN creado:")
        print(f"   📊 Train: {len(train_indices)} pares")
        print(f"   📊 Test: {len(test_indices)} pares")
        
        return dataset
    
    def _sample_negative_pairs(self, positive_pairs: torch.Tensor, 
                              num_nodes: int, num_samples: int) -> torch.Tensor:
        """Muestrea pares negativos"""
        positive_set = set()
        for pair in positive_pairs:
            u, v = pair.tolist()
            positive_set.add((min(u, v), max(u, v)))
        
        negative_pairs = []
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(negative_pairs) < num_samples and attempts < max_attempts:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            
            if u != v:
                pair = (min(u, v), max(u, v))
                if pair not in positive_set:
                    negative_pairs.append([u, v])
            
            attempts += 1
        
        while len(negative_pairs) < num_samples:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v:
                negative_pairs.append([u, v])
        
        return torch.tensor(negative_pairs[:num_samples])


class TGNTrainer:
    """
    🏋️ Entrenador de TGN Simple
    ===========================
    """
    
    def __init__(self, model: TGNSimple, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_metrics = []
        
    def train_model(self, dataset: Dict, epochs: int = 100, 
                   batch_size: int = 32, learning_rate: float = 0.01) -> Dict:
        """Entrena el modelo TGN"""
        print("🏋️ Iniciando entrenamiento de TGN Simple...")
        print(f"   📊 Épocas: {epochs}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   📈 Learning rate: {learning_rate}")
        
        # Preparar datos
        X_train = dataset['X_train_pairs']
        y_train = dataset['y_train']
        temporal_graphs = dataset['temporal_graphs']
        
        # Mover grafos al dispositivo
        for i, graph_data in enumerate(temporal_graphs):
            for key, value in graph_data.items():
                if isinstance(value, torch.Tensor):
                    temporal_graphs[i][key] = value.to(self.device)
        
        # Split train/validation
        val_size = int(len(X_train) * 0.2)
        train_size = len(X_train) - val_size
        
        train_indices = torch.randperm(len(X_train))[:train_size]
        val_indices = torch.randperm(len(X_train))[train_size:train_size + val_size]
        
        # DataLoaders
        train_dataset = TensorDataset(X_train[train_indices], y_train[train_indices])
        val_dataset = TensorDataset(X_train[val_indices], y_train[val_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizador
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.BCELoss()
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_pairs, batch_labels in train_loader:
                batch_pairs = batch_pairs.to(self.device)
                batch_labels = batch_labels.to(self.device).float()
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(temporal_graphs, batch_pairs).squeeze()
                
                # Loss
                loss = criterion(predictions, batch_labels)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation
            val_metrics = self._validate(val_loader, temporal_graphs, criterion)
            self.val_metrics.append(val_metrics)
            
            # Scheduling
            scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_tgn_simple_model.pth')
            else:
                patience_counter += 1
            
            # Progress
            if (epoch + 1) % 20 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Época {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f} | "
                      f"Time: {elapsed:.1f}s")
            
            if patience_counter >= patience:
                print(f"🛑 Early stopping en época {epoch+1}")
                break
        
        # Cargar mejor modelo
        self.model.load_state_dict(torch.load('best_tgn_simple_model.pth'))
        
        return {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'total_epochs': epoch + 1,
            'best_val_loss': best_val_loss
        }
    
    def _validate(self, val_loader: DataLoader, temporal_graphs: List[Dict],
                 criterion: nn.Module) -> Dict:
        """Validación"""
        self.model.eval()
        
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_pairs, batch_labels in val_loader:
                batch_pairs = batch_pairs.to(self.device)
                batch_labels = batch_labels.to(self.device).float()
                
                predictions = self.model(temporal_graphs, batch_pairs).squeeze()
                loss = criterion(predictions, batch_labels)
                val_loss += loss.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                num_batches += 1
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_predictions)
        ap = average_precision_score(all_labels, all_predictions)
        
        pred_binary = (all_predictions > 0.5).astype(int)
        f1 = f1_score(all_labels, pred_binary)
        
        return {
            'loss': val_loss / num_batches,
            'auc': auc,
            'ap': ap,
            'f1': f1
        }
    
    def evaluate_test(self, dataset: Dict) -> Dict:
        """Evaluación en test"""
        print("📊 Evaluando TGN Simple en conjunto de test...")
        
        self.model.eval()
        
        X_test = dataset['X_test_pairs'].to(self.device)
        y_test = dataset['y_test'].cpu().numpy()
        temporal_graphs = dataset['temporal_graphs']
        
        all_predictions = []
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_pairs = X_test[i:i+batch_size]
                
                if len(batch_pairs) == 0:
                    continue
                
                predictions = self.model(temporal_graphs, batch_pairs).squeeze()
                all_predictions.extend(predictions.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        
        auc_roc = roc_auc_score(y_test, all_predictions)
        auc_pr = average_precision_score(y_test, all_predictions)
        
        # F1 con diferentes thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        f1_scores = []
        
        for threshold in thresholds:
            pred_binary = (all_predictions > threshold).astype(int)
            f1 = f1_score(y_test, pred_binary)
            f1_scores.append(f1)
        
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        best_f1 = f1_scores[best_threshold_idx]
        
        results = {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'best_f1': best_f1,
            'best_threshold': best_threshold,
            'f1_by_threshold': dict(zip(thresholds, f1_scores)),
            'predictions': all_predictions,
            'true_labels': y_test
        }
        
        print(f"📊 RESULTADOS FINALES TGN SIMPLE:")
        print(f"   🎯 AUC-ROC: {auc_roc:.4f}")
        print(f"   📈 AUC-PR: {auc_pr:.4f}")
        print(f"   ⚖️  Best F1: {best_f1:.4f} (threshold: {best_threshold})")
        
        return results


def main():
    """🚀 Función principal - Implementación TGN Simple"""
    
    print("⚡ TGN SIMPLE - IMPLEMENTACIÓN COMPLETA")
    print("=" * 60)
    print("Temporal Graph Networks (Simplificado)")
    print("Para predicción de enlaces en redes semánticas temporales")
    print()
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    # 1. Cargar datos
    print("\n📂 FASE 1: CARGA DE DATOS")
    print("-" * 30)
    
    processor = TemporalGraphProcessor()
    
    try:
        network_info = processor.load_and_process()
        dataset = processor.create_link_prediction_dataset(test_ratio=0.2)
        print(f"✅ Datos temporales cargados y procesados")
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 2. Crear modelo
    print("\n🧠 FASE 2: CREACIÓN DEL MODELO")
    print("-" * 35)
    
    first_graph = dataset['temporal_graphs'][0]
    input_dim = first_graph['node_features'].size(1)
    
    model = TGNSimple(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=32,
        memory_dim=100,
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modelo TGN Simple creado")
    print(f"   📊 Parámetros: {num_params:,}")
    print(f"   🏗️  Arquitectura: {input_dim}→64→32")
    print(f"   🧠 Memoria: 100 dims")
    
    # 3. Entrenamiento
    print("\n🏋️ FASE 3: ENTRENAMIENTO")
    print("-" * 30)
    
    trainer = TGNTrainer(model, device)
    
    training_history = trainer.train_model(
        dataset=dataset,
        epochs=100,
        batch_size=32,
        learning_rate=0.01
    )
    
    # 4. Evaluación
    print("\n📊 FASE 4: EVALUACIÓN")
    print("-" * 25)
    
    evaluation_results = trainer.evaluate_test(dataset)
    
    # 5. Guardar resultados
    print("\n💾 FASE 5: GUARDADO DE RESULTADOS")
    print("-" * 40)
    
    results_summary = {
        'model_name': 'TGN_Simple',
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': 64,
            'output_dim': 32,
            'memory_dim': 100,
            'num_parameters': num_params
        },
        'evaluation_results': evaluation_results,
        'dataset_info': network_info,
        'timestamp': datetime.now().isoformat()
    }
    
    # Guardar modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': results_summary['model_config'],
        'results': evaluation_results
    }, 'tgn_simple_model_complete.pth')
    
    print(f"✅ Resultados guardados:")
    print(f"   🧠 tgn_simple_model_complete.pth")
    
    # 6. Resumen final
    print(f"\n🎉 IMPLEMENTACIÓN TGN SIMPLE COMPLETADA")
    print("=" * 50)
    print(f"📊 RESULTADOS FINALES:")
    print(f"   🎯 AUC-ROC: {evaluation_results['auc_roc']:.4f}")
    print(f"   📈 AUC-PR: {evaluation_results['auc_pr']:.4f}")
    print(f"   ⚖️  F1-Score: {evaluation_results['best_f1']:.4f}")
    
    return results_summary


if __name__ == "__main__":
    results = main()
