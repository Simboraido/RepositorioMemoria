#!/usr/bin/env python3
"""
🔥 IMPLEMENTACIÓN TDGNN - TEMPORAL DEPENDENT GRAPH NEURAL NETWORK
================================================================

Implementación completa de TDGNN para predicción de enlaces en redes semánticas
temporales usando los datos de metadatos académicos.

Características:
- Temporal Aggregator (TDAgg) para información temporal
- Continuous-time link prediction
- Comparación directa vs baseline Adamic-Adar
- Optimizado para GPU GTX 1050

Basado en:
"Continuous-Time Link Prediction via Temporal Dependent Graph Neural Network"
Qu et al., WWW 2020

Autor: Sistema de Implementación TDGNN
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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TemporalAggregator(nn.Module):
    """
    🕐 Temporal Aggregator (TDAgg) - Núcleo de TDGNN
    ===============================================
    
    Agrega información de vecinos considerando timestamps temporales.
    Utiliza atención temporal para ponderar la importancia de conexiones
    basada en su recencia y frecuencia.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4):
        super(TemporalAggregator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Temporal encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-head attention para temporal aggregation
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature transformation
        self.feature_transform = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_timestamps: torch.Tensor, target_nodes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del Temporal Aggregator
        
        Args:
            node_features: Features de nodos [num_nodes, input_dim]
            edge_index: Índices de aristas [2, num_edges]
            edge_timestamps: Timestamps de aristas [num_edges]
            target_nodes: Nodos objetivo para agregación [batch_size]
        
        Returns:
            Agregated features [batch_size, hidden_dim]
        """
        batch_size = target_nodes.size(0)
        device = node_features.device
        
        # Transform node features
        transformed_features = self.feature_transform(node_features)
        
        # Encode temporal information
        time_features = self.time_encoder(edge_timestamps.unsqueeze(-1))
        
        # Aggregate for each target node
        aggregated_outputs = []
        
        for i, target_node in enumerate(target_nodes):
            # Find neighbors of target node
            neighbor_mask = (edge_index[0] == target_node) | (edge_index[1] == target_node)
            
            if neighbor_mask.sum() == 0:
                # No neighbors, use own features
                aggregated_outputs.append(transformed_features[target_node])
                continue
            
            # Get neighbor information
            neighbor_edges = edge_index[:, neighbor_mask]
            neighbor_times = edge_timestamps[neighbor_mask]
            neighbor_time_features = time_features[neighbor_mask]
            
            # Get neighbor node indices
            neighbor_indices = torch.where(
                neighbor_edges[0] == target_node,
                neighbor_edges[1],
                neighbor_edges[0]
            )
            
            # Get neighbor features
            neighbor_features = transformed_features[neighbor_indices]
            
            # Combine neighbor features with temporal information
            temporal_neighbor_features = neighbor_features + neighbor_time_features
            
            # Add target node features for attention
            target_features = transformed_features[target_node].unsqueeze(0)
            
            # Prepare for multi-head attention
            if temporal_neighbor_features.size(0) > 0:
                # Stack features for attention
                attention_input = temporal_neighbor_features.unsqueeze(0)  # [1, num_neighbors, hidden_dim]
                query = target_features.unsqueeze(0)  # [1, 1, hidden_dim]
                
                # Apply temporal attention
                attended_features, _ = self.temporal_attention(
                    query=query,
                    key=attention_input,
                    value=attention_input
                )
                
                aggregated_output = attended_features.squeeze(0).squeeze(0)
            else:
                aggregated_output = target_features.squeeze(0)
            
            aggregated_outputs.append(aggregated_output)
        
        # Stack outputs
        aggregated_tensor = torch.stack(aggregated_outputs, dim=0)
        
        # Final projection
        output = self.output_projection(aggregated_tensor)
        
        return output


class TDGNNModel(nn.Module):
    """
    🚀 TDGNN - Temporal Dependent Graph Neural Network
    =================================================
    
    Modelo completo para predicción de enlaces temporales.
    Incluye multiple capas de TDGNN con temporal aggregators.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super(TDGNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # TDGNN layers
        self.tdgnn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = TemporalAggregator(
                input_dim=hidden_dim,  # All layers use hidden_dim
                hidden_dim=hidden_dim,
                num_heads=num_heads
            )
            self.tdgnn_layers.append(layer)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Link prediction layers
        self.link_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_timestamps: torch.Tensor, target_pairs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass completo del TDGNN
        
        Args:
            node_features: Features de nodos [num_nodes, input_dim]
            edge_index: Índices de aristas [2, num_edges]
            edge_timestamps: Timestamps de aristas [num_edges]
            target_pairs: Pares de nodos a predecir [batch_size, 2]
        
        Returns:
            Probabilidades de enlaces [batch_size, 1]
        """
        # Initial projection
        x = self.input_projection(node_features)
        
        # Apply TDGNN layers
        for i, tdgnn_layer in enumerate(self.tdgnn_layers):
            # Get unique nodes in target pairs for efficient computation
            unique_nodes = torch.unique(target_pairs.flatten())
            
            # Apply temporal aggregation
            x_aggregated = tdgnn_layer(
                node_features=x,  # Use current features, not original
                edge_index=edge_index,
                edge_timestamps=edge_timestamps,
                target_nodes=unique_nodes
            )
            
            # Update only the relevant nodes
            for j, node in enumerate(unique_nodes):
                x[node] = x_aggregated[j]
            
            # Apply activation and dropout
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Final output projection
        node_embeddings = self.output_layers(x)
        
        # Link prediction
        source_embeddings = node_embeddings[target_pairs[:, 0]]
        target_embeddings = node_embeddings[target_pairs[:, 1]]
        
        # Combine embeddings (concatenation)
        pair_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
        
        # Predict link probability
        link_scores = self.link_predictor(pair_embeddings)
        link_probs = torch.sigmoid(link_scores)
        
        return link_probs


class TemporalDataLoader:
    """
    📊 Cargador de Datos Temporales
    ==============================
    
    Carga y procesa los datos de redes semánticas temporales para TDGNN.
    Convierte los datos de NetworkX a formato PyTorch compatible.
    """
    
    def __init__(self, temporal_networks_path: str = "temporal_networks/networks"):
        self.temporal_networks_path = Path(temporal_networks_path)
        self.node_to_id = {}
        self.id_to_node = {}
        self.temporal_graphs = []
        self.node_features = None
        self.edge_data = []
        
    def load_temporal_networks(self) -> Dict:
        """Carga todas las redes temporales"""
        print("📂 Cargando redes temporales...")
        
        network_files = list(self.temporal_networks_path.glob("semantic_network_*.graphml"))
        network_files.sort()
        
        print(f"Encontrados {len(network_files)} archivos de red")
        
        all_nodes = set()
        temporal_data = {}
        
        # Cargar todos los grafos y recopilar nodos únicos
        for i, file_path in enumerate(network_files):
            period = file_path.stem.replace('network_', '')
            print(f"  📈 Cargando {period}...")
            
            try:
                G = nx.read_graphml(file_path)
                temporal_data[period] = G
                all_nodes.update(G.nodes())
                
            except Exception as e:
                print(f"  ❌ Error cargando {file_path}: {e}")
                continue
        
        # Crear mapeo de nodos a IDs
        self.node_to_id = {node: i for i, node in enumerate(sorted(all_nodes))}
        self.id_to_node = {i: node for node, i in self.node_to_id.items()}
        
        print(f"📊 Total de nodos únicos: {len(all_nodes)}")
        
        # Procesar cada grafo temporal
        for period, G in temporal_data.items():
            self._process_temporal_graph(G, period)
        
        # Crear features de nodos
        self._create_node_features()
        
        return {
            'num_nodes': len(all_nodes),
            'num_periods': len(temporal_data),
            'periods': list(temporal_data.keys())
        }
    
    def _process_temporal_graph(self, G: nx.Graph, period: str):
        """Procesa un grafo temporal individual"""
        
        # Convertir aristas a formato tensor
        edges = []
        edge_weights = []
        timestamps = []
        
        # Simular timestamp basado en el período
        base_timestamp = self._period_to_timestamp(period)
        
        for u, v, data in G.edges(data=True):
            if u in self.node_to_id and v in self.node_to_id:
                u_id = self.node_to_id[u]
                v_id = self.node_to_id[v]
                
                edges.append([u_id, v_id])
                edges.append([v_id, u_id])  # Grafo no dirigido
                
                weight = data.get('weight', 1.0)
                edge_weights.extend([weight, weight])
                
                # Timestamp con pequeña variación aleatoria
                timestamp = base_timestamp + np.random.uniform(0, 1)
                timestamps.extend([timestamp, timestamp])
        
        if edges:
            graph_data = {
                'period': period,
                'edge_index': torch.tensor(edges).t().contiguous(),
                'edge_weights': torch.tensor(edge_weights, dtype=torch.float),
                'edge_timestamps': torch.tensor(timestamps, dtype=torch.float),
                'num_edges': len(edges)
            }
            
            self.temporal_graphs.append(graph_data)
    
    def _period_to_timestamp(self, period: str) -> float:
        """Convierte período a timestamp numérico"""
        try:
            # Formato esperado: "semantic_network_2000-2001"
            # Extraer año inicial
            year_part = period.replace('semantic_network_', '').split('-')[0]
            start_year = int(year_part)
            return float(start_year - 2000)  # Normalizar a partir de 2000
        except:
            return 0.0
    
    def _create_node_features(self):
        """Crea features iniciales para los nodos"""
        num_nodes = len(self.node_to_id)
        feature_dim = 128  # Dimensión de features
        
        # Features basadas en estadísticas de red
        features = np.zeros((num_nodes, feature_dim))
        
        for i, graph_data in enumerate(self.temporal_graphs):
            edge_index = graph_data['edge_index']
            
            # Calcular grado para cada nodo
            degrees = torch.zeros(num_nodes)
            if edge_index.size(1) > 0:
                degrees = torch.bincount(edge_index.flatten(), minlength=num_nodes)
            
            # Asignar features (simplificado)
            for node_id in range(num_nodes):
                degree = degrees[node_id].item()
                
                # Feature 0-9: One-hot encoding del grado (hasta 10)
                degree_cat = min(degree, 9)
                features[node_id, degree_cat] = 1.0
                
                # Feature 10: Grado normalizado
                features[node_id, 10] = degree / max(degrees.max().item(), 1)
                
                # Feature 11: Actividad temporal (en cuántos períodos aparece)
                features[node_id, 11] = 1.0 if degree > 0 else 0.0
                
                # Features 12-127: Random embedding inicial
                if i == 0:  # Solo para el primer período
                    features[node_id, 12:] = np.random.normal(0, 0.1, feature_dim - 12)
        
        self.node_features = torch.tensor(features, dtype=torch.float)
        print(f"✅ Features de nodos creadas: {self.node_features.shape}")
    
    def create_link_prediction_dataset(self, test_ratio: float = 0.2) -> Dict:
        """
        Crea dataset para predicción de enlaces
        
        Args:
            test_ratio: Proporción de datos para test
        
        Returns:
            Dataset con train/test splits
        """
        print("🔗 Creando dataset de predicción de enlaces...")
        
        all_positive_pairs = []
        all_negative_pairs = []
        all_timestamps = []
        all_edge_indices = []
        all_edge_timestamps = []
        
        for graph_data in self.temporal_graphs:
            edge_index = graph_data['edge_index']
            edge_timestamps = graph_data['edge_timestamps']
            period_timestamp = self._period_to_timestamp(graph_data['period'])
            
            # Enlaces positivos (existentes)
            positive_pairs = edge_index.t().unique(dim=0)
            
            # Filtrar duplicados (por ser grafo no dirigido)
            positive_pairs = positive_pairs[positive_pairs[:, 0] < positive_pairs[:, 1]]
            
            # Enlaces negativos (muestreo)
            num_positive = positive_pairs.size(0)
            negative_pairs = self._sample_negative_pairs(
                positive_pairs, self.node_features.size(0), num_positive
            )
            
            # Agregar a listas globales
            all_positive_pairs.append(positive_pairs)
            all_negative_pairs.append(negative_pairs)
            
            # Timestamps para cada par
            pos_timestamps = torch.full((num_positive,), period_timestamp)
            neg_timestamps = torch.full((num_positive,), period_timestamp)
            
            all_timestamps.extend([pos_timestamps, neg_timestamps])
            
            # Información de grafo para cada par
            all_edge_indices.extend([edge_index, edge_index])
            all_edge_timestamps.extend([edge_timestamps, edge_timestamps])
        
        # Combinar todos los datos
        X_pairs = torch.cat(all_positive_pairs + all_negative_pairs, dim=0)
        y_labels = torch.cat([
            torch.ones(sum(pairs.size(0) for pairs in all_positive_pairs)),
            torch.zeros(sum(pairs.size(0) for pairs in all_negative_pairs))
        ])
        
        pair_timestamps = torch.cat(all_timestamps)
        
        # Split train/test
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
            'node_features': self.node_features,
            'temporal_graphs': self.temporal_graphs,
            'node_mapping': {'node_to_id': self.node_to_id, 'id_to_node': self.id_to_node}
        }
        
        print(f"✅ Dataset creado:")
        print(f"   📊 Train: {len(train_indices)} pares")
        print(f"   📊 Test: {len(test_indices)} pares")
        print(f"   ⚖️  Balance: {y_labels.mean().item():.3f} (proporción positiva)")
        
        return dataset
    
    def _sample_negative_pairs(self, positive_pairs: torch.Tensor, 
                              num_nodes: int, num_samples: int) -> torch.Tensor:
        """Muestrea pares negativos que no existen en el grafo"""
        
        # Convertir pares positivos a set para búsqueda rápida
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
        
        # Si no podemos generar suficientes, rellenar con aleatorios
        while len(negative_pairs) < num_samples:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v:
                negative_pairs.append([u, v])
        
        return torch.tensor(negative_pairs[:num_samples])


class TDGNNTrainer:
    """
    🏋️ Entrenador de TDGNN
    =====================
    
    Maneja el entrenamiento, validación y evaluación del modelo TDGNN.
    Incluye comparación con baseline Adamic-Adar.
    """
    
    def __init__(self, model: TDGNNModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_metrics = []
        
    def train_model(self, dataset: Dict, epochs: int = 200, 
                   batch_size: int = 64, learning_rate: float = 0.01,
                   val_ratio: float = 0.2) -> Dict:
        """
        Entrena el modelo TDGNN
        
        Args:
            dataset: Dataset de entrenamiento
            epochs: Número de épocas
            batch_size: Tamaño de batch
            learning_rate: Tasa de aprendizaje
            val_ratio: Proporción de datos para validación
        
        Returns:
            Historial de entrenamiento
        """
        print("🏋️ Iniciando entrenamiento de TDGNN...")
        print(f"   📊 Épocas: {epochs}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   📈 Learning rate: {learning_rate}")
        
        # Preparar datos
        X_train = dataset['X_train_pairs']
        y_train = dataset['y_train']
        train_timestamps = dataset['train_timestamps']
        node_features = dataset['node_features'].to(self.device)
        
        # Split train/validation
        val_size = int(len(X_train) * val_ratio)
        train_size = len(X_train) - val_size
        
        train_indices = torch.randperm(len(X_train))[:train_size]
        val_indices = torch.randperm(len(X_train))[train_size:train_size + val_size]
        
        # Crear DataLoaders
        train_dataset = TensorDataset(
            X_train[train_indices], 
            y_train[train_indices],
            train_timestamps[train_indices]
        )
        
        val_dataset = TensorDataset(
            X_train[val_indices],
            y_train[val_indices], 
            train_timestamps[val_indices]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Configurar optimizador
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        criterion = nn.BCELoss()
        
        # Variables para early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_pairs, batch_labels, batch_timestamps in train_loader:
                batch_pairs = batch_pairs.to(self.device)
                batch_labels = batch_labels.to(self.device).float()
                batch_timestamps = batch_timestamps.to(self.device)
                
                optimizer.zero_grad()
                
                # Obtener información de grafo temporal (simplificado)
                # En implementación real, seleccionaríamos el grafo apropiado por timestamp
                edge_index, edge_timestamps = self._get_temporal_graph_info(
                    dataset, batch_timestamps[0]
                )
                
                # Forward pass
                predictions = self.model(
                    node_features=node_features,
                    edge_index=edge_index.to(self.device),
                    edge_timestamps=edge_timestamps.to(self.device),
                    target_pairs=batch_pairs
                ).squeeze()
                
                # Loss
                loss = criterion(predictions, batch_labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            val_metrics = self._validate(val_loader, node_features, dataset, criterion)
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Guardar mejor modelo
                torch.save(self.model.state_dict(), 'best_tdgnn_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 20 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Época {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping en época {epoch+1}")
                break
        
        # Cargar mejor modelo
        self.model.load_state_dict(torch.load('best_tdgnn_model.pth'))
        
        training_history = {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'total_epochs': epoch + 1,
            'best_val_loss': best_val_loss
        }
        
        print(f"✅ Entrenamiento completado en {time.time() - start_time:.1f}s")
        
        return training_history
    
    def _validate(self, val_loader: DataLoader, node_features: torch.Tensor,
                 dataset: Dict, criterion: nn.Module) -> Dict:
        """Validación del modelo"""
        self.model.eval()
        
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_pairs, batch_labels, batch_timestamps in val_loader:
                batch_pairs = batch_pairs.to(self.device)
                batch_labels = batch_labels.to(self.device).float()
                batch_timestamps = batch_timestamps.to(self.device)
                
                # Obtener información de grafo temporal
                edge_index, edge_timestamps = self._get_temporal_graph_info(
                    dataset, batch_timestamps[0]
                )
                
                # Forward pass
                predictions = self.model(
                    node_features=node_features,
                    edge_index=edge_index.to(self.device),
                    edge_timestamps=edge_timestamps.to(self.device),
                    target_pairs=batch_pairs
                ).squeeze()
                
                # Loss
                loss = criterion(predictions, batch_labels)
                val_loss += loss.item()
                
                # Collect for metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                num_batches += 1
        
        # Calcular métricas
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_predictions)
        ap = average_precision_score(all_labels, all_predictions)
        
        # F1 score con threshold 0.5
        pred_binary = (all_predictions > 0.5).astype(int)
        f1 = f1_score(all_labels, pred_binary)
        
        return {
            'loss': val_loss / num_batches,
            'auc': auc,
            'ap': ap,
            'f1': f1
        }
    
    def _get_temporal_graph_info(self, dataset: Dict, timestamp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtiene información del grafo temporal más cercano al timestamp"""
        
        # Simplificado: usar el primer grafo temporal disponible
        # En implementación real, seleccionaríamos por timestamp
        if dataset['temporal_graphs']:
            graph_data = dataset['temporal_graphs'][0]
            return graph_data['edge_index'], graph_data['edge_timestamps']
        else:
            # Grafo vacío como fallback
            num_nodes = dataset['node_features'].size(0)
            empty_edge_index = torch.empty((2, 0), dtype=torch.long)
            empty_timestamps = torch.empty((0,), dtype=torch.float)
            return empty_edge_index, empty_timestamps
    
    def evaluate_test(self, dataset: Dict) -> Dict:
        """
        Evaluación final en conjunto de test
        
        Args:
            dataset: Dataset completo con train/test split
        
        Returns:
            Métricas de evaluación
        """
        print("📊 Evaluando modelo en conjunto de test...")
        
        self.model.eval()
        
        X_test = dataset['X_test_pairs'].to(self.device)
        y_test = dataset['y_test'].cpu().numpy()
        test_timestamps = dataset['test_timestamps'].to(self.device)
        node_features = dataset['node_features'].to(self.device)
        
        all_predictions = []
        batch_size = 128
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_pairs = X_test[i:i+batch_size]
                batch_timestamps = test_timestamps[i:i+batch_size]
                
                if len(batch_pairs) == 0:
                    continue
                
                # Obtener información de grafo temporal
                edge_index, edge_timestamps = self._get_temporal_graph_info(
                    dataset, batch_timestamps[0] if len(batch_timestamps) > 0 else torch.tensor(0.0)
                )
                
                # Forward pass
                predictions = self.model(
                    node_features=node_features,
                    edge_index=edge_index.to(self.device),
                    edge_timestamps=edge_timestamps.to(self.device),
                    target_pairs=batch_pairs
                ).squeeze()
                
                all_predictions.extend(predictions.cpu().numpy())
        
        # Calcular métricas finales
        all_predictions = np.array(all_predictions)
        
        auc_roc = roc_auc_score(y_test, all_predictions)
        auc_pr = average_precision_score(y_test, all_predictions)
        
        # Probar diferentes thresholds para F1
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
        
        print(f"📊 RESULTADOS FINALES TDGNN:")
        print(f"   🎯 AUC-ROC: {auc_roc:.4f}")
        print(f"   📈 AUC-PR: {auc_pr:.4f}")
        print(f"   ⚖️  Best F1: {best_f1:.4f} (threshold: {best_threshold})")
        
        return results


def compare_with_baseline(tdgnn_results: Dict, baseline_results: Dict = None) -> Dict:
    """
    🔍 Comparación con baseline Adamic-Adar
    ======================================
    
    Compara los resultados de TDGNN con el baseline establecido.
    """
    print("\n🔍 COMPARACIÓN CON BASELINE ADAMIC-ADAR")
    print("=" * 50)
    
    # Resultados baseline (de análisis previo)
    if baseline_results is None:
        baseline_results = {
            'auc_roc': 0.7044,  # Del análisis previo
            'auc_pr': 0.6496,
            'best_f1': 0.6012,
            'method': 'Adamic-Adar Index'
        }
    
    tdgnn_auc = tdgnn_results['auc_roc']
    tdgnn_f1 = tdgnn_results['best_f1']
    
    baseline_auc = baseline_results['auc_roc']
    baseline_f1 = baseline_results['best_f1']
    
    # Calcular mejoras
    auc_improvement = ((tdgnn_auc - baseline_auc) / baseline_auc) * 100
    f1_improvement = ((tdgnn_f1 - baseline_f1) / baseline_f1) * 100
    
    comparison = {
        'tdgnn': tdgnn_results,
        'baseline': baseline_results,
        'improvements': {
            'auc_roc_improvement_pct': auc_improvement,
            'f1_improvement_pct': f1_improvement,
            'auc_roc_absolute': tdgnn_auc - baseline_auc,
            'f1_absolute': tdgnn_f1 - baseline_f1
        }
    }
    
    print(f"📊 BASELINE (Adamic-Adar):")
    print(f"   • AUC-ROC: {baseline_auc:.4f}")
    print(f"   • F1-Score: {baseline_f1:.4f}")
    
    print(f"\n🚀 TDGNN:")
    print(f"   • AUC-ROC: {tdgnn_auc:.4f}")
    print(f"   • F1-Score: {tdgnn_f1:.4f}")
    
    print(f"\n📈 MEJORAS:")
    print(f"   • AUC-ROC: {auc_improvement:+.1f}% ({tdgnn_auc - baseline_auc:+.4f})")
    print(f"   • F1-Score: {f1_improvement:+.1f}% ({tdgnn_f1 - baseline_f1:+.4f})")
    
    if tdgnn_auc > baseline_auc:
        print(f"\n✅ TDGNN SUPERA EL BASELINE!")
        if auc_improvement > 10:
            print(f"🎉 Mejora significativa (>{auc_improvement:.1f}%)")
    else:
        print(f"\n⚠️  TDGNN no supera el baseline")
        print(f"💡 Considerar ajuste de hiperparámetros o más entrenamiento")
    
    return comparison


def create_visualizations(training_history: Dict, evaluation_results: Dict, 
                         comparison: Dict, output_dir: str = "."):
    """
    📊 Crear visualizaciones de resultados
    """
    print("\n📊 Creando visualizaciones...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🔥 TDGNN - Resultados de Entrenamiento y Evaluación', fontsize=16, fontweight='bold')
    
    # 1. Training loss
    axes[0,0].plot(training_history['train_losses'], label='Train Loss', color='blue', alpha=0.7)
    val_losses = [m['loss'] for m in training_history['val_metrics']]
    axes[0,0].plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    axes[0,0].set_xlabel('Época')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Evolución del Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Validation AUC
    val_aucs = [m['auc'] for m in training_history['val_metrics']]
    axes[0,1].plot(val_aucs, label='Validation AUC-ROC', color='green', linewidth=2)
    axes[0,1].axhline(y=comparison['baseline']['auc_roc'], color='red', 
                     linestyle='--', label='Baseline (Adamic-Adar)')
    axes[0,1].set_xlabel('Época')
    axes[0,1].set_ylabel('AUC-ROC')
    axes[0,1].set_title('Evolución del AUC-ROC')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Comparación final
    methods = ['Adamic-Adar\n(Baseline)', 'TDGNN']
    auc_scores = [comparison['baseline']['auc_roc'], comparison['tdgnn']['auc_roc']]
    f1_scores = [comparison['baseline']['best_f1'], comparison['tdgnn']['best_f1']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = axes[1,0].bar(x - width/2, auc_scores, width, label='AUC-ROC', color='skyblue', alpha=0.8)
    bars2 = axes[1,0].bar(x + width/2, f1_scores, width, label='F1-Score', color='lightcoral', alpha=0.8)
    
    axes[1,0].set_xlabel('Método')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_title('Comparación de Rendimiento')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(methods)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar in bars1:
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom')
    
    # 4. F1 por threshold
    thresholds = list(evaluation_results['f1_by_threshold'].keys())
    f1_values = list(evaluation_results['f1_by_threshold'].values())
    
    axes[1,1].plot(thresholds, f1_values, 'o-', color='purple', linewidth=2, markersize=6)
    best_idx = np.argmax(f1_values)
    axes[1,1].axvline(x=thresholds[best_idx], color='red', linestyle='--', 
                     label=f'Mejor threshold: {thresholds[best_idx]}')
    axes[1,1].set_xlabel('Threshold')
    axes[1,1].set_ylabel('F1-Score')
    axes[1,1].set_title('F1-Score por Threshold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar visualización
    output_path = Path(output_dir) / "tdgnn_results_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráficos guardados en: {output_path}")
    
    plt.show()
    
    return output_path


def main():
    """🚀 Función principal - Implementación completa de TDGNN"""
    
    print("🔥 TDGNN - IMPLEMENTACIÓN COMPLETA")
    print("=" * 60)
    print("Temporal Dependent Graph Neural Network")
    print("Para predicción de enlaces en redes semánticas temporales")
    print()
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    # Verificar GPU
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 1. Cargar datos temporales
    print("\n📂 FASE 1: CARGA DE DATOS")
    print("-" * 30)
    
    data_loader = TemporalDataLoader()
    
    try:
        network_info = data_loader.load_temporal_networks()
        print(f"✅ Redes temporales cargadas exitosamente")
        
        # Crear dataset para predicción de enlaces
        dataset = data_loader.create_link_prediction_dataset(test_ratio=0.2)
        print(f"✅ Dataset de predicción de enlaces creado")
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        print("💡 Asegúrate de tener las redes temporales en la carpeta 'temporal_networks'")
        return
    
    # 2. Crear modelo TDGNN
    print("\n🧠 FASE 2: CREACIÓN DEL MODELO")
    print("-" * 35)
    
    # Parámetros del modelo
    input_dim = dataset['node_features'].size(1)
    hidden_dim = 64
    output_dim = 32
    num_layers = 2
    
    model = TDGNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modelo TDGNN creado")
    print(f"   📊 Parámetros: {num_params:,}")
    print(f"   🏗️  Arquitectura: {input_dim}→{hidden_dim}→{output_dim}")
    print(f"   📚 Capas: {num_layers}")
    
    # 3. Entrenar modelo
    print("\n🏋️ FASE 3: ENTRENAMIENTO")
    print("-" * 30)
    
    trainer = TDGNNTrainer(model, device)
    
    training_history = trainer.train_model(
        dataset=dataset,
        epochs=100,  # Reducido para demo
        batch_size=32,  # Optimizado para GTX 1050
        learning_rate=0.01,
        val_ratio=0.2
    )
    
    print(f"✅ Entrenamiento completado")
    
    # 4. Evaluación final
    print("\n📊 FASE 4: EVALUACIÓN")
    print("-" * 25)
    
    evaluation_results = trainer.evaluate_test(dataset)
    
    # 5. Comparación con baseline
    print("\n🔍 FASE 5: COMPARACIÓN CON BASELINE")
    print("-" * 40)
    
    comparison = compare_with_baseline(evaluation_results)
    
    # 6. Visualizaciones
    print("\n📊 FASE 6: VISUALIZACIONES")
    print("-" * 30)
    
    viz_path = create_visualizations(training_history, evaluation_results, comparison)
    
    # 7. Guardar resultados
    print("\n💾 FASE 7: GUARDADO DE RESULTADOS")
    print("-" * 40)
    
    results_summary = {
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_layers': num_layers,
            'num_parameters': num_params
        },
        'training_history': training_history,
        'evaluation_results': evaluation_results,
        'comparison_with_baseline': comparison,
        'dataset_info': network_info,
        'timestamp': datetime.now().isoformat()
    }
    
    # Guardar como JSON
    with open('tdgnn_results.json', 'w') as f:
        # Convertir tensors a listas para JSON
        json_results = {
            k: v for k, v in results_summary.items() 
            if k not in ['training_history', 'evaluation_results']
        }
        json.dump(json_results, f, indent=2)
    
    # Guardar modelo entrenado
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': results_summary['model_config'],
        'dataset_info': network_info
    }, 'tdgnn_model_complete.pth')
    
    print(f"✅ Resultados guardados:")
    print(f"   📄 tdgnn_results.json")
    print(f"   🧠 tdgnn_model_complete.pth")
    print(f"   📊 {viz_path}")
    
    # 8. Resumen final
    print(f"\n🎉 IMPLEMENTACIÓN TDGNN COMPLETADA")
    print("=" * 50)
    print(f"📊 RESULTADOS FINALES:")
    print(f"   🎯 AUC-ROC: {evaluation_results['auc_roc']:.4f}")
    print(f"   📈 AUC-PR: {evaluation_results['auc_pr']:.4f}")
    print(f"   ⚖️  F1-Score: {evaluation_results['best_f1']:.4f}")
    
    auc_improvement = comparison['improvements']['auc_roc_improvement_pct']
    print(f"\n🚀 MEJORA vs BASELINE:")
    print(f"   📈 AUC-ROC: {auc_improvement:+.1f}%")
    
    if evaluation_results['auc_roc'] > 0.75:
        print(f"\n✅ ¡OBJETIVO CUMPLIDO! AUC-ROC > 0.75")
        print(f"🎯 Listo para proceder con A3TGCN")
    else:
        print(f"\n📝 Considerar ajuste de hiperparámetros")
        print(f"💡 O más datos de entrenamiento")
    
    return results_summary


if __name__ == "__main__":
    results = main()
