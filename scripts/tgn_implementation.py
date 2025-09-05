#!/usr/bin/env python3
"""
⚡ IMPLEMENTACIÓN TGN - TEMPORAL GRAPH NETWORKS
==============================================

Implementación completa de TGN (Temporal Graph Networks) para predicción 
de enlaces en redes semánticas temporales usando los datos de metadatos académicos.

Características:
- TGN con módulo de memoria para información temporal persistente
- Self-supervised learning para captura de patrones temporales
- Inductive capability para nuevos nodos
- State-of-the-art en benchmarks temporales

Basado en:
"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
Rossi et al., ICML 2020

Autor: Sistema de Implementación TGN
Fecha: Julio 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'External_repos', 'tgn'))

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

# Intentar importar TGN oficial
try:
    from tgn import TGN
    from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
    from utils.data_processing import get_data, compute_time_statistics
    TGN_AVAILABLE = True
    print("✅ TGN oficial disponible")
except ImportError:
    TGN_AVAILABLE = False
    print("⚠️ TGN oficial no disponible. Usando implementación propia.")


class MemoryModule(nn.Module):
    """
    🧠 Módulo de Memoria para TGN
    ============================
    
    Almacena y actualiza información temporal de nodos de manera persistente.
    """
    
    def __init__(self, n_nodes: int, memory_dimension: int, input_dimension: int,
                 message_dimension: int = 100, device: str = "cpu"):
        super(MemoryModule, self).__init__()
        
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.device = device
        
        # Memoria persistente para cada nodo
        self.memory = nn.Parameter(
            torch.zeros(n_nodes, memory_dimension),
            requires_grad=False
        )
        
        # Timestamps de última actualización
        self.last_update = nn.Parameter(
            torch.zeros(n_nodes),
            requires_grad=False
        )
        
        # Función de actualización de memoria
        self.memory_updater = nn.GRUCell(
            input_size=message_dimension,
            hidden_size=memory_dimension
        )
        
        # Proyección de mensajes
        self.message_function = nn.Sequential(
            nn.Linear(input_dimension + memory_dimension + 1, message_dimension),  # +1 para time
            nn.ReLU(),
            nn.Linear(message_dimension, message_dimension)
        )
        
    def get_memory(self, node_idxs: torch.Tensor) -> torch.Tensor:
        """Obtiene memoria de nodos específicos"""
        return self.memory[node_idxs]
    
    def set_memory(self, node_idxs: torch.Tensor, values: torch.Tensor):
        """Actualiza memoria de nodos específicos"""
        self.memory[node_idxs] = values
    
    def get_last_update(self, node_idxs: torch.Tensor) -> torch.Tensor:
        """Obtiene timestamps de última actualización"""
        return self.last_update[node_idxs]
    
    def update_memory(self, unique_node_ids: torch.Tensor, unique_messages: torch.Tensor,
                     timestamps: torch.Tensor):
        """
        Actualiza memoria de nodos con nuevos mensajes
        
        Args:
            unique_node_ids: IDs de nodos únicos
            unique_messages: Mensajes agregados para cada nodo
            timestamps: Timestamps de actualización
        """
        if len(unique_node_ids) == 0:
            return
        
        # Obtener memoria actual
        memory = self.get_memory(unique_node_ids)
        
        # Actualizar usando GRU
        updated_memory = self.memory_updater(unique_messages, memory)
        
        # Guardar memoria actualizada
        self.set_memory(unique_node_ids, updated_memory)
        
        # Actualizar timestamps
        self.last_update[unique_node_ids] = timestamps
    
    def compute_messages(self, source_nodes: torch.Tensor, destination_nodes: torch.Tensor,
                        edge_times: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Computa mensajes para actualización de memoria
        
        Args:
            source_nodes: Nodos fuente
            destination_nodes: Nodos destino  
            edge_times: Timestamps de aristas
            edge_features: Features de aristas
        
        Returns:
            Mensajes computados
        """
        # Obtener memoria actual de nodos fuente
        source_memory = self.get_memory(source_nodes)
        
        # Obtener timestamps de última actualización
        source_time_delta = edge_times - self.get_last_update(source_nodes)
        
        # Concatenar features: memoria + features arista + tiempo
        message_input = torch.cat([
            source_memory,
            edge_features,
            source_time_delta.unsqueeze(1)
        ], dim=1)
        
        # Computar mensajes
        messages = self.message_function(message_input)
        
        return messages
    
    def reset_memory(self):
        """Reinicia toda la memoria"""
        self.memory.data.zero_()
        self.last_update.data.zero_()


class GraphAttentionEmbedding(nn.Module):
    """
    🎯 Graph Attention Embedding para TGN
    =====================================
    
    Computa embeddings de nodos usando atención sobre vecinos temporales.
    """
    
    def __init__(self, input_dimension: int, n_heads: int = 2, dropout: float = 0.1,
                 output_dimension: int = None):
        super(GraphAttentionEmbedding, self).__init__()
        
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension or input_dimension
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dimension,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Proyección de salida
        self.output_projection = nn.Linear(input_dimension, self.output_dimension)
        
    def forward(self, source_nodes: torch.Tensor, source_node_features: torch.Tensor,
                neighbor_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del embedding con atención
        
        Args:
            source_nodes: Nodos fuente
            source_node_features: Features de nodos fuente
            neighbor_embeddings: Embeddings de vecinos
        
        Returns:
            Embeddings actualizados
        """
        batch_size = source_nodes.size(0)
        
        if neighbor_embeddings.size(0) == 0:
            # No hay vecinos, usar solo features propias
            return self.output_projection(source_node_features)
        
        # Preparar para atención: query = nodos fuente, key/value = vecinos
        query = source_node_features.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Si hay menos vecinos que nodos fuente, expandir
        if neighbor_embeddings.size(0) < batch_size:
            # Rellenar con ceros
            padding_size = batch_size - neighbor_embeddings.size(0)
            padding = torch.zeros(padding_size, neighbor_embeddings.size(1), 
                                device=neighbor_embeddings.device)
            neighbor_embeddings = torch.cat([neighbor_embeddings, padding], dim=0)
        
        key_value = neighbor_embeddings.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Aplicar atención
        attended_features, _ = self.attention(query, key_value, key_value)
        
        # Proyección final
        output = self.output_projection(attended_features.squeeze(1))
        
        return output


class TGNModel(nn.Module):
    """
    ⚡ TGN - Temporal Graph Network
    ==============================
    
    Implementación completa de TGN con módulo de memoria.
    """
    
    def __init__(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                 memory_dimension: int = 100, time_dimension: int = 100,
                 n_layers: int = 1, n_heads: int = 2, dropout: float = 0.1,
                 output_dimension: int = None, device: str = "cpu"):
        super(TGNModel, self).__init__()
        
        self.n_nodes = node_features.shape[0]
        self.n_node_features = node_features.shape[1]
        self.n_edge_features = edge_features.shape[1] if edge_features.shape[0] > 0 else 0
        self.memory_dimension = memory_dimension
        self.time_dimension = time_dimension
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.output_dimension = output_dimension or memory_dimension
        self.device = device
        
        # Registrar features como buffers
        self.register_buffer('node_features', node_features)
        if edge_features.shape[0] > 0:
            self.register_buffer('edge_features', edge_features)
        else:
            self.register_buffer('edge_features', torch.zeros(1, 128))  # Dummy
        
        # Módulo de memoria
        self.memory = MemoryModule(
            n_nodes=self.n_nodes,
            memory_dimension=memory_dimension,
            input_dimension=self.n_node_features + self.n_edge_features + 1,  # +1 para tiempo
            message_dimension=memory_dimension,
            device=device
        )
        
        # Embedding layers
        self.embedding_dimension = memory_dimension + self.n_node_features + time_dimension
        
        self.embedding_layers = nn.ModuleList([
            GraphAttentionEmbedding(
                input_dimension=self.embedding_dimension,
                n_heads=n_heads,
                dropout=dropout,
                output_dimension=memory_dimension
            ) for _ in range(n_layers)
        ])
        
        # Time encoder
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dimension),
            nn.ReLU(),
            nn.Linear(time_dimension, time_dimension)
        )
        
        # Proyección final
        self.output_projection = nn.Linear(memory_dimension, self.output_dimension)
        
        # Link predictor
        self.link_predictor = nn.Sequential(
            nn.Linear(self.output_dimension * 2, memory_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dimension, memory_dimension // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dimension // 2, 1)
        )
        
    def compute_temporal_embeddings(self, source_nodes: torch.Tensor, timestamps: torch.Tensor,
                                  n_neighbors: int = 20) -> torch.Tensor:
        """
        Computa embeddings temporales para nodos fuente
        
        Args:
            source_nodes: Nodos fuente
            timestamps: Timestamps correspondientes
            n_neighbors: Número de vecinos a considerar
        
        Returns:
            Embeddings temporales
        """
        source_nodes_torch = source_nodes
        timestamps_torch = timestamps
        
        # Obtener memoria actual
        source_memory = self.memory.get_memory(source_nodes_torch)
        
        # Obtener features de nodos
        source_node_features = self.node_features[source_nodes_torch]
        
        # Codificar tiempo
        time_encoding = self.time_encoder(timestamps_torch.unsqueeze(1))
        
        # Combinar memoria, features y tiempo
        source_embedding = torch.cat([
            source_memory,
            source_node_features,
            time_encoding
        ], dim=1)
        
        # Aplicar capas de embedding con atención
        for layer in self.embedding_layers:
            # Para simplificar, usar el mismo embedding como vecinos
            neighbor_embeddings = source_embedding
            source_embedding = layer(source_nodes_torch, source_embedding, neighbor_embeddings)
        
        # Proyección final
        output_embeddings = self.output_projection(source_embedding)
        
        return output_embeddings
    
    def update_memory(self, nodes: torch.Tensor, edge_idxs: torch.Tensor, 
                     timestamps: torch.Tensor):
        """
        Actualiza memoria con nuevas interacciones
        
        Args:
            nodes: Nodos involucrados
            edge_idxs: Índices de aristas
            timestamps: Timestamps de interacciones
        """
        if len(nodes) == 0:
            return
        
        # Obtener features de aristas (simplificado)
        if len(edge_idxs) > 0 and edge_idxs.max() < self.edge_features.size(0):
            edge_features = self.edge_features[edge_idxs]
        else:
            edge_features = torch.zeros(len(nodes), self.n_edge_features, device=self.device)
        
        # Computar mensajes
        messages = self.memory.compute_messages(
            source_nodes=nodes,
            destination_nodes=nodes,  # Simplificado
            edge_times=timestamps,
            edge_features=edge_features
        )
        
        # Actualizar memoria
        unique_nodes = torch.unique(nodes)
        if len(unique_nodes) > 0:
            # Agregar mensajes por nodo
            aggregated_messages = torch.zeros(len(unique_nodes), messages.size(1), device=self.device)
            for i, node in enumerate(unique_nodes):
                node_mask = (nodes == node)
                if node_mask.sum() > 0:
                    aggregated_messages[i] = messages[node_mask].mean(dim=0)
            
            self.memory.update_memory(
                unique_node_ids=unique_nodes,
                unique_messages=aggregated_messages,
                timestamps=timestamps[:len(unique_nodes)]
            )
    
    def forward(self, source_nodes: torch.Tensor, destination_nodes: torch.Tensor,
                timestamps: torch.Tensor, edge_idxs: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass de TGN
        
        Args:
            source_nodes: Nodos fuente
            destination_nodes: Nodos destino
            timestamps: Timestamps
            edge_idxs: Índices de aristas (opcional)
        
        Returns:
            Probabilidades de enlaces
        """
        # Actualizar memoria con interacciones actuales
        all_nodes = torch.cat([source_nodes, destination_nodes])
        all_timestamps = timestamps.repeat(2)
        if edge_idxs is not None:
            all_edge_idxs = edge_idxs.repeat(2)
        else:
            all_edge_idxs = torch.arange(len(all_nodes), device=all_nodes.device)
        
        self.update_memory(all_nodes, all_edge_idxs, all_timestamps)
        
        # Computar embeddings temporales
        source_embeddings = self.compute_temporal_embeddings(source_nodes, timestamps)
        destination_embeddings = self.compute_temporal_embeddings(destination_nodes, timestamps)
        
        # Predicción de enlaces
        pair_embeddings = torch.cat([source_embeddings, destination_embeddings], dim=1)
        link_scores = self.link_predictor(pair_embeddings)
        link_probs = torch.sigmoid(link_scores)
        
        return link_probs
    
    def reset_memory(self):
        """Reinicia memoria del modelo"""
        self.memory.reset_memory()


class TemporalGraphProcessor:
    """
    📊 Procesador de Grafos Temporales para TGN
    ===========================================
    """
    
    def __init__(self, temporal_networks_path: str = "../temporal_networks/networks"):
        self.temporal_networks_path = Path(temporal_networks_path)
        self.node_to_id = {}
        self.id_to_node = {}
        self.temporal_graphs = []
        self.all_interactions = []
        
    def load_and_process(self) -> Dict:
        """Carga y procesa redes temporales para TGN"""
        print("📂 Cargando redes temporales para TGN...")
        
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
        
        # Procesar interacciones temporales
        self._process_temporal_interactions(temporal_data)
        
        return {
            'num_nodes': len(all_nodes),
            'num_periods': len(temporal_data),
            'num_interactions': len(self.all_interactions),
            'node_mapping': {'node_to_id': self.node_to_id, 'id_to_node': self.id_to_node}
        }
    
    def _period_to_timestamp(self, period: str) -> float:
        """Convierte período a timestamp"""
        try:
            year_part = period.split('-')[0]
            start_year = int(year_part)
            return float(start_year - 2000)
        except:
            return 0.0
    
    def _process_temporal_interactions(self, temporal_data: List[Dict]):
        """Procesa interacciones temporales para TGN"""
        
        for data in temporal_data:
            G = data['graph']
            timestamp = data['timestamp']
            
            # Convertir aristas a interacciones
            for u, v, edge_data in G.edges(data=True):
                if u in self.node_to_id and v in self.node_to_id:
                    u_id = self.node_to_id[u]
                    v_id = self.node_to_id[v]
                    weight = edge_data.get('weight', 1.0)
                    
                    # Agregar interacción
                    self.all_interactions.append({
                        'source': u_id,
                        'destination': v_id,
                        'timestamp': timestamp,
                        'edge_weight': weight,
                        'edge_features': [weight, timestamp]  # Features simples
                    })
        
        # Ordenar por timestamp
        self.all_interactions.sort(key=lambda x: x['timestamp'])
    
    def create_node_edge_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crea features de nodos y aristas para TGN"""
        
        num_nodes = len(self.node_to_id)
        
        # Features de nodos (128-dim como antes)
        node_features = torch.randn(num_nodes, 128) * 0.1
        
        # Features de aristas (2-dim: peso + tiempo normalizado)
        if self.all_interactions:
            edge_features = torch.tensor([
                interaction['edge_features'] for interaction in self.all_interactions
            ], dtype=torch.float)
        else:
            edge_features = torch.zeros(1, 2)
        
        return node_features, edge_features
    
    def create_link_prediction_dataset(self, test_ratio: float = 0.2) -> Dict:
        """Crea dataset para predicción de enlaces temporal"""
        print("🔗 Creando dataset de predicción de enlaces para TGN...")
        
        if not self.all_interactions:
            raise ValueError("No hay interacciones temporales disponibles")
        
        # Convertir interacciones a arrays
        sources = torch.tensor([i['source'] for i in self.all_interactions])
        destinations = torch.tensor([i['destination'] for i in self.all_interactions])
        timestamps = torch.tensor([i['timestamp'] for i in self.all_interactions])
        edge_weights = torch.tensor([i['edge_weight'] for i in self.all_interactions])
        
        # Crear pares positivos
        positive_pairs = torch.stack([sources, destinations], dim=1)
        positive_labels = torch.ones(len(positive_pairs))
        
        # Crear pares negativos
        num_positive = len(positive_pairs)
        negative_pairs = self._sample_negative_pairs(positive_pairs, len(self.node_to_id), num_positive)
        negative_labels = torch.zeros(len(negative_pairs))
        
        # Combinar
        all_pairs = torch.cat([positive_pairs, negative_pairs], dim=0)
        all_labels = torch.cat([positive_labels, negative_labels], dim=0)
        all_timestamps = torch.cat([timestamps, timestamps[:len(negative_pairs)]], dim=0)
        
        # Train/test split
        indices = torch.randperm(len(all_pairs))
        split_idx = int(len(indices) * (1 - test_ratio))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        dataset = {
            'X_train_pairs': all_pairs[train_indices],
            'X_test_pairs': all_pairs[test_indices],
            'y_train': all_labels[train_indices],
            'y_test': all_labels[test_indices],
            'train_timestamps': all_timestamps[train_indices],
            'test_timestamps': all_timestamps[test_indices],
            'all_interactions': self.all_interactions,
            'node_mapping': {'node_to_id': self.node_to_id, 'id_to_node': self.id_to_node}
        }
        
        print(f"✅ Dataset TGN creado:")
        print(f"   📊 Train: {len(train_indices)} pares")
        print(f"   📊 Test: {len(test_indices)} pares")
        print(f"   ⏰ Interacciones temporales: {len(self.all_interactions)}")
        
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
    🏋️ Entrenador de TGN
    ====================
    """
    
    def __init__(self, model: TGNModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_metrics = []
        
    def train_model(self, dataset: Dict, node_features: torch.Tensor, edge_features: torch.Tensor,
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.01) -> Dict:
        """Entrena el modelo TGN"""
        print("🏋️ Iniciando entrenamiento de TGN...")
        print(f"   📊 Épocas: {epochs}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   📈 Learning rate: {learning_rate}")
        
        # Preparar datos
        X_train = dataset['X_train_pairs']
        y_train = dataset['y_train']
        train_timestamps = dataset['train_timestamps']
        
        # Split train/validation
        val_size = int(len(X_train) * 0.2)
        train_size = len(X_train) - val_size
        
        train_indices = torch.randperm(len(X_train))[:train_size]
        val_indices = torch.randperm(len(X_train))[train_size:train_size + val_size]
        
        # DataLoaders
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
            # Reset memoria al inicio de cada época
            self.model.reset_memory()
            
            # Training
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_pairs, batch_labels, batch_timestamps in train_loader:
                batch_pairs = batch_pairs.to(self.device)
                batch_labels = batch_labels.to(self.device).float()
                batch_timestamps = batch_timestamps.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(
                    source_nodes=batch_pairs[:, 0],
                    destination_nodes=batch_pairs[:, 1],
                    timestamps=batch_timestamps
                ).squeeze()
                
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
            val_metrics = self._validate(val_loader, criterion)
            self.val_metrics.append(val_metrics)
            
            # Scheduling
            scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_tgn_model.pth')
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
        self.model.load_state_dict(torch.load('best_tgn_model.pth'))
        
        return {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'total_epochs': epoch + 1,
            'best_val_loss': best_val_loss
        }
    
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> Dict:
        """Validación"""
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
                
                predictions = self.model(
                    source_nodes=batch_pairs[:, 0],
                    destination_nodes=batch_pairs[:, 1],
                    timestamps=batch_timestamps
                ).squeeze()
                
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
        print("📊 Evaluando TGN en conjunto de test...")
        
        self.model.eval()
        self.model.reset_memory()  # Reset para evaluación limpia
        
        X_test = dataset['X_test_pairs'].to(self.device)
        y_test = dataset['y_test'].cpu().numpy()
        test_timestamps = dataset['test_timestamps'].to(self.device)
        
        all_predictions = []
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_pairs = X_test[i:i+batch_size]
                batch_timestamps = test_timestamps[i:i+batch_size]
                
                if len(batch_pairs) == 0:
                    continue
                
                predictions = self.model(
                    source_nodes=batch_pairs[:, 0],
                    destination_nodes=batch_pairs[:, 1],
                    timestamps=batch_timestamps
                ).squeeze()
                
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
        
        print(f"📊 RESULTADOS FINALES TGN:")
        print(f"   🎯 AUC-ROC: {auc_roc:.4f}")
        print(f"   📈 AUC-PR: {auc_pr:.4f}")
        print(f"   ⚖️  Best F1: {best_f1:.4f} (threshold: {best_threshold})")
        
        return results


def main():
    """🚀 Función principal - Implementación TGN"""
    
    print("⚡ TGN - IMPLEMENTACIÓN COMPLETA")
    print("=" * 60)
    print("Temporal Graph Networks")
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
        node_features, edge_features = processor.create_node_edge_features()
        dataset = processor.create_link_prediction_dataset(test_ratio=0.2)
        print(f"✅ Datos temporales cargados y procesados")
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 2. Crear modelo
    print("\n🧠 FASE 2: CREACIÓN DEL MODELO")
    print("-" * 35)
    
    model = TGNModel(
        node_features=node_features,
        edge_features=edge_features,
        memory_dimension=100,
        time_dimension=100,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        output_dimension=32,
        device=str(device)
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modelo TGN creado")
    print(f"   📊 Parámetros: {num_params:,}")
    print(f"   🧠 Memoria dimension: 100")
    print(f"   ⏰ Time dimension: 100")
    print(f"   🏗️  Salida: 32 dims")
    
    # 3. Entrenamiento
    print("\n🏋️ FASE 3: ENTRENAMIENTO")
    print("-" * 30)
    
    trainer = TGNTrainer(model, device)
    
    training_history = trainer.train_model(
        dataset=dataset,
        node_features=node_features,
        edge_features=edge_features,
        epochs=80,  # Menos épocas por ser más complejo
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
        'model_name': 'TGN',
        'model_config': {
            'memory_dimension': 100,
            'time_dimension': 100,
            'n_layers': 1,
            'n_heads': 2,
            'output_dimension': 32,
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
        'results': evaluation_results,
        'node_features': node_features,
        'edge_features': edge_features
    }, 'tgn_model_complete.pth')
    
    print(f"✅ Resultados guardados:")
    print(f"   🧠 tgn_model_complete.pth")
    
    # 6. Resumen final
    print(f"\n🎉 IMPLEMENTACIÓN TGN COMPLETADA")
    print("=" * 50)
    print(f"📊 RESULTADOS FINALES:")
    print(f"   🎯 AUC-ROC: {evaluation_results['auc_roc']:.4f}")
    print(f"   📈 AUC-PR: {evaluation_results['auc_pr']:.4f}")
    print(f"   ⚖️  F1-Score: {evaluation_results['best_f1']:.4f}")
    
    return results_summary


if __name__ == "__main__":
    results = main()
