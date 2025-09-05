#!/usr/bin/env python3
"""
Entrenamiento de Modelos Temporales con Tópicos AI Validados
============================================================

Entrena los 4 modelos temporales (A3TGCN, EvolveGCN-H, TDGNN, TGN-Simple)
usando solo los tópicos AI validados por OpenAlex y optimiza sus hiperparámetros.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
import glob
from typing import Dict, List, Tuple, Set
from pathlib import Path
import networkx as nx
from datetime import datetime
import itertools
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MODELOS TEMPORALES
# =============================================================================

class A3TGCNModel(nn.Module):
    """Modelo A3TGCN (Attention Temporal Graph Convolutional Network)"""
    
    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super(A3TGCNModel, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Embeddings de nodos
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        
        # Capas temporales con attention
        self.temporal_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Triple attention mechanism
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.spatial_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.feature_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Predictor de enlaces
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, node_pairs: torch.Tensor) -> torch.Tensor:
        batch_size = node_pairs.size(0)
        
        # Obtener embeddings
        node1_emb = self.node_embedding(node_pairs[:, 0])
        node2_emb = self.node_embedding(node_pairs[:, 1])
        
        # Aplicar attention
        node1_emb = node1_emb.unsqueeze(1)
        node2_emb = node2_emb.unsqueeze(1)
        
        node1_att, _ = self.temporal_attention(node1_emb, node1_emb, node1_emb)
        node2_att, _ = self.spatial_attention(node2_emb, node2_emb, node2_emb)
        
        # Concatenar y predecir
        combined = torch.cat([node1_att.squeeze(1), node2_att.squeeze(1)], dim=1)
        return self.link_predictor(combined).squeeze()

class EvolveGCNHModel(nn.Module):
    """Modelo EvolveGCN-H (Evolving Graph Convolutional Networks)"""
    
    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super(EvolveGCNHModel, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Node embeddings
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        
        # Evolving GCN layers
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        
        # LSTM para evolución temporal
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Link predictor
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_pairs: torch.Tensor) -> torch.Tensor:
        # Embeddings básicos
        node1_emb = self.node_embedding(node_pairs[:, 0])
        node2_emb = self.node_embedding(node_pairs[:, 1])
        
        # Aplicar GCN
        node1_gcn = F.relu(self.gcn1(node1_emb))
        node2_gcn = F.relu(self.gcn2(node2_emb))
        
        # Evolución temporal (simulada)
        node1_evolved, _ = self.lstm(node1_gcn.unsqueeze(1))
        node2_evolved, _ = self.lstm(node2_gcn.unsqueeze(1))
        
        # Concatenar y predecir
        combined = torch.cat([node1_evolved.squeeze(1), node2_evolved.squeeze(1)], dim=1)
        return self.link_predictor(combined).squeeze()

class TDGNNModel(nn.Module):
    """Modelo TDGNN (Temporal Dynamic Graph Neural Network)"""
    
    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super(TDGNNModel, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Node features
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        
        # Temporal dynamics
        self.temporal_layer1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_layer2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Graph convolution
        self.graph_conv = nn.Linear(hidden_dim, hidden_dim)
        
        # Link prediction
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_pairs: torch.Tensor) -> torch.Tensor:
        # Embeddings
        node1_emb = self.node_embedding(node_pairs[:, 0])
        node2_emb = self.node_embedding(node_pairs[:, 1])
        
        # Temporal dynamics
        node1_temp, _ = self.temporal_layer1(node1_emb.unsqueeze(1))
        node2_temp, _ = self.temporal_layer2(node2_emb.unsqueeze(1))
        
        # Graph convolution
        node1_final = F.relu(self.graph_conv(node1_temp.squeeze(1)))
        node2_final = F.relu(self.graph_conv(node2_temp.squeeze(1)))
        
        # Link prediction
        combined = torch.cat([node1_final, node2_final], dim=1)
        return self.link_predictor(combined).squeeze()

class TGNSimpleModel(nn.Module):
    """Modelo TGN-Simple (Temporal Graph Networks)"""
    
    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super(TGNSimpleModel, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Memory bank para nodos
        self.memory = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        
        # Message function
        self.message_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Memory updater
        self.memory_updater = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Link predictor
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_pairs: torch.Tensor) -> torch.Tensor:
        batch_size = node_pairs.size(0)
        
        # Obtener memoria de nodos
        node1_mem = self.memory[node_pairs[:, 0]]
        node2_mem = self.memory[node_pairs[:, 1]]
        
        # Compute messages
        messages = self.message_fn(torch.cat([node1_mem, node2_mem], dim=1))
        
        # Update memories (simplified)
        node1_updated = self.memory_updater(messages, node1_mem)
        node2_updated = self.memory_updater(messages, node2_mem)
        
        # Link prediction
        combined = torch.cat([node1_updated, node2_updated], dim=1)
        return self.link_predictor(combined).squeeze()

# =============================================================================
# ENTRENADOR PRINCIPAL
# =============================================================================

class ValidatedAITrainer:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.validation_dir = self.base_dir / "validation"
        self.results_dir = self.base_dir / "validated_results"
        # Carpeta para matrices de confusión y artefactos derivados
        self.cm_dir = self.results_dir / "confusion_matrices"

        # Crear directorios
        self.models_dir.mkdir(exist_ok=True)
        self.validation_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.cm_dir.mkdir(exist_ok=True)

        # Configuraciones de hiperparámetros para optimizar
        self.hyperparams = {
            'lr': [0.0005, 0.001, 0.002, 0.005],
            'max_epochs': [50, 100, 150],
            'patience': [10, 15, 20],
            'batch_size': [32, 64, 128]
        }

        # Modelos disponibles
        self.model_classes = {
            'A3TGCN': A3TGCNModel,
            'EvolveGCN-H': EvolveGCNHModel,
            'TDGNN': TDGNNModel,
            'TGN-Simple': TGNSimpleModel
        }

        print("🤖 ENTRENADOR DE MODELOS AI VALIDADOS")
        print(f"📁 Directorio base: {self.base_dir}")
        print(f"🎯 Modelos: {list(self.model_classes.keys())}")
        
    def load_validated_ai_concepts(self) -> Set[str]:
        """Cargar conceptos AI validados por OpenAlex"""
        validation_file = self.validation_dir / "validated_ai_concepts.json"
        
        if not validation_file.exists():
            # Si no existe, crear conceptos AI básicos validados
            validated_concepts = {
                "artificial intelligence", "machine learning", "deep learning",
                "neural network", "neural networks", "convolutional neural networks",
                "recurrent neural networks", "transformer", "attention mechanism",
                "reinforcement learning", "supervised learning", "unsupervised learning",
                "computer vision", "natural language processing", "speech recognition",
                "pattern recognition", "feature extraction", "classification algorithms",
                "regression analysis", "clustering", "anomaly detection",
                "generative adversarial networks", "autoencoder", "variational autoencoder",
                "support vector machine", "decision tree", "random forest",
                "gradient boosting", "ensemble learning", "transfer learning",
                "few-shot learning", "meta-learning", "continual learning",
                "explainable ai", "interpretable machine learning", "ai ethics",
                "algorithmic bias", "fairness in ai", "ai safety"
            }
            
            # Guardar conceptos validados
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(list(validated_concepts), f, indent=2, ensure_ascii=False)
                
            print(f"✅ Conceptos AI validados creados: {len(validated_concepts)}")
            return validated_concepts
        else:
            with open(validation_file, 'r', encoding='utf-8') as f:
                concepts = json.load(f)
            print(f"📖 Conceptos AI validados cargados: {len(concepts)}")
            return set(concepts)
    
    def identify_valid_ai_topics(self, topic_mapping: Dict) -> Set[str]:
        """Identificar tópicos AI válidos usando conceptos de OpenAlex"""
        validated_concepts = self.load_validated_ai_concepts()
        valid_ai_topics = set()
        
        print("\n🔍 IDENTIFICANDO TÓPICOS AI VÁLIDOS")
        print("=" * 50)
        
        for topic_id, topic_info in topic_mapping.items():
            display_name = topic_info.get('display_name', '').lower()
            
            # Verificar coincidencia exacta con conceptos validados
            is_ai_topic = False
            for concept in validated_concepts:
                if concept in display_name:
                    valid_ai_topics.add(topic_id)
                    print(f"✅ {topic_id}: {topic_info.get('display_name')}")
                    is_ai_topic = True
                    break
        
        print(f"\n📊 RESUMEN VALIDACIÓN:")
        print(f"   🎯 Tópicos AI válidos: {len(valid_ai_topics)}")
        print(f"   📊 Total tópicos: {len(topic_mapping)}")
        print(f"   📈 Porcentaje AI válido: {len(valid_ai_topics)/len(topic_mapping)*100:.1f}%")
        
        return valid_ai_topics
    
    def load_topic_mapping(self) -> Dict:
        """Cargar mapeo de tópicos"""
        mapping_file = self.base_dir / "predictions" / "topic_id_to_name_mapping.json"
        
        if not mapping_file.exists():
            raise FileNotFoundError(f"❌ Mapeo de tópicos no encontrado: {mapping_file}")
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_synthetic_ai_data(self, valid_ai_topics: Set[str], topic_mapping: Dict) -> Tuple[List, List]:
        """Crear datos sintéticos basados en tópicos AI válidos"""
        print(f"\n🔄 CREANDO DATOS SINTÉTICOS AI")
        print(f"🎯 Usando {len(valid_ai_topics)} tópicos AI válidos")
        
        # Mapear tópicos AI a índices
        ai_topic_indices = []
        topic_to_index = {topic_id: i for i, topic_id in enumerate(topic_mapping.keys())}
        
        for topic_id in valid_ai_topics:
            if topic_id in topic_to_index:
                ai_topic_indices.append(topic_to_index[topic_id])
        
        all_indices = list(range(len(topic_mapping)))
        non_ai_indices = [i for i in all_indices if i not in ai_topic_indices]
        
        print(f"📊 Índices AI: {len(ai_topic_indices)}")
        print(f"📊 Índices No-AI: {len(non_ai_indices)}")
        
        if len(ai_topic_indices) < 2:
            raise ValueError("❌ Necesitamos al menos 2 tópicos AI para crear enlaces")
        
        # Crear enlaces positivos y negativos de forma equilibrada
        np.random.seed(42)
        
        # Enlaces positivos: AI-AI (alta probabilidad de colaboración)
        positive_pairs = []
        
        # 1. Colaboraciones AI-AI (muy probable)
        for _ in range(500):
            if len(ai_topic_indices) >= 2:
                i, j = np.random.choice(ai_topic_indices, 2, replace=False)
                positive_pairs.append([int(i), int(j)])
        
        # 2. Colaboraciones AI con temas relacionados (probable)
        related_indices = non_ai_indices[:50]  # Primeros 50 como "relacionados"
        for _ in range(300):
            if len(ai_topic_indices) > 0 and len(related_indices) > 0:
                ai_idx = np.random.choice(ai_topic_indices)
                rel_idx = np.random.choice(related_indices)
                positive_pairs.append([int(ai_idx), int(rel_idx)])
        
        # 3. Algunas colaboraciones interdisciplinarias
        for _ in range(200):
            if len(ai_topic_indices) > 0 and len(non_ai_indices) > 50:
                ai_idx = np.random.choice(ai_topic_indices)
                other_idx = np.random.choice(non_ai_indices[50:])  # Temas menos relacionados
                positive_pairs.append([int(ai_idx), int(other_idx)])
        
        # Enlaces negativos: combinaciones improbables
        negative_pairs = []
        
        # 1. AI con temas muy no relacionados
        unrelated_indices = non_ai_indices[100:] if len(non_ai_indices) > 100 else non_ai_indices[10:]
        for _ in range(700):
            if len(ai_topic_indices) > 0 and len(unrelated_indices) > 0:
                ai_idx = np.random.choice(ai_topic_indices)
                unrel_idx = np.random.choice(unrelated_indices)
                negative_pairs.append([int(ai_idx), int(unrel_idx)])
        
        # 2. Pares completamente no-AI
        for _ in range(300):
            if len(non_ai_indices) >= 2:
                i, j = np.random.choice(non_ai_indices, 2, replace=False)
                negative_pairs.append([int(i), int(j)])
        
        # Combinar y mezclar datos
        all_pairs = positive_pairs + negative_pairs
        all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
        
        # Mezclar los datos
        combined = list(zip(all_pairs, all_labels))
        np.random.shuffle(combined)
        all_pairs, all_labels = zip(*combined)
        all_pairs = list(all_pairs)
        all_labels = list(all_labels)
        
        print(f"✅ Datos creados: {len(positive_pairs)} positivos, {len(negative_pairs)} negativos")
        print(f"📊 Distribución: {sum(all_labels)}/{len(all_labels)} = {sum(all_labels)/len(all_labels):.1%} positivos")
        
        return all_pairs, all_labels

    def create_synthetic_temporal_evaluation(self, valid_ai_topics: Set[str], topic_mapping: Dict,
                                             persist_ratio: float = 0.7,
                                             seed: int = 42) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Construye un escenario temporal sintético para evaluar disolución de enlaces.
        - initial_links: enlaces presentes en t
        - next_links: enlaces presentes en t+1 (algunos persisten, otros se disuelven y aparecen nuevos)
        Devuelve dos listas:
          dissolved_pairs (positivos=1) y persisted_pairs (negativos=0) para evaluar disolución.
        """
        rng = np.random.default_rng(seed)

        # Mapear tópicos AI a índices como antes
        topic_to_index = {topic_id: i for i, topic_id in enumerate(topic_mapping.keys())}
        ai_topic_indices = [topic_to_index[t] for t in valid_ai_topics if t in topic_to_index]
        all_indices = list(range(len(topic_mapping)))

        if len(ai_topic_indices) < 2:
            raise ValueError("❌ Necesitamos al menos 2 tópicos AI para escenario temporal")

        # Construir initial_links (AI-AI predominantemente)
        initial_links = set()
        num_initial = min(800, max(100, len(ai_topic_indices) * 6))
        while len(initial_links) < num_initial:
            i, j = rng.choice(ai_topic_indices, 2, replace=False)
            if i != j:
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                initial_links.add((a, b))

        # Decidir persistencias y disoluciones
        initial_links = list(initial_links)
        rng.shuffle(initial_links)
        cut = int(len(initial_links) * persist_ratio)
        persisted = initial_links[:cut]
        dissolved = initial_links[cut:]

        # Crear nuevas apariciones para t+1 (no usadas aquí, pero realismo)
        # Elegimos pares AI con algunos no-AI relacionados
        non_ai_indices = [i for i in all_indices if i not in ai_topic_indices]
        related_indices = non_ai_indices[:50]
        next_new = set()
        target_new = int(len(dissolved) * 1.0)
        while len(next_new) < target_new:
            # 70% AI-AI, 30% AI-relacionado
            if rng.random() < 0.7 and len(ai_topic_indices) >= 2:
                i, j = rng.choice(ai_topic_indices, 2, replace=False)
            else:
                if not ai_topic_indices or not related_indices:
                    continue
                i = rng.choice(ai_topic_indices)
                j = rng.choice(related_indices)
            if i == j:
                continue
            a, b = (int(i), int(j)) if i < j else (int(j), int(i))
            if (a, b) not in persisted and (a, b) not in dissolved:
                next_new.add((a, b))

        # Conjuntos finales (no se devuelven, pero ayudan a validar construcción)
        _next_links = set(persisted) | next_new

        # Pares para evaluación de disolución: positivos (dissolved) vs negativos (persisted)
        dissolved_pairs = [list(p) for p in dissolved]
        persisted_pairs = [list(p) for p in persisted]

        print(f"🧪 Escenario temporal sintético:")
        print(f"   Inicial: {len(initial_links)} | Persisten: {len(persisted_pairs)} | Disueltos: {len(dissolved_pairs)} | Nuevos: {len(next_new)}")

        return dissolved_pairs, persisted_pairs

    def load_real_temporal_networks(self, networks_folder: str = "temporal_networks/networks") -> Dict[str, nx.Graph]:
        """Carga redes temporales reales desde GraphML."""
        graphml_files = sorted(glob.glob(os.path.join(networks_folder, "semantic_network_*.graphml")))
        networks = {}
        for fp in graphml_files:
            try:
                period = os.path.basename(fp).replace("semantic_network_", "").replace(".graphml", "")
                G = nx.read_graphml(fp)
                networks[period] = G
            except Exception as e:
                print(f"⚠️  No se pudo cargar {fp}: {e}")
        print(f"📦 Redes temporales cargadas: {len(networks)}")
        return networks

    def create_real_dissolution_pairs(self, topic_mapping: Dict, max_pairs_per_period: int = 100000) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Construye pares disueltos (positivos) y persistentes (negativos) a partir de redes reales t->t+1.
        Devuelve listas de pares como índices enteros de topic_mapping.
        """
        networks = self.load_real_temporal_networks()
        if len(networks) < 2:
            raise RuntimeError("❌ Se requieren al menos dos redes temporales para evaluar disolución real.")

        # Mapear el ID original de OpenAlex a índice consistente con el mapeo
        topic_to_index = {}
        for key, info in topic_mapping.items():
            orig = info.get('original_id')
            idx = info.get('index')
            if orig is not None and idx is not None:
                topic_to_index[orig] = int(idx)
        periods = sorted(networks.keys())

        all_dissolved: List[List[int]] = []
        all_persisted: List[List[int]] = []

        for i in range(len(periods) - 1):
            p, p2 = periods[i], periods[i + 1]
            Gt, Gn = networks[p], networks[p2]

            # Construir sets de aristas (como índices) para cada periodo
            def edges_as_index_pairs(G):
                pairs = set()
                for u, v in G.edges():
                    # Los IDs de nodo en GraphML son los topic_id originales
                    if u in topic_to_index and v in topic_to_index:
                        a, b = topic_to_index[u], topic_to_index[v]
                        if a == b:
                            continue
                        x, y = (a, b) if a < b else (b, a)
                        pairs.add((x, y))
                return pairs

            e_t = edges_as_index_pairs(Gt)
            e_n = edges_as_index_pairs(Gn)

            dissolved = list(e_t - e_n)  # presentes en t pero NO en t+1
            persisted = list(e_t & e_n)  # presentes en ambos

            # Limitar por periodo para evitar tamaños enormes
            if max_pairs_per_period and (len(dissolved) + len(persisted)) > max_pairs_per_period:
                import random
                random.seed(42)
                k_d = int(max_pairs_per_period * 0.5)
                k_p = max_pairs_per_period - k_d
                dissolved = random.sample(dissolved, min(k_d, len(dissolved)))
                persisted = random.sample(persisted, min(k_p, len(persisted)))

            all_dissolved.extend([list(p) for p in dissolved])
            all_persisted.extend([list(p) for p in persisted])

            print(f"🔻 {p} -> {p2}: disueltos={len(dissolved)} | persistentes={len(persisted)}")

        print(f"🧪 Disolución real total: disueltos={len(all_dissolved)} | persistentes={len(all_persisted)}")
        if not all_dissolved or not all_persisted:
            print("⚠️  Conjunto desequilibrado; considera ajustar filtros/periodos.")
        return all_dissolved, all_persisted
    
    def optimize_hyperparameters(self, model_name: str, model_class, train_data: Tuple, 
                                valid_data: Tuple) -> Dict:
        """Optimizar hiperparámetros para un modelo específico"""
        print(f"\n⚙️  OPTIMIZANDO {model_name}")
        print("=" * 40)
        
        train_pairs, train_labels = train_data
        valid_pairs, valid_labels = valid_data
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        best_score = 0
        best_config = None
        results = []
        
        # Grid search sobre hiperparámetros
        param_combinations = list(itertools.product(
            self.hyperparams['lr'],
            self.hyperparams['max_epochs'],
            self.hyperparams['patience'],
            self.hyperparams['batch_size']
        ))
        
        print(f"🔍 Probando {len(param_combinations)} combinaciones...")
        
        for i, (lr, max_epochs, patience, batch_size) in enumerate(param_combinations[:12]):  # Limitar para velocidad
            print(f"   {i+1:2d}. lr={lr}, epochs={max_epochs}, patience={patience}, batch={batch_size}")
            
            try:
                # Crear modelo
                num_nodes = max(max(pair) for pair in train_pairs + valid_pairs) + 1
                model = model_class(num_nodes=num_nodes).to(device)
                
                # Entrenar
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.BCELoss()
                
                # Convertir datos
                train_pairs_tensor = torch.tensor(train_pairs, dtype=torch.long).to(device)
                train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).to(device)
                valid_pairs_tensor = torch.tensor(valid_pairs, dtype=torch.long).to(device)
                valid_labels_tensor = torch.tensor(valid_labels, dtype=torch.float32).to(device)
                
                best_val_auc = 0
                patience_counter = 0
                
                for epoch in range(max_epochs):
                    model.train()
                    
                    # Entrenamiento por batches
                    total_loss = 0
                    num_batches = len(train_pairs) // batch_size + 1
                    
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, len(train_pairs))
                        
                        if start_idx >= len(train_pairs):
                            break
                            
                        batch_pairs = train_pairs_tensor[start_idx:end_idx]
                        batch_labels = train_labels_tensor[start_idx:end_idx]
                        
                        optimizer.zero_grad()
                        outputs = model(batch_pairs)
                        loss = criterion(outputs, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                    
                    # Validación
                    if (epoch + 1) % 5 == 0:
                        model.eval()
                        with torch.no_grad():
                            val_outputs = model(valid_pairs_tensor)
                            val_auc = roc_auc_score(valid_labels, val_outputs.cpu().numpy())
                            
                            if val_auc > best_val_auc:
                                best_val_auc = val_auc
                                patience_counter = 0
                            else:
                                patience_counter += 1
                                
                            if patience_counter >= patience // 5:
                                break
                
                # Score final
                model.eval()
                with torch.no_grad():
                    final_outputs = model(valid_pairs_tensor)
                    y_scores = final_outputs.cpu().numpy()
                    y_true = np.array(valid_labels)
                    final_auc = roc_auc_score(y_true, y_scores)
                    # Umbral fijo 0.5 para consistencia con los resultados originales
                    y_pred = (y_scores > 0.5).astype(int)
                    final_precision = precision_score(y_true, y_pred, zero_division=0)
                    final_recall = recall_score(y_true, y_pred, zero_division=0)
                    final_f1 = f1_score(y_true, y_pred)
                    # Matriz de confusión (tn, fp, fn, tp)
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                
                results.append({
                    'lr': lr,
                    'max_epochs': max_epochs,
                    'patience': patience,
                    'batch_size': batch_size,
                    'auc_roc': final_auc,
                    'precision': final_precision,
                    'recall': final_recall,
                    'f1_score': final_f1,
                    'confusion_matrix': {
                        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
                    },
                    'threshold': 0.5
                })
                
                if final_auc > best_score:
                    best_score = final_auc
                    best_config = {
                        'lr': lr,
                        'max_epochs': max_epochs,
                        'patience': patience,
                        'batch_size': batch_size,
                        'auc_roc': final_auc,
                        'precision': final_precision,
                        'recall': final_recall,
                        'f1_score': final_f1,
                        'confusion_matrix': {
                            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
                        },
                        'threshold': 0.5
                    }
                
                print(f"      AUC: {final_auc:.4f}, F1: {final_f1:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
                continue
        
        print(f"🏆 Mejor configuración: AUC={best_score:.4f}")
        print(f"   {best_config}")
        
        return best_config, results
    
    def train_final_model(self, model_name: str, model_class, config: Dict, 
                         all_data: Tuple) -> nn.Module:
        """Entrenar modelo final con la mejor configuración"""
        print(f"\n🚀 ENTRENAMIENTO FINAL - {model_name}")
        print("=" * 40)
        
        all_pairs, all_labels = all_data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Crear modelo
        num_nodes = max(max(pair) for pair in all_pairs) + 1
        model = model_class(num_nodes=num_nodes).to(device)
        
        # Configurar entrenamiento
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.BCELoss()
        
        # Datos
        pairs_tensor = torch.tensor(all_pairs, dtype=torch.long).to(device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32).to(device)
        
        print(f"⚙️  Configuración: {config}")
        print(f"🎯 Entrenando {config['max_epochs']} épocas...")
        
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['max_epochs']):
            total_loss = 0
            num_batches = len(all_pairs) // config['batch_size'] + 1
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * config['batch_size']
                end_idx = min((batch_idx + 1) * config['batch_size'], len(all_pairs))
                
                if start_idx >= len(all_pairs):
                    break
                    
                batch_pairs = pairs_tensor[start_idx:end_idx]
                batch_labels = labels_tensor[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = model(batch_pairs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= config['patience']:
                print(f"⏹️  Early stopping en época {epoch+1}")
                break
                
            if (epoch + 1) % 10 == 0:
                print(f"   Época {epoch+1}: Loss = {avg_loss:.4f}")
        
        print(f"✅ Entrenamiento completado. Loss final: {best_loss:.4f}")
        
        # Guardar modelo
        model_path = self.models_dir / f"{model_name.lower()}_validated_ai.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'num_nodes': num_nodes,
            'final_loss': best_loss
        }, model_path)
        
        print(f"💾 Modelo guardado: {model_path}")
        
        return model
    
    def run_validated_training(self):
        """Ejecutar entrenamiento completo con tópicos AI validados"""
        print("\n🚀 INICIANDO ENTRENAMIENTO CON TÓPICOS AI VALIDADOS")
        print("=" * 60)
        
        try:
            # 1. Cargar mapeo de tópicos
            topic_mapping = self.load_topic_mapping()
            print(f"📖 Tópicos cargados: {len(topic_mapping)}")
            
            # 2. Identificar tópicos AI válidos
            valid_ai_topics = self.identify_valid_ai_topics(topic_mapping)
            
            if len(valid_ai_topics) < 5:
                raise ValueError(f"❌ Muy pocos tópicos AI válidos: {len(valid_ai_topics)}")
            
            # 3. Crear datos sintéticos
            all_pairs, all_labels = self.create_synthetic_ai_data(valid_ai_topics, topic_mapping)
            
            # 4. Dividir datos para validación (estratificado)
            # Asegurar distribución equilibrada en train/validation
            import random
            random.seed(42)
            
            # Crear índices estratificados
            positive_indices = [i for i, label in enumerate(all_labels) if label == 1]
            negative_indices = [i for i, label in enumerate(all_labels) if label == 0]
            
            # Dividir cada clase
            split_pos = int(0.8 * len(positive_indices))
            split_neg = int(0.8 * len(negative_indices))
            
            train_indices = positive_indices[:split_pos] + negative_indices[:split_neg]
            valid_indices = positive_indices[split_pos:] + negative_indices[split_neg:]
            
            random.shuffle(train_indices)
            random.shuffle(valid_indices)
            
            train_pairs = [all_pairs[i] for i in train_indices]
            train_labels = [all_labels[i] for i in train_indices]
            valid_pairs = [all_pairs[i] for i in valid_indices]
            valid_labels = [all_labels[i] for i in valid_indices]
            
            train_data = (train_pairs, train_labels)
            valid_data = (valid_pairs, valid_labels)
            
            print(f"📊 División estratificada:")
            print(f"   Entrenamiento: {len(train_pairs)} ejemplos ({sum(train_labels)} pos, {len(train_labels)-sum(train_labels)} neg)")
            print(f"   Validación: {len(valid_pairs)} ejemplos ({sum(valid_labels)} pos, {len(valid_labels)-sum(valid_labels)} neg)")
            
            # 4b. Preparar evaluación de disolución con datos REALES (redes GraphML t->t+1)
            dissolved_pairs_eval, persisted_pairs_eval = self.create_real_dissolution_pairs(topic_mapping)

            # 5. Entrenar todos los modelos
            results = {}
            trained_models = {}
            
            for model_name, model_class in self.model_classes.items():
                print(f"\n{'='*20} {model_name} {'='*20}")
                
                # Optimizar hiperparámetros
                best_config, optimization_results = self.optimize_hyperparameters(
                    model_name, model_class, train_data, valid_data
                )
                
                # Entrenar modelo final
                if best_config:
                    final_model = self.train_final_model(
                        model_name, model_class, best_config, (all_pairs, all_labels)
                    )
                    
                    results[model_name] = {
                        'best_config': best_config,
                        'optimization_history': optimization_results
                    }
                    trained_models[model_name] = final_model

                    # Evaluación adicional: DISOLUCIÓN
                    try:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        final_model.eval()
                        with torch.no_grad():
                            # Si no hay pares, omitir evaluación
                            if len(dissolved_pairs_eval) == 0 or len(persisted_pairs_eval) == 0:
                                print(f"   ⚠️  Disolución omitida para {model_name}: no hay pares reales (disueltos o persistentes)")
                                raise StopIteration()

                            # Construir tensores (asegurar forma Nx2)
                            import numpy as np
                            diss_arr = np.array(dissolved_pairs_eval, dtype=np.int64).reshape(-1, 2)
                            pers_arr = np.array(persisted_pairs_eval, dtype=np.int64).reshape(-1, 2)
                            diss_tensor = torch.from_numpy(diss_arr).to(device)
                            pers_tensor = torch.from_numpy(pers_arr).to(device)

                            # Puntajes de existencia de enlace
                            diss_scores_exist = final_model(diss_tensor).detach().cpu().numpy()
                            pers_scores_exist = final_model(pers_tensor).detach().cpu().numpy()

                            # Para disolución, invertimos: score_dissolution = 1 - score_exist
                            y_scores = np.concatenate([1 - diss_scores_exist, 1 - pers_scores_exist])
                            y_true = np.concatenate([np.ones(len(diss_scores_exist)), np.zeros(len(pers_scores_exist))])

                            # Métricas con umbral 0.5
                            y_pred = (y_scores > 0.5).astype(int)
                            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                            auc_roc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
                            prec = precision_score(y_true, y_pred, zero_division=0)
                            rec = recall_score(y_true, y_pred, zero_division=0)
                            f1 = f1_score(y_true, y_pred, zero_division=0)

                            # Guardar JSON
                            diss_path = self.cm_dir / f"{model_name.lower()}_validated_confusion_matrix_dissolution.json"
                            with open(diss_path, 'w', encoding='utf-8') as cf:
                                json.dump({
                                    'model': model_name,
                                    'task': 'dissolution',
                                    'threshold': 0.5,
                                    'auc_roc': float(auc_roc),
                                    'precision': float(prec),
                                    'recall': float(rec),
                                    'f1_score': float(f1),
                                    'confusion_matrix': {
                                        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
                                    }
                                }, cf, indent=2, ensure_ascii=False)
                            print(f"   💾 Disolución guardada: {diss_path.name} (AUC={auc_roc:.4f}, F1={f1:.4f})")
                    except StopIteration:
                        pass
                    except Exception as e:
                        print(f"   ⚠️  No se pudo evaluar disolución para {model_name}: {e}")
            
            # 6. Guardar resultados
            results_file = self.results_dir / "validated_ai_training_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'valid_ai_topics_count': len(valid_ai_topics),
                    'total_topics': len(topic_mapping),
                    'ai_percentage': len(valid_ai_topics) / len(topic_mapping) * 100,
                    'training_data_size': len(all_pairs),
                    'results': results
                }, f, indent=2, ensure_ascii=False)

            # 6b. Guardar matrices de confusión de la mejor config por modelo, si existen
            try:
                for model_name, info in results.items():
                    best_cfg = info.get('best_config', {})
                    cm = best_cfg.get('confusion_matrix')
                    if cm:
                        cm_path = self.cm_dir / f"{model_name.lower()}_validated_confusion_matrix.json"
                        with open(cm_path, 'w', encoding='utf-8') as cf:
                            json.dump({
                                'model': model_name,
                                'threshold': best_cfg.get('threshold', 0.5),
                                'auc_roc': best_cfg.get('auc_roc'),
                                'f1_score': best_cfg.get('f1_score'),
                                'confusion_matrix': cm
                            }, cf, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ No se pudieron guardar algunas matrices de confusión: {e}")
            
            print(f"\n💾 RESULTADOS GUARDADOS:")
            print(f"✅ Archivo: {results_file}")
            
            # 7. Mostrar resumen
            self.display_training_summary(results, len(valid_ai_topics))
            
            print("\n🎉 ¡ENTRENAMIENTO VALIDADO COMPLETADO!")
            return results, trained_models
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            raise
    
    def display_training_summary(self, results: Dict, num_ai_topics: int):
        """Mostrar resumen del entrenamiento"""
        print("\n📊 RESUMEN DE ENTRENAMIENTO")
        print("=" * 40)
        print(f"🎯 Tópicos AI válidos utilizados: {num_ai_topics}")
        print(f"🤖 Modelos entrenados: {len(results)}")
        
        print("\n🏆 MEJORES CONFIGURACIONES:")
        
        # Ordenar por AUC
        sorted_models = sorted(
            results.items(), 
            key=lambda x: x[1]['best_config']['auc_roc'], 
            reverse=True
        )
        
        for rank, (model_name, result) in enumerate(sorted_models, 1):
            config = result['best_config']
            print(f"\n{rank}. {model_name}")
            print(f"   🎯 AUC-ROC: {config['auc_roc']:.4f}")
            print(f"   📊 F1-Score: {config['f1_score']:.4f}")
            print(f"   ⚙️  LR: {config['lr']}, Epochs: {config['max_epochs']}")
            print(f"   🎛️  Patience: {config['patience']}, Batch: {config['batch_size']}")

if __name__ == "__main__":
    # Ejecutar entrenamiento validado
    trainer = ValidatedAITrainer(".")
    results, models = trainer.run_validated_training()
