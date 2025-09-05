# Resultados corregidos desde matrices de confusión

## Modelos validados (F1 real)

| Modelo | AUC-ROC | Threshold | Precision | Recall | F1 real | F1 original | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EvolveGCN-H | 0.9270 | 0.50 | 0.8357 | 0.8650 | 0.8501 | 0.8501 | 166 | 34 | 27 | 173 |
| A3TGCN | 0.9176 | 0.50 | 0.8140 | 0.8750 | 0.8434 | 0.8434 | 160 | 40 | 25 | 175 |
| TDGNN | 0.9247 | 0.50 | 0.8122 | 0.8650 | 0.8378 | 0.8378 | 160 | 40 | 27 | 173 |
| TGN-Simple | 0.9063 | 0.50 | 0.8028 | 0.8550 | 0.8281 | 0.8281 | 158 | 42 | 29 | 171 |

### Matrices de confusión (modelos validados)
![A3TGCN_validated_cm](validated_results/confusion_matrices/plots/A3TGCN_validated_cm.png)
![EvolveGCN-H_validated_cm](validated_results/confusion_matrices/plots/EvolveGCN-H_validated_cm.png)
![TDGNN_validated_cm](validated_results/confusion_matrices/plots/TDGNN_validated_cm.png)
![TGN-Simple_validated_cm](validated_results/confusion_matrices/plots/TGN-Simple_validated_cm.png)

## Modelos validados - Disolución (F1 real)

| Modelo | AUC-ROC | Threshold | Precision | Recall | F1 real | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EvolveGCN-H | 0.4286 | 0.50 | 0.6829 | 0.9277 | 0.7867 | 2 | 137 | 23 | 295 |
| TGN-Simple | 0.4806 | 0.50 | 0.6821 | 0.9245 | 0.7850 | 2 | 137 | 24 | 294 |
| TDGNN | 0.3937 | 0.50 | 0.6784 | 0.9088 | 0.7769 | 2 | 137 | 29 | 289 |
| A3TGCN | 0.4523 | 0.50 | 0.6745 | 0.8994 | 0.7709 | 1 | 138 | 32 | 286 |

### Matrices de confusión (modelos validados - disolución)
![A3TGCN_validated_cm_dissolution](validated_results/confusion_matrices/plots/A3TGCN_validated_cm_dissolution.png)
![EvolveGCN-H_validated_cm_dissolution](validated_results/confusion_matrices/plots/EvolveGCN-H_validated_cm_dissolution.png)
![TDGNN_validated_cm_dissolution](validated_results/confusion_matrices/plots/TDGNN_validated_cm_dissolution.png)
![TGN-Simple_validated_cm_dissolution](validated_results/confusion_matrices/plots/TGN-Simple_validated_cm_dissolution.png)

## Baselines (agregado por algoritmo)

| Algoritmo | TN | FP | FN | TP | Precision | Recall | F1 real |
|---|---:|---:|---:|---:|---:|---:|---:|
| adamic_adar | 128 | 8 | 30 | 35 | 0.8140 | 0.5385 | 0.6481 |
| common_neighbors | 126 | 10 | 30 | 35 | 0.7778 | 0.5385 | 0.6364 |
| jaccard | 129 | 7 | 30 | 35 | 0.8333 | 0.5385 | 0.6542 |
| preferential_attachment | 95 | 41 | 31 | 34 | 0.4533 | 0.5231 | 0.4857 |
| resource_allocation | 128 | 8 | 30 | 35 | 0.8140 | 0.5385 | 0.6481 |

## Baselines - Disolución (agregado por algoritmo)

| Algoritmo | TN | FP | FN | TP | Precision | Recall | F1 real |
|---|---:|---:|---:|---:|---:|---:|---:|
| adamic_adar | 68 | 37 | 21 | 41 | 0.5256 | 0.6613 | 0.5857 |
| common_neighbors | 68 | 37 | 26 | 36 | 0.4932 | 0.5806 | 0.5333 |
| jaccard | 67 | 38 | 22 | 40 | 0.5128 | 0.6452 | 0.5714 |
| preferential_attachment | 53 | 52 | 15 | 47 | 0.4747 | 0.7581 | 0.5839 |
| resource_allocation | 69 | 36 | 22 | 40 | 0.5263 | 0.6452 | 0.5797 |

## Matrices de confusión (baselines - apariciones)
![2006-2007_to_2008-2009_adamic_adar_cm](baseline_predictions/evaluations/confusion_matrices/plots/2006-2007_to_2008-2009_adamic_adar_cm.png)
![2006-2007_to_2008-2009_adamic_adar_dissolution_cm](baseline_predictions/evaluations/confusion_matrices/plots/2006-2007_to_2008-2009_adamic_adar_dissolution_cm.png)
![2006-2007_to_2008-2009_common_neighbors_cm](baseline_predictions/evaluations/confusion_matrices/plots/2006-2007_to_2008-2009_common_neighbors_cm.png)
![2006-2007_to_2008-2009_common_neighbors_dissolution_cm](baseline_predictions/evaluations/confusion_matrices/plots/2006-2007_to_2008-2009_common_neighbors_dissolution_cm.png)
![2006-2007_to_2008-2009_jaccard_cm](baseline_predictions/evaluations/confusion_matrices/plots/2006-2007_to_2008-2009_jaccard_cm.png)
![2006-2007_to_2008-2009_jaccard_dissolution_cm](baseline_predictions/evaluations/confusion_matrices/plots/2006-2007_to_2008-2009_jaccard_dissolution_cm.png)

## Matrices de confusión (baselines - disolución)
![2006-2007_to_2008-2009_adamic_adar_dissolution_cm](baseline_predictions/evaluations/confusion_matrices/plots/2006-2007_to_2008-2009_adamic_adar_dissolution_cm.png)
![2006-2007_to_2008-2009_common_neighbors_dissolution_cm](baseline_predictions/evaluations/confusion_matrices/plots/2006-2007_to_2008-2009_common_neighbors_dissolution_cm.png)
![2006-2007_to_2008-2009_jaccard_dissolution_cm](baseline_predictions/evaluations/confusion_matrices/plots/2006-2007_to_2008-2009_jaccard_dissolution_cm.png)
![2006-2007_to_2008-2009_preferential_attachment_dissolution_cm](baseline_predictions/evaluations/confusion_matrices/plots/2006-2007_to_2008-2009_preferential_attachment_dissolution_cm.png)
![2006-2007_to_2008-2009_resource_allocation_dissolution_cm](baseline_predictions/evaluations/confusion_matrices/plots/2006-2007_to_2008-2009_resource_allocation_dissolution_cm.png)
![2008-2009_to_2010-2011_adamic_adar_dissolution_cm](baseline_predictions/evaluations/confusion_matrices/plots/2008-2009_to_2010-2011_adamic_adar_dissolution_cm.png)