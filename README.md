# RepositorioMemoria — Predicción de enlaces en redes temporales de tópicos

Memoria de Science of Data Science.

Este repositorio contiene el pipeline reproducible para entrenar y evaluar modelos (GNN y baselines) de predicción de aparición y disolución de enlaces, y generar predicciones futuras (2024-2025, 2025-2026) usando el mejor modelo validado.

## Estructura
- `temporal_networks/networks/`: redes GraphML (por periodo) y CSV auxiliares.
- `scripts/`: entrenamiento, evaluación, reporte y predicción.
- `validated_results/`: resultados consolidados y matrices de confusión (JSON/PNG).
- `baseline_predictions/`: matrices de confusión de baselines agregadas.
- `predictions/`: salidas de predicción y análisis.

## Requisitos
Instala dependencias:

```bash
pip install -r requirements.txt
```

## Reproducir resultados
1) Recalcular métricas y reporte desde matrices de confusión:
```bash
python scripts/recompute_metrics_from_confusion.py
```
2) Generar predicciones con el mejor modelo:
```bash
python scripts/predict_future_with_best_model.py
```
3) Analizar predicciones (estadísticas y top-K):
```bash
python scripts/analyze_future_predictions.py
```

Checkpoints `.pth` no se versionan; súbelos como assets de Release y documenta su descarga si deseas reproducir entrenamiento.

## Notas
- Los periodos se basan en la última red disponible (`semantic_network_2022-2023.graphml`).
- Las evaluaciones incluyen aparición y disolución, con F1 reales a partir de matrices de confusión.
