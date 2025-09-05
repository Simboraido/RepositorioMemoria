# RepositorioMemoria — Predicción de enlaces en redes temporales de tópicos

Memoria de Science of Data Science.

Este repositorio contiene el pipeline reproducible para entrenar y evaluar modelos (GNN y baselines) de predicción de Aparición y Desaparición de enlaces en redes temporales de tópicos, y generar predicciones futuras (2024-2025, 2025-2026) usando el mejor modelo validado.

## Estructura
- `temporal_networks/networks/`: redes GraphML (por periodo) y CSV auxiliares.
- `scripts/`: entrenamiento, evaluación, reporte y predicción.
- `validated_results/`: resultados consolidados y matrices de confusión (JSON/PNG).
- `baseline_predictions/`: matrices de confusión de baselines agregadas.
- `predictions/`: salidas de predicción y análisis.

## Definiciones: Apariciones y Desapariciones de enlaces
- Aparición (link appearance): pares de tópicos que no tenían arista en el periodo t y sí la tienen en t+1. Es una tarea de enlace positivo (y_true=1 si aparece; el modelo puntúa probabilidad de existencia del enlace).
- Desaparición (link dissolution): pares de tópicos que tenían arista en t y dejan de tenerla en t+1. La evaluamos como clasificación binaria invirtiendo el score de existencia: score_diss = 1 − score_exist, con y_true=1 si el enlace desaparece.

Ambas tareas generan sus propias matrices de confusión, métricas (Precision/Recall/F1) y AUC.

## Requisitos
Instala dependencias:

```bash
pip install -r requirements.txt

Este repositorio usa submódulos externos para algunos modelos (TGN, TDGNN, EvolveGCN). Para clonar o actualizar todo correctamente:

Clonación (recursiva):
```bash
git clone --recurse-submodules https://github.com/Simboraido/RepositorioMemoria.git
```

Si ya clonaste sin submódulos, inicializa/actualiza así:
```bash
git submodule update --init --recursive
```

Para traer cambios en submódulos posteriormente:
```bash
git submodule update --remote --recursive
```
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

Umbrales (thresholds): puedes optimizarlos desde scores de validación con:
```bash
python scripts/optimize_threshold.py --csv validated_results/<modelo>_validation_scores.csv --out validated_results/thresholds/<modelo>/
```
Luego, reconstruye matrices al nuevo umbral:
```bash
python scripts/rebuild_confusion_from_scores.py --csv validated_results/<modelo>_validation_scores.csv --model-name <Modelo> --threshold <thr>
python scripts/recompute_metrics_from_confusion.py
```

Checkpoints `.pth` no se versionan; súbelos como assets de Release y documenta su descarga si deseas reproducir entrenamiento.

## Notas
- Los periodos se basan en la última red disponible (`semantic_network_2022-2023.graphml`).
- Las evaluaciones incluyen aparición y disolución, con F1 reales a partir de matrices de confusión.
- Para replicación exacta, asegúrate de tener submódulos actualizados (ver sección de submódulos) y las redes GraphML generadas en `temporal_networks/networks/`.
