# -*- coding: utf-8 -*-
"""
Accuracy assessment for building use classification.
Exports ONE CSV table with per-class AND overall metrics.

Class coding:
  0 = Non-residential
  1 = Residential
"""

import geopandas as gpd
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
IN_SHP  = r"D:\PhD_main\chapter_2\outputs\bldg_type_accuracy_samples.shp"
OUT_CSV = r"D:\PhD_main\chapter_2\outputs\bldg_type_accuracy_metrics.csv"

# --------------------------------------------------
# Load data
# --------------------------------------------------
gdf = gpd.read_file(IN_SHP)

PRED_COL = "Prediction"

# Auto-detect truncated GroundTruth field (shapefile-safe)
gt_col = next(
    (c for c in gdf.columns if c.lower().startswith("ground")),
    None
)

if PRED_COL not in gdf.columns or gt_col is None:
    raise ValueError(f"Prediction or GroundTruth field not found.\nColumns: {list(gdf.columns)}")

GROUP_FIELDS = ["shapeGroup", "Community", "Setting"]
for c in GROUP_FIELDS:
    if c not in gdf.columns:
        raise ValueError(f"Missing grouping field: {c}")

df = gdf.dropna(subset=[PRED_COL, gt_col]).copy()
df[PRED_COL] = df[PRED_COL].astype(int)
df[gt_col]   = df[gt_col].astype(int)

# --------------------------------------------------
# Metric function
# --------------------------------------------------
def compute_metrics(y_true, y_pred):
    return {
        "n": len(y_true),

        # Overall
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),

        # Residential (1)
        "precision_residential": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_residential":    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_residential":        f1_score(y_true, y_pred, pos_label=1, zero_division=0),

        # Non-residential (0)
        "precision_nonres": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_nonres":    recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_nonres":        f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

rows = []

# --------------------------------------------------
# Overall
# --------------------------------------------------
m = compute_metrics(df[gt_col], df[PRED_COL])
m.update({"group_type": "Overall", "group_value": "All"})
rows.append(m)

# --------------------------------------------------
# Grouped metrics
# --------------------------------------------------
for field in GROUP_FIELDS:
    for val, g in df.groupby(field):
        m = compute_metrics(g[gt_col], g[PRED_COL])
        m.update({"group_type": field, "group_value": val})
        rows.append(m)

# --------------------------------------------------
# Export CSV
# --------------------------------------------------
out_df = pd.DataFrame(rows)[
    [
        "group_type", "group_value", "n",
        "overall_accuracy", "macro_f1",
        "precision_residential", "recall_residential", "f1_residential",
        "precision_nonres", "recall_nonres", "f1_nonres",
    ]
].sort_values(["group_type", "group_value"])

out_df.to_csv(OUT_CSV, index=False)

print(f"Accuracy table written to:\n{OUT_CSV}")
