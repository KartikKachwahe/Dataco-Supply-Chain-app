import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

NB_DIR = "notebooks/"

nb1 = nbformat.read(NB_DIR + "01_eda_data_cleaning.ipynb", as_version=4)
nb2 = nbformat.read(NB_DIR + "02_sql_kpi_dashboard.ipynb", as_version=4) 
nb3 = nbformat.read(NB_DIR + "03_predictive_modeling.ipynb", as_version=4)

merged = new_notebook()
cells = []

# ---------------------------------------------------------------
# Title
# ---------------------------------------------------------------
cells.append(new_markdown_cell(
"""# DataCo Smart Supply Chain — Full Analysis Notebook

This single notebook combines all three project deliverables end-to-end:

1. **Data Cleaning & Exploratory Data Analysis** (formerly `01_eda_data_cleaning.ipynb`)
2. **SQL KPI Dashboard Analysis** (formerly `02_sql_kpi_dashboard.ipynb`)
3. **Predictive Modeling — Late Delivery & Fraud Detection** (formerly `03_predictive_modeling.ipynb`)

The dataframe produced in Part 1 is reused directly by Parts 2 and 3, so run the
notebook top-to-bottom in a single session.

> **Data:** place `DataCoSupplyChainDataset.csv` in `data/raw/` before running
> (see the load cell directly below for the exact path)."""
))

# ---------------------------------------------------------------
# Part 1 — EDA & Data Cleaning
# ---------------------------------------------------------------
cells.append(new_markdown_cell("---\n# Part 1 — Data Cleaning & Exploratory Data Analysis"))

part1_cells = list(nb1.cells)
# Fix the first (Colab upload) cell to load from local disk instead
fixed_load = new_code_cell(
"""import pandas as pd
import numpy as np

# Path to the raw dataset (see data/raw/ in the repo structure)
DATA_PATH = "data/raw/DataCoSupplyChainDataset.csv"

df = pd.read_csv(DATA_PATH, encoding="latin-1")

print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")"""
)
part1_cells[0] = fixed_load
cells.extend(part1_cells)

# ---------------------------------------------------------------
# Part 2 — SQL KPI Dashboard
# ---------------------------------------------------------------
cells.append(new_markdown_cell(
"""---
# Part 2 — SQL KPI Dashboard Analysis

Reuses the cleaned `df` from Part 1 — no need to reload or re-clean the raw CSV.
The dataframe is loaded into an in-memory SQLite database so the original SQL
queries (`sql/supply_chain_queries.sql`) can run unmodified."""
))

part2_cells = list(nb2.cells)
# Replace the redundant reload/re-clean cell with a lightweight reuse of `df`
fixed_reuse = new_code_cell(
"""import sqlite3

# `df` already exists (cleaned in Part 1) — just make sure order_month is a
# plain string so SQLite can store it.
df["order_month"] = df["order_month"].astype(str)

conn = sqlite3.connect(":memory:")
df.to_sql("orders", conn, if_exists="replace", index=False)

test = pd.read_sql("SELECT COUNT(*) as total_rows FROM orders", conn)
print(f"Database created with {test['total_rows'][0]:,} rows")"""
)
part2_cells[0] = fixed_reuse
cells.extend(part2_cells)

# ---------------------------------------------------------------
# Part 3 — Predictive Modeling
# ---------------------------------------------------------------
cells.append(new_markdown_cell(
"""---
# Part 3 — Predictive Modeling: Late Delivery & Fraud Detection

Reuses the same cleaned `df` from Part 1. Trains and compares Logistic
Regression, Random Forest, and Gradient Boosting for both the late-delivery
and fraud-detection tasks.

> The Streamlit app (`app.py` + `pages/`) uses model artifacts already
> exported to `models/*.pkl` by `train_and_export.py`, so this section is for
> analysis/reproducibility rather than a required step before deploying."""
))

part3_cells = list(nb3.cells)
fixed_reuse3 = new_code_cell(
"""from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score
)

# `df` already exists (cleaned in Part 1) — reused directly, no reload needed."""
)
part3_cells[0] = fixed_reuse3
cells.extend(part3_cells)

# ---------------------------------------------------------------
# Strip outputs / execution counts so the notebook is light to store in git
# ---------------------------------------------------------------
for c in cells:
    if c.cell_type == "code":
        c.outputs = []
        c.execution_count = None

merged.cells = cells
merged.metadata = nb1.metadata if nb1.metadata else {}
merged.metadata.setdefault("kernelspec", {"name": "python3", "display_name": "Python 3", "language": "python"})

nbformat.write(merged, "dataco-supply-chain-analysis/notebooks/full_analysis.ipynb")
print("Merged notebook written. Total cells:", len(cells))
