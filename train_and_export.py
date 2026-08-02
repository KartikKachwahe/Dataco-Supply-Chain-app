"""
train_and_export.py
Reproduces the notebook pipeline end-to-end:
  raw CSV -> cleaning -> feature engineering -> train 3 models per task
  -> save best model (.pkl) + label encoders + metrics + a sampled processed CSV

Run once locally to populate models/ and data/ before deploying the Streamlit app.
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from data_cleaning import run_full_cleaning_pipeline
from feature_engineering import (
    create_delivery_prediction_features,
    create_fraud_detection_features,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)

RAW_CSV = "data/raw/DataCoSupplyChainDataset.csv"
OUT_DIR = "."

# ------------------------------------------------------------------
# 1. Clean
# ------------------------------------------------------------------
df = run_full_cleaning_pipeline(RAW_CSV)

# ------------------------------------------------------------------
# 2. Feature engineering (task-specific)
# ------------------------------------------------------------------
df = create_delivery_prediction_features(df)
df = create_fraud_detection_features(df)

# Category name -> normalize col name used elsewhere in repo
if "category_name" not in df.columns and "category_name_x" in df.columns:
    df.rename(columns={"category_name_x": "category_name"}, inplace=True)

print("Columns available:", len(df.columns))

# ------------------------------------------------------------------
# Shared categorical encoding (kept simple + consistent for both tasks)
# ------------------------------------------------------------------
cat_cols = ["type", "shipping_mode", "market", "category_name", "order_status"]
encoders = {}
df_enc = df.copy()
for col in cat_cols:
    if col in df_enc.columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

# ------------------------------------------------------------------
# TASK 1: Late Delivery Prediction
# ------------------------------------------------------------------
late_features = [
    "benefit_per_order", "order_profit_per_order", "order_item_discount",
    "days_for_shipment_scheduled", "shipping_mode", "order_status",
    "market", "sales", "category_name", "product_price", "type",
    "order_item_quantity",
]
late_features = [c for c in late_features if c in df_enc.columns]
target_late = "late_delivery_risk"

data_late = df_enc[late_features + [target_late]].dropna()
X = data_late[late_features]
y = data_late[target_late]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

late_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

late_results = {}
late_fitted = {}
for name, model in late_models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, proba)
    late_results[name] = {
        "accuracy": round(float(acc), 4),
        "auc": round(float(auc), 4),
        "confusion_matrix": cm,
        "roc": {"fpr": fpr[::max(1, len(fpr)//100)].tolist(),
                "tpr": tpr[::max(1, len(tpr)//100)].tolist()},
    }
    late_fitted[name] = model
    print(f"[Late Delivery] {name}: acc={acc:.4f} auc={auc:.4f}")

best_late_name = max(late_results, key=lambda n: late_results[n]["auc"])
best_late_model = late_fitted[best_late_name]
late_importance = None
if hasattr(best_late_model, "feature_importances_"):
    late_importance = dict(zip(late_features, best_late_model.feature_importances_.tolist()))

joblib.dump(
    {"model": best_late_model, "features": late_features,
     "model_name": best_late_name, "encoders": {c: encoders[c] for c in cat_cols if c in encoders}},
    f"{OUT_DIR}/models/late_delivery_model.pkl",
)

# ------------------------------------------------------------------
# TASK 2: Fraud Detection
# ------------------------------------------------------------------
fraud_features = [
    "type", "late_delivery_risk", "shipping_delay_days", "days_for_shipping_real",
    "order_item_discount", "benefit_per_order", "order_profit_per_order",
    "market", "days_for_shipment_scheduled", "shipping_mode", "sales",
    "product_price", "category_name", "order_item_quantity",
]
fraud_features = [c for c in fraud_features if c in df_enc.columns]

df_enc["is_fraud"] = (df["order_status"].astype(str) == "SUSPECTED_FRAUD").astype(int)
target_fraud = "is_fraud"

data_fraud = df_enc[fraud_features + [target_fraud]].dropna()
Xf = data_fraud[fraud_features]
yf = data_fraud[target_fraud]
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    Xf, yf, test_size=0.25, random_state=42, stratify=yf
)

fraud_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

fraud_results = {}
fraud_fitted = {}
for name, model in fraud_models.items():
    model.fit(Xf_train, yf_train)
    proba = model.predict_proba(Xf_test)[:, 1]
    pred = model.predict(Xf_test)
    prec = precision_score(yf_test, pred, zero_division=0)
    rec = recall_score(yf_test, pred, zero_division=0)
    auc = roc_auc_score(yf_test, proba)
    cm = confusion_matrix(yf_test, pred).tolist()
    fpr, tpr, _ = roc_curve(yf_test, proba)
    fraud_results[name] = {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "auc": round(float(auc), 4),
        "confusion_matrix": cm,
        "roc": {"fpr": fpr[::max(1, len(fpr)//100)].tolist(),
                "tpr": tpr[::max(1, len(tpr)//100)].tolist()},
    }
    fraud_fitted[name] = model
    print(f"[Fraud] {name}: precision={prec:.4f} recall={rec:.4f} auc={auc:.4f}")

# Best = highest AUC (Random Forest tends to win on precision but GB/LR on recall/AUC)
best_fraud_name = max(fraud_results, key=lambda n: fraud_results[n]["auc"])
best_fraud_model = fraud_fitted[best_fraud_name]
fraud_importance = None
if hasattr(best_fraud_model, "feature_importances_"):
    fraud_importance = dict(zip(fraud_features, best_fraud_model.feature_importances_.tolist()))

joblib.dump(
    {"model": best_fraud_model, "features": fraud_features,
     "model_name": best_fraud_name, "encoders": {c: encoders[c] for c in cat_cols if c in encoders}},
    f"{OUT_DIR}/models/fraud_model.pkl",
)

# ------------------------------------------------------------------
# Save metrics (used by the Streamlit pages so we don't retrain live)
# ------------------------------------------------------------------
metrics = {
    "late_delivery": {"results": late_results, "best_model": best_late_name,
                      "feature_importance": late_importance, "features": late_features},
    "fraud": {"results": fraud_results, "best_model": best_fraud_name,
              "feature_importance": fraud_importance, "features": fraud_features},
}
with open(f"{OUT_DIR}/models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ------------------------------------------------------------------
# Save a sampled + processed CSV for the app (keeps deploy size small)
# ------------------------------------------------------------------
keep_cols = [
    "type", "days_for_shipping_real", "days_for_shipment_scheduled",
    "benefit_per_order", "sales_per_customer", "delivery_status",
    "late_delivery_risk", "category_name", "customer_segment", "market",
    "order_region", "order_country", "order_state", "order_city",
    "order_date_dateorders", "order_item_discount", "order_item_discount_rate",
    "order_item_product_price", "order_item_profit_ratio", "order_item_quantity",
    "sales", "order_item_total", "order_profit_per_order", "order_status",
    "product_name", "product_price", "shipping_mode", "shipping_date_dateorders",
    "shipping_delay_days", "order_month", "order_year", "order_day_of_week",
    "profit_margin_pct",
]
keep_cols = [c for c in keep_cols if c in df.columns]
df_sample = df[keep_cols].copy()

# order_month is a Period -> stringify for CSV portability
if "order_month" in df_sample.columns:
    df_sample["order_month"] = df_sample["order_month"].astype(str)

# Sample down for a light deploy artifact while keeping distribution
if len(df_sample) > 60000:
    df_sample = df_sample.sample(n=60000, random_state=42).reset_index(drop=True)

df_sample.to_csv(f"{OUT_DIR}/data/processed_supply_chain.csv", index=False)

print("\nDone.")
print("Best late delivery model:", best_late_name)
print("Best fraud model:", best_fraud_name)
print("Sampled CSV rows:", len(df_sample))
