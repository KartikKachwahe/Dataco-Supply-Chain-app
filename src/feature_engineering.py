"""
feature_engineering.py
Feature engineering functions for predictive modeling (Deliverable 3).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_categorical_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Label encode specified categorical columns."""
    df = df.copy()
    encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def create_delivery_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features specifically for late delivery prediction model."""
    df = df.copy()

    # Order value buckets
    df['order_value_bucket'] = pd.qcut(df['sales'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])

    # Is high discount
    if 'order_item_discount' in df.columns:
        df['is_high_discount'] = (df['order_item_discount'] > df['order_item_discount'].median()).astype(int)

    # Items per order indicator
    if 'order_item_quantity' in df.columns:
        df['is_bulk_order'] = (df['order_item_quantity'] > 3).astype(int)

    # Day of week encoded
    if 'order_day_of_week' in df.columns:
        df['is_weekend_order'] = df['order_day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

    return df


def create_fraud_detection_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features specifically for fraud detection model."""
    df = df.copy()

    # Unusually high profit margin flag
    if 'profit_margin_pct' in df.columns:
        q95 = df['profit_margin_pct'].quantile(0.95)
        df['extreme_margin'] = (df['profit_margin_pct'] > q95).astype(int)

    # Order-to-ship speed
    if 'days_for_shipping_real' in df.columns:
        df['fast_ship'] = (df['days_for_shipping_real'] <= 1).astype(int)

    # Negative profit flag
    if 'order_profit_per_order' in df.columns:
        df['negative_profit'] = (df['order_profit_per_order'] < 0).astype(int)

    return df


def prepare_model_data(df: pd.DataFrame, target_col: str, feature_cols: list,
                       scale: bool = True) -> tuple:
    """Prepare X and y arrays for model training."""
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle any remaining NaN
    X = X.fillna(X.median(numeric_only=True))

    if scale:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        return X_scaled, y, scaler

    return X, y, None
