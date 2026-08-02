"""
data_cleaning.py
Reusable data cleaning and preprocessing functions for the DataCo Supply Chain dataset.
"""


import pandas as pd
import numpy as np


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw DataCo dataset from CSV."""
    df = pd.read_csv(filepath, encoding='latin-1')
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: lowercase, replace spaces with underscores, remove special chars."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('.', '_', regex=False)
    )
    return df


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are irrelevant or contain PII/placeholder data."""
    cols_to_drop = [
        'customer_email', 'customer_password', 'product_image',
        'customer_fname', 'customer_lname'
    ]
    existing_drops = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"Dropped {len(existing_drops)} redundant columns: {existing_drops}")
    return df


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date string columns to datetime objects."""
    date_cols = ['order_date_dateorders', 'shipping_date_dateorders']
    df = df.copy()
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Converted '{col}' to datetime")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Report and handle missing values."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    }).query('missing_count > 0').sort_values('missing_pct', ascending=False)

    if len(missing_report) > 0:
        print(f"\nColumns with missing values:\n{missing_report}")
    else:
        print("No missing values found.")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"Removed {removed:,} duplicate rows ({removed/initial_rows*100:.2f}%)")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful calculated columns for analysis."""
    df = df.copy()

    # Shipping delay (actual - scheduled)
    if 'days_for_shipping_real' in df.columns and 'days_for_shipment_scheduled' in df.columns:
        df['shipping_delay_days'] = df['days_for_shipping_real'] - df['days_for_shipment_scheduled']

    # Order month and year for time series
    if 'order_date_dateorders' in df.columns:
        df['order_month'] = df['order_date_dateorders'].dt.to_period('M')
        df['order_year'] = df['order_date_dateorders'].dt.year
        df['order_day_of_week'] = df['order_date_dateorders'].dt.day_name()

    # Profit margin percentage
    if 'order_profit_per_order' in df.columns and 'sales' in df.columns:
        df['profit_margin_pct'] = np.where(
            df['sales'] != 0,
            (df['order_profit_per_order'] / df['sales'] * 100).round(2),
            0
        )

    print("Added derived features: shipping_delay_days, order_month, order_year, order_day_of_week, profit_margin_pct")
    return df


def run_full_cleaning_pipeline(filepath: str) -> pd.DataFrame:
    """Execute the complete cleaning pipeline and return a clean DataFrame."""
    print("=" * 60)
    print("RUNNING DATA CLEANING PIPELINE")
    print("=" * 60)

    df = load_raw_data(filepath)
    df = standardize_columns(df)
    df = drop_redundant_columns(df)
    df = convert_date_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = add_derived_features(df)

    print("=" * 60)
    print(f"PIPELINE COMPLETE — Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("=" * 60)
    return df
