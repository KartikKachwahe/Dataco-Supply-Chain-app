"""
app_utils.py
Shared helpers for the Streamlit app: cached data/model loading + one
consistent visual theme ("control tower") injected on every page.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed_supply_chain.csv"
MODELS_DIR = ROOT / "models"

# ----------------------------------------------------------------------
# Theme tokens — "control tower": dark slate base, amber = attention /
# in-transit, teal = on-time / healthy, coral = late / risk / fraud.
# ----------------------------------------------------------------------
COLOR_BG = "#12161F"
COLOR_PANEL = "#1B212E"
COLOR_BORDER = "#2A3242"
COLOR_TEXT = "#E8EAED"
COLOR_MUTED = "#8B93A7"
COLOR_AMBER = "#F2A44C"
COLOR_TEAL = "#2DD4BF"
COLOR_CORAL = "#EF6461"
COLOR_BLUE = "#5B8DEF"

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_COLORWAY = [COLOR_AMBER, COLOR_TEAL, COLOR_CORAL, COLOR_BLUE, "#B892FF", "#8B93A7"]


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background-color: {COLOR_BG};
        }}

        [data-testid="stSidebar"] {{
            background-color: {COLOR_PANEL};
            border-right: 1px solid {COLOR_BORDER};
        }}

        .manifest-header {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            border-bottom: 1px solid {COLOR_BORDER};
            padding-bottom: 0.6rem;
            margin-bottom: 1.4rem;
        }}
        .manifest-header .tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            color: {COLOR_AMBER};
            text-transform: uppercase;
        }}
        .manifest-header h1 {{
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0.15rem 0 0 0;
            color: {COLOR_TEXT};
        }}
        .manifest-header .sub {{
            color: {COLOR_MUTED};
            font-size: 0.92rem;
            margin-top: 0.2rem;
        }}

        div[data-testid="stMetric"] {{
            background-color: {COLOR_PANEL};
            border: 1px solid {COLOR_BORDER};
            border-radius: 10px;
            padding: 0.9rem 1rem 0.6rem 1rem;
        }}
        div[data-testid="stMetricLabel"] {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem !important;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {COLOR_MUTED} !important;
        }}
        div[data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace;
            color: {COLOR_TEXT} !important;
        }}

        .panel {{
            background-color: {COLOR_PANEL};
            border: 1px solid {COLOR_BORDER};
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }}
        .pill {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.05em;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            margin-right: 0.4rem;
        }}
        .pill-amber {{ background: rgba(242,164,76,0.15); color: {COLOR_AMBER}; border: 1px solid rgba(242,164,76,0.4); }}
        .pill-teal  {{ background: rgba(45,212,191,0.15); color: {COLOR_TEAL}; border: 1px solid rgba(45,212,191,0.4); }}
        .pill-coral {{ background: rgba(239,100,97,0.15); color: {COLOR_CORAL}; border: 1px solid rgba(239,100,97,0.4); }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(tag: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="manifest-header">
            <div>
                <div class="tag">{tag}</div>
                <h1>{title}</h1>
                <div class="sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading supply chain data...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["order_date_dateorders"] = pd.to_datetime(df["order_date_dateorders"], errors="coerce")
    df["shipping_date_dateorders"] = pd.to_datetime(df["shipping_date_dateorders"], errors="coerce")
    return df


@st.cache_resource(show_spinner="Loading model...")
def load_model(task: str):
    """task: 'late_delivery' or 'fraud'"""
    filename = "late_delivery_model.pkl" if task == "late_delivery" else "fraud_model.pkl"
    path = MODELS_DIR / filename
    return joblib.load(path)


@st.cache_data
def load_metrics() -> dict:
    with open(MODELS_DIR / "metrics.json") as f:
        return json.load(f)


def encode_input(raw: dict, encoders: dict) -> dict:
    """Apply the saved LabelEncoders to raw categorical inputs, falling back
    to the most frequent class if a value wasn't seen during training."""
    out = dict(raw)
    for col, le in encoders.items():
        if col in out:
            val = str(out[col])
            if val in le.classes_:
                out[col] = int(le.transform([val])[0])
            else:
                out[col] = 0
    return out
