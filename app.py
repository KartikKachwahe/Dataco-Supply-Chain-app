import sys
from pathlib import Path 

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from app_utils import inject_css, page_header, load_data, load_metrics  # noqa: E402

st.set_page_config(
    page_title="DataCo Supply Chain Control Tower",
    page_icon="🚛",
    layout="wide",
)
inject_css()

page_header(
    "DataCo Global · Live Manifest",
    "Supply Chain Control Tower",
    "180K+ orders across 5 markets — descriptive, diagnostic, and predictive analytics in one place.",
)

df = load_data()
metrics = load_metrics()

# ---------------------------------------------------------------------
# Top-line snapshot
# ---------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Orders analyzed", f"{len(df):,}")
with col2:
    on_time_rate = 100 - df["late_delivery_risk"].mean() * 100
    st.metric("On-time delivery rate", f"{on_time_rate:.1f}%")
with col3:
    total_revenue = df["sales"].sum()
    st.metric("Total revenue (sample)", f"${total_revenue/1e6:.1f}M")
with col4:
    fraud_rate = (df["order_status"] == "SUSPECTED_FRAUD").mean() * 100
    st.metric("Suspected fraud rate", f"{fraud_rate:.2f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Navigation cards
# ---------------------------------------------------------------------
st.markdown("### Explore the control tower")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        <div class="panel">
        <span class="pill pill-amber">01</span><b>Overview</b>
        <p style="color:#8B93A7; margin-top:0.5rem;">
        KPI dashboard — revenue, profit margin, on-time rate, delivery status,
        with live filters by market, region, shipping mode, and time period.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="panel">
        <span class="pill pill-teal">03</span><b>Late Delivery Prediction</b>
        <p style="color:#8B93A7; margin-top:0.5rem;">
        Enter order details and get a live late-delivery risk score from the
        best-performing model (Gradient Boosting), plus feature importance.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="panel">
        <span class="pill pill-amber">02</span><b>Exploratory Data Analysis</b>
        <p style="color:#8B93A7; margin-top:0.5rem;">
        Interactive Plotly versions of the original EDA — monthly trends,
        regional revenue, shipping delay distribution, correlation heatmap.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="panel">
        <span class="pill pill-coral">04</span><b>Fraud Detection</b>
        <p style="color:#8B93A7; margin-top:0.5rem;">
        Score a transaction for fraud risk in real time and inspect which
        signals (order type, delivery risk, discount, margin) drove it.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.info("Use the sidebar to jump between pages: **Overview → EDA → Late Delivery → Fraud Detection**.")

with st.expander("About this project"):
    st.markdown(
        """
This app is the deployed companion to the **DataCo Smart Supply Chain Analytics**
project: an end-to-end analytics pipeline covering data cleaning, EDA, SQL KPI
extraction, and two classification models (late delivery risk, fraud risk)
trained on 180K+ orders from DataCo Global.

**Tech stack:** Python, Pandas, Plotly, Scikit-learn, Streamlit.
        """
    )
