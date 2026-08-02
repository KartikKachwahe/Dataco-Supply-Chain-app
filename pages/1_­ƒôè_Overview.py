import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from app_utils import (  # noqa: E402
    inject_css, page_header, load_data,
    PLOTLY_TEMPLATE, PLOTLY_COLORWAY, COLOR_TEAL, COLOR_CORAL,
)

st.set_page_config(page_title="Overview · DataCo Control Tower", page_icon="📊", layout="wide")
inject_css()
page_header("01 · KPI Dashboard", "Supply Chain Overview", "Filter the manifest and watch every KPI update live.")

df = load_data()

# ---------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filters")
    markets = st.multiselect("Market", sorted(df["market"].unique()), default=list(sorted(df["market"].unique())))
    regions = st.multiselect("Order region", sorted(df["order_region"].unique()))
    modes = st.multiselect("Shipping mode", sorted(df["shipping_mode"].unique()))
    categories = st.multiselect("Category", sorted(df["category_name"].unique()))
    years = st.multiselect("Order year", sorted(df["order_year"].unique()))

f = df.copy()
if markets:
    f = f[f["market"].isin(markets)]
if regions:
    f = f[f["order_region"].isin(regions)]
if modes:
    f = f[f["shipping_mode"].isin(modes)]
if categories:
    f = f[f["category_name"].isin(categories)]
if years:
    f = f[f["order_year"].isin(years)]

if f.empty:
    st.warning("No orders match the current filters — widen your selection in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total revenue", f"${f['sales'].sum()/1e6:.2f}M")
c2.metric("Total profit", f"${f['order_profit_per_order'].sum()/1e3:.0f}K")
c3.metric("Avg profit margin", f"{f['profit_margin_pct'].mean():.1f}%")
c4.metric("On-time rate", f"{100 - f['late_delivery_risk'].mean()*100:.1f}%")
c5.metric("Avg shipping delay", f"{f['shipping_delay_days'].mean():.2f} days")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.markdown("**Revenue by market**")
    rev_market = f.groupby("market", as_index=False)["sales"].sum().sort_values("sales")
    fig = px.bar(rev_market, x="sales", y="market", orientation="h",
                 template=PLOTLY_TEMPLATE, color_discrete_sequence=PLOTLY_COLORWAY)
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Total Sales ($)", yaxis_title="")
    st.plotly_chart(fig, width='stretch')

with right:
    st.markdown("**Delivery status distribution**")
    status_counts = f["delivery_status"].value_counts().reset_index()
    status_counts.columns = ["delivery_status", "count"]
    fig = px.pie(status_counts, names="delivery_status", values="count", hole=0.55,
                 template=PLOTLY_TEMPLATE, color_discrete_sequence=PLOTLY_COLORWAY)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width='stretch')

left2, right2 = st.columns(2)

with left2:
    st.markdown("**Revenue & profit by category (top 10)**")
    cat = f.groupby("category_name", as_index=False).agg(
        total_revenue=("sales", "sum"), total_profit=("order_profit_per_order", "sum")
    ).sort_values("total_revenue", ascending=False).head(10)
    fig = px.bar(cat, x="category_name", y=["total_revenue", "total_profit"], barmode="group",
                 template=PLOTLY_TEMPLATE, color_discrete_sequence=[PLOTLY_COLORWAY[0], PLOTLY_COLORWAY[1]])
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), xaxis_title="", yaxis_title="$", legend_title="")
    st.plotly_chart(fig, width='stretch')

with right2:
    st.markdown("**On-time vs. late rate by shipping mode**")
    mode_perf = f.groupby("shipping_mode", as_index=False)["late_delivery_risk"].mean()
    mode_perf["late_delivery_risk"] *= 100
    mode_perf = mode_perf.sort_values("late_delivery_risk")
    fig = px.bar(mode_perf, x="late_delivery_risk", y="shipping_mode", orientation="h",
                 template=PLOTLY_TEMPLATE, color="late_delivery_risk",
                 color_continuous_scale=[COLOR_TEAL, COLOR_CORAL])
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Late Delivery Rate (%)",
                       yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')

st.markdown("**Monthly revenue trend**")
monthly = f.groupby("order_month", as_index=False)["sales"].sum().sort_values("order_month")
fig = px.area(monthly, x="order_month", y="sales", template=PLOTLY_TEMPLATE,
              color_discrete_sequence=PLOTLY_COLORWAY)
fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Month", yaxis_title="Revenue ($)")
st.plotly_chart(fig, width='stretch')

with st.expander("View filtered data"):
    st.dataframe(f.head(500), width='stretch')
