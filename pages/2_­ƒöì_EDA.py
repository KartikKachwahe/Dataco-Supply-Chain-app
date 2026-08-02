import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from app_utils import inject_css, page_header, load_data, PLOTLY_TEMPLATE, PLOTLY_COLORWAY  # noqa: E402

st.set_page_config(page_title="EDA · DataCo Control Tower", page_icon="🔍", layout="wide")
inject_css()
page_header("02 · Exploratory Data Analysis", "Patterns in the Manifest",
            "The original notebook's plots, rebuilt as interactive Plotly charts.")

df = load_data()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Order trends", "Delivery & shipping", "Profitability", "Correlations"]
)

# ---------------------------------------------------------------------
with tab1:
    st.markdown("**Monthly order volume**")
    monthly = df.groupby("order_month").size().reset_index(name="orders").sort_values("order_month")
    fig = px.line(monthly, x="order_month", y="orders", markers=True,
                  template=PLOTLY_TEMPLATE, color_discrete_sequence=PLOTLY_COLORWAY)
    fig.update_traces(fill="tozeroy")
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Month", yaxis_title="Orders")
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Orders by day of week**")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = df["order_day_of_week"].value_counts().reindex(dow_order).reset_index()
        dow.columns = ["day", "orders"]
        fig = px.bar(dow, x="day", y="orders", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=PLOTLY_COLORWAY)
        fig.update_layout(margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.markdown("**Total sales by market region**")
        region_sales = df.groupby("market", as_index=False)["sales"].sum().sort_values("sales")
        fig = px.bar(region_sales, x="sales", y="market", orientation="h",
                     template=PLOTLY_TEMPLATE, color_discrete_sequence=PLOTLY_COLORWAY)
        fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Total Sales ($)", yaxis_title="")
        st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------
with tab2:
    st.markdown("**Order distribution by delivery status**")
    counts = df["delivery_status"].value_counts().reset_index()
    counts.columns = ["delivery_status", "orders"]
    fig = px.bar(counts, x="delivery_status", y="orders", template=PLOTLY_TEMPLATE,
                 color="delivery_status", color_discrete_sequence=PLOTLY_COLORWAY)
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), showlegend=False, xaxis_title="", yaxis_title="Orders")
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Late delivery risk by shipping mode**")
        late_rate = (df.groupby("shipping_mode")["late_delivery_risk"].mean() * 100).sort_values(ascending=False).reset_index()
        late_rate.columns = ["shipping_mode", "late_rate_pct"]
        fig = px.bar(late_rate, x="late_rate_pct", y="shipping_mode", orientation="h",
                     template=PLOTLY_TEMPLATE, color_discrete_sequence=[PLOTLY_COLORWAY[2]])
        fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Late Delivery Rate (%)", yaxis_title="")
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.markdown("**Distribution of shipping delay (days)**")
        fig = px.histogram(df, x="shipping_delay_days", nbins=30, template=PLOTLY_TEMPLATE,
                            color_discrete_sequence=[PLOTLY_COLORWAY[0]])
        fig.add_vline(x=0, line_dash="dash", line_color=PLOTLY_COLORWAY[1],
                      annotation_text="On time", annotation_position="top")
        fig.update_layout(margin=dict(l=0, r=10, t=10, b=0),
                           xaxis_title="Delay (days) — negative = early, positive = late", yaxis_title="Frequency")
        st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------
with tab3:
    st.markdown("**Profit distribution by top 10 product categories**")
    top_cats = df["category_name"].value_counts().head(10).index
    subset = df[df["category_name"].isin(top_cats)]
    fig = px.box(subset, x="category_name", y="order_profit_per_order", template=PLOTLY_TEMPLATE,
                 color="category_name", color_discrete_sequence=PLOTLY_COLORWAY)
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), showlegend=False,
                       xaxis_title="", yaxis_title="Profit per order ($)")
    st.plotly_chart(fig, width='stretch')

    st.markdown("**Sales vs. profit per order** (bubble size = quantity)")
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(sample, x="sales", y="order_profit_per_order", size="order_item_quantity",
                      color="market", template=PLOTLY_TEMPLATE, color_discrete_sequence=PLOTLY_COLORWAY,
                      opacity=0.6)
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Sales ($)", yaxis_title="Profit per order ($)")
    st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------
with tab4:
    st.markdown("**Correlation heatmap — numeric features**")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0,
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=0, r=10, t=10, b=0), height=650)
    st.plotly_chart(fig, width='stretch')
