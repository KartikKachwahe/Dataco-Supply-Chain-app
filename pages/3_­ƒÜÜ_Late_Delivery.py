import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from app_utils import (  # noqa: E402
    inject_css, page_header, load_data, load_model, load_metrics, encode_input,
    PLOTLY_TEMPLATE, PLOTLY_COLORWAY, COLOR_TEAL, COLOR_CORAL,
)

st.set_page_config(page_title="Late Delivery · DataCo Control Tower", page_icon="🚚", layout="wide")
inject_css()
page_header("03 · Predictive Model", "Late Delivery Risk",
            "Score an order in real time using the best-performing classifier.")

df = load_data()
metrics = load_metrics()
bundle = load_model("late_delivery")
model = bundle["model"]
features = bundle["features"]
encoders = bundle.get("encoders", {})
best_name = metrics["late_delivery"]["best_model"]

left, right = st.columns([1.1, 1])

# ---------------------------------------------------------------------
# Live scoring form
# ---------------------------------------------------------------------
with left:
    st.markdown(f"#### Score an order &nbsp;<span class='pill pill-amber'>{best_name}</span>", unsafe_allow_html=True)
    with st.form("late_delivery_form"):
        c1, c2 = st.columns(2)
        with c1:
            shipping_mode = st.selectbox("Shipping mode", sorted(df["shipping_mode"].unique()))
            order_status = st.selectbox("Order status", sorted(df["order_status"].unique()))
            market = st.selectbox("Market", sorted(df["market"].unique()))
            order_type = st.selectbox("Payment type", sorted(df["type"].unique()))
            category_name = st.selectbox("Category", sorted(df["category_name"].unique()))
        with c2:
            days_scheduled = st.slider("Days for shipment (scheduled)", 0, 10, 4)
            sales = st.number_input("Sales ($)", min_value=0.0, value=float(df["sales"].median()), step=10.0)
            product_price = st.number_input("Product price ($)", min_value=0.0, value=float(df["product_price"].median()), step=10.0)
            discount = st.number_input("Order item discount ($)", min_value=0.0, value=float(df["order_item_discount"].median()), step=1.0)
            quantity = st.slider("Order item quantity", 1, 10, 2)
            profit = st.number_input("Benefit per order ($)", value=float(df["benefit_per_order"].median()), step=5.0)

        submitted = st.form_submit_button("Predict late-delivery risk", width='stretch')

    if submitted:
        raw = {
            "benefit_per_order": profit,
            "order_profit_per_order": profit,
            "order_item_discount": discount,
            "days_for_shipment_scheduled": days_scheduled,
            "shipping_mode": shipping_mode,
            "order_status": order_status,
            "market": market,
            "sales": sales,
            "category_name": category_name,
            "product_price": product_price,
            "type": order_type,
            "order_item_quantity": quantity,
        }
        encoded = encode_input(raw, encoders)
        X = pd.DataFrame([{f: encoded.get(f, 0) for f in features}])[features]
        proba = model.predict_proba(X)[0, 1]
        pred = int(proba >= 0.5)

        st.markdown("---")
        if pred:
            st.error(f"⚠️ **High risk of late delivery** — predicted probability {proba*100:.1f}%")
        else:
            st.success(f"✅ **Likely on time** — predicted late-delivery probability {proba*100:.1f}%")
        st.progress(min(max(proba, 0.0), 1.0))

# ---------------------------------------------------------------------
# Model comparison + feature importance
# ---------------------------------------------------------------------
with right:
    st.markdown("#### Model comparison")
    results = metrics["late_delivery"]["results"]
    comp = pd.DataFrame(results).T[["accuracy", "auc"]].reset_index().rename(columns={"index": "model"})
    fig = px.bar(comp, x="model", y=["accuracy", "auc"], barmode="group",
                 template=PLOTLY_TEMPLATE, color_discrete_sequence=[PLOTLY_COLORWAY[0], PLOTLY_COLORWAY[1]])
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), yaxis_title="Score", xaxis_title="", legend_title="")
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### ROC curve")
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Scatter(x=r["roc"]["fpr"], y=r["roc"]["tpr"], mode="lines",
                                  name=f"{name} (AUC={r['auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
    fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=0, r=10, t=10, b=0),
                       xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", legend=dict(x=0.5, y=0.05))
    st.plotly_chart(fig, width='stretch')

    st.markdown(f"#### Feature importance — {best_name}")
    imp = metrics["late_delivery"]["feature_importance"]
    if imp:
        imp_df = pd.DataFrame(sorted(imp.items(), key=lambda x: x[1]), columns=["feature", "importance"])
        fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                     template=PLOTLY_TEMPLATE, color_discrete_sequence=[PLOTLY_COLORWAY[1]])
        fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), yaxis_title="", xaxis_title="Importance")
        st.plotly_chart(fig, width='stretch')
    else:
        st.caption(f"{best_name} does not expose native feature importances.")

with st.expander("Confusion matrix — best model"):
    cm = results[best_name]["confusion_matrix"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Predicted On Time", "Predicted Late"], y=["Actual On Time", "Actual Late"],
        colorscale=[[0, COLOR_TEAL], [1, COLOR_CORAL]], texttemplate="%{z}",
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=0, r=10, t=10, b=0), height=350)
    st.plotly_chart(fig, width='stretch')
