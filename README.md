# DataCo Smart Supply Chain — Analytics & Streamlit App

An end-to-end Supply Chain Analytics project: data cleaning, EDA, SQL KPI
extraction, and two predictive models (late delivery risk, fraud risk),
deployed as an interactive **Streamlit** app.

**Live app pages:** Overview → EDA → Late Delivery Prediction → Fraud Detection

---

## 🗂️ Repository Structure

```
dataco-supply-chain-analysis/
├── app.py                        ← Streamlit entry point (landing page)
├── pages/
│   ├── 1_📊_Overview.py           ← KPI dashboard: revenue, margin, on-time rate, filters
│   ├── 2_🔍_EDA.py                ← Interactive Plotly EDA (trends, delays, correlations)
│   ├── 3_🚚_Late_Delivery.py      ← Live prediction form + model comparison + feature importance
│   └── 4_🚨_Fraud_Detection.py    ← Live prediction form + model comparison + feature importance
│
├── models/
│   ├── late_delivery_model.pkl   ← Best model (Gradient Boosting) + encoders + feature list
│   ├── fraud_model.pkl           ← Best model (Logistic Regression) + encoders + feature list
│   └── metrics.json              ← Accuracy/AUC/precision/recall/ROC/confusion matrix for all 3 models per task
│
├── data/
│   ├── processed_supply_chain.csv   ← Cleaned + sampled dataset used by the app (~20MB, 60K rows)
│   └── raw/                         ← Put the full Kaggle CSV here for local re-training (gitignored)
│
├── notebooks/
│   ├── full_analysis.ipynb          ← ⭐ All 3 original notebooks merged into one, run top-to-bottom
│   ├── 01_eda_data_cleaning.ipynb   ← Original, kept for reference
│   ├── 02_sql_kpi_dashboard.ipynb   ← Original, kept for reference
│   └── 03_predictive_modeling.ipynb ← Original, kept for reference
│
├── src/
│   ├── data_cleaning.py          ← Reusable cleaning pipeline
│   ├── feature_engineering.py    ← Feature creation for both ML tasks
│   ├── visualization.py          ← Original matplotlib/seaborn plotting functions
│   └── app_utils.py              ← Shared Streamlit helpers: caching, theming, model I/O
│
├── sql/
│   └── supply_chain_queries.sql  ← 7 KPI queries used in the SQL dashboard notebook section
│
├── reports/figures/              ← Exported confusion matrices, ROC curves, feature importance charts
├── train_and_export.py           ← Re-run this to retrain models from raw data and refresh models/*.pkl
├── merge_notebooks.py            ← Script used to produce notebooks/full_analysis.ipynb
├── .streamlit/config.toml        ← App theme (dark, amber/teal/coral accents)
├── requirements.txt
└── README.md
```

---

## 🚀 Run locally

```bash
git clone https://github.com/YOUR_USERNAME/dataco-supply-chain-analysis.git
cd dataco-supply-chain-analysis

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

The app reads `data/processed_supply_chain.csv` and `models/*.pkl`, both of
which are already committed — **no training step is required** to run the app.

---

## 🔁 Retraining the models (optional)

Only needed if you change the feature engineering or want to retrain on the
full dataset:

1. Download the dataset from [Kaggle — DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
   and place it at `data/raw/DataCoSupplyChainDataset.csv`.
2. Run:
   ```bash
   python train_and_export.py
   ```
   This regenerates `models/late_delivery_model.pkl`, `models/fraud_model.pkl`,
   `models/metrics.json`, and refreshes `data/processed_supply_chain.csv`.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (make sure `models/*.pkl` and
   `data/processed_supply_chain.csv` are committed — check `.gitignore`,
   they're intentionally **not** ignored).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point it at your repo, branch `main`, main file path **`app.py`**.
4. Deploy. Streamlit installs `requirements.txt` and the app is live — no
   secrets or environment variables needed for this project.

**Repo size check:** the largest tracked file is the sampled CSV (~20MB) —
comfortably under Streamlit Cloud's limits and GitHub's 100MB hard cap. The
full raw ~92MB Kaggle CSV is intentionally **not** committed.

---

## 📊 Models

| Task | Best model | Key metric |
|---|---|---|
| Late Delivery Prediction | Gradient Boosting | ~71% accuracy, 0.77 AUC |
| Fraud Detection | Logistic Regression | 0.97 AUC (high recall, low precision — fraud is rare) |

Full comparisons (all 3 algorithms per task, confusion matrices, ROC curves)
are in `models/metrics.json` and rendered live on the app's prediction pages.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data manipulation |
| Plotly | Interactive charts (app) |
| Matplotlib / Seaborn | Static charts (notebooks) |
| Scikit-learn | Machine learning |
| Streamlit | App framework & deployment |
| SQLite | SQL-based KPI analysis |

---

## 📜 License

MIT — see [LICENSE](LICENSE).
