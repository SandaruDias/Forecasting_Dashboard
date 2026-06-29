# Streamlit forecast dashboard

The dashboard loads `hybrid_lstm_xgb_model.pkl` and generates forecasts without
retraining.

## Run

Use Python 3.11, then install the dependencies:

```powershell
python -m pip install -r requirements.txt

```

Upload:

1. A lookback CSV containing at least 288 rows.
2. A next-day CSV containing one row per required forecast point.

Both files must contain the engineered feature columns expected by the saved
model. The generated CSV contains `timestamp` and
`Forecasted Average Power (kW)`.
python -m streamlit run streamlit_app_final.py