from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import xgboost as xgb  # noqa: F401 - required when unpickling the XGBoost model
from tensorflow.keras.models import model_from_json


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "hybrid_lstm_xgb_model.pkl"
TIMESTAMP_COL = "timestamp"
FORECAST_COL = "Forecasted Average Power (kW)"


st.set_page_config(
    page_title="EV Average Power Forecast",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the saved hybrid model...")
def load_hybrid_model(model_path: str) -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path}")

    # Pickle files can execute code while loading. Only load this trusted local bundle.
    with path.open("rb") as file:
        bundle = pickle.load(file)

    required_keys = {
        "lstm_json",
        "lstm_weights",
        "xgb_corrector",
        "scaler_X",
        "scaler_y",
        "seq_len",
        "feature_cols",
        "target_col",
    }
    missing = required_keys.difference(bundle)
    if missing:
        raise KeyError(f"The model bundle is missing: {sorted(missing)}")

    lstm_model = model_from_json(bundle["lstm_json"])
    lstm_model.set_weights(bundle["lstm_weights"])

    return {
        **bundle,
        "lstm_model": lstm_model,
        "seq_len": int(bundle["seq_len"]),
        "feature_cols": list(bundle["feature_cols"]),
    }


def read_uploaded_csv(uploaded_file, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"{label} could not be read as a CSV: {exc}") from exc

    if frame.empty:
        raise ValueError(f"{label} is empty.")
    return frame


def prepare_features(
    lookback_df: pd.DataFrame,
    next_day_df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    missing_lookback = [c for c in feature_cols if c not in lookback_df.columns]
    missing_next_day = [c for c in feature_cols if c not in next_day_df.columns]

    if missing_lookback or missing_next_day:
        messages = []
        if missing_lookback:
            messages.append(
                "Missing from lookback CSV: " + ", ".join(missing_lookback)
            )
        if missing_next_day:
            messages.append(
                "Missing from next-day CSV: " + ", ".join(missing_next_day)
            )
        raise ValueError(" | ".join(messages))

    if len(lookback_df) < sequence_length:
        raise ValueError(
            f"The lookback CSV needs at least {sequence_length} rows; "
            f"it contains {len(lookback_df)}."
        )

    history = (
        lookback_df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .tail(sequence_length)
    )
    future = next_day_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    history_bad = history.columns[history.isna().any()].tolist()
    future_bad = future.columns[future.isna().any()].tolist()
    if history_bad or future_bad:
        messages = []
        if history_bad:
            messages.append(
                "Lookback columns containing missing/non-numeric values: "
                + ", ".join(history_bad)
            )
        if future_bad:
            messages.append(
                "Next-day columns containing missing/non-numeric values: "
                + ", ".join(future_bad)
            )
        raise ValueError(" | ".join(messages))

    return history.to_numpy(dtype=float), future.to_numpy(dtype=float)


def forecast(
    bundle: dict,
    history_values: np.ndarray,
    future_values: np.ndarray,
) -> np.ndarray:
    scaler_x = bundle["scaler_X"]
    scaler_y = bundle["scaler_y"]
    sequence_length = bundle["seq_len"]

    history_scaled = scaler_x.transform(history_values)
    future_scaled = scaler_x.transform(future_values)
    combined_scaled = np.vstack((history_scaled, future_scaled))

    sequences = np.stack(
        [
            combined_scaled[index : index + sequence_length]
            for index in range(len(future_scaled))
        ]
    )

    lstm_scaled = (
        bundle["lstm_model"].predict(sequences, verbose=0).reshape(-1)
    )
    correction_scaled = (
        bundle["xgb_corrector"].predict(future_scaled).reshape(-1)
    )
    hybrid_scaled = lstm_scaled + correction_scaled

    return scaler_y.inverse_transform(hybrid_scaled.reshape(-1, 1)).reshape(-1)


def build_output(next_day_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    if TIMESTAMP_COL in next_day_df.columns:
        timestamps = next_day_df[TIMESTAMP_COL].copy()
    else:
        timestamps = pd.Series(
            np.arange(1, len(next_day_df) + 1), name=TIMESTAMP_COL
        )

    return pd.DataFrame(
        {
            TIMESTAMP_COL: timestamps,
            FORECAST_COL: predictions,
        }
    )


st.title("⚡ EV Average Power Forecast")
st.caption(
    "Upload a lookback CSV and a next-day CSV to forecast Average Power "
    "with the saved LSTM + XGBoost model. The model is loaded without retraining."
)

try:
    model_bundle = load_hybrid_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

with st.sidebar:
    st.header("Model information")
    st.metric("Required lookback rows", model_bundle["seq_len"])
    st.metric("Expected features", len(model_bundle["feature_cols"]))
    st.write(f"Target: `{model_bundle['target_col']}`")
    st.info(
        "The uploaded files must contain the same engineered feature columns "
        "used to train the saved model."
    )

left, right = st.columns(2)
with left:
    lookback_upload = st.file_uploader(
        "Upload lookback CSV",
        type=["csv"],
        key="lookback_csv",
        help=(
            f"Must contain at least {model_bundle['seq_len']} rows and all "
            "saved-model feature columns."
        ),
    )

with right:
    next_day_upload = st.file_uploader(
        "Upload next-day CSV",
        type=["csv"],
        key="next_day_csv",
        help="Each row in this file produces one forecast value.",
    )

if lookback_upload is not None and next_day_upload is not None:
    upload_signature = (
        lookback_upload.name,
        lookback_upload.size,
        next_day_upload.name,
        next_day_upload.size,
    )
    if st.session_state.get("forecast_signature") != upload_signature:
        st.session_state.pop("forecast_result", None)

    try:
        lookback_data = read_uploaded_csv(lookback_upload, "Lookback CSV")
        next_day_data = read_uploaded_csv(next_day_upload, "Next-day CSV")

        summary_left, summary_right = st.columns(2)
        summary_left.success(
            f"Lookback CSV loaded: {len(lookback_data):,} rows"
        )
        summary_right.success(
            f"Next-day CSV loaded: {len(next_day_data):,} rows"
        )

        with st.expander("Preview uploaded data"):
            preview_left, preview_right = st.columns(2)
            with preview_left:
                st.write("Lookback CSV")
                st.dataframe(lookback_data.head(), use_container_width=True)
            with preview_right:
                st.write("Next-day CSV")
                st.dataframe(next_day_data.head(), use_container_width=True)

        if st.button("Generate forecast", type="primary", use_container_width=True):
            with st.spinner("Generating forecast..."):
                history_array, future_array = prepare_features(
                    lookback_data,
                    next_day_data,
                    model_bundle["feature_cols"],
                    model_bundle["seq_len"],
                )
                forecast_values = forecast(
                    model_bundle, history_array, future_array
                )
                result = build_output(next_day_data, forecast_values)

            st.session_state["forecast_result"] = result
            st.session_state["forecast_signature"] = upload_signature

    except Exception as exc:
        st.error(str(exc))

if "forecast_result" in st.session_state:
    result = st.session_state["forecast_result"]

    st.divider()
    st.subheader("Forecast results")

    minimum_index = result[FORECAST_COL].idxmin()
    maximum_index = result[FORECAST_COL].idxmax()

    def format_occurrence(index: int) -> str:
        value = result.loc[index, TIMESTAMP_COL]
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d %H:%M")
        if pd.notna(value):
            return str(value)
        return f"Forecast step {index + 1}"

    minimum_time = format_occurrence(minimum_index)
    maximum_time = format_occurrence(maximum_index)

    metric_points, metric_average = st.columns(2)
    metric_points.metric("Forecast points", f"{len(result):,}")
    metric_average.metric(
        "Average forecast", f"{result[FORECAST_COL].mean():,.2f} kW"
    )

    metric_minimum, metric_maximum = st.columns(2)
    metric_minimum.metric(
        "Minimum demand",
        f"{result.loc[minimum_index, FORECAST_COL]:,.2f} kW",
        help=f"Occurred at {minimum_time}",
    )
    metric_minimum.caption(f"Occurred at: {minimum_time}")
    metric_maximum.metric(
        "Maximum demand",
        f"{result.loc[maximum_index, FORECAST_COL]:,.2f} kW",
        help=f"Occurred at {maximum_time}",
    )
    metric_maximum.caption(f"Occurred at: {maximum_time}")

    chart_data = result.copy()
    parsed_time = pd.to_datetime(chart_data[TIMESTAMP_COL], errors="coerce")
    if parsed_time.notna().all():
        chart_data[TIMESTAMP_COL] = parsed_time
        chart_data = chart_data.set_index(TIMESTAMP_COL)
    else:
        chart_data = chart_data.set_index(
            pd.Index(np.arange(1, len(chart_data) + 1), name="Forecast step")
        )

    st.line_chart(
        chart_data[[FORECAST_COL]],
        y=FORECAST_COL,
        color="#7C3AED",
        use_container_width=True,
    )

    st.dataframe(
        result.style.format({FORECAST_COL: "{:,.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forecast CSV",
        data=csv_bytes,
        file_name="average_power_forecast.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
