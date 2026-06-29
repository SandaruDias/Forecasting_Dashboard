from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb  # noqa: F401 - required when unpickling the XGBoost model
import cvxpy as cp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    # Keras 3 / TF 2.16+: standalone keras package resolves cleanly for type checkers
    from keras.models import model_from_json  # type: ignore[import-untyped]
except ModuleNotFoundError:
    # Older TF (< 2.16): fall back to the tf.keras namespace at runtime
    from tensorflow.keras.models import model_from_json  # type: ignore[no-redef]


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "hybrid_lstm_xgb_model_validation.pkl"
TIMESTAMP_COL = "timestamp"
FORECAST_COL = "Forecasted Average Power (kW)"

# ─────────────────────────────────────────────────────────────────────────────
# BESS & Tariff constants (from notebook Cell 4)
# ─────────────────────────────────────────────────────────────────────────────
DELTA_T   = 0.25
N_STEPS   = 96
E_BESS    = 220
P_CH_MAX  = E_BESS * 0.25
P_DIS_MAX = E_BESS * 0.25
ETA_CH    = 0.95
ETA_DIS   = 0.95
SOC_MIN   = 0.20 * E_BESS
SOC_MAX   = 0.90 * E_BESS
SOC_INIT  = 0.20 * E_BESS
SOC_FINAL = 0.20 * E_BESS
C_DEG           = 0.05
LAM_DEMAND      = 4.81
LAM_DEMAND_DAILY= LAM_DEMAND / 30.0
C_BESS          = 200.0
C_BESS_TOTAL    = C_BESS * E_BESS
DISCOUNT_RATE   = 0.10
N_YEARS         = 10
P_GRID_MAX      = 240.0
CRF             = DISCOUNT_RATE * (1 + DISCOUNT_RATE)**N_YEARS / ((1 + DISCOUNT_RATE)**N_YEARS - 1)
BESS_ANNUAL_COST= C_BESS_TOTAL * CRF
BESS_DAILY_COST = BESS_ANNUAL_COST / 365.0

PARAMS = {
    'E_bess'          : E_BESS,
    'P_ch_max'        : P_CH_MAX,
    'P_dis_max'       : P_DIS_MAX,
    'eta_ch'          : ETA_CH,
    'eta_dis'         : ETA_DIS,
    'soc_min'         : SOC_MIN,
    'soc_max'         : SOC_MAX,
    'soc_final'       : SOC_FINAL,
    'c_deg'           : C_DEG,
    'lam_demand_daily': LAM_DEMAND_DAILY,
    'bess_daily_cost' : BESS_DAILY_COST,
    'delta_t'         : DELTA_T,
}


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Day-Ahead EV Charging Hub Forecast & Two Layer Peak Shaving Optimization",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid;
        margin-bottom: 8px;
    }
    .metric-card.blue  { border-color: #4f8ef7; }
    .metric-card.green { border-color: #2ecc71; }
    .metric-card.red   { border-color: #e74c3c; }
    .metric-card.purple{ border-color: #9b59b6; }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        padding-bottom: 4px;
        border-bottom: 2px solid #333;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper — TOU masks from a 96-step hours_axis
# ─────────────────────────────────────────────────────────────────────────────
def build_tou_masks(n: int, delta_t: float = DELTA_T):
    h = np.arange(n) * delta_t
    mask_peak    = ((h >= 18) & (h < 22)) 
    mask_offpeak = (h >= 0) & (h < 8) | ((h >= 22) & (h < 24))
    mask_day     = ~mask_peak & ~mask_offpeak
    tariff = np.where(mask_peak, 0.220, 0.048)
    return h, mask_peak, mask_offpeak, mask_day, tariff


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 helpers (unchanged from original app)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading the saved hybrid model...")
def load_hybrid_model(model_path: str) -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path}")
    with path.open("rb") as file:
        bundle = pickle.load(file)
    required_keys = {
        "lstm_json", "lstm_weights", "xgb_corrector",
        "scaler_X", "scaler_y", "seq_len", "feature_cols", "target_col",
    }
    missing = required_keys.difference(bundle)
    if missing:
        raise KeyError(f"The model bundle is missing: {sorted(missing)}")
    lstm_model = model_from_json(bundle["lstm_json"])
    lstm_model.set_weights(bundle["lstm_weights"])
    return {**bundle, "lstm_model": lstm_model,
            "seq_len": int(bundle["seq_len"]),
            "feature_cols": list(bundle["feature_cols"])}


def read_uploaded_csv(uploaded_file, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"{label} could not be read as a CSV: {exc}") from exc
    if frame.empty:
        raise ValueError(f"{label} is empty.")
    return frame


def prepare_features(lookback_df, next_day_df, feature_cols, sequence_length):
    missing_lookback = [c for c in feature_cols if c not in lookback_df.columns]
    missing_next_day = [c for c in feature_cols if c not in next_day_df.columns]
    if missing_lookback or missing_next_day:
        messages = []
        if missing_lookback:
            messages.append("Missing from lookback CSV: " + ", ".join(missing_lookback))
        if missing_next_day:
            messages.append("Missing from next-day CSV: " + ", ".join(missing_next_day))
        raise ValueError(" | ".join(messages))
    if len(lookback_df) < sequence_length:
        raise ValueError(
            f"The lookback CSV needs at least {sequence_length} rows; "
            f"it contains {len(lookback_df)}.")
    history = (lookback_df[feature_cols].apply(pd.to_numeric, errors="coerce")
               .tail(sequence_length))
    future  = next_day_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    history_bad = history.columns[history.isna().any()].tolist()
    future_bad  = future.columns[future.isna().any()].tolist()
    if history_bad or future_bad:
        messages = []
        if history_bad:
            messages.append("Lookback columns with missing/non-numeric values: " + ", ".join(history_bad))
        if future_bad:
            messages.append("Next-day columns with missing/non-numeric values: " + ", ".join(future_bad))
        raise ValueError(" | ".join(messages))
    return history.to_numpy(dtype=float), future.to_numpy(dtype=float)


def forecast(bundle, history_values, future_values):
    scaler_x = bundle["scaler_X"]
    scaler_y = bundle["scaler_y"]
    sequence_length = bundle["seq_len"]
    history_scaled  = scaler_x.transform(history_values)
    future_scaled   = scaler_x.transform(future_values)
    combined_scaled = np.vstack((history_scaled, future_scaled))
    sequences = np.stack(
        [combined_scaled[i: i + sequence_length] for i in range(len(future_scaled))])
    lstm_scaled       = bundle["lstm_model"].predict(sequences, verbose=0).reshape(-1)
    correction_scaled = bundle["xgb_corrector"].predict(future_scaled).reshape(-1)
    hybrid_scaled     = lstm_scaled + correction_scaled
    return scaler_y.inverse_transform(hybrid_scaled.reshape(-1, 1)).reshape(-1)


def build_output(next_day_df, predictions):
    if TIMESTAMP_COL in next_day_df.columns:
        timestamps = next_day_df[TIMESTAMP_COL].copy()
    else:
        timestamps = pd.Series(np.arange(1, len(next_day_df) + 1), name=TIMESTAMP_COL)
    return pd.DataFrame({TIMESTAMP_COL: timestamps, FORECAST_COL: predictions})


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 helpers — Peak Shaving Optimisation
# ─────────────────────────────────────────────────────────────────────────────
def calculate_baseline(P_demand, tariff_arr, lam_demand_daily, delta_t, mask_pk):
    J_energy        = float(np.sum(tariff_arr * P_demand) * delta_t)
    peak_demand_max = float(P_demand.max())
    J_demand        = lam_demand_daily * peak_demand_max
    return {
        'J_energy'        : J_energy,
        'J_demand'        : J_demand,
        'total_cost'      : J_energy + J_demand,
        'peak_max'        : peak_demand_max,
        'total_energy_kWh': float(np.sum(P_demand) * delta_t),
    }


def solve_peak_shaving(P_demand, tariff_arr, mask_pk, mask_op, mask_day_,
                        soc_init, params, verbose=False):
    N          = len(P_demand)
    dt         = params['delta_t']
    P_ch_max   = params['P_ch_max']
    P_dis_max  = params['P_dis_max']
    eta_ch     = params['eta_ch']
    eta_dis    = params['eta_dis']
    soc_min    = params['soc_min']
    soc_max    = params['soc_max']
    soc_final  = params['soc_final']
    c_deg      = params['c_deg']
    lam_dem    = params['lam_demand_daily']
    bess_daily = params['bess_daily_cost']

    P_ch   = cp.Variable(N, nonneg=True)
    P_dis  = cp.Variable(N, nonneg=True)
    SoC    = cp.Variable(N + 1)
    P_grid = cp.Variable(N, nonneg=True)
    T      = cp.Variable(nonneg=True)

    J_energy = cp.sum(cp.multiply(tariff_arr, P_grid)) * dt
    J_demand = lam_dem * T
    J_deg    = c_deg * cp.sum(P_dis) * dt
    J_bess   = bess_daily

    objective   = cp.Minimize(J_energy + J_demand + J_deg + J_bess)
    constraints = []
    constraints += [P_grid == P_demand - P_dis / eta_dis + P_ch * eta_ch]
    constraints += [P_grid >= 0, P_grid <= P_GRID_MAX]
    constraints += [P_grid <= T]
    constraints += [P_dis[mask_op] == 0]
    constraints += [P_ch[~mask_op]   == 0]
    constraints += [SoC[0] == soc_init]
    for t in range(N):
        constraints += [SoC[t + 1] == SoC[t] + (P_ch[t] * eta_ch - P_dis[t] / eta_dis) * dt]
    constraints += [SoC >= soc_min, SoC <= soc_max]
    constraints += [P_ch  <= P_ch_max]
    constraints += [P_dis <= P_dis_max]
    constraints += [P_ch + P_dis <= max(P_ch_max, P_dis_max)]
    constraints += [SoC[N] == soc_final]
    constraints += [SoC[32]==SOC_MAX]
    constraints += [T >= 50, T <= float(P_demand.max())]
    for t in range(1, N):
        constraints += [(P_dis[t] - P_dis[t-1]) <=  0.1 * P_dis_max]
        constraints += [(P_dis[t-1] - P_dis[t]) <=  0.1 * P_dis_max]
        constraints += [(P_ch[t]  - P_ch[t-1])  <=  0.1 * P_ch_max]
        constraints += [(P_ch[t-1] - P_ch[t])   <=  0.1 * P_ch_max]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.HIGHS, verbose=verbose)

    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f'Optimisation failed: {problem.status}')

    return {
        'status'      : problem.status,
        'threshold_kW': float(T.value),
        'P_grid'      : P_grid.value,
        'P_ch'        : P_ch.value,
        'P_dis'       : P_dis.value,
        'SoC'         : SoC.value,
        'J_energy'    : float(J_energy.value),
        'J_demand'    : float(J_demand.value),
        'J_deg'       : float(J_deg.value),
        'J_bess'      : float(bess_daily),
        'total_cost'  : float(problem.value),
    }


def run_sensitivity(P_demand, tariff_arr, mask_pk, mask_op, mask_dy,
                    bess_sizes, delta_t=DELTA_T):
    rows = []
    for sz in bess_sizes:
        p_ch_max  = sz * 0.25
        p_dis_max = sz * 0.25
        soc_min   = 0.20 * sz
        soc_max   = 0.90 * sz
        soc_init  = 0.60 * sz
        bd        = C_BESS * sz * CRF / 365.0
        N         = len(P_demand)

        P_ch_v   = cp.Variable(N, nonneg=True)
        P_dis_v  = cp.Variable(N, nonneg=True)
        SoC_v    = cp.Variable(N + 1)
        P_grid_v = cp.Variable(N, nonneg=True)
        T_v      = cp.Variable(nonneg=True)

        J_e  = cp.sum(cp.multiply(tariff_arr, P_grid_v)) * delta_t
        J_d  = LAM_DEMAND_DAILY * T_v
        J_dg = C_DEG * cp.sum(P_dis_v) * delta_t
        obj  = cp.Minimize(J_e + J_d + J_dg + bd)

        cons = [
            P_grid_v == P_demand - P_dis_v / ETA_DIS + P_ch_v * ETA_CH,
            P_grid_v >= 0, P_grid_v[mask_pk] <= T_v,
            P_dis_v[~mask_pk] == 0, P_ch_v[mask_pk] == 0,
            SoC_v[0] == soc_init, SoC_v >= soc_min, SoC_v <= soc_max,
            P_ch_v <= p_ch_max, P_dis_v <= p_dis_max,
            P_ch_v + P_dis_v <= max(p_ch_max, p_dis_max),
            SoC_v[N] == soc_init, T_v >= 500, T_v <= float(P_demand.max()),
        ]
        for t in range(N):
            cons.append(SoC_v[t+1] == SoC_v[t] + (P_ch_v[t]*ETA_CH - P_dis_v[t]/ETA_DIS)*delta_t)

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.HIGHS, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            rows.append({'BESS Capacity (kWh)': sz,
                         'Optimal Threshold (kW)': float(T_v.value),
                         'Total Daily Cost ($)': float(prob.value),
                         'Demand Charge ($)': float(J_d.value)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart builders
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'demand'    : '#f0f0f0',
    'grid'      : '#4f8ef7',
    'discharge' : '#2ecc71',
    'charge'    : '#f39c12',
    'threshold' : '#e74c3c',
    'shaved'    : 'rgba(231,76,60,0.25)',
    'soc'       : '#9b59b6',
    'baseline'  : '#e74c3c',
    'bess'      : '#4f8ef7',
    'offpeak_bg': 'rgba(46,204,113,0.07)',
    'peak_bg'   : 'rgba(231,76,60,0.07)',
    'day_bg'    : 'rgba(243,156,18,0.07)',
}


def _window_shapes(hours_axis, y_max, mask_peak, mask_offpeak, mask_day):
    """Return a list of Plotly shape dicts for TOU window shading."""
    shapes = []

    def spans(mask):
        """Convert a boolean mask over hours_axis to list of (start, end) hour pairs."""
        segs = []
        in_seg = False
        for i, v in enumerate(mask):
            if v and not in_seg:
                start = hours_axis[i]; in_seg = True
            elif not v and in_seg:
                segs.append((start, hours_axis[i])); in_seg = False
        if in_seg:
            segs.append((start, hours_axis[-1] + DELTA_T))
        return segs

    for color, mask in [
        (COLORS['offpeak_bg'], mask_offpeak),
        (COLORS['peak_bg'],    mask_peak),
        (COLORS['day_bg'],     mask_day),
    ]:
        for s, e in spans(mask):
            shapes.append(dict(
                type='rect', xref='x', yref='y',
                x0=s, x1=e, y0=0, y1=y_max,
                fillcolor=color, line_width=0, layer='below',
            ))
    return shapes


def plot_power_flows(hours_axis, P_dem, P_grid, P_dis, P_ch, T_opt,
                     mask_peak, mask_offpeak, mask_day):
    y_max = float(P_dem.max()) * 1.18
    fig = go.Figure()

    # Shaded region BESS fills
    shaved = np.where((P_dem > T_opt) & mask_peak, P_dem - T_opt, 0)
    fig.add_trace(go.Scatter(
        x=np.concatenate([hours_axis, hours_axis[::-1]]),
        y=np.concatenate([np.where((P_dem > T_opt) & mask_peak, P_dem, T_opt),
                          np.full(len(hours_axis), T_opt)[::-1]]),
        fill='toself', fillcolor=COLORS['shaved'],
        line=dict(width=0), name='Demand filled by BESS', hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(x=hours_axis, y=P_dem,  mode='lines',
        line=dict(color=COLORS['demand'], width=2.5), name='Forecasted Demand (kW)'))
    fig.add_trace(go.Scatter(x=hours_axis, y=P_grid, mode='lines',
        line=dict(color=COLORS['grid'], width=2), name='Grid Import (kW)'))
    fig.add_trace(go.Scatter(x=hours_axis, y=P_dis,  mode='lines',
        line=dict(color=COLORS['discharge'], width=1.8, dash='dash'), name='BESS Discharge (kW)'))
    fig.add_trace(go.Scatter(x=hours_axis, y=P_ch,   mode='lines',
        line=dict(color=COLORS['charge'], width=1.8, dash='dot'), name='BESS Charge (kW)'))
    fig.add_hline(y=T_opt, line=dict(color=COLORS['threshold'], width=2.5, dash='dash'),
                  annotation_text=f"Optimal Threshold = {T_opt:.1f} kW",
                  annotation_font_color=COLORS['threshold'])

    fig.update_layout(
        shapes=_window_shapes(hours_axis, y_max, mask_peak, mask_offpeak, mask_day),
        xaxis=dict(title='Time of Day', tickvals=list(range(0, 25, 2)),
                   ticktext=[f'{h:02d}:00' for h in range(0, 25, 2)]),
        yaxis=dict(title='Power (kW)', range=[0, y_max]),
        title=dict(text=f'Power Flows & Optimal Peak Shaving Threshold  [T = {T_opt:.1f} kW]',
                   font=dict(size=16, color='white')),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_dark', height=420, margin=dict(t=80, b=60),
    )
    return fig


def plot_soc(hours_axis, SoC, E_bess=E_BESS, soc_min=SOC_MIN, soc_max_val=SOC_MAX,
             soc_init=SOC_INIT, mask_peak=None, mask_offpeak=None, mask_day=None):
    soc_time = np.arange(len(SoC)) * DELTA_T
    SoC_pct  = SoC / E_bess * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=soc_time, y=SoC_pct, mode='lines',
        line=dict(color=COLORS['soc'], width=2.5), name='SoC (%)'))
    fig.add_hline(y=20, line=dict(color='red',   width=1.5, dash='dot'),
                  annotation_text='Min 20%', annotation_font_color='red')
    fig.add_hline(y=90, line=dict(color='green', width=1.5, dash='dot'),
                  annotation_text='Max 90%', annotation_font_color='green')
    fig.add_hline(y=60, line=dict(color='grey',  width=1.2, dash='dashdot'),
                  annotation_text='Init/Final 60%', annotation_font_color='grey')
    if mask_peak is not None:
        fig.update_layout(
            shapes=_window_shapes(hours_axis, 105, mask_peak, mask_offpeak, mask_day))
    fig.update_layout(
        xaxis=dict(title='Time (hours)', tickvals=list(range(0, 25, 2)),
                   ticktext=[f'{h:02d}:00' for h in range(0, 25, 2)]),
        yaxis=dict(title='State of Charge (%)', range=[0, 105]),
        title=dict(text='Battery State of Charge Profile', font=dict(size=15, color='white')),
        template='plotly_dark', height=340, margin=dict(t=60, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def plot_bess_schedule(hours_axis, P_dis, P_ch, mask_peak, mask_offpeak, mask_day):
    y_max = float(P_DIS_MAX) * 1.2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours_axis, y=P_dis, fill='tozeroy',
        fillcolor='rgba(231,76,60,0.35)', line=dict(color=COLORS['threshold'], width=1.5),
        name='BESS Discharge (kW)'))
    fig.add_trace(go.Scatter(x=hours_axis, y=P_ch, fill='tozeroy',
        fillcolor='rgba(79,142,247,0.35)', line=dict(color=COLORS['grid'], width=1.5),
        name='BESS Charge (kW)'))
    fig.add_hline(y=P_DIS_MAX, line=dict(color='red',   width=1, dash='dot'),
                  annotation_text=f'Max Discharge {P_DIS_MAX:.0f} kW', annotation_font_color='red')
    fig.add_hline(y=P_CH_MAX,  line=dict(color='blue',  width=1, dash='dot'),
                  annotation_text=f'Max Charge {P_CH_MAX:.0f} kW', annotation_font_color='blue')
    fig.update_layout(
        shapes=_window_shapes(hours_axis, y_max, mask_peak, mask_offpeak, mask_day),
        xaxis=dict(title='Time of Day', tickvals=list(range(0, 25, 2)),
                   ticktext=[f'{h:02d}:00' for h in range(0, 25, 2)]),
        yaxis=dict(title='Power (kW)', range=[0, y_max]),
        title=dict(text='BESS Charge & Discharge Schedule', font=dict(size=15, color='white')),
        template='plotly_dark', height=340, margin=dict(t=60, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def plot_cost_comparison(baseline, result):
    categories = ['Energy Cost', 'Demand Charge', 'Total Cost']
    base_vals  = [baseline['J_energy'], baseline['J_demand'], baseline['total_cost']]
    bess_vals  = [result['J_energy'],   result['J_demand'],
                  result['J_energy'] + result['J_demand']]
    savings = baseline['total_cost'] - (result['J_energy'] + result['J_demand']+result['J_deg'] + result['J_bess'])

    fig = make_subplots(rows=1, cols=2,specs=[[{"type": "xy"}, {"type": "domain"}]], subplot_titles=['Baseline vs With BESS', 'Total Cost Breakdown (With BESS)'])

    fig.add_trace(go.Bar(name='Without BESS', x=categories, y=base_vals,
        marker_color='rgba(231,76,60,0.8)',
        text=[f'${v:,.0f}' for v in base_vals], textposition='outside'), row=1, col=1)
    fig.add_trace(go.Bar(name='With BESS', x=categories, y=bess_vals,
        marker_color='rgba(79,142,247,0.8)',
        text=[f'${v:,.0f}' for v in bess_vals], textposition='outside'), row=1, col=1)

    pie_labels = ['Energy Cost', 'Demand Charge', 'Degradation', 'BESS Capital']
    pie_vals   = [result['J_energy'], result['J_demand'], result['J_deg'], result['J_bess']]
    fig.add_trace(go.Pie(labels=pie_labels, values=pie_vals, hole=0.4,
        marker=dict(colors=['#4f8ef7', '#e74c3c', '#f39c12', '#9b59b6'])), row=1, col=2)

    fig.add_annotation(
        text=f"Daily Savings: ${savings:,.2f}", showarrow=False,
        x=2, y=max(base_vals) * 1.05, xref='x', yref='y',
        font=dict(color='#2ecc71', size=13), row=1, col=1,
    )
    fig.update_layout(template='plotly_dark', height=380,
                      margin=dict(t=80, b=40), barmode='group',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    return fig


def plot_sensitivity(sens_df, baseline_cost):
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Optimal Threshold vs BESS Size', 'Total Daily Cost vs BESS Size'])
    fig.add_trace(go.Scatter(
        x=sens_df['BESS Capacity (kWh)'], y=sens_df['Optimal Threshold (kW)'],
        mode='lines+markers', line=dict(color=COLORS['threshold'], width=2),
        marker=dict(size=8), name='Optimal Threshold'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sens_df['BESS Capacity (kWh)'], y=sens_df['Total Daily Cost ($)'],
        mode='lines+markers', line=dict(color=COLORS['grid'], width=2),
        marker=dict(size=8), name='Total Cost (incl. BESS capital)'), row=1, col=2)
    fig.add_hline(y=baseline_cost, line=dict(color=COLORS['baseline'], width=1.5, dash='dash'),
                  annotation_text='Baseline (no BESS)', annotation_font_color=COLORS['baseline'],
                  row=1, col=2)
    fig.update_xaxes(title_text='BESS Capacity (kWh)')
    fig.update_yaxes(title_text='Threshold (kW)', row=1, col=1)
    fig.update_yaxes(title_text='Cost ($)', row=1, col=2)
    fig.update_layout(template='plotly_dark', height=360,
                      margin=dict(t=70, b=40), showlegend=False)
    return fig


def plot_peak_shaving_bars(hours_axis, P_dem, P_grid, T_opt, mask_peak, mask_offpeak, mask_day):
    bar_w     = DELTA_T * 0.9
    shaved_m  = (P_dem > T_opt) & mask_peak
    bar_colors = np.where(mask_peak, '#4f8ef7',
                 np.where(mask_offpeak, '#2ecc71', '#f39c12'))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hours_axis, y=P_dem, width=bar_w,
        marker_color='rgba(200,200,200,0.25)', name='Total Demand', offset=0))
    fig.add_trace(go.Bar(x=hours_axis, y=P_grid, width=bar_w,
        marker_color=bar_colors, opacity=0.85, name='Grid Import', offset=0))
    if shaved_m.any():
        fig.add_trace(go.Bar(
            x=hours_axis[shaved_m], y=(P_dem - T_opt)[shaved_m],
            base=np.full(shaved_m.sum(), T_opt),
            width=bar_w, marker_color='rgba(231,76,60,0.75)',
            name='Peak Shaved by BESS', offset=0))
    fig.add_hline(y=T_opt, line=dict(color=COLORS['threshold'], width=2.5, dash='dash'),
                  annotation_text=f'Threshold = {T_opt:.1f} kW',
                  annotation_font_color=COLORS['threshold'])
    fig.update_layout(
        shapes=_window_shapes(hours_axis, float(P_dem.max()) * 1.18, mask_peak, mask_offpeak, mask_day),
        xaxis=dict(title='Time of Day', tickvals=list(range(0, 25, 2)),
                   ticktext=[f'{h:02d}:00' for h in range(0, 25, 2)]),
        yaxis=dict(title='Power (kW)', range=[0, float(P_dem.max()) * 1.18]),
        title=dict(text=f'Peak Shaving Bar Chart  [{int(shaved_m.sum())} slots shaved]',
                   font=dict(size=15, color='white')),
        template='plotly_dark', barmode='overlay', height=360,
        margin=dict(t=60, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚡ Day-Ahead EV Charging Hub Forecast & Two Layer Peak Shaving Optimization")
st.caption(
    "Forecast: LSTM + XGBoost hybrid model forecasts average power demand for next 24-hours. "
    "Layer 2: Convex optimization determines the optimal static peak-shaving threshold."
)

# ─────────────────────────────────────────────────────────────────────────────
# Load LSTM+XGB model
# ─────────────────────────────────────────────────────────────────────────────
try:
    model_bundle = load_hybrid_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — model info + BESS parameters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Forecast Model")
    st.metric("Required lookback rows", model_bundle["seq_len"])
    st.metric("Expected features",      len(model_bundle["feature_cols"]))
    st.write(f"Target: `{model_bundle['target_col']}`")

    st.divider()
    st.subheader("BESS Parameters")
    E_bess_ui      = st.number_input("BESS Capacity (kWh)",    value=float(E_BESS),    step=500.0)
    c_deg_ui       = st.number_input("Degradation cost ($/kWh)", value=C_DEG,           step=0.005, format="%.3f")
    lam_demand_ui  = st.number_input("Monthly demand charge ($/kW)", value=LAM_DEMAND, step=0.1)
    c_bess_ui      = st.number_input("BESS specific cost ($/kWh)", value=C_BESS,        step=10.0)
    p_grid_max_ui  = st.number_input("Max grid connection (kW)",  value=P_GRID_MAX,     step=100.0)

    st.divider()
    st.subheader("Sensitivity Analysis")
    run_sensitivity_flag = st.checkbox("Run BESS size sensitivity", value=False)
    sens_sizes_str = st.text_input("BESS sizes to test (kWh, comma-sep)",
                                    value="2000,4000,6000,8000,10000,12000")

    st.info("Upload CSVs in the main panel, generate forecast, then run optimisation.")

# Rebuild params from sidebar
P_CH_MAX_UI  = E_bess_ui * 0.25
P_DIS_MAX_UI = E_bess_ui * 0.25
SOC_MIN_UI   = 0.20 * E_bess_ui
SOC_MAX_UI   = 0.90 * E_bess_ui
SOC_INIT_UI  = 0.20 * E_bess_ui
LAM_DEMAND_DAILY_UI = lam_demand_ui / 30.0
CRF_UI       = DISCOUNT_RATE * (1 + DISCOUNT_RATE)**N_YEARS / ((1 + DISCOUNT_RATE)**N_YEARS - 1)
BESS_DAILY_UI= c_bess_ui * E_bess_ui * CRF_UI / 365.0

PARAMS_UI = {
    'E_bess'          : E_bess_ui,
    'P_ch_max'        : P_CH_MAX_UI,
    'P_dis_max'       : P_DIS_MAX_UI,
    'eta_ch'          : ETA_CH,
    'eta_dis'         : ETA_DIS,
    'soc_min'         : SOC_MIN_UI,
    'soc_max'         : SOC_MAX_UI,
    'soc_final'       : SOC_INIT_UI,
    'c_deg'           : c_deg_ui,
    'lam_demand_daily': LAM_DEMAND_DAILY_UI,
    'bess_daily_cost' : BESS_DAILY_UI,
    'delta_t'         : DELTA_T,
}

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════  LAYER 1 — DEMAND FORECAST  ══════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 📈 24h-ahead EV Charging Demand Forecast")

left, right = st.columns(2)
with left:
    lookback_upload = st.file_uploader(
        "Upload lookback CSV", type=["csv"], key="lookback_csv",
        help=f"Must contain at least {model_bundle['seq_len']} rows and all feature columns.")
with right:
    next_day_upload = st.file_uploader(
        "Upload next-day CSV", type=["csv"], key="next_day_csv",
        help="Each row produces one forecast value.")

if lookback_upload is not None and next_day_upload is not None:
    upload_signature = (lookback_upload.name, lookback_upload.size,
                        next_day_upload.name, next_day_upload.size)
    if st.session_state.get("forecast_signature") != upload_signature:
        st.session_state.pop("forecast_result", None)
        st.session_state.pop("opt_result", None)

    try:
        lookback_data = read_uploaded_csv(lookback_upload, "Lookback CSV")
        next_day_data = read_uploaded_csv(next_day_upload, "Next-day CSV")

        sum_l, sum_r = st.columns(2)
        sum_l.success(f"Lookback CSV loaded: {len(lookback_data):,} rows")
        sum_r.success(f"Next-day CSV loaded: {len(next_day_data):,} rows")

        with st.expander("Preview uploaded data"):
            pl, pr = st.columns(2)
            with pl:
                st.write("Lookback CSV"); st.dataframe(lookback_data.head(), use_container_width=True)
            with pr:
                st.write("Next-day CSV");  st.dataframe(next_day_data.head(), use_container_width=True)

        if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating forecast..."):
                history_array, future_array = prepare_features(
                    lookback_data, next_day_data,
                    model_bundle["feature_cols"], model_bundle["seq_len"])
                forecast_values = forecast(model_bundle, history_array, future_array)
                result_df = build_output(next_day_data, forecast_values)
            st.session_state["forecast_result"]    = result_df
            st.session_state["forecast_signature"] = upload_signature
            st.session_state.pop("opt_result", None)

    except Exception as exc:
        st.error(str(exc))

# ── Forecast results display ──────────────────────────────────────────────
if "forecast_result" in st.session_state:
    result_df = st.session_state["forecast_result"]

    st.divider()
    st.subheader("Forecast Results")

    minimum_index = result_df[FORECAST_COL].idxmin()
    maximum_index = result_df[FORECAST_COL].idxmax()

    def format_occurrence(idx):
        val    = result_df.loc[idx, TIMESTAMP_COL]
        parsed = pd.to_datetime(val, errors="coerce")
        if pd.notna(parsed): return parsed.strftime("%Y-%m-%d %H:%M")
        if pd.notna(val):    return str(val)
        return f"Step {idx + 1}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast Points",    f"{len(result_df):,}")
    c2.metric("Average Forecast",   f"{result_df[FORECAST_COL].mean():,.2f} kW")
    c3.metric("Minimum Demand",     f"{result_df.loc[minimum_index, FORECAST_COL]:,.2f} kW",
              help=f"At {format_occurrence(minimum_index)}")
    c4.metric("Maximum Demand",     f"{result_df.loc[maximum_index, FORECAST_COL]:,.2f} kW",
              help=f"At {format_occurrence(maximum_index)}")

    chart_data = result_df.copy()
    parsed_time = pd.to_datetime(chart_data[TIMESTAMP_COL], errors="coerce")
    if parsed_time.notna().all():
        chart_data[TIMESTAMP_COL] = parsed_time
        chart_data = chart_data.set_index(TIMESTAMP_COL)
    else:
        chart_data = chart_data.set_index(
            pd.Index(np.arange(1, len(chart_data) + 1), name="Forecast step"))

    st.line_chart(chart_data[[FORECAST_COL]], y=FORECAST_COL,
                  color="#7C3AED", use_container_width=True)

    col_tbl, col_dl = st.columns([3, 1])
    with col_tbl:
        st.dataframe(result_df.style.format({FORECAST_COL: "{:,.2f}"}),
                     use_container_width=True, hide_index=True)
    with col_dl:
        st.download_button("⬇️ Download Forecast CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="average_power_forecast.csv", mime="text/csv",
            type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════  LAYER 2 — PEAK SHAVING OPTIMISATION  ════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
if "forecast_result" in st.session_state:
    result_df = st.session_state["forecast_result"]

    st.divider()
    st.markdown("## 🔋 Peak Shaving Threshold Optimization")
    st.caption(
        "Uses the forecasted demand above as input. Convex optimization (CVXPY / HiGHS LP solver) "
        "determines the optimal flat threshold that minimises energy cost + demand charge + "
        "battery degradation + BESS capital cost."
    )

    # TOU info table
    with st.expander("📋 TOU Tariff & BESS Constraints Reference"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Time-of-Use Tariff (PUCSL — Apr. 2026)**")
            st.dataframe(pd.DataFrame({
                'Period'         : ['Off-peak', 'Day (shoulder)', 'TOU Peak'],
                'Hours'          : ['00:00–08:00, 22:00–24:00', '08:00–18:00', '18:00–22:00'],
                'Rate (LKR/kWh)' : [15, 15, 70],
                'Rate (USD/kWh)' : [0.048, 0.048, 0.220],
            }), hide_index=True, use_container_width=True)
        with col_b:
            st.markdown("**BESS Constraints**")
            st.dataframe(pd.DataFrame({
                'Parameter': ['Capacity', 'DoD limit', 'SoC range', 'Max C/D rate',
                              'Charging window', 'Discharging window'],
                'Value'    : [f'{E_bess_ui:,.0f} kWh', '70%', '20%–90%',
                              f'C/4 = {P_CH_MAX_UI:,.0f} kW',
                              'Off-peak ',
                              'Day and Peak only '],
            }), hide_index=True, use_container_width=True)

    if st.button("⚡ Run Peak Shaving Optimisation", type="primary", use_container_width=True):
        P_forecast = result_df[FORECAST_COL].values.astype(float)

        if len(P_forecast) != N_STEPS:
            st.warning(
                f"Optimisation expects exactly {N_STEPS} time steps (15-min resolution, 24h). "
                f"Forecast has {len(P_forecast)} steps. Interpolating to {N_STEPS} steps.")
            P_forecast = np.interp(
                np.linspace(0, 1, N_STEPS),
                np.linspace(0, 1, len(P_forecast)), P_forecast)

        hours_ax, mask_pk, mask_op, mask_dy, tariff_arr = build_tou_masks(len(P_forecast))
        baseline_res = calculate_baseline(P_forecast, tariff_arr, LAM_DEMAND_DAILY_UI,
                                          DELTA_T, mask_pk)

        with st.spinner("Solving static optimization (HiGHS LP)... this may take 15–30 seconds."):
            try:
                opt = solve_peak_shaving(P_forecast, tariff_arr, mask_pk, mask_op, mask_dy,
                                         SOC_INIT_UI, PARAMS_UI)
                st.session_state["opt_result"]   = opt
                st.session_state["opt_baseline"] = baseline_res
                st.session_state["opt_hours"]    = hours_ax
                st.session_state["opt_masks"]    = (mask_pk, mask_op, mask_dy)
                st.session_state["opt_demand"]   = P_forecast
                # Store tariff_arr too so sensitivity can use it after rerender
                st.session_state["opt_tariff"]   = tariff_arr
                st.success(f"✅ Optimisation complete — Solver status: **{opt['status']}**")
            except Exception as exc:
                st.error(f"Optimisation failed: {exc}")

    # ── Optimisation results — shown whenever they exist in session state ──
    if "opt_result" in st.session_state:
        opt        = st.session_state["opt_result"]
        baseline   = st.session_state["opt_baseline"]
        hours_ax   = st.session_state["opt_hours"]
        mask_pk, mask_op, mask_dy = st.session_state["opt_masks"]
        P_forecast = st.session_state["opt_demand"]
        tariff_arr = st.session_state["opt_tariff"]

        T_opt              = opt['threshold_kW']
        P_grid_opt         = opt['P_grid']
        P_dis_opt          = opt['P_dis']
        P_ch_opt           = opt['P_ch']
        SoC_opt            = opt['SoC']
        peak_reduction_kW  = P_forecast[mask_pk].max() - T_opt
        peak_reduction_pct = peak_reduction_kW / P_forecast[mask_pk].max() * 100
        energy_shaved_kWh  = float(np.sum(np.maximum(P_forecast[mask_pk] - T_opt, 0)) * DELTA_T)
        bess_discharge_kWh = float(np.sum(P_dis_opt) * DELTA_T)
        bess_charge_kWh    = float(np.sum(P_ch_opt)  * DELTA_T)
        daily_savings      = baseline['total_cost'] - (opt['J_energy'] + opt['J_demand']+ opt['J_deg'] + opt['J_bess'])
        annual_savings     = daily_savings * 365

        # ── KPI row ───────────────────────────────────────────────────────
        st.subheader("Key Performance Indicators")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("🎯 Optimal Threshold",   f"{T_opt:,.1f} kW")
        k2.metric("📉 Peak Reduction",      f"{peak_reduction_pct:.1f}%",
                  delta=f"-{peak_reduction_kW:,.1f} kW")
        k3.metric("💰 Daily Savings",       f"${daily_savings:,.2f}")
        k4.metric("📅 Annual Savings Est.", f"${annual_savings:,.0f}")
        k5.metric("⚡ Energy Shaved",       f"{energy_shaved_kWh:,.1f} kWh")

        # ── Tabbed charts ─────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔌 Power Flows", "📊 Peak Shaving Bars",
            "🔋 State of Charge", "⚡ BESS Schedule", "💵 Cost Analysis"])

        with tab1:
            st.plotly_chart(
                plot_power_flows(hours_ax, P_forecast, P_grid_opt, P_dis_opt, P_ch_opt,
                                 T_opt, mask_pk, mask_op, mask_dy),
                use_container_width=True)

        with tab2:
            st.plotly_chart(
                plot_peak_shaving_bars(hours_ax, P_forecast, P_grid_opt, T_opt,
                                       mask_pk, mask_op, mask_dy),
                use_container_width=True)

        with tab3:
            st.plotly_chart(
                plot_soc(hours_ax, SoC_opt, E_bess_ui, SOC_MIN_UI, SOC_MAX_UI,
                         SOC_INIT_UI, mask_pk, mask_op, mask_dy),
                use_container_width=True)

        with tab4:
            st.plotly_chart(
                plot_bess_schedule(hours_ax, P_dis_opt, P_ch_opt, mask_pk, mask_op, mask_dy),
                use_container_width=True)

        with tab5:
            st.plotly_chart(plot_cost_comparison(baseline, opt), use_container_width=True)

        # ── Economic summary tables ───────────────────────────────────────
        st.subheader("Economic Summary")
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.markdown("**Cost Breakdown (With BESS)**")
            cost_df = pd.DataFrame({
                'Cost Component': ['Energy Cost (grid import)', 'Demand Charge',
                                   'Battery Degradation', 'BESS Capital (daily)',
                                   'Total Optimised Cost'],
                'Amount ($)'    : [opt['J_energy'], opt['J_demand'],
                                   opt['J_deg'], opt['J_bess'],
                                   opt['J_energy'] + opt['J_demand'] + opt['J_deg'] + opt['J_bess']],
            })
            st.dataframe(cost_df.style.format({'Amount ($)': '${:,.2f}'}),
                         hide_index=True, use_container_width=True)
        with ecol2:
            st.markdown("**Baseline vs With BESS**")
            comp_df = pd.DataFrame({
                'Metric'   : ['Energy Cost ($)', 'Demand Charge ($)', 'Total Operational ($)',
                              'Daily Savings ($)', 'Annual Savings ($)'],
                'Baseline' : [baseline['J_energy'], baseline['J_demand'],
                              baseline['total_cost'], '—', '—'],
                'With BESS': [opt['J_energy'], opt['J_demand'],
                              opt['J_energy'] + opt['J_demand'],
                              daily_savings, annual_savings],
            })
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

        # ── Battery operation summary ─────────────────────────────────────
        st.subheader("Battery Operation Summary")
        SoC_pct  = SoC_opt / E_bess_ui * 100
        shaved_m = (P_forecast > T_opt) & mask_pk
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Energy Discharged", f"{bess_discharge_kWh:,.1f} kWh")
        b2.metric("Energy Charged",    f"{bess_charge_kWh:,.1f} kWh")
        b3.metric("SoC Min Reached",   f"{SoC_pct.min():.1f}%")
        b4.metric("SoC Max Reached",   f"{SoC_pct.max():.1f}%")
        b5.metric("Slots Shaved",      f"{int(shaved_m.sum())} × 15 min")

        # ── Schedule CSV download ─────────────────────────────────────────
        sched_df = pd.DataFrame({
            'Time (hours)'          : hours_ax,
            'Forecasted Demand (kW)': P_forecast,
            'Grid Import (kW)'      : P_grid_opt,
            'BESS Discharge (kW)'   : P_dis_opt,
            'BESS Charge (kW)'      : P_ch_opt,
            'SoC (kWh)'             : SoC_opt[:-1],
            'SoC (%)'               : SoC_pct[:-1],
            'TOU Period'            : np.where(mask_pk, 'Peak',
                                      np.where(mask_op, 'Off-peak', 'Day')),
        })
        st.download_button(
            "⬇️ Download Full Schedule CSV",
            data=sched_df.to_csv(index=False).encode("utf-8"),
            file_name="peak_shaving_schedule.csv", mime="text/csv",
            type="primary", use_container_width=True)

        # ── Sensitivity analysis ──────────────────────────────────────────
        if run_sensitivity_flag:
            st.divider()
            st.subheader("📐 Sensitivity Analysis — BESS Size")
            try:
                sizes = [int(x.strip()) for x in sens_sizes_str.split(",") if x.strip()]
                with st.spinner(f"Running sensitivity for {len(sizes)} BESS sizes..."):
                    sens_df = run_sensitivity(P_forecast, tariff_arr, mask_pk, mask_op,
                                             mask_dy, sizes)
                if not sens_df.empty:
                    st.plotly_chart(
                        plot_sensitivity(sens_df, baseline['total_cost']),
                        use_container_width=True)
                    st.dataframe(
                        sens_df.style.format({
                            'Optimal Threshold (kW)': '{:,.1f}',
                            'Total Daily Cost ($)'  : '${:,.2f}',
                            'Demand Charge ($)'     : '${:,.2f}',
                        }),
                        hide_index=True, use_container_width=True)
                    st.download_button(
                        "⬇️ Download Sensitivity CSV",
                        data=sens_df.to_csv(index=False).encode("utf-8"),
                        file_name="sensitivity_analysis.csv", mime="text/csv")
            except Exception as exc:
                st.error(f"Sensitivity analysis failed: {exc}")

