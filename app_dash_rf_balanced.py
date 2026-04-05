import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import io
import os
import ipaddress

# =============================================================================
# 1. APP INITIALIZATION
# =============================================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "ML-DDoS Detector"

# =============================================================================
# 2. LOAD MODEL 
# =============================================================================
MODEL_PATH = os.path.join(_BASE_DIR, "model", "rf_src_ip_model.pkl")

def load_model(model_path):
    try:
        data = joblib.load(model_path)
        if isinstance(data, dict):
            return data['model'], data['features']
        return data, None
    except FileNotFoundError:
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

model, feature_cols = load_model(MODEL_PATH)

if feature_cols is None:
    feature_cols = [
        "pkt_rate",
        "byte_rate",
        "syn_ack_ratio",
        "pkt_ratio",
        "payload_ratio",
        "dst_port_ratio",
        "mean_iat",
        "avg_idle",
        "size_consistency",
        "dst_ip_ratio",
        "active_duration_sec",
    ]

# Auto-detect model name from loaded model type
def _detect_model_name(m):
    if m is None:
        return "Unknown"
    name = type(m).__name__
    if "RandomForest" in name:
        return "Random Forest"
    if "XGB" in name or "Booster" in name:
        return "XGBoost"
    if "SVC" in name or "SVM" in name:
        return "SVM"
    return name

MODEL_NAME = _detect_model_name(model)

# =============================================================================
# 3. DATA PROCESSING 
# =============================================================================

def build_features(df):
    """Transform raw data into 11-feature behavioral profile per Source IP.

    Synchronized with the Random Forest model trained on:
      pkt_rate, byte_rate, syn_ack_ratio, pkt_ratio, payload_ratio,
      dst_port_ratio, mean_iat, avg_idle, size_consistency,
      dst_ip_ratio, active_duration_sec
    """
    # ── Column name normalisation: strip whitespace from headers ──
    df.columns = df.columns.str.strip()

    # ── Helper: safe aggregation — uses column only if it exists ──
    def _safe_agg(col, func, fallback=0):
        """Return (col, func) tuple if col present, else add synthetic column of fallback value."""
        if col in df.columns:
            return col, func
        df[f'__missing_{col}'] = fallback
        return f'__missing_{col}', func

    # 1. Parse Timestamp (optional — used only for first_seen / last_seen / active_duration)
    _has_timestamp = "Timestamp" in df.columns
    if _has_timestamp:
        try:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
        except Exception:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # 2. Aggregate per Source IP
    # For Flow ID counter: if missing, use any column as proxy count
    _flow_id_col = "Flow ID" if "Flow ID" in df.columns else df.columns[0]

    agg_kwargs = dict(
        total_flows      = (_flow_id_col,        "count"),
        total_fwd_pkts   = _safe_agg("Tot Fwd Pkts",   "sum", 0),
        total_bwd_pkts   = _safe_agg("Tot Bwd Pkts",   "sum", 0),
        total_fwd_bytes  = _safe_agg("TotLen Fwd Pkts", "sum", 0),
        total_bwd_bytes  = _safe_agg("TotLen Bwd Pkts", "sum", 0),
        unique_dst_ips   = _safe_agg("Dst IP",         "nunique", 1),
        unique_dst_ports = _safe_agg("Dst Port",       "nunique", 1),
        mean_flow_duration = _safe_agg("Flow Duration", "mean", 0),
        mean_iat         = _safe_agg("Flow IAT Mean",  "mean", 0),
        avg_idle         = _safe_agg("Idle Mean",      "mean", 0),
        avg_pkt_size     = _safe_agg("Pkt Size Avg",   "mean", 0),
        std_pkt_len      = _safe_agg("Pkt Len Std",    "mean", 0),
        total_syn_flags  = _safe_agg("SYN Flag Cnt",   "sum", 0),
        total_ack_flags  = _safe_agg("ACK Flag Cnt",   "sum", 0),
        total_rst_flags  = _safe_agg("RST Flag Cnt",   "sum", 0),
    )
    # Add Timestamp-dependent aggregations only if column exists
    if _has_timestamp:
        agg_kwargs['first_seen'] = ("Timestamp", "min")
        agg_kwargs['last_seen']  = ("Timestamp", "max")

    src_features = df.groupby("Src IP").agg(**agg_kwargs).reset_index()
    # Drop synthetic fallback columns
    src_features.drop(columns=[c for c in src_features.columns if c.startswith('__missing_')], inplace=True)

    src_ip_dataset = src_features.copy()

    # 3. Active duration
    if _has_timestamp and 'first_seen' in src_ip_dataset.columns and 'last_seen' in src_ip_dataset.columns:
        duration = (src_ip_dataset["last_seen"] - src_ip_dataset["first_seen"]).dt.total_seconds()
        src_ip_dataset["active_duration_sec"] = duration.clip(lower=1)
        src_ip_dataset["first_seen"] = src_ip_dataset["first_seen"].astype(str)
        src_ip_dataset["last_seen"]  = src_ip_dataset["last_seen"].astype(str)
    elif "mean_flow_duration" in src_ip_dataset.columns:
        src_ip_dataset["active_duration_sec"] = src_ip_dataset["mean_flow_duration"].clip(lower=1e-6) / 1e6
        src_ip_dataset["active_duration_sec"] = src_ip_dataset["active_duration_sec"].clip(lower=1)
        src_ip_dataset["first_seen"] = "N/A"
        src_ip_dataset["last_seen"]  = "N/A"
    else:
        src_ip_dataset["active_duration_sec"] = 1.0
        src_ip_dataset["first_seen"] = "N/A"
        src_ip_dataset["last_seen"]  = "N/A"

    # 4. Derived Behavioral Features
    # -- Volumetric --
    src_ip_dataset["byte_rate"] = (
        (src_ip_dataset["total_fwd_bytes"] + src_ip_dataset["total_bwd_bytes"])
        / src_ip_dataset["active_duration_sec"]
    )
    src_ip_dataset["pkt_rate"] = (
        (src_ip_dataset["total_fwd_pkts"] + src_ip_dataset["total_bwd_pkts"])
        / src_ip_dataset["active_duration_sec"]
    )

    # -- Protocol & Asymmetry --
    src_ip_dataset["syn_ack_ratio"] = (
        src_ip_dataset["total_syn_flags"] / (src_ip_dataset["total_ack_flags"] + 1)
    )
    src_ip_dataset["pkt_ratio"] = (
        src_ip_dataset["total_fwd_pkts"] / (src_ip_dataset["total_bwd_pkts"] + 1)
    )
    src_ip_dataset["payload_ratio"] = (
        src_ip_dataset["total_fwd_bytes"] / (src_ip_dataset["total_fwd_pkts"] + 1)
    )

    # -- Application Behavior --
    src_ip_dataset["dst_ip_ratio"] = (
        src_ip_dataset["unique_dst_ips"] / src_ip_dataset["total_flows"]
    )
    src_ip_dataset["dst_port_ratio"] = (
        src_ip_dataset["unique_dst_ports"] / src_ip_dataset["total_flows"]
    )
    src_ip_dataset["size_consistency"] = (
        src_ip_dataset["std_pkt_len"] / (src_ip_dataset["avg_pkt_size"] + 1)
    )

    # 5. Clean up infinities & NaN
    src_ip_dataset = src_ip_dataset.replace([np.inf, -np.inf], np.nan).fillna(0)

    return src_ip_dataset

# =============================================================================
# 4. LAYOUT HELPER FUNCTIONS
# =============================================================================
def make_metric_card(title, value, subtitle, icon_html, color_class):
    """Create a single metric card matching Figma design"""
    icon_element = icon_html if not isinstance(icon_html, str) else html.Span(icon_html)
    return html.Div([
        html.Div([
            html.Div([
                html.Div(title, className="metric-title"),
                html.Div(icon_element, className=f"metric-icon {color_class}"),
            ], className="metric-top"),
            html.Div(value, className="metric-value"),
            html.Div(subtitle, className="metric-subtitle"),
        ], className="metric-inner"),
    ], className="metric-card")

# SVG-style icon helpers for metric cards
def icon_chart():
    return html.Div([
        html.Div(className="icon-svg-line icon-svg-line-1"),
        html.Div(className="icon-svg-line icon-svg-line-2"),
    ], className="icon-svg-chart")

def icon_warning():
    return html.Div("⚠", className="icon-svg-warning")

def icon_check():
    return html.Div("✓", className="icon-svg-check")

def icon_target():
    return html.Div("◎", className="icon-svg-target")

def make_status_row(label, value_element):
    """Create a status row for system status card"""
    return html.Div([
        html.Span(label, className="status-label"),
        html.Span(value_element if not isinstance(value_element, str) else value_element, className="status-value"),
    ], className="status-row")

def make_activity_item(dot_color, text, time_text):
    """Create an activity timeline item"""
    return html.Div([
        html.Div(className=f"activity-dot dot-{dot_color}"),
        html.Div([
            html.Div(text, className="activity-text"),
            html.Div(time_text, className="activity-time"),
        ], className="activity-content"),
    ], className="activity-item")

# =============================================================================
# 4B. CHART CONSTANTS & ANALYTICS CHARTS (before layout)
# =============================================================================
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='rgba(255,255,255,0.8)', family='Inter, system-ui, sans-serif'),
    margin=dict(t=40, b=40, l=40, r=40),
)

def create_traffic_over_time():
    """Line chart: Traffic Over Time (demo data)"""
    hours = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '23:59']
    benign = [3800, 3000, 2200, 2800, 2200, 2400, 3400]
    ddos = [200, 100, 800, 3600, 4800, 3600, 4000]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=benign, mode='lines+markers', name='Benign Traffic',
        line=dict(color='#4ADE80', width=2), marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=ddos, mode='lines+markers', name='DDoS Attacks',
        line=dict(color='#EF4444', width=2), marker=dict(size=8)
    ))
    fig.update_layout(
        **CHART_LAYOUT, height=350,
        legend=dict(orientation='h', yanchor='bottom', y=-0.22, xanchor='center', x=0.5),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', griddash='dot'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', griddash='dot'),
    )
    return fig

def create_analytics_distribution():
    """Pie chart: Traffic Distribution (demo data)"""
    fig = go.Figure(go.Pie(
        values=[1271655, 12847],
        labels=['Benign Traffic', 'DDoS Attacks'],
        marker=dict(colors=['#4ADE80', '#EF4444']),
        hole=0.55,
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=11, color='rgba(255,255,255,0.8)'),
    ))
    layout = {k: v for k, v in CHART_LAYOUT.items() if k != 'margin'}
    fig.update_layout(**layout, height=280, showlegend=False,
                       margin=dict(t=20, b=20, l=80, r=80))
    return fig

def create_feature_importance():
    """Horizontal bar chart: Feature Importance (demo data)"""
    features = ['Protocol Type', 'Packet Length', 'Flow IAT Mean', 'Byte Rate', 'Packet Rate', 'Flow Duration']
    importance = [0.52, 0.62, 0.68, 0.75, 0.82, 0.92]
    fig = go.Figure(go.Bar(
        x=importance, y=features, orientation='h',
        marker=dict(color='#22D3EE'),
    ))
    layout = {k: v for k, v in CHART_LAYOUT.items() if k != 'margin'}
    fig.update_layout(
        **layout, height=350,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', griddash='dot', tickformat='.0%'),
        yaxis=dict(showgrid=False),
        margin=dict(t=10, b=30, l=120, r=20),
    )
    return fig

# =============================================================================
# 5. MAIN LAYOUT
# =============================================================================
app.layout = html.Div([

    # ─── Fixed analysis-complete toast overlay ───────────────────
    # Displayed in the centre of the viewport by the clientside callback
    # whenever analysis finishes, then auto-dismissed after ~3 s.
    html.Div([
        html.Div("✅", style={'fontSize': '2.5rem', 'marginBottom': '0.6rem'}),
        html.Div("Analysis Complete!", style={
            'fontWeight': '700', 'fontSize': '1.35rem', 'color': '#4ADE80',
            'marginBottom': '0.35rem',
        }),
        html.Div("Scrolling to results...", style={
            'color': 'rgba(255,255,255,0.55)', 'fontSize': '0.88rem',
        }),
    ], id='analysis-toast'),

    # Hidden store used as dummy Output for the toast clientside callback
    dcc.Store(id='_toast-trigger'),

    # ─── NAVIGATION BAR ─────────────────────────────────────────
    html.Nav([
        html.Div([
            html.Div([
                html.Span("🛡️", className="nav-logo-icon"),
                html.Span("ML-DDoS Detector", className="nav-logo-text"),
            ], className="nav-logo"),
            html.Div([
                html.Button("Detection", id="nav-btn-detection", n_clicks=0, className="nav-link active"),
                html.Button("Models", id="nav-btn-models", n_clicks=0, className="nav-link"),
                html.Button("About", id="nav-btn-about", n_clicks=0, className="nav-link"),
            ], className="nav-links"),
        ], className="nav-inner"),
    ], className="navbar"),

    # ─── PAGE CONTENT ────────────────────────────────────────────
    html.Div([

        # ═════════════════════════════════════════════════════════
        # PAGE: DETECTION
        # ═════════════════════════════════════════════════════════
        html.Div([
            # ── Hero Header ──
            html.Div([
                html.P(
                    "Monitor and analyze network traffic patterns. "
                    "Upload logs to detect anomalies and security threats using advanced ML algorithms.",
                    className="hero-desc",
                ),
                # Quick-info badges
                html.Div([
                    html.Div([
                        html.Span("🤖", style={'marginRight': '0.4rem'}),
                        html.Span(f"Model: {MODEL_NAME}", style={'fontWeight': '600'}),
                    ], className="hero-badge"),
                    html.Div([
                        html.Span("📊", style={'marginRight': '0.4rem'}),
                        html.Span(f"{len(feature_cols)} Behavioral Features", style={'fontWeight': '600'}),
                    ], className="hero-badge"),
                    html.Div([
                        html.Span("🟢", style={'marginRight': '0.4rem'}),
                        html.Span("Status: Active", style={'fontWeight': '600'}),
                    ], className="hero-badge"),
                ], className="hero-badges"),
            ], className="hero-section"),

            # ── Traffic Analysis Dashboard (centered full-width) ──
            html.Div([
                html.H2("Traffic Analysis Dashboard", className="dash-panel-title"),

                # Upload zone
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.Div("📁", style={'fontSize': '2.2rem', 'marginBottom': '0.5rem'}),
                        html.Div([
                            html.Span("Drag and Drop or "),
                            html.A("Browse files", className="upload-link"),
                        ], className="upload-text"),
                        html.Div("Limit 200MB per file  •  CSV, PARQUET", className="upload-hint"),
                    ]),
                    className="upload-zone",
                    multiple=False,
                ),
                html.Div(id='file-info', style={'marginTop': '0.75rem'}),

                # Threshold control
                html.Div([
                    html.Div([
                        html.Span("🎯 Detection Threshold", className="threshold-label"),
                        html.Span(id='threshold-value', children="0.25", className="threshold-val-badge"),
                    ], className="threshold-header-row"),
                    dcc.Slider(
                        id='threshold-slider',
                        min=0.01,
                        max=1.0,
                        step=0.01,
                        value=0.25,
                        marks={
                            0.01: {'label': '1%',  'style': {'color': 'rgba(255,255,255,0.45)'}},
                            0.25: {'label': '25%', 'style': {'color': 'rgba(255,255,255,0.45)'}},
                            0.50: {'label': '50%', 'style': {'color': 'rgba(255,255,255,0.45)'}},
                            0.75: {'label': '75%', 'style': {'color': 'rgba(255,255,255,0.45)'}},
                            1.0:  {'label': '100%','style': {'color': 'rgba(255,255,255,0.45)'}},
                        },
                        tooltip={'always_visible': False, 'placement': 'bottom'},
                    ),
                    html.Div(
                        "⬇ Lower = more sensitive (more alerts)   |   ⬆ Higher = fewer false positives",
                        className="threshold-hint-text"
                    ),
                ], className="threshold-section"),

                # ── Whitelist (Safe IPs) ──
                html.Div([
                    html.Div([
                        html.Span("🛡️", style={'fontSize': '1.1rem', 'marginRight': '0.45rem'}),
                        html.Span("Whitelist (Safe IPs)", className="threshold-label"),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '0.5rem'}),
                    html.P(
                        "IPs in this list will be marked as safe regardless of ML prediction.",
                        style={'color': 'rgba(255,255,255,0.35)', 'fontSize': '0.75rem',
                               'margin': '0 0 0.5rem 0'}
                    ),
                    dcc.Textarea(
                        id='whitelist-input',
                        value='',
                        placeholder='Enter safe IPs (one per line):\n192.168.10.1\n8.8.8.8',
                        style={
                            'width': '100%', 'height': '90px',
                            'backgroundColor': 'rgba(255,255,255,0.04)',
                            'border': '1px solid rgba(255,255,255,0.1)',
                            'borderRadius': '8px', 'color': '#fff',
                            'padding': '0.6rem 0.8rem', 'fontSize': '0.85rem',
                            'fontFamily': 'JetBrains Mono, monospace',
                            'resize': 'vertical',
                        },
                    ),
                ], className="threshold-section"),

                # Analyze button
                html.Button("🔍  Analyze Traffic", id='scan-button', n_clicks=0, className="analyze-btn"),
                html.Div(id='scan-btn-container', style={'display': 'none'}),

                # Privacy note
                html.P(
                    "All uploaded network traffic data is processed locally and used solely for analysis purposes.",
                    className="privacy-note",
                ),
            ], className="dash-panel"),

            # ── Results Section (appears below after analysis) ──
            html.Div(id='results-section', children=[]),

        ], id="page-detection", className="page", style={'display': 'block'}),

        # ═════════════════════════════════════════════════════════
        # PAGE: ML MODELS
        # ═════════════════════════════════════════════════════════
        html.Div([
            html.Div([
                html.H1("ML Models", className="page-title"),
                html.P("Comparison of machine learning models for DDoS detection", className="page-subtitle"),
            ], className="page-header"),

            # Model Comparison Cards
            html.Div([
                # ── XGBoost ──
                html.Div([
                    html.Div([
                        html.Span("🚀", className="model-icon"),
                        html.Span("XGBoost", className="model-name"),
                    ], className="model-card-header"),
                    html.Div([
                        html.Div([
                            html.Div([html.Div("Accuracy", className="model-metric-label"), html.Div("99.97%", className="model-metric-val model-metric-cyan")]),
                            html.Div([html.Div("F1-Score", className="model-metric-label"), html.Div("85.71%", className="model-metric-val model-metric-green")]),
                        ], className="model-metrics-row"),
                        html.Div([
                            html.Div([html.Div("Precision", className="model-metric-label"), html.Div("85.71%", className="model-metric-val")]),
                            html.Div([html.Div("Recall", className="model-metric-label"), html.Div("85.71%", className="model-metric-val")]),
                        ], className="model-metrics-row"),
                    ], className="model-metrics-section"),
                    html.Div([
                        html.Div([html.Span("✅"), html.Span(" Strengths", className="sw-title")], className="sw-header"),
                        html.Ul([
                            html.Li("Highest CV stability (88.74% Mean F1)"),
                            html.Li("Extremely resilient to imbalanced traffic"),
                            html.Li("Exceptional feature importance ranking"),
                            html.Li("Isolates definitive behavioral signatures"),
                            html.Li("Highly optimized inference speed"),
                        ], className="sw-list"),
                    ], className="sw-section"),
                    html.Div([
                        html.Div([html.Span("❌"), html.Span(" Weaknesses", className="sw-title sw-title-red")], className="sw-header"),
                        html.Ul([
                           html.Li("Slightly lower static recall (85.71%) vs RF"),
                           html.Li("Requires careful hyperparameter tuning"),
                           html.Li("More complex tree ensembling structure"),
                        ], className="sw-list"),
                    ], className="sw-section"),
                    html.Div([
                        html.Div([html.Span("⚡"), html.Span(" Use Cases", className="sw-title sw-title-yellow")], className="sw-header"),
                        html.Ul([
                            html.Li("Secondary validation engine"),
                            html.Li("Imbalanced dataset research"),
                            html.Li("Identifying core volumetric metrics"),
                            html.Li("Robust baseline performance benchmarking"),
                        ], className="sw-list"),
                    ], className="sw-section"),
                ], className="model-card"),

                # ── Support Vector Machine ──
                html.Div([
                    html.Div([
                        html.Span("🧠", className="model-icon"),
                        html.Span("Support Vector Machine (SVC)", className="model-name"),
                    ], className="model-card-header"),
                    html.Div([
                        html.Div([
                            html.Div([html.Div("Accuracy", className="model-metric-label"), html.Div("99.97%", className="model-metric-val model-metric-cyan")]),
                            html.Div([html.Div("F1-Score", className="model-metric-label"), html.Div("83.33%", className="model-metric-val model-metric-green")]),
                        ], className="model-metrics-row"),
                        html.Div([
                            html.Div([html.Div("Precision", className="model-metric-label"), html.Div("100%", className="model-metric-val")]),
                            html.Div([html.Div("Recall", className="model-metric-label"), html.Div("71.43%", className="model-metric-val")]),
                        ], className="model-metrics-row"),
                    ], className="model-metrics-section"),
                    html.Div([
                        html.Div([html.Span("✅"), html.Span(" Strengths", className="sw-title")], className="sw-header"),
                        html.Ul([
                            html.Li("Perfect Precision (zero false alarms)"),
                            html.Li("Maps non-linear boundaries using RBF kernel"),
                            html.Li("Highly accurate on benign majority class"),
                        ], className="sw-list"),
                    ], className="sw-section"),
                    html.Div([
                        html.Div([html.Span("❌"), html.Span(" Weaknesses", className="sw-title sw-title-red")], className="sw-header"),
                        html.Ul([
                            html.Li("Severe performance drop on imbalanced data"),
                            html.Li("Failed to detect complex attack vectors"),
                            html.Li("Lacks direct feature importance coefficients"),
                            html.Li("High standard deviation in Cross-Validation"),
                        ], className="sw-list"),
                    ], className="sw-section"),
                    html.Div([
                        html.Div([html.Span("⚡"), html.Span(" Use Cases", className="sw-title sw-title-yellow")], className="sw-header"),
                        html.Ul([
                            html.Li("Baseline distance-based model comparison"),
                            html.Li("Strictly zero-false-positive environments"),
                            html.Li("Secondary metric validation"),
                        ], className="sw-list"),
                    ], className="sw-section"),
                ], className="model-card"),

                # ── Random Forest (Best) ──
                html.Div([
                    html.Div([
                        html.Div([
                            html.Span("🌲", className="model-icon"),
                            html.Span("Random Forest", className="model-name"),
                        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '0.75rem'}),
                        html.Span(["🏆 Best"], className="best-badge"),
                    ], className="model-card-header model-card-header-best"),
                    html.Div([
                        html.Div([
                            html.Div([html.Div("Accuracy", className="model-metric-label"), html.Div("100%", className="model-metric-val model-metric-cyan")]),
                            html.Div([html.Div("F1-Score", className="model-metric-label"), html.Div("100%", className="model-metric-val model-metric-green")]),
                        ], className="model-metrics-row"),
                        html.Div([
                            html.Div([html.Div("Precision", className="model-metric-label"), html.Div("100%", className="model-metric-val")]),
                            html.Div([html.Div("Recall", className="model-metric-label"), html.Div("100%", className="model-metric-val")]),
                        ], className="model-metrics-row"),
                    ], className="model-metrics-section"),
                    html.Div([
                        html.Div([html.Span("✅"), html.Span(" Strengths", className="sw-title")], className="sw-header"),
                        html.Ul([
                            html.Li("Achieved perfect 100% Recall metric"),
                            html.Li("Zero missed malicious attacks in testing"),
                            html.Li("Effectively interprets 11 custom features"),
                            html.Li("Highly aggressive threat detection capability"),
                        ], className="sw-list"),
                    ], className="sw-section"),
                    html.Div([
                        html.Div([html.Span("❌"), html.Span(" Weaknesses", className="sw-title sw-title-red")], className="sw-header"),
                        html.Ul([
                            html.Li("Trades precision for maximum recall"),
                            html.Li("Slightly lower CV stability vs XGBoost"),
                            html.Li("Generates larger exported model size"),
                        ], className="sw-list"),
                    ], className="sw-section"),
                    html.Div([
                        html.Div([html.Span("⚡"), html.Span(" Use Cases", className="sw-title sw-title-yellow")], className="sw-header"),
                        html.Ul([
                            html.Li("Core inference engine for Detection Page"),
                            html.Li("Forensic analysis requiring max security"),
                            html.Li("Zero-tolerance environments for misses"),
                        ], className="sw-list"),
                    ], className="sw-section"),
                ], className="model-card model-card-best"),
            ], className="models-grid"),

            # Performance Comparison Table
            html.Div([
                html.H3("Performance Comparison", className="comparison-title"),
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("MODEL"),
                            html.Th("ACCURACY"),
                            html.Th("PRECISION"),
                            html.Th("RECALL"),
                            html.Th("F1-SCORE"),
                            html.Th("STATUS"),
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td("Random Forest"),
                            html.Td("100%"),
                            html.Td("100%"),
                            html.Td("100%"),
                            html.Td("100%"),
                            html.Td(html.Span("Production", className="status-production")),
                        ]),
                        html.Tr([
                            html.Td("Support Vector Machine (SVC)"),
                            html.Td("99.97%"),
                            html.Td("100%"),
                            html.Td("71.43%"),
                            html.Td("83.33%"),
                            html.Td(html.Span("Testing", className="status-testing")),
                        ]),
                        html.Tr([
                            html.Td("XGBoost"),
                            html.Td("99.97%"),
                            html.Td("85.71%"),
                            html.Td("85.71%"),
                            html.Td("85.71%"),
                            html.Td(html.Span("Testing", className="status-testing")),
                        ]),
                    ]),
                ], className="comparison-table"),
            ], className="analytics-card comparison-section"),

        ], id="page-models", className="page", style={'display': 'none'}),

        # ═════════════════════════════════════════════════════════
        # PAGE: ABOUT
        # ═════════════════════════════════════════════════════════
        html.Div([
            # Hero Section
            html.Div([
                html.Div([
                    html.Span("🎓", className="hero-icon"),
                    html.Div([
                        html.H1("Applying Machine Learning for DDoS Attack Detection", className="hero-title"),
                        html.P("Based on Network Traffic Analysis", className="hero-subtitle"),
                        html.Div([
                            html.Span("🎓 "),
                            html.Span("Graduation Project 2026"),
                        ], className="hero-badge"),
                    ]),
                ], className="hero-inner"),
            ], className="about-hero"),

            # Project Overview
            html.Div([
                html.Div([
                    html.Span("📖", className="section-icon"),
                    html.Span("Project Overview", className="section-title"),
                ], className="card-header"),
                html.P(
                    "This project implements a comprehensive machine learning-based system for detecting " \
                    "Distributed Denial of Service (DDoS) attacks through network traffic analysis. " \
                    "After rigorously evaluating advanced ML algorithms-including Support Vector Machine (SVC), Random Forest, and XGBoost "
                    "- the system utilizes Random Forest as its core inference engine due to its superior stability and resilience on highly imbalanced data. " \
                    "To facilitate incident response, the web application features a dedicated Detection page. " \
                    "This streamlined interface allows network administrators to upload raw traffic logs (.csv or .parquet), automate behavioral feature engineering, "
                    "and instantly generate an actionable Blacklist of malicious IP addresses to support cybersecurity operations and research.",
                    className="about-text"
                ),
            ], className="analytics-card"),

            # Team Members
            html.Div([
                html.Div([
                    html.Span("👥", className="section-icon"),
                    html.Span("Team Members", className="section-title"),
                ], className="card-header"),
                html.Div([
                    html.Div([
                        html.Div("L", className="member-avatar avatar-blue"),
                        html.Div([
                            html.Div("Lê Bửu Trình - SE172430", className="member-name"),
                            html.Div("Team Leader", className="member-role"),
                        ]),
                    ], className="member-card"),
                    html.Div([
                        html.Div("N", className="member-avatar avatar-purple"),
                        html.Div([
                            html.Div("Nguyễn Chí Thành - SE182953", className="member-name"),
                            html.Div("Team Member", className="member-role"),
                        ]),
                    ], className="member-card"),
                    html.Div([
                        html.Div("V", className="member-avatar avatar-teal"),
                        html.Div([
                            html.Div("Võ Hoàng Mỹ Nhung - SE183053", className="member-name"),
                            html.Div("Team Member", className="member-role"),
                        ]),
                    ], className="member-card"),
                    html.Div([
                        html.Div("N", className="member-avatar avatar-pink"),
                        html.Div([
                            html.Div("Nguyễn Cao Đức An - SE181870", className="member-name"),
                            html.Div("Team Member", className="member-role"),
                        ]),
                    ], className="member-card"),
                ], className="members-grid"),
            ], className="analytics-card"),

            # Supervisor
            html.Div([
                html.Div([
                    html.Span("👨‍🏫", className="section-icon"),
                    html.Span("Supervisor", className="section-title"),
                ], className="card-header"),
                html.Div([
                    html.Div("Hồ Hải", className="supervisor-name"),
                    html.Div("Lecturer", className="supervisor-role"),
                ]),
            ], className="analytics-card"),

            # Dataset
            html.Div([
                html.Div([
                    html.Span("📁", className="section-icon"),
                    html.Span("Dataset", className="section-title"),
                ], className="card-header"),
                html.Div([
                    html.H4("Devendra DDoS Dataset", className="dataset-name"),
                    html.P(
                        "A massive-scale dataset sourced from Kaggle, containing extensive network traffic logs for DDoS attack detection. " \
                        "It provides the foundational flow-based metrics encompassing both benign and malicious patterns, " \
                        "enabling our system's deep behavioral analysis.",
                        className="about-text"
                    ),
                    html.A("View on Kaggle ↗", href="https://www.kaggle.com/datasets/devendra416/ddos-datasets", target="_blank", className="kaggle-link"),
                ], className="dataset-info"),
                html.Div([
                    html.Div([
                        html.Div("12.80M", className="stat-value stat-cyan"),
                        html.Div("Total Samples", className="stat-label"),
                    ], className="stat-card"),
                    html.Div([
                        html.Div("85", className="stat-value stat-cyan"),
                        html.Div("Features", className="stat-label"),
                    ], className="stat-card"),
                    html.Div([
                        html.Div("Balanced", className="stat-value stat-green"),
                        html.Div("Dataset Type", className="stat-label"),
                    ], className="stat-card"),
                ], className="stats-grid"),
            ], className="analytics-card"),

            # Technologies Used
            html.Div([
                html.Div([
                    html.Span("💻", className="section-icon"),
                    html.Span("Technologies Used", className="section-title"),
                ], className="card-header"),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Span("🧠 ", className="tech-icon"),
                            html.Span("Machine Learning", className="tech-title"),
                        ], className="tech-header"),
                        html.Ul([
                            html.Li("Python 3.12"),
                            html.Li("Scikit-learn"),
                            html.Li("Joblib"),
                        ], className="tech-list"),
                    ], className="tech-card"),
                    html.Div([
                        html.Div([
                            html.Span("📊 ", className="tech-icon"),
                            html.Span("Data Processing", className="tech-title"),
                        ], className="tech-header"),
                        html.Ul([
                            html.Li("Pandas"),
                            html.Li("NumPy"),
                            html.Li("Matplotlib"),
                            html.Li("Plotly"),
                        ], className="tech-list"),
                    ], className="tech-card"),
                    html.Div([
                        html.Div([
                            html.Span("🌐 ", className="tech-icon"),
                            html.Span("Web Framework", className="tech-title"),
                        ], className="tech-header"),
                        html.Ul([
                            html.Li("Dash by Plotly"),
                            html.Li("Dash Bootstrap Components"),
                            html.Li("Base64 & IO"),
                        ], className="tech-list"),
                    ], className="tech-card"),
                ], className="tech-grid"),
            ], className="analytics-card"),

            # Key Features
            html.Div([
                html.Div([
                    html.Span("🔑", className="section-icon"),
                    html.Span("Key Features", className="section-title"),
                ], className="card-header"),
                html.Div([
                    html.Div([html.Span("●", className="feature-dot"), " Batch Log Processing"], className="feature-item"),
                    html.Div([html.Span("●", className="feature-dot"), " Automated Feature Extraction"], className="feature-item"),
                    html.Div([html.Span("●", className="feature-dot"), " Random Forest Inference Engine"], className="feature-item"),
                    html.Div([html.Span("●", className="feature-dot"), " Actionable IP Blacklisting"], className="feature-item"),
                    html.Div([html.Span("●", className="feature-dot"), " Interactive Traffic Analytics"], className="feature-item"),
                ], className="features-grid"),
            ], className="analytics-card"),

            # Academic Footer
            html.Div([
                html.P([
                    "This project is submitted as part of the requirements for the Bachelor's Degree in Information Security at ",
                    html.Strong("FPT University, Ho Chi Minh City"),
                ], className="academic-text"),
                html.P("Academic Year 2025-2026", className="academic-year"),
            ], className="academic-footer"),

        ], id="page-about", className="page", style={'display': 'none'}),

    ], className="main-content"),

    # ─── FOOTER ──────────────────────────────────────────────────
    html.Footer([
        html.P([
            "ML-DDoS Detector © 2026 • ",
            html.A("Terms of Service", href="#", className="footer-link"),
            " • ",
            html.A("Privacy Policy", href="#", className="footer-link"),
        ]),
    ], className="app-footer"),

    # ─── HIDDEN STORES ───────────────────────────────────────────
    dcc.Store(id='stored-data'),
    dcc.Store(id='scan-metrics', data=None),
    dcc.Store(id='_do-analysis', data=0),

    # Stores to hold download payloads (set once during analysis)
    dcc.Store(id='_bl-csv-data',    data=None),
    dcc.Store(id='_fw-linux-data',  data=None),
    dcc.Store(id='_fw-win-data',    data=None),
    dcc.Store(id='_full-csv-data',  data=None),

    # ─── WHITELIST CONFIRMATION MODAL ────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(
            dbc.ModalTitle([
                html.Span("⚠️", style={'marginRight': '0.5rem'}),
                "Whitelist Confirmation",
            ]),
            close_button=False,
            className="wl-modal-header",
        ),
        dbc.ModalBody(
            html.Div([
                html.P(
                    "You have entered IP addresses in the Whitelist.",
                    style={'fontWeight': '600', 'marginBottom': '0.6rem',
                           'fontSize': '0.95rem'},
                ),
                html.P(
                    "These IPs will be marked as safe regardless of ML prediction results. "
                    "Are you sure you want to proceed?",
                    style={'color': 'rgba(255,255,255,0.6)', 'fontSize': '0.88rem',
                           'lineHeight': '1.5'},
                ),
            ]),
            className="wl-modal-body",
        ),
        dbc.ModalFooter([
            dbc.Button(
                "Cancel",
                id="wl-cancel-btn",
                n_clicks=0,
                className="me-2",
                color="secondary",
                outline=True,
            ),
            dbc.Button(
                [html.Span("✅", style={'marginRight': '0.4rem'}), "Confirm & Analyze"],
                id="wl-confirm-btn",
                n_clicks=0,
                color="success",
            ),
        ], className="wl-modal-footer"),
    ], id="wl-confirm-modal", is_open=False, centered=True, backdrop="static",
       className="wl-modal-dark"),

], className="app-wrapper")

# =============================================================================
# 6. CALLBACKS
# =============================================================================

# ─── Navigation (clientside — instant, no server round-trip) ────
app.clientside_callback(
    """
    function(n1, n2, n3) {
        const ctx = dash_clientside.callback_context;
        let idx = 0;
        if (ctx.triggered.length) {
            const btnId = ctx.triggered[0].prop_id.split('.')[0];
            const map = {'nav-btn-detection': 0, 'nav-btn-models': 1, 'nav-btn-about': 2};
            idx = (btnId in map) ? map[btnId] : 0;
        }
        const styles = [{display:'none'}, {display:'none'}, {display:'none'}];
        const classes = ['nav-link', 'nav-link', 'nav-link'];
        styles[idx] = {display: 'block'};
        classes[idx] = 'nav-link active';
        return styles.concat(classes);
    }
    """,
    [Output('page-detection', 'style'),
     Output('page-models', 'style'),
     Output('page-about', 'style'),
     Output('nav-btn-detection', 'className'),
     Output('nav-btn-models', 'className'),
     Output('nav-btn-about', 'className')],
    [Input('nav-btn-detection', 'n_clicks'),
     Input('nav-btn-models', 'n_clicks'),
     Input('nav-btn-about', 'n_clicks')],
)

# ─── Threshold Display (clientside — instant) ───────────────────
app.clientside_callback(
    """
    function(value) { return value.toFixed(2); }
    """,
    Output('threshold-value', 'children'),
    Input('threshold-slider', 'value'),
)

# ─── Show File Info on Upload ───────────────────────────────────
@app.callback(
    Output('file-info', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def show_file_info(contents, filename):
    if contents is None:
        return None

    try:
        content_string = contents.split(',')[1]
        decoded = base64.b64decode(content_string)
        file_size = len(decoded)

        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.2f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.2f} MB"

        file_info = html.Div([
            html.Div([
                html.Span("✅ ", style={'marginRight': '0.5rem'}),
                html.Span("File uploaded successfully", style={'color': '#4ADE80', 'fontWeight': '600'}),
            ], style={'marginBottom': '0.5rem'}),
            html.Div([
                html.Span(f"📄 {filename}", style={'color': '#fff', 'fontWeight': '500'}),
                html.Span(f"  •  {size_str}", style={'color': 'rgba(255,255,255,0.5)', 'marginLeft': '0.75rem'}),
            ]),
        ], className="file-info-box")

        return file_info
    except Exception:
        return html.Div("File uploaded", style={'color': '#4ADE80'})

# ─── Whitelist confirmation flow ────────────────────────────────
# Handles: scan-button click, modal Confirm, modal Cancel
@app.callback(
    [Output('wl-confirm-modal', 'is_open'),
     Output('_do-analysis', 'data'),
     Output('whitelist-input', 'value')],
    [Input('scan-button', 'n_clicks'),
     Input('wl-confirm-btn', 'n_clicks'),
     Input('wl-cancel-btn', 'n_clicks')],
    [State('whitelist-input', 'value'),
     State('_do-analysis', 'data')],
    prevent_initial_call=True,
)
def handle_scan_flow(scan_clicks, confirm_clicks, cancel_clicks,
                     whitelist_text, current_count):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    cnt = (current_count or 0)

    if trigger == 'scan-button':
        has_wl = bool(whitelist_text and whitelist_text.strip())
        if has_wl:
            # Whitelist not empty → show confirmation modal
            return True, no_update, no_update
        # Whitelist empty → skip modal, trigger analysis directly
        return False, cnt + 1, no_update

    if trigger == 'wl-confirm-btn':
        # User confirmed → close modal + trigger analysis
        return False, cnt + 1, no_update

    if trigger == 'wl-cancel-btn':
        # User cancelled → close modal + clear whitelist
        return False, no_update, ''

    return no_update, no_update, no_update

# ─── Process File & Generate Results ────────────────────────────
@app.callback(
    [Output('results-section', 'children'),
     Output('stored-data', 'data'),
     Output('scan-metrics', 'data'),
     Output('_bl-csv-data',   'data'),
     Output('_fw-linux-data', 'data'),
     Output('_fw-win-data',   'data'),
     Output('_full-csv-data', 'data')],
    [Input('_do-analysis', 'data')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename'),
     State('threshold-slider', 'value'),
     State('whitelist-input', 'value')],
    prevent_initial_call=True,
)
def process_file(analysis_trigger, contents, filename, threshold, whitelist_text):
    if not analysis_trigger or contents is None:
        return (no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    try:
        # Parse file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(decoded))
        else:
            return (html.Div("❌ Unsupported file format", className="error-msg"),
                    None, no_update, no_update, no_update, no_update, no_update)

        raw_count = len(df)

        # ── Pre-validate: check minimum required columns ──
        REQUIRED_COLS = ['Src IP']
        RECOMMENDED_COLS = ['Dst IP', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
                             'SYN Flag Cnt', 'ACK Flag Cnt', 'Flow IAT Mean', 'Idle Mean']
        df.columns = df.columns.str.strip()  # normalise header whitespace
        missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing_required:
            missing_rec = [c for c in RECOMMENDED_COLS if c not in df.columns]
            err_lines = [
                html.Div([
                    html.Span("❌ ", style={'fontSize': '1.4rem'}),
                    html.Span("Wrong file format",
                              style={'fontWeight': '700', 'color': '#EF4444', 'fontSize': '1rem'}),
                ], style={'marginBottom': '0.75rem', 'display': 'flex', 'alignItems': 'center'}),
                html.P(
                    f"File '{filename}' is missing required columns: {', '.join(missing_required)}.",
                    style={'color': 'rgba(255,255,255,0.75)', 'marginBottom': '0.5rem'}
                ),
                html.P(
                    "Please upload a raw network traffic CSV file (CIC-FlowMeter format) — "
                    "NOT an exported analysis result or blacklist file.",
                    style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '0.85rem', 'marginBottom': '0.5rem'}
                ),
            ]
            if missing_rec:
                err_lines.append(html.P(
                    f"Missing columns (optional but recommended): {', '.join(missing_rec)}",
                    style={'color': 'rgba(251,191,36,0.85)', 'fontSize': '0.8rem'}
                ))
            return (html.Div(err_lines, className="error-msg", style={'padding': '1.2rem'}),
                    None, no_update, no_update, no_update, no_update, no_update)

        # Process features
        processed_df = build_features(df)

        # Predict
        if model is not None:
            X_input = processed_df[feature_cols]
            probs = model.predict_proba(X_input)[:, 1]
            predictions = (probs > threshold).astype(int)

            processed_df['is_attacker'] = predictions
            processed_df['attack_probability'] = probs
            processed_df['Status'] = processed_df['is_attacker'].map({0: 'Normal', 1: 'Malicious'})

            # ── WHITELIST: override ML prediction for trusted IPs ──
            whitelist_ips = []
            if whitelist_text:
                whitelist_ips = [ip.strip() for ip in whitelist_text.strip().split('\n') if ip.strip()]
            if whitelist_ips:
                is_wl = processed_df['Src IP'].isin(whitelist_ips)
                processed_df.loc[is_wl, 'is_attacker'] = 0
                processed_df.loc[is_wl, 'attack_probability'] = 0.0
                processed_df.loc[is_wl, 'Status'] = 'Whitelisted'
        else:
            return (html.Div("❌ Model not loaded. Please check model path.", className="error-msg"),
                    None, no_update, no_update, no_update, no_update, no_update)

        # ────── CALCULATE METRICS ──────
        total_ips = len(processed_df)
        attackers = processed_df[processed_df['is_attacker'] == 1]
        attacker_count = len(attackers)
        benign_count = total_ips - attacker_count
        attack_percent = (attacker_count / total_ips) * 100 if total_ips > 0 else 0
        avg_conf = attackers['attack_probability'].mean() if not attackers.empty else 0

        # Risk levels
        processed_df['Risk Level'] = pd.cut(
            processed_df['attack_probability'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

        # Re-slice attackers AFTER Risk Level is assigned so charts see the column
        attackers = processed_df[processed_df['is_attacker'] == 1]

        # Risk breakdown counts (for threat tooltip)
        risk_series = processed_df[processed_df['is_attacker'] == 1]['Risk Level']
        rb_low      = int((risk_series == 'Low').sum())
        rb_medium   = int((risk_series == 'Medium').sum())
        rb_high     = int((risk_series == 'High').sum())
        rb_critical = int((risk_series == 'Critical').sum())

        # Threat level label
        threat_level = (
            'Critical' if attack_percent > 50 else
            'High'     if attack_percent > 20 else
            'Medium'   if attack_percent > 5  else
            'Low'      if attacker_count > 0  else
            'Secure'
        )
        threat_color = {
            'Critical': '#EF4444', 'High': '#F97316',
            'Medium': '#FBBF24',  'Low': '#4ADE80', 'Secure': '#4ADE80'
        }[threat_level]

        # ────── VICTIM ANALYTICS (flow-level, based on attacker Source IPs) ──────
        victim_summary = pd.DataFrame(columns=['Dst IP', 'attacked_flows', 'unique_attackers', 'attack_type'])
        ddos_victims = pd.DataFrame(columns=['Dst IP', 'attacked_flows', 'unique_attackers', 'attack_type'])
        top_victims = pd.DataFrame(columns=['Dst IP', 'attacked_flows', 'unique_attackers', 'attack_type'])

        if attacker_count > 0 and 'Src IP' in df.columns and 'Dst IP' in df.columns:
            attack_src_set = set(attackers['Src IP'].astype(str).tolist())
            attack_flows = df[['Src IP', 'Dst IP']].copy()
            attack_flows['Src IP'] = attack_flows['Src IP'].astype(str)
            attack_flows['Dst IP'] = attack_flows['Dst IP'].astype(str)
            attack_flows = attack_flows[attack_flows['Src IP'].isin(attack_src_set)]

            if not attack_flows.empty:
                # Layer-2 correlation: classify by unique attacker count per victim
                victim_summary = (
                    attack_flows.groupby('Dst IP', as_index=False)
                    .agg(
                        attacked_flows=('Dst IP', 'size'),
                        unique_attackers=('Src IP', 'nunique'),
                    )
                    .sort_values(['attacked_flows', 'unique_attackers'], ascending=[False, False])
                )
                # 1 attacker → DoS, >1 attackers → DDoS
                victim_summary['attack_type'] = victim_summary['unique_attackers'].apply(
                    lambda x: 'DoS' if x == 1 else 'DDoS'
                )
                ddos_victims = victim_summary[victim_summary['attack_type'] == 'DDoS'].copy()
                top_victims = victim_summary.head(10).copy()

        # Generate firewall scripts for BOTH platforms (user picks OS after analysis)
        _atk_df = attackers if attacker_count > 0 else pd.DataFrame()
        _vic_ips = top_victims['Dst IP'].astype(str).tolist() if not top_victims.empty else []
        fw_script_linux   = build_firewall_script(_atk_df, _vic_ips, 'linux')
        fw_script_windows = build_firewall_script(_atk_df, _vic_ips, 'windows')

        # ────── BUILD DETECTION RESULTS ──────
        results = html.Div([
            html.Hr(className="section-divider"),

            # ── Analysis-complete banner ──
            html.Div([
                html.Span("✅", style={'fontSize': '1.3rem', 'marginRight': '0.6rem'}),
                html.Span("Analysis Complete! ",
                          style={'fontWeight': '700', 'color': '#4ADE80'}),
                html.Span("Full results displayed below.",
                          style={'color': 'rgba(255,255,255,0.6)'}),
            ], className="analysis-complete-banner"),

            # Results Header
            html.Div([
                html.H2("Scan Results", className="results-title"),
                html.P(f"Analyzed {raw_count:,} flows from {filename}", className="results-subtitle"),
            ], className="results-header"),

            # Result Metrics
            dbc.Row([
                # ── Total Source IPs ──
                dbc.Col(html.Div([
                    html.Div("TOTAL SOURCE IPS", className="r-metric-label"),
                    html.Div(f"{total_ips:,}", className="r-metric-value"),
                    html.Div("Nodes Scanned", className="r-metric-sub"),
                ], className="r-metric-card"), width=3),

                # ── Attackers (with risk breakdown) ──
                dbc.Col(html.Div([
                    html.Div("ATTACKERS", className="r-metric-label"),
                    html.Div(f"{attacker_count:,}", className="r-metric-value",
                             style={'color': '#EF4444' if attacker_count > 0 else '#4ADE80'}),
                    html.Div(f"{attack_percent:.1f}% of total", className="r-metric-sub"),
                    # Risk breakdown row (always visible under the count)
                    html.Div([
                        html.Span([
                            html.Span("● ", style={'color': '#4ADE80'}),
                            html.Span(f"Low: {rb_low}"),
                        ], style={'marginRight': '0.6rem', 'fontSize': '0.72rem',
                                  'color': 'rgba(255,255,255,0.65)'}),
                        html.Span([
                            html.Span("● ", style={'color': '#FBBF24'}),
                            html.Span(f"Med: {rb_medium}"),
                        ], style={'marginRight': '0.6rem', 'fontSize': '0.72rem',
                                  'color': 'rgba(255,255,255,0.65)'}),
                        html.Span([
                            html.Span("● ", style={'color': '#F97316'}),
                            html.Span(f"High: {rb_high}"),
                        ], style={'marginRight': '0.6rem', 'fontSize': '0.72rem',
                                  'color': 'rgba(255,255,255,0.65)'}),
                        html.Span([
                            html.Span("● ", style={'color': '#EF4444'}),
                            html.Span(f"Crit: {rb_critical}"),
                        ], style={'fontSize': '0.72rem', 'color': 'rgba(255,255,255,0.65)'}),
                    ], style={'marginTop': '0.4rem', 'display': 'flex', 'flexWrap': 'wrap',
                              'gap': '0.1rem'}) if attacker_count > 0 else None,
                ], className="r-metric-card"), width=3),

                # ── Network Status ──
                dbc.Col(html.Div([
                    html.Div("NETWORK STATUS", className="r-metric-label"),
                    html.Div(threat_level.upper(), style={
                        'fontSize': '1.15rem',
                        'fontWeight': '700',
                        'color': threat_color,
                        'border': f'2px solid {threat_color}',
                        'borderRadius': '999px',
                        'padding': '4px 18px',
                        'display': 'inline-block',
                        'marginTop': '0.35rem',
                        'letterSpacing': '0.08em',
                        'background': f'{threat_color}18',
                    }),
                ], className="r-metric-card"), width=3),

                # ── Avg Threat Confidence ──
                dbc.Col(html.Div([
                    html.Div("AVG THREAT CONFIDENCE", className="r-metric-label"),
                    html.Div(f"{avg_conf:.0%}", className="r-metric-value"),
                    html.Div(
                        "High certainty" if avg_conf >= 0.75 else
                        "Moderate certainty" if avg_conf >= 0.5 else
                        "Low certainty",
                        className="r-metric-sub"
                    ),
                ], className="r-metric-card"), width=3),
            ], className="results-metrics-row"),

            # Charts
            html.Div([
                html.H3("📊 Traffic Distribution", className="section-heading"),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=create_traffic_chart(processed_df))], width=6),
                    dbc.Col([dcc.Graph(figure=create_risk_chart(attackers))], width=6),
                ]),
            ], className="charts-section"),

            # ── Behavioral Feature Analysis (new 11-feature model) ──
            html.Div([
                html.H3("🧠 Behavioral Feature Analysis", className="section-heading"),
                html.P(
                    "Radar comparison of 11 behavioral features between attackers and normal traffic. "
                    "Features farther from center indicate higher anomaly signals.",
                    style={'color': 'rgba(255,255,255,0.4)', 'fontSize': '0.82rem',
                           'marginBottom': '1rem', 'marginTop': '-0.5rem'}
                ),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=create_behavior_radar(processed_df))
                    ], width=7),
                    dbc.Col([
                        _build_feature_insights(attackers, processed_df),
                    ], width=5),
                ]),
            ], className="charts-section") if attacker_count > 0 else None,

            # Top Threats
            html.Div([
                html.H3("🔥 Top 10 Most Dangerous IPs", className="section-heading"),
                dcc.Graph(figure=create_threats_chart(processed_df)),
            ], className="charts-section") if attacker_count > 0 else html.Div(
                "✅ No threats detected", className="no-threats-msg"
            ),

            # Top victim chart
            html.Div([
                html.H3("🎯 Top 10 Most Targeted Servers (Top Victims)", className="section-heading"),
                dcc.Graph(figure=create_top_victims_chart(top_victims)),
            ], className="charts-section") if not top_victims.empty else None,

            # Data Tables
            dbc.Tabs([
                dbc.Tab(
                    build_blacklist_tab(attackers, attacker_count, threshold,
                                       fw_script_linux, fw_script_windows),
                    label="🚨 BLACKLIST",
                    tab_id="tab-bl",
                ),
                dbc.Tab(
                    build_victims_tab(ddos_victims, victim_summary),
                    label="🎯 VICTIMS",
                    tab_id="tab-victims",
                ),
                dbc.Tab(
                    build_full_tab(processed_df),
                    label="📊 FULL REPORT",
                    tab_id="tab-full",
                ),
            ], id="result-tabs", active_tab="tab-bl", className="result-tabs"),
        ])

        scan_data = {
            'total_flows': raw_count,
            'total_ips': total_ips,
            'attacker_count': attacker_count,
            'benign_count': benign_count,
            'attack_percent': attack_percent,
            'avg_conf': avg_conf,
        }

        # ── Prepare download payloads ──
        # Blacklist CSV (same columns shown in the blacklist tab table)
        bl_csv_str = None
        if attacker_count > 0:
            base_cols = ['Src IP', 'attack_probability']
            optional_cols = [
                'pkt_rate', 'byte_rate', 'syn_ack_ratio',
                'mean_iat', 'size_consistency', 'active_duration_sec',
            ]
            _bl_cols = base_cols + [c for c in optional_cols if c in attackers.columns]
            bl_csv_str = attackers[_bl_cols].sort_values(
                'attack_probability', ascending=False
            ).to_csv(index=False)

        return (results,
                processed_df.to_dict('records'), scan_data,
                bl_csv_str,
                fw_script_linux or None,
                fw_script_windows or None,
                processed_df.to_csv(index=False))

    except Exception as e:
        return (html.Div(f"❌ Error: {str(e)}", className="error-msg"),
                None, no_update, no_update, no_update, no_update, no_update)

# =============================================================================
# 7. CHART HELPER FUNCTIONS (callback charts)
# =============================================================================

# ── Feature labels for radar & insights ──
_RADAR_FEATURES = [
    ('pkt_rate',           'Pkt Rate'),
    ('byte_rate',          'Byte Rate'),
    ('syn_ack_ratio',      'SYN/ACK'),
    ('pkt_ratio',          'Pkt Ratio'),
    ('payload_ratio',      'Payload Ratio'),
    ('dst_port_ratio',     'Dst Port Ratio'),
    ('mean_iat',           'Mean IAT'),
    ('avg_idle',           'Avg Idle'),
    ('size_consistency',   'Size Consist.'),
    ('dst_ip_ratio',       'Dst IP Ratio'),
    ('active_duration_sec','Active Dur'),
]

def create_behavior_radar(df):
    """Radar chart comparing mean feature values: Attackers vs Normal.
    Values are min-max normalized per feature so all axes share [0,1] scale."""
    atk = df[df['is_attacker'] == 1]
    nor = df[df['is_attacker'] == 0]

    cols   = [c for c, _ in _RADAR_FEATURES if c in df.columns]
    labels = [l for c, l in _RADAR_FEATURES if c in df.columns]

    atk_means = atk[cols].mean() if not atk.empty else pd.Series(0, index=cols)
    nor_means = nor[cols].mean() if not nor.empty else pd.Series(0, index=cols)

    # Min-max normalize so radar is readable
    combined = pd.DataFrame({'atk': atk_means, 'nor': nor_means})
    mn = combined.min(axis=1)
    mx = combined.max(axis=1)
    rng = (mx - mn).replace(0, 1)
    atk_norm = ((combined['atk'] - mn) / rng).tolist()
    nor_norm = ((combined['nor'] - mn) / rng).tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=atk_norm + [atk_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(239,68,68,0.12)',
        line=dict(color='#EF4444', width=2),
        name='Attackers',
    ))
    fig.add_trace(go.Scatterpolar(
        r=nor_norm + [nor_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(74,222,128,0.08)',
        line=dict(color='#4ADE80', width=2),
        name='Normal',
    ))
    radar_layout = {k: v for k, v in CHART_LAYOUT.items() if k != 'margin'}
    fig.update_layout(
        **radar_layout,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor='rgba(255,255,255,0.08)',
                tickfont=dict(size=9, color='rgba(255,255,255,0.3)'),
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.08)',
                tickfont=dict(size=10, color='rgba(255,255,255,0.65)'),
            ),
        ),
        legend=dict(orientation='h', yanchor='bottom', y=-0.18,
                    xanchor='center', x=0.5,
                    font=dict(size=11)),
        margin=dict(t=30, b=60, l=60, r=60),
        height=380,
    )
    return fig


def _build_feature_insights(attackers, full_df):
    """Build mini insight cards highlighting what the 11-feature model detected."""
    normals = full_df[full_df['is_attacker'] == 0]

    # Define insight rules: (feature, label, unit, description, higher_is_bad)
    INSIGHT_DEFS = [
        ('pkt_rate',        'Packet Rate',      'pkt/s',  'Volumetric flood signal',       True),
        ('byte_rate',       'Byte Rate',        'B/s',    'Bandwidth saturation',          True),
        ('syn_ack_ratio',   'SYN/ACK Ratio',    '',       'SYN Flood indicator',           True),
        ('mean_iat',        'Mean IAT',         's',      'Inter-arrival timing anomaly',  False),
        ('size_consistency','Size Consistency', '',       'Botnet-like uniform packets',   False),
        ('payload_ratio',   'Payload Ratio',    'B/pkt',  'Empty vs heavy payloads',       True),
    ]

    cards = []
    for feat, label, unit, desc, higher_bad in INSIGHT_DEFS:
        if feat not in full_df.columns:
            continue
        atk_val = attackers[feat].mean() if not attackers.empty else 0
        nor_val = normals[feat].mean() if not normals.empty else 0

        if nor_val != 0:
            diff_pct = ((atk_val - nor_val) / abs(nor_val)) * 100
        else:
            diff_pct = 100 if atk_val > 0 else 0

        is_anomaly = (diff_pct > 30 and higher_bad) or (diff_pct < -30 and not higher_bad)
        arrow = '↑' if diff_pct > 0 else '↓'
        color = '#EF4444' if is_anomaly else '#4ADE80'

        cards.append(
            html.Div([
                html.Div([
                    html.Span(label, style={
                        'color': 'rgba(255,255,255,0.6)', 'fontSize': '0.75rem',
                        'fontWeight': '500', 'textTransform': 'uppercase',
                        'letterSpacing': '0.03em',
                    }),
                    html.Span(f"{arrow} {abs(diff_pct):.0f}%", style={
                        'color': color, 'fontSize': '0.75rem', 'fontWeight': '700',
                    }),
                ], style={'display': 'flex', 'justifyContent': 'space-between',
                          'alignItems': 'center', 'marginBottom': '0.25rem'}),
                html.Div([
                    html.Span(f"Atk: {atk_val:.2f}{unit}", style={
                        'color': '#EF4444', 'fontSize': '0.82rem', 'fontWeight': '600',
                    }),
                    html.Span("  vs  ", style={'color': 'rgba(255,255,255,0.25)', 'fontSize': '0.75rem'}),
                    html.Span(f"Norm: {nor_val:.2f}{unit}", style={
                        'color': '#4ADE80', 'fontSize': '0.82rem', 'fontWeight': '600',
                    }),
                ]),
                html.Div(desc, style={
                    'color': 'rgba(255,255,255,0.3)', 'fontSize': '0.7rem',
                    'marginTop': '0.15rem',
                }),
            ], className="feat-insight-card")
        )

    return html.Div(cards, className="feat-insight-grid")
def create_traffic_chart(df):
    """Pie chart: Normal vs Malicious"""
    counts = df['Status'].value_counts().reset_index()
    counts.columns = ['Status', 'Count']
    fig = px.pie(counts, values='Count', names='Status', color='Status',
                 color_discrete_map={'Normal': '#4ADE80', 'Malicious': '#EF4444'},
                 hole=0.55, title="Traffic Classification")
    fig.update_layout(**CHART_LAYOUT)
    return fig

def create_risk_chart(df):
    """Bar chart: Risk level distribution — df must be the ATTACKERS-only subset.
    This ensures numbers match exactly what the DDOS ATTACKERS metric card shows."""
    if df is None or df.empty or 'Risk Level' not in df.columns:
        # Return an empty chart with a message when no attackers detected
        fig = go.Figure()
        fig.update_layout(
            **CHART_LAYOUT,
            title="Risk Level Distribution (No Attackers)",
            annotations=[dict(text="No attackers detected", showarrow=False,
                              font=dict(color='rgba(255,255,255,0.35)', size=14),
                              xref="paper", yref="paper", x=0.5, y=0.5)]
        )
        return fig

    # Enforce display order: Low → Medium → High → Critical
    LEVEL_ORDER   = ['Low', 'Medium', 'High', 'Critical']
    LEVEL_COLORS  = {'Low': '#4ADE80', 'Medium': '#FBBF24',
                     'High': '#F97316', 'Critical': '#EF4444'}

    risk_counts = (
        df['Risk Level']
        .value_counts()
        .reindex(LEVEL_ORDER, fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ['Risk Level', 'Count']
    # Remove levels with zero count for a cleaner chart
    risk_counts = risk_counts[risk_counts['Count'] > 0]

    fig = px.bar(risk_counts, x='Risk Level', y='Count', color='Risk Level',
                 color_discrete_map=LEVEL_COLORS,
                 category_orders={'Risk Level': LEVEL_ORDER},
                 title="Attacker Risk Level Distribution",
                 text='Count')
    fig.update_traces(textposition='outside', textfont=dict(color='rgba(255,255,255,0.7)', size=12))
    fig.update_layout(**CHART_LAYOUT, showlegend=False,
                      xaxis=dict(showgrid=False, categoryorder='array',
                                 categoryarray=LEVEL_ORDER),
                      yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'))
    return fig

def create_threats_chart(df):
    """Bar chart: Top 10 most dangerous IPs with rich hover tooltips on mouse-over."""
    top = df.nlargest(10, 'attack_probability').copy()

    # --- Define which columns to surface in the tooltip (label, format) ---
    HOVER_COLS = [
        ('flows_per_sec',       'Flows / sec',       '{:.2f}'),
        ('pkt_rate',            'Packet Rate',       '{:.2f} pkt/s'),
        ('byte_rate',           'Byte Rate',         '{:.2f} B/s'),
        ('syn_ack_ratio',       'SYN/ACK Ratio',     '{:.4f}'),
        ('dst_ip_ratio',        'Dst IP Ratio',      '{:.4f}'),
        ('dst_port_ratio',      'Dst Port Ratio',    '{:.4f}'),
        ('active_duration_sec', 'Active Duration',   '{:.1f} s'),
        ('mean_iat',            'Mean IAT',          '{:.4f}'),
        ('Status',              'Status',            '{}'),
        ('Risk Level',          'Risk Level',        '{}'),
    ]
    available = [(col, lbl, fmt) for col, lbl, fmt in HOVER_COLS if col in top.columns]

    # --- Build customdata: pre-format every value as a display string ---
    custom_rows = []
    for _, row in top.iterrows():
        cell = []
        for col, lbl, fmt in available:
            try:
                cell.append(fmt.format(row[col]) if '{}' not in fmt else str(row[col]))
            except Exception:
                cell.append(str(row.get(col, 'N/A')))
        custom_rows.append(cell)
    custom_array = np.array(custom_rows) if custom_rows else None

    # --- Build hovertemplate that fires when user hovers over each bar ---
    tip_lines = [
        "<b style='font-size:13px;color:#FCA5A5'>⚠️  %{x}</b>",
        "Attack Probability: <b style='color:#EF4444'>%{y:.2%}</b>",
        "<span style='color:rgba(255,255,255,0.25)'>────────────────────</span>",
    ]
    for i, (col, lbl, _) in enumerate(available):
        tip_lines.append(f"<span style='color:rgba(255,255,255,0.55)'>{lbl}:</span>  <b>%{{customdata[{i}]}}</b>")
    tip_lines.append("<extra></extra>")
    hovertemplate = "<br>".join(tip_lines)

    fig = go.Figure(go.Bar(
        x=top['Src IP'],
        y=top['attack_probability'],
        customdata=custom_array,
        hovertemplate=hovertemplate,
        marker=dict(
            color=top['attack_probability'],
            colorscale=[[0, '#F97316'], [0.5, '#EF4444'], [1, '#B91C1C']],
            showscale=True,
            colorbar=dict(
                tickformat='.0%',
                title=dict(text='Risk', font=dict(color='rgba(255,255,255,0.55)', size=11)),
                tickfont=dict(color='rgba(255,255,255,0.55)'),
                len=0.9,
            ),
        ),
        hoverlabel=dict(
            bgcolor='rgba(8,12,30,0.97)',
            bordercolor='rgba(239,68,68,0.55)',
            font=dict(color='#E2E8F0', size=12, family='Inter, system-ui, sans-serif'),
            namelength=0,
        ),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        showlegend=False,
        xaxis=dict(showgrid=False, tickangle=-30, tickfont=dict(size=11)),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.06)',
            tickformat='.0%',
        ),
        hoverdistance=50,
    )
    return fig


def create_top_victims_chart(top_victims_df):
    """Bar chart for top targeted destination servers (victims)."""
    if top_victims_df is None or top_victims_df.empty:
        fig = go.Figure()
        fig.update_layout(
            **CHART_LAYOUT,
            title="Top Victims (No data)",
            annotations=[dict(
                text="No victim targets found",
                showarrow=False,
                font=dict(color='rgba(255,255,255,0.35)', size=14),
                xref="paper", yref="paper", x=0.5, y=0.5
            )]
        )
        return fig

    color_map = {'DDoS': '#DC2626', 'DoS': '#F59E0B', 'Unknown': '#64748B'}
    fig = px.bar(
        top_victims_df,
        x='Dst IP',
        y='attacked_flows',
        color='attack_type',
        color_discrete_map=color_map,
        title="Number of attack flows per destination server",
        text='attacked_flows',
        labels={'Dst IP': 'Destination IP', 'attacked_flows': 'Attack Flows'}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)')
    )
    return fig


def build_top_victims_table(top_victims_df):
    """Compact table for top 10 victims with attack intensity fields."""
    if top_victims_df is None or top_victims_df.empty:
        return html.Div("No victim targets to display.", className="tab-safe")

    show = top_victims_df.copy().reset_index(drop=True)
    show.insert(0, 'Rank', show.index + 1)
    show.rename(columns={
        'Dst IP': 'Victim Server IP',
        'attacked_flows': 'Attack Flows',
        'unique_attackers': 'Unique Attackers',
        'attack_type': 'Attack Type'
    }, inplace=True)

    return html.Div(
        dbc.Table.from_dataframe(show, striped=True, bordered=True, hover=True, size='sm', className='custom-table'),
        style={'overflowX': 'auto', 'maxWidth': '100%'}
    )


def build_firewall_script(attackers_df, victim_ips, os_choice='linux'):
    """Generate platform-specific firewall mitigation script.

    Linux  → iptables + ipset (hybrid: block high-risk, rate-limit suspects)
    Windows → PowerShell NetFirewallRule
    """
    if attackers_df is None or attackers_df.empty:
        return '# No attackers detected — nothing to block.\n'

    if os_choice == 'windows':
        return _fw_windows(attackers_df, victim_ips)
    return _fw_linux(attackers_df, victim_ips)


def _fw_linux(attackers_df, victim_ips):
    lines = [
        '#!/bin/bash',
        '# Auto-generated Hybrid Defense Script — ML-DDoS Detector',
        '# Review before applying in production',
        '',
        'set -e',
        '',
        "echo 'Initialising multi-layer defence (iptables + ipset)...'",
        '',
        '# Install ipset if missing',
        'if ! command -v ipset &>/dev/null; then sudo apt-get install -y ipset || sudo dnf install -y ipset; fi',
        '',
        'ipset destroy ddos_blacklist 2>/dev/null || true',
        'ipset create ddos_blacklist hash:ip hashsize 4096 maxelem 200000',
        '',
        '# --- Rules classified by ML risk score ---',
    ]

    for _, row in attackers_df.iterrows():
        ip = str(row['Src IP'])
        prob = float(row.get('attack_probability', 1.0))
        if prob > 0.80:
            lines.append(f"ipset add ddos_blacklist {ip}  # Risk: {prob:.2%}")
        else:
            lines.append(f"iptables -A INPUT -s {ip} -m limit --limit 20/sec --limit-burst 50 -j ACCEPT  # Rate-limit: {prob:.2%}")
            lines.append(f"iptables -A INPUT -s {ip} -j DROP  # Drop excess")

    lines += [
        '',
        '# --- Activate ipset permanent block ---',
        'iptables -D INPUT -m set --match-set ddos_blacklist src -j DROP 2>/dev/null || true',
        'iptables -I INPUT 1 -m set --match-set ddos_blacklist src -j DROP',
    ]

    lines += ['', "echo 'Mitigation rules applied successfully.'", '']
    return '\n'.join(lines)


def _fw_windows(attackers_df, victim_ips):
    lines = [
        '# Auto-generated Windows Firewall Rules — ML-DDoS Detector',
        '# Run in PowerShell as Administrator',
        '',
        "Write-Host 'Initialising Windows Server defence...' -ForegroundColor Cyan",
        '',
        "Remove-NetFirewallRule -DisplayName 'DDoS_BlockList*' -ErrorAction SilentlyContinue",
        '',
    ]

    high_risk = attackers_df[attackers_df['attack_probability'] > 0.80]['Src IP'].astype(str).tolist()
    suspects  = attackers_df[attackers_df['attack_probability'] <= 0.80]['Src IP'].astype(str).tolist()

    if high_risk:
        ip_str = ', '.join([f"'{ip}'" for ip in high_risk])
        lines += [
            '# 1. High-Risk IPs (>80%) — permanent block',
            f'$HighRiskIPs = @({ip_str})',
            "New-NetFirewallRule -DisplayName 'DDoS_BlockList_High' -Direction Inbound -Action Block -RemoteAddress $HighRiskIPs",
            '',
        ]

    if suspects:
        ip_str = ', '.join([f"'{ip}'" for ip in suspects])
        lines += [
            '# 2. Suspect IPs (<=80%) — temporary block (review recommended)',
            '# NOTE: Windows Firewall does not support native rate-limiting.',
            f'$SuspiciousIPs = @({ip_str})',
            "New-NetFirewallRule -DisplayName 'DDoS_BlockList_Medium' -Direction Inbound -Action Block -RemoteAddress $SuspiciousIPs",
            '',
        ]

    lines += ["Write-Host 'Firewall rules applied.' -ForegroundColor Green", '']
    return '\n'.join(lines)


def build_victims_tab(ddos_victims, all_victims):
    """Victim-focused tab: DDoS target list and aggregate targets."""
    ddos_view = ddos_victims.copy() if ddos_victims is not None else pd.DataFrame()
    all_view = all_victims.copy() if all_victims is not None else pd.DataFrame()

    if not ddos_view.empty:
        ddos_view = ddos_view.reset_index(drop=True)
        ddos_view.insert(0, 'No.', ddos_view.index + 1)
    if not all_view.empty:
        all_view = all_view.reset_index(drop=True)
        all_view.insert(0, 'No.', all_view.index + 1)

    ddos_title = "🚨 DDOS ATTACK TARGET LIST (DISTRIBUTED)"
    all_title = "🎯 ALL ATTACK TARGETS SUMMARY"

    # Rename columns for display
    col_map = {'unique_attackers': 'Number of Attackers', 'attack_type': 'Attack Type'}
    ddos_cols = ['No.', 'Dst IP', 'Number of Attackers', 'Attack Type']
    all_cols  = ['No.', 'Dst IP', 'Number of Attackers', 'Attack Type']

    ddos_disp = ddos_view.rename(columns=col_map) if not ddos_view.empty else pd.DataFrame(columns=ddos_cols)
    all_disp  = all_view.rename(columns=col_map)  if not all_view.empty  else pd.DataFrame(columns=all_cols)

    # Helper: build table rows with color-coded Attack Type column
    _type_colors = {'DDoS': '#EF4444', 'DoS': '#FBBF24', 'Unknown': '#64748B'}
    def _make_rows(frame, cols):
        rows = []
        for _, r in frame.iterrows():
            cells = []
            for c in cols:
                val = r.get(c, '')
                if c == 'Attack Type':
                    color = _type_colors.get(str(val), '#64748B')
                    cells.append(html.Td(
                        html.Span(str(val), style={
                            'color': color, 'fontWeight': '700',
                            'padding': '2px 10px', 'borderRadius': '6px',
                            'background': f'{color}18', 'border': f'1px solid {color}44',
                            'fontSize': '0.82rem',
                        })
                    ))
                else:
                    cells.append(html.Td(str(val)))
            rows.append(html.Tr(cells))
        return rows

    def _build_colored_table(frame, cols):
        header = html.Thead(html.Tr([html.Th(c) for c in cols]))
        body = html.Tbody(_make_rows(frame, cols) if not frame.empty else [])
        return dbc.Table([header, body], striped=True, bordered=True,
                         hover=True, size='sm', className='custom-table')

    return html.Div([
        html.Div([
            html.H4(ddos_title, style={'color': '#FCA5A5', 'fontSize': '0.95rem', 'marginBottom': '0.65rem'}),
            html.P("Destination servers receiving traffic labelled DDoS from detected sources.", className="tab-warning"),
            html.Div(
                _build_colored_table(ddos_disp, ddos_cols),
                style={'overflowX': 'auto', 'maxWidth': '100%'}
            ),
        ], style={'marginBottom': '1.5rem'}),

        html.Div([
            html.H4(all_title, style={'color': '#FDE68A', 'fontSize': '0.95rem', 'marginBottom': '0.65rem'}),
            html.P("All targeted destination servers (including DDoS, DoS, and Unknown).", className="tab-warning"),
            html.Div(
                _build_colored_table(all_disp, all_cols),
                style={'overflowX': 'auto', 'maxWidth': '100%'}
            ),
        ], style={'marginBottom': '1.5rem'}),
    ], className="tab-content-inner")

# =============================================================================
# 8. TABLE TAB FUNCTIONS
# =============================================================================
def build_blacklist_tab(attackers, count, threshold,
                       fw_linux='', fw_windows=''):
    """Build the attacker blacklist tab with embedded Mitigation Tools."""
    if count > 0:
        # Dynamically pick available columns for blacklist table
        base_cols = ['Src IP', 'attack_probability']
        optional_cols = [
            'pkt_rate', 'byte_rate', 'syn_ack_ratio',
            'mean_iat', 'size_consistency', 'active_duration_sec',
        ]
        display_cols = base_cols + [c for c in optional_cols if c in attackers.columns]

        bl = attackers[display_cols].copy()
        bl = bl.sort_values('attack_probability', ascending=False)

        bl['attack_probability'] = bl['attack_probability'].apply(lambda x: f"{x:.2%}")
        for c in ['pkt_rate', 'byte_rate', 'syn_ack_ratio']:
            if c in bl.columns:
                bl[c] = bl[c].apply(lambda x: f"{x:.2f}")
        if 'mean_iat' in bl.columns:
            bl['mean_iat'] = bl['mean_iat'].apply(lambda x: f"{x:.4f}")
        if 'size_consistency' in bl.columns:
            bl['size_consistency'] = bl['size_consistency'].apply(lambda x: f"{x:.4f}")
        if 'active_duration_sec' in bl.columns:
            bl['active_duration_sec'] = bl['active_duration_sec'].apply(lambda x: f"{x:.1f}s")

        # ── Mitigation Tools section (below attacker table) ──

        mitigation_section = html.Div([
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.08)', 'margin': '1.5rem 0 1rem'}),
            html.Div([
                html.Span("🛠️", style={'fontSize': '1.2rem', 'marginRight': '0.55rem'}),
                html.Span("INCIDENT RESPONSE TOOLS (MITIGATION TOOLS)",
                          style={'fontWeight': '700', 'fontSize': '0.95rem',
                                 'letterSpacing': '0.04em'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '0.65rem'}),

            html.P(
                "What OS is the victim server running?",
                style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '0.82rem',
                       'marginBottom': '0.6rem'}
            ),

            dbc.RadioItems(
                id='os-choice',
                options=[
                    {'label': ' Linux (iptables / ipset)', 'value': 'linux'},
                    {'label': ' Windows Server (PowerShell)', 'value': 'windows'},
                ],
                value='linux',
                inline=True,
                className='os-radio-group',
            ),

            # Two download rows — toggled by clientside callback
            html.Div([
                html.Button("📥 Download Blacklist (.csv)",
                            id='btn-dl-bl-csv-linux', n_clicks=0,
                            className="download-btn"),
                html.Button("🛡️ Download Firewall Script (.sh)",
                            id='btn-dl-fw-linux', n_clicks=0,
                            className="download-btn"),
            ], id='fw-linux-row',
               style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '0.65rem',
                      'marginTop': '0.85rem'}),

            html.Div([
                html.Button("📥 Download Blacklist (.csv)",
                            id='btn-dl-bl-csv-win', n_clicks=0,
                            className="download-btn"),
                html.Button("🛡️ Download Firewall Script (.ps1)",
                            id='btn-dl-fw-win', n_clicks=0,
                            className="download-btn"),
            ], id='fw-windows-row',
               style={'display': 'none', 'flexWrap': 'wrap', 'gap': '0.65rem',
                      'marginTop': '0.85rem'}),
        ], className='mitigation-tools-section')

        return html.Div([
            html.P(f"⚠️ {count} IPs detected with attack behavior (threshold {threshold})", className="tab-warning"),
            dbc.Table.from_dataframe(bl, striped=True, bordered=True, hover=True, size='sm', className='custom-table'),
            mitigation_section,
        ], className="tab-content-inner")
    return html.Div("✅ No threats detected. System is secure.", className="tab-safe")

def build_full_tab(df):
    """Build full dataset report tab — single table showing ALL columns from processed_df."""
    full_display = df.copy()

    return html.Div([
        html.Div([
            html.H4("Full Network Data",
                    style={'color': '#fff', 'marginBottom': '0', 'fontSize': '1rem'}),
            html.Span(f"{len(df):,} records",
                      style={'color': 'rgba(255,255,255,0.4)', 'fontSize': '0.78rem'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between',
                  'alignItems': 'center', 'marginBottom': '0.85rem'}),
        html.Div([
            html.Button("📥 Download Full Report (CSV)",
                        id='btn-dl-full-csv', n_clicks=0,
                        className="download-btn"),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '0.65rem',
                  'marginBottom': '0.85rem'}),
        html.Div(
            dbc.Table.from_dataframe(
                full_display,
                striped=True, bordered=True, hover=True,
                size='sm', className='custom-table',
                style={'tableLayout': 'auto', 'width': 'max-content', 'minWidth': '100%'},
            ),
            style={
                'overflowX': 'auto',
                'overflowY': 'hidden',
                'width': '100%',
                'maxWidth': '100%',
                '-webkit-overflow-scrolling': 'touch',
            }
        ),
    ], className="tab-content-inner")

# ─── Clientside callback: show centred toast then scroll to results ───────
# When results-section gets new children (analysis finished), pop up a
# centred overlay toast for ~3 s, then smoothly scroll to results.
app.clientside_callback(
    """
    function(children) {
        if (!children || children === null) return null;

        var toast = document.getElementById('analysis-toast');
        if (toast) {
            /* reset any previous animation state */
            toast.classList.remove('toast-hide');
            toast.classList.add('toast-show');

            /* after 2.8 s: start fade-out AND scroll simultaneously */
            setTimeout(function() {
                toast.classList.remove('toast-show');
                toast.classList.add('toast-hide');

                var el = document.getElementById('results-section');
                if (el) {
                    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }

                /* fully hide once fade-out animation has finished (0.45 s) */
                setTimeout(function() {
                    toast.classList.remove('toast-hide');
                }, 500);
            }, 2800);
        }
        return null;
    }
    """,
    Output('_toast-trigger', 'data'),
    Input('results-section', 'children'),
    prevent_initial_call=True,
)

# ─── Clientside callback: toggle Linux / Windows download rows ───
app.clientside_callback(
    """
    function(os) {
        if (os === 'windows') {
            return [{display: 'none'}, {display: 'flex', flexWrap: 'wrap', gap: '0.65rem', marginTop: '0.85rem'}];
        }
        return [{display: 'flex', flexWrap: 'wrap', gap: '0.65rem', marginTop: '0.85rem'}, {display: 'none'}];
    }
    """,
    [Output('fw-linux-row', 'style'),
     Output('fw-windows-row', 'style')],
    Input('os-choice', 'value'),
)

# ─── Clientside helper: Blob-based file download (bypasses React data: URI block) ───
_BLOB_DOWNLOAD_JS = """
    function(n, content) {
        if (!n || !content) return window.dash_clientside.no_update;
        var blob = new Blob([content], {type: '%MIME%'});
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href = url;  a.download = '%FNAME%';
        document.body.appendChild(a);  a.click();
        setTimeout(function() { document.body.removeChild(a); URL.revokeObjectURL(url); }, 200);
        return window.dash_clientside.no_update;
    }
"""

# Blacklist CSV  —  Linux row button
app.clientside_callback(
    _BLOB_DOWNLOAD_JS.replace('%MIME%', 'text/csv').replace('%FNAME%', 'ddos_blacklist.csv'),
    Output('btn-dl-bl-csv-linux', 'title'),
    Input('btn-dl-bl-csv-linux', 'n_clicks'),
    State('_bl-csv-data', 'data'),
    prevent_initial_call=True,
)

# Blacklist CSV  —  Windows row button
app.clientside_callback(
    _BLOB_DOWNLOAD_JS.replace('%MIME%', 'text/csv').replace('%FNAME%', 'ddos_blacklist.csv'),
    Output('btn-dl-bl-csv-win', 'title'),
    Input('btn-dl-bl-csv-win', 'n_clicks'),
    State('_bl-csv-data', 'data'),
    prevent_initial_call=True,
)

# Firewall script  —  Linux (.sh)
app.clientside_callback(
    _BLOB_DOWNLOAD_JS.replace('%MIME%', 'text/plain').replace('%FNAME%', 'ddos_firewall_mitigation.sh'),
    Output('btn-dl-fw-linux', 'title'),
    Input('btn-dl-fw-linux', 'n_clicks'),
    State('_fw-linux-data', 'data'),
    prevent_initial_call=True,
)

# Firewall script  —  Windows (.ps1)
app.clientside_callback(
    _BLOB_DOWNLOAD_JS.replace('%MIME%', 'text/plain').replace('%FNAME%', 'ddos_firewall_mitigation.ps1'),
    Output('btn-dl-fw-win', 'title'),
    Input('btn-dl-fw-win', 'n_clicks'),
    State('_fw-win-data', 'data'),
    prevent_initial_call=True,
)

# Full report CSV
app.clientside_callback(
    _BLOB_DOWNLOAD_JS.replace('%MIME%', 'text/csv').replace('%FNAME%', 'full_network_traffic_analysis.csv'),
    Output('btn-dl-full-csv', 'title'),
    Input('btn-dl-full-csv', 'n_clicks'),
    State('_full-csv-data', 'data'),
    prevent_initial_call=True,
)

# =============================================================================
# 9. RUN
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=8051, use_reloader=False)
