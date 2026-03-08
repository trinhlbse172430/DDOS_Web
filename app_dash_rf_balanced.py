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

# =============================================================================
# 1. KHỞI TẠO APP
# =============================================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
    
)
server = app.server
app.title = "ML-DDoS Detector"

# =============================================================================
# 2. TẢI MODEL 
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
# 3. XỬ LÝ DỮ LIỆU 
# =============================================================================

def build_features(df):
    """Transform raw data into 11-feature behavioral profile per Source IP.

    Synchronized with the Random Forest model trained on:
      pkt_rate, byte_rate, syn_ack_ratio, pkt_ratio, payload_ratio,
      dst_port_ratio, mean_iat, avg_idle, size_consistency,
      dst_ip_ratio, active_duration_sec
    """
    # 1. Parse Timestamp
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    except Exception:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # 2. Aggregate per Source IP  (17 raw aggregation columns)
    src_features = df.groupby("Src IP").agg(
        total_flows     = ("Flow ID",       "count"),
        total_fwd_pkts  = ("Tot Fwd Pkts",  "sum"),
        total_bwd_pkts  = ("Tot Bwd Pkts",  "sum"),
        total_fwd_bytes = ("TotLen Fwd Pkts","sum"),
        total_bwd_bytes = ("TotLen Bwd Pkts","sum"),
        unique_dst_ips  = ("Dst IP",        "nunique"),
        unique_dst_ports= ("Dst Port",      "nunique"),
        mean_flow_duration=("Flow Duration", "mean"),
        mean_iat        = ("Flow IAT Mean", "mean"),
        avg_idle        = ("Idle Mean",     "mean"),
        first_seen      = ("Timestamp",     "min"),
        last_seen       = ("Timestamp",     "max"),
        avg_pkt_size    = ("Pkt Size Avg",  "mean"),
        std_pkt_len     = ("Pkt Len Std",   "mean"),
        total_syn_flags = ("SYN Flag Cnt",  "sum"),
        total_ack_flags = ("ACK Flag Cnt",  "sum"),
        total_rst_flags = ("RST Flag Cnt",  "sum"),
    ).reset_index()

    src_ip_dataset = src_features.copy()

    # 3. Active duration
    duration = (src_ip_dataset["last_seen"] - src_ip_dataset["first_seen"]).dt.total_seconds()
    src_ip_dataset["active_duration_sec"] = duration.clip(lower=1)

    # Convert timestamps to string to avoid PyArrow issues downstream
    src_ip_dataset["first_seen"] = src_ip_dataset["first_seen"].astype(str)
    src_ip_dataset["last_seen"]  = src_ip_dataset["last_seen"].astype(str)

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
                        html.Span(id='threshold-value', children="0.20", className="threshold-val-badge"),
                    ], className="threshold-header-row"),
                    dcc.Slider(
                        id='threshold-slider',
                        min=0.01,
                        max=1.0,
                        step=0.01,
                        value=0.20,
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

                # ── Random Forest  ──
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

], className="app-wrapper")

# =============================================================================
# 6. CALLBACKS
# =============================================================================

# ─── Navigation ─────────────────────────────────────────────────
@app.callback(
    [Output('page-detection', 'style'),
     Output('page-models', 'style'),
     Output('page-about', 'style'),
     Output('nav-btn-detection', 'className'),
     Output('nav-btn-models', 'className'),
     Output('nav-btn-about', 'className')],
    [Input('nav-btn-detection', 'n_clicks'),
     Input('nav-btn-models', 'n_clicks'),
     Input('nav-btn-about', 'n_clicks')]
)
def navigate(n1, n2, n3):
    ctx = callback_context
    if not ctx.triggered:
        return ({'display': 'block'}, {'display': 'none'}, {'display': 'none'},
                'nav-link active', 'nav-link', 'nav-link')

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    page_map = {
        'nav-btn-detection': 0,
        'nav-btn-models': 1,
        'nav-btn-about': 2,
    }

    styles = [{'display': 'none'}] * 3
    classes = ['nav-link'] * 3

    idx = page_map.get(button_id, 0)
    styles[idx] = {'display': 'block'}
    classes[idx] = 'nav-link active'

    return tuple(styles + classes)

# ─── Threshold Display ──────────────────────────────────────────
@app.callback(
    Output('threshold-value', 'children'),
    Input('threshold-slider', 'value')
)
def update_threshold(value):
    return f"{value:.2f}"  # displayed inside threshold-val-badge

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

# ─── Process File & Generate Results ────────────────────────────
@app.callback(
    [Output('results-section', 'children'),
     Output('stored-data', 'data'),
     Output('scan-metrics', 'data')],
    [Input('scan-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename'),
     State('threshold-slider', 'value')]
)
def process_file(n_clicks, contents, filename, threshold):
    if n_clicks == 0 or contents is None:
        return (no_update, no_update, no_update)

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
                    None, no_update)

        raw_count = len(df)

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
        else:
            return (html.Div("❌ Model not loaded. Please check model path.", className="error-msg"),
                    None, no_update)

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

        # ────── BUILD DETECTION RESULTS ──────
        results = html.Div([
            html.Hr(className="section-divider"),

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
                ], className="r-metric-card")),

                # ── DDoS Attackers (with risk breakdown) ──
                dbc.Col(html.Div([
                    html.Div("DDOS ATTACKERS", className="r-metric-label"),
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
                ], className="r-metric-card")),

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
                ], className="r-metric-card")),

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
                ], className="r-metric-card")),
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

            # Data Tables
            dbc.Tabs([
                dbc.Tab(
                    build_blacklist_tab(attackers, attacker_count, threshold),
                    label="🚨 BLACKLIST",
                    tab_id="tab-bl",
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

        return (results,
                processed_df.to_dict('records'), scan_data)

    except Exception as e:
        return (html.Div(f"❌ Error: {str(e)}", className="error-msg"),
                None, no_update)

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

# =============================================================================
# 8. TABLE TAB FUNCTIONS
# =============================================================================
def build_blacklist_tab(attackers, count, threshold):
    """Build the attacker blacklist tab"""
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

        return html.Div([
            html.P(f"⚠️ {count} IPs detected with attack behavior (threshold {threshold})", className="tab-warning"),
            dbc.Table.from_dataframe(bl, striped=True, bordered=True, hover=True, size='sm', className='custom-table'),
            html.A("📥 Download Blacklist CSV", download="blacklist.csv",
                   href="data:text/csv;charset=utf-8," + bl.to_csv(index=False),
                   className="download-btn"),
        ], className="tab-content-inner")
    return html.Div("✅ No threats detected. System is secure.", className="tab-safe")

def build_full_tab(df):
    """Build full dataset report tab — fixed-column layout with overflow scroll."""
    # Select ONLY the columns we want to show (avoids wide internal cols that break layout)
    SHOW_COLS  = ['Src IP', 'Status', 'Risk Level', 'attack_probability']
    EXTRA_COLS = [
        'pkt_rate', 'byte_rate', 'syn_ack_ratio',
        'mean_iat', 'size_consistency',
        'dst_ip_ratio', 'dst_port_ratio', 'active_duration_sec',
    ]
    cols = SHOW_COLS + [c for c in EXTRA_COLS if c in df.columns]
    display = df[cols].copy()

    display['attack_probability'] = display['attack_probability'].apply(lambda x: f"{x:.2%}")
    for c in ['pkt_rate', 'byte_rate', 'syn_ack_ratio']:
        if c in display.columns:
            display[c] = display[c].apply(lambda x: f"{x:.2f}")
    for c in ['mean_iat', 'size_consistency', 'dst_ip_ratio', 'dst_port_ratio']:
        if c in display.columns:
            display[c] = display[c].apply(lambda x: f"{x:.4f}")
    if 'active_duration_sec' in display.columns:
        display['active_duration_sec'] = display['active_duration_sec'].apply(lambda x: f"{x:.1f}s")

    # Rename columns for display
    rename_map = {
        'attack_probability': 'Attack Prob',
        'pkt_rate':           'Pkt/s',
        'byte_rate':          'Byte/s',
        'syn_ack_ratio':      'SYN/ACK',
        'mean_iat':           'Mean IAT',
        'size_consistency':   'Size Consist.',
        'dst_ip_ratio':       'Dst IP Ratio',
        'dst_port_ratio':     'Dst Port Ratio',
        'active_duration_sec':'Active Dur',
    }
    display.rename(columns=rename_map, inplace=True)

    # Wrap table in overflow container — THIS is the root fix for the zoom bug
    table_div = html.Div(
        dbc.Table.from_dataframe(
            display.head(100),
            striped=True, bordered=True, hover=True,
            size='sm', className='custom-table',
            style={'tableLayout': 'auto', 'width': 'max-content', 'minWidth': '100%'},
        ),
        style={
            'overflowX': 'auto',
            'overflowY': 'hidden',
            'width':     '100%',
            'maxWidth':  '100%',
            '-webkit-overflow-scrolling': 'touch',
        }
    )

    return html.Div([
        html.Div([
            html.H4("Complete Traffic Analysis",
                    style={'color': '#fff', 'marginBottom': '0', 'fontSize': '1rem'}),
            html.Span(f"{min(len(df), 100):,} of {len(df):,} records",
                      style={'color': 'rgba(255,255,255,0.4)', 'fontSize': '0.78rem'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between',
                  'alignItems': 'center', 'marginBottom': '0.85rem'}),
        table_div,
        html.A("📥 Download Full Report (CSV)", download="full_report.csv",
               href="data:text/csv;charset=utf-8," + df[cols].to_csv(index=False),
               className="download-btn"),
    ], className="tab-content-inner")

# =============================================================================
# 9. RUN
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=8051)
