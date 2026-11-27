"""
SmartLeak — Intelligent Water Network Leakage Detection & Risk Monitoring System
Modern 2D Animated Streamlit Web Dashboard

Design: Industrial SaaS WaterTech analytics platform
Tagline: Every Drop Matters.
100% Pure 2D — Zero WebGL / Zero 3D — Engineering Accuracy First
"""

from __future__ import annotations

import io
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so engine imports reliably
_PROJ_DIR = str(Path(__file__).resolve().parent)
if _PROJ_DIR not in sys.path:
    sys.path.insert(0, _PROJ_DIR)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# ── Import the existing SmartLeak engine ──────────────────────────────────────
try:
    import main as smartleak
    _ENGINE_OK = True
    _ENGINE_ERR = None
except Exception as _exc:
    smartleak = None          # type: ignore[assignment]
    _ENGINE_OK = False
    _ENGINE_ERR = str(_exc)

# ── Tabler (MIT Open-Source Template) Design Tokens (Dark & Light) ────────────
THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "name": "dark",
        "bg": "#151f2c",
        "surface": "#1b2838",
        "elevated": "#223246",
        "sidebar_bg": "#111a25",
        "cyan": "#06b6d4",
        "teal": "#0ca678",
        "purple": "#7950f2",
        "success": "#2fb344",
        "warning": "#f59f00",
        "critical": "#d63939",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "border": "#2b3c51",
        "plotly_template": "plotly_dark",
        "card_shadow": "0 1px 3px rgba(0, 0, 0, 0.25)",
        "alert_bg": "rgba(214, 57, 57, 0.10)",
        "warn_bg": "rgba(245, 159, 0, 0.10)",
        "sim_bg": "rgba(6, 182, 212, 0.12)",
    },
    "light": {
        "name": "light",
        "bg": "#f1f5f9",
        "surface": "#ffffff",
        "elevated": "#f8fafc",
        "sidebar_bg": "#ffffff",
        "cyan": "#206bc4",
        "teal": "#0ca678",
        "purple": "#7048e8",
        "success": "#2fb344",
        "warning": "#f59f00",
        "critical": "#d63939",
        "text": "#182433",
        "muted": "#626976",
        "border": "#dce1e7",
        "plotly_template": "plotly_white",
        "card_shadow": "0 1px 2px rgba(24, 36, 51, 0.04)",
        "alert_bg": "rgba(214, 57, 57, 0.06)",
        "warn_bg": "rgba(245, 159, 0, 0.08)",
        "sim_bg": "rgba(32, 107, 196, 0.08)",
    },
}

C_NAVY     = "#071321"
C_DARK     = "#0D1B2A"
C_AQUA     = "#22D3EE"
C_TURQ     = "#14B8A6"
C_PURPLE   = "#A78BFA"
C_SUCCESS  = "#22C55E"
C_WARNING  = "#F59E0B"
C_DANGER   = "#EF4444"
C_MUTED    = "#94A3B8"
C_PANEL    = "#0D1B2A"
C_BORDER   = "#1E3A52"
C_CYAN     = "#22D3EE"
C_BLUE     = "#0891B2"
PLOTLY_TEMPLATE = "plotly_dark"


def _theme() -> dict[str, str]:
    mode = st.session_state.get("theme", "dark")
    return THEMES.get(mode, THEMES["dark"])


def _apply_plotly_theme(fig: go.Figure, height: int = 240) -> go.Figure:
    t = _theme()
    fig.update_layout(
        template=t["plotly_template"],
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["elevated"],
        font=dict(family="Inter, sans-serif", color=t["text"]),
        height=height,
    )
    fig.update_xaxes(gridcolor=t["border"], zerolinecolor=t["border"])
    fig.update_yaxes(gridcolor=t["border"], zerolinecolor=t["border"])
    return fig


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMARTLEAK — AquaTech & NeuralFlow Water Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS with Theme Tokens, Google Fonts & Lightweight Animations ───────
def _inject_css() -> None:
    t = _theme()
    st.html(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
    <style>
    /* ── Root theme tokens ── */
    :root {{
        --bg: {t["bg"]};
        --surface: {t["surface"]};
        --elevated: {t["elevated"]};
        --cyan: {t["cyan"]};
        --teal: {t["teal"]};
        --purple: {t["purple"]};
        --success: {t["success"]};
        --warning: {t["warning"]};
        --critical: {t["critical"]};
        --text: {t["text"]};
        --muted: {t["muted"]};
        --border: {t["border"]};
    }}

    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background: {t["bg"]} !important;
        color: {t["text"]} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    h1, h2, h3, h4, .brand-name {{
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        color: {t["text"]} !important;
    }}
    code, pre, .mono, .kpi-value {{
        font-family: 'IBM Plex Mono', monospace !important;
    }}

    /* ── Smooth page fade-in ── */
    @keyframes page-fade-in {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .main .block-container {{
        padding-top: 0.9rem !important;
        max-width: 100% !important;
        animation: page-fade-in 0.3s ease-out;
    }}

    /* ── Respect reduced motion preferences ── */
    @media (prefers-reduced-motion: reduce) {{
        *, ::before, ::after {{
            animation-delay: -1ms !important;
            animation-duration: 1ms !important;
            animation-iteration-count: 1 !important;
            background-attachment: initial !important;
            scroll-behavior: auto !important;
            transition-duration: 0s !important;
            transition-delay: 0s !important;
        }}
    }}

    /* ── Hide default Streamlit chrome (keep sidebar collapse button) ── */
    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {t["sidebar_bg"]} !important;
        border-right: 1px solid {t["border"]};
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 0.8rem !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {t["text"]};
    }}

    /* ── Tabler-style Top Navbar ── */
    .topbar {{
        background: {t["surface"]};
        border: 1px solid {t["border"]};
        border-radius: 6px;
        padding: 10px 16px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        box-shadow: {t["card_shadow"]};
    }}
    .topbar-left {{
        display: flex; align-items: center; gap: 12px;
    }}
    .topbar-title {{
        font-family: 'Inter', sans-serif;
        font-size: 16px; font-weight: 700;
        color: {t["text"]}; letter-spacing: 0.4px;
    }}
    .topbar-sub {{
        font-size: 11px; color: {t["muted"]};
    }}
    .topbar-pills {{
        display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    }}
    .pill {{
        font-size: 11px; font-weight: 600;
        padding: 3px 9px; border-radius: 4px;
        background: {t["elevated"]}; border: 1px solid {t["border"]};
        color: {t["text"]};
        font-family: 'IBM Plex Mono', monospace;
    }}

    /* ── Tabler Card / KPI Card ── */
    .kpi-card {{
        background: {t["surface"]};
        border: 1px solid {t["border"]};
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
        box-shadow: {t["card_shadow"]};
        transition: border-color 0.15s ease;
    }}
    .kpi-card:hover {{
        border-color: {t["cyan"]};
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: {t["cyan"]};
    }}
    .kpi-label {{
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: {t["muted"]};
        text-transform: uppercase;
        margin-bottom: 5px;
    }}
    .kpi-value {{
        font-size: 22px;
        font-weight: 700;
        color: {t["cyan"]};
        line-height: 1.15;
        margin-bottom: 4px;
        font-variant-numeric: tabular-nums;
    }}
    .kpi-detail {{
        font-size: 11px;
        color: {t["muted"]};
        margin-top: 2px;
    }}
    .kpi-card.success .kpi-value {{ color: {t["success"]}; }}
    .kpi-card.warning .kpi-value {{ color: {t["warning"]}; }}
    .kpi-card.danger  .kpi-value {{ color: {t["critical"]}; }}
    .kpi-card.purple  .kpi-value {{ color: {t["purple"]}; }}
    .kpi-card.success::before {{ background: {t["success"]}; }}
    .kpi-card.warning::before {{ background: {t["warning"]}; }}
    .kpi-card.danger::before  {{ background: {t["critical"]}; }}
    .kpi-card.purple::before  {{ background: {t["purple"]}; }}

    /* ── KPI Card slide-in animation ── */
    @keyframes kpi-slide-in {{
        0%   {{ opacity: 0; transform: translateY(10px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    .kpi-card {{ animation: kpi-slide-in 0.35s ease-out both; }}

    /* ── Pulsing monitoring dot ── */
    @keyframes live-dot-pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50%      {{ opacity: 0.5; transform: scale(1.35); }}
    }}
    .live-dot {{
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
        animation: live-dot-pulse 1.6s ease-in-out infinite;
    }}
    .live-dot-normal  {{ background: {t["success"]}; }}
    .live-dot-warning {{ background: {t["warning"]}; }}
    .live-dot-danger  {{ background: {t["critical"]}; }}
    .live-dot-anomaly {{ background: {t["purple"]}; }}

    /* ── Loading skeleton ── */
    @keyframes shimmer {{
        0%   {{ opacity: 0.55; }}
        50%  {{ opacity: 0.95; }}
        100% {{ opacity: 0.55; }}
    }}
    .smartleak-spinner {{
        display: inline-block;
        width: 20px; height: 20px;
        border: 3px solid {t["border"]};
        border-top-color: {t["cyan"]};
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        vertical-align: middle;
        margin-right: 8px;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

    /* ── Sidebar nav button transitions ── */
    section[data-testid="stSidebar"] button {{
        transition: background 0.2s ease, transform 0.15s ease, border-color 0.2s ease !important;
    }}
    section[data-testid="stSidebar"] button:hover {{
        transform: translateX(3px) !important;
    }}

    /* ── Progress bar fill animation ── */
    @keyframes progress-fill {{
        0%   {{ width: 0%; }}
        100% {{ width: var(--fill-pct); }}
    }}
    .risk-progress-bar {{
        height: 6px;
        border-radius: 3px;
        background: {t["elevated"]};
        overflow: hidden;
        margin-top: 6px;
    }}
    .risk-progress-bar .fill {{
        height: 100%;
        border-radius: 3px;
        animation: progress-fill 0.6s ease-out both;
    }}

    /* ── Pulsing leak banner ── */
    @keyframes subtle-pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.35); }}
        70% {{ box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
    }}
    .leak-alert-banner {{
        background: {t["alert_bg"]};
        border: 1px solid {t["critical"]};
        border-left: 5px solid {t["critical"]};
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0 14px 0;
        animation: subtle-pulse 2s infinite ease-in-out;
    }}

    /* ── Status badge ── */
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }}
    .badge-normal  {{ background: rgba(34,197,94,.16);  color: {t["success"]};  border: 1px solid {t["success"]}; }}
    .badge-warning {{ background: rgba(245,158,11,.16); color: {t["warning"]};  border: 1px solid {t["warning"]}; }}
    .badge-danger  {{ background: rgba(239,68,68,.16);  color: {t["critical"]}; border: 1px solid {t["critical"]}; }}
    .badge-info    {{ background: {t["sim_bg"]};        color: {t["cyan"]};     border: 1px solid {t["cyan"]}; }}
    .badge-purple  {{ background: rgba(167,139,250,.16); color: {t["purple"]};  border: 1px solid {t["purple"]}; }}

    /* ── Section header ── */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0 6px;
        border-bottom: 1px solid {t["border"]};
        margin-bottom: 12px;
    }}
    .section-header h3 {{
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: {t["text"]} !important;
        text-transform: uppercase;
        margin: 0;
    }}

    /* ── Info panel ── */
    .info-panel {{
        background: {t["surface"]};
        border: 1px solid {t["border"]};
        border-left: 3px solid {t["cyan"]};
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        box-shadow: {t["card_shadow"]};
        color: {t["text"]};
    }}

    /* ── Simulated data tag ── */
    .sim-tag {{
        background: {t["sim_bg"]};
        color: {t["cyan"]};
        border: 1px solid {t["cyan"]};
        border-radius: 4px;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 2px 7px;
        text-transform: uppercase;
        vertical-align: middle;
    }}

    /* ── Brand header ── */
    .brand-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 0 12px;
    }}
    .brand-logo {{ font-size: 28px; }}
    .brand-name {{
        font-size: 20px;
        font-weight: 800;
        color: {t["cyan"]} !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        line-height: 1.1;
    }}
    .brand-tagline {{
        font-size: 10px;
        color: {t["muted"]};
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }}
    </style>
    """)


# ── State Initialization ──────────────────────────────────────────────────────
def _init_state() -> None:
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    if "datasets" not in st.session_state:
        st.session_state.datasets = []
    if "anim_enabled" not in st.session_state:
        st.session_state.anim_enabled = True
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "3D Interactive"
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = "B2"
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = [
            {
                "time": "T+000s",
                "level": "NORMAL",
                "message": "WNTR/EPANET hydraulic digital twin initialized. All 4 zones nominal.",
            }
        ]
    if "simulation" not in st.session_state:
        st.session_state.simulation = None
        st.session_state.sim_error = None
        if _ENGINE_OK:
            with st.spinner("Initializing WNTR/EPANET hydraulic engine…"):
                try:
                    cal = smartleak.HydraulicCalibration()
                    st.session_state.simulation = smartleak.SmartLeakSimulation(cal)
                except Exception as exc:
                    st.session_state.sim_error = str(exc)
        else:
            st.session_state.sim_error = _ENGINE_ERR
    if "trend_pressure" not in st.session_state:
        st.session_state.trend_pressure = []
        st.session_state.trend_flow = []
        st.session_state.trend_risk = []
        st.session_state.trend_time = []
        if st.session_state.simulation is not None:
            _record_trend(st.session_state.simulation)


def _render_top_bar() -> None:
    t = _theme()
    engine = _engine()
    conn_html = (
        f'<span class="badge badge-normal">● ENGINE ONLINE</span>'
        if engine else
        f'<span class="badge badge-danger">● ENGINE OFFLINE</span>'
    )
    sim_txt = f"T+{engine.seconds:03d}s / 150s · {engine.network_status}" if engine else "Offline"
    alert_cnt = (
        sum(1 for z, r in engine.risk.items() if r >= 31 or engine.anomaly_flags.get(z, False))
        if engine else 0
    )

    tb_left, tb_right = st.columns([3.5, 2.2])
    with tb_left:
        st.markdown(f"""
        <div class="topbar">
            <div class="topbar-left">
                <span style="font-size:24px;">💧</span>
                <div>
                    <div class="topbar-title">SMARTLEAK <span style="font-size:10px;color:{t['purple']};letter-spacing:1px;">AQUATECH · HYDROVISION · NEURALFLOW</span></div>
                    <div class="topbar-sub">Every Drop Matters. &nbsp;·&nbsp; <span class="sim-tag">SIMULATED TELEMETRY</span></div>
                </div>
            </div>
            <div class="topbar-pills">
                {conn_html}
                <span class="pill">⏱ {sim_txt}</span>
                <span class="pill">🔔 {alert_cnt} Alert(s)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with tb_right:
        c_theme, c_alerts, c_settings = st.columns([1.2, 1.1, 1.0])
        with c_theme:
            lbl = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
            if st.button(lbl, key="topbar_theme_btn", use_container_width=True):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
        with c_alerts:
            if st.button(f"🔔 Alerts ({alert_cnt})", key="topbar_alerts_btn", use_container_width=True):
                st.session_state.page = "Alerts and Maintenance"
                st.rerun()
        with c_settings:
            if st.button("⚙️ Settings", key="topbar_settings_btn", use_container_width=True):
                st.session_state.page = "Settings"
                st.rerun()


def _log_activity(engine: Any, message: str, level: str = "INFO") -> None:
    log = st.session_state.setdefault("activity_log", [])
    t_str = f"T+{engine.seconds:03d}s" if engine else "T+000s"
    log.insert(0, {"time": t_str, "level": level, "message": message})
    if len(log) > 40:
        st.session_state.activity_log = log[:40]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _engine() -> Any | None:
    return st.session_state.simulation


def _risk_color(score: float) -> str:
    if score >= 61:
        return C_DANGER
    if score >= 31:
        return C_WARNING
    return C_SUCCESS


def _kpi_card(label: str, value: str, detail: str = "", variant: str = "default") -> str:
    return f"""
    <div class="kpi-card {variant if variant != 'default' else ''}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-detail">{detail}</div>
    </div>"""


def _section_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header"><span style="font-size:16px">{icon}</span>'
        f'<h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


def _record_trend(engine: Any) -> None:
    t = st.session_state.trend_time
    p = st.session_state.trend_pressure
    f = st.session_state.trend_flow
    r = st.session_state.trend_risk
    # Get values for whichever zone is leak target or Zone B
    target_z = engine.probable_zone or "B"
    t.append(engine.seconds)
    p.append(engine.measurements[target_z].pressure_m)
    f.append(engine.measurements[target_z].flow_lps)
    r.append(engine.risk[target_z])
    if len(t) > 60:
        st.session_state.trend_time     = t[-60:]
        st.session_state.trend_pressure = p[-60:]
        st.session_state.trend_flow     = f[-60:]
        st.session_state.trend_risk     = r[-60:]


def _quality_score(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return dict(rows=0, cols=0, missing=0, dups=0, complete=0,
                    completeness=0.0, score=0.0, status="No data", issues=[])
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum())
    complete = int(df.dropna().shape[0])
    total_cells = rows * cols
    completeness = (1 - missing / total_cells) * 100.0 if total_cells else 0.0
    dup_pct = (dups / rows * 100.0) if rows else 0.0
    num_ok = 0; num_col = 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_col += 1
            num_ok += int(pd.to_numeric(df[col], errors="coerce").notna().sum())
    num_cov = (num_ok / max(1, rows * max(1, num_col))) * 100.0
    score = max(0.0, min(100.0,
        completeness * 0.60 + (100.0 - dup_pct) * 0.20 + num_cov * 0.20))
    issues = []
    if missing > 0:
        issues.append(f"{missing} missing value(s)")
    if dups > 0:
        issues.append(f"{dups} duplicate row(s)")
    if completeness < 95:
        issues.append(f"Completeness {completeness:.1f}%")
    status = "Good" if score >= 85 else "Needs attention" if score >= 60 else "Poor"
    return dict(rows=rows, cols=cols, missing=missing, dups=dups,
                complete=complete, completeness=round(completeness, 2),
                score=round(score, 2), status=status, issues=issues)


def _read_upload(uploaded_file: Any) -> pd.DataFrame | None:
    try:
        name = str(uploaded_file.name).lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        st.warning("Unsupported file type. Upload CSV or Excel.")
        return None
    except Exception as exc:
        st.error(f"Could not read '{uploaded_file.name}': {exc}")
        return None


# ── ANIMATED 2D VECTOR SVG NETWORK VISUALIZER ──────────────────────────────────
def render_animated_2d_network(
    engine: Any,
    selected_node: str | None = None,
    anim_enabled: bool = True,
    width: int = 1000,
    height: int = 500,
) -> None:
    """
    Renders a high-precision, responsive 2D SVG water distribution network with
    CSS flow animations, moving pulses, dynamic particle velocities, and warning indicators.
    100% 2D - Clean vector SVG - No WebGL / Three.js.
    """
    snapshot = engine.snapshot
    active_leak_junc = engine.leak_junction or engine.active_leak_junction
    leak_active = (
        (engine.leak_severity > 0)
        or (max(engine.risk.values(), default=0.0) >= 31)
        or (engine._manual_leak_severity is not None)
    )
    if leak_active and not active_leak_junc:
        active_leak_junc = f"{engine.probable_zone or 'B'}2"

    # 1. Precise 2D Coordinates (scaled for SVG viewBox 0 0 1000 500)
    # Reservoir -> A1 -> A2 -> B1 -> B2 -> C1 -> C2 -> D1 -> D2
    # Branch nodes A3, B3, C3, D3 drop vertically below trunk
    coords = {
        "RESERVOIR": (75,  220),
        "A1":        (190, 220),
        "A2":        (290, 220),
        "A3":        (240, 370),
        "B1":        (410, 220),
        "B2":        (510, 220),
        "B3":        (460, 370),
        "C1":        (630, 220),
        "C2":        (730, 220),
        "C3":        (680, 370),
        "D1":        (840, 220),
        "D2":        (940, 220),
        "D3":        (890, 370),
    }

    pipes_def = [
        ("P_SUPPLY",   "RESERVOIR", "A1", "A"),
        ("P_A_CORE",   "A1",  "A2",  "A"),
        ("P_A_BRANCH", "A1",  "A3",  "A"),
        ("P_A_TO_B",   "A2",  "B1",  "A"),
        ("P_B_CORE",   "B1",  "B2",  "B"),
        ("P_B_BRANCH", "B1",  "B3",  "B"),
        ("P_B_TO_C",   "B2",  "C1",  "B"),
        ("P_C_CORE",   "C1",  "C2",  "C"),
        ("P_C_BRANCH", "C1",  "C3",  "C"),
        ("P_C_TO_D",   "C2",  "D1",  "C"),
        ("P_D_CORE",   "D1",  "D2",  "D"),
        ("P_D_BRANCH", "D1",  "D3",  "D"),
    ]

    # Zone boundaries (left, right, title, zone_id)
    zone_bounds = [
        ("A", 135, 345, "ZONE A · INLET & SUPPLY"),
        ("B", 355, 565, "ZONE B · CORE DISTRIBUTION"),
        ("C", 575, 785, "ZONE C · RESIDENTIAL SECTOR"),
        ("D", 795, 985, "ZONE D · ELEVATED DISTRICT"),
    ]

    svg_parts = []
    svg_parts.append(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 500" width="100%" height="{height}" style="background-color: #07111B; border-radius: 10px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; user-select: none;">''')

    # Definitions for SVG Markers, Gradients and Keyframe Styles
    svg_parts.append('''
    <defs>
        <marker id="arrow-norm" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#00C8C8"/>
        </marker>
        <marker id="arrow-warn" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#F59E0B"/>
        </marker>
        <marker id="arrow-danger" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#DC2626"/>
        </marker>
        <filter id="glow-danger" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
            </feMerge>
        </filter>
        <filter id="glow-aqua" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
            </feMerge>
        </filter>
        <linearGradient id="pipe-norm" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#14344D" />
            <stop offset="100%" stop-color="#00C8C8" />
        </linearGradient>
    </defs>
    <style>
        .zone-card { transition: fill 0.4s ease, stroke 0.4s ease; }
        .node-circle { transition: transform 0.2s ease, fill 0.3s ease; cursor: pointer; }
        .node-circle:hover { transform: scale(1.15); }
        .pipe-line { transition: stroke 0.3s ease, stroke-width 0.3s ease; }
        @keyframes dash-flow {
            to { stroke-dashoffset: -28; }
        }
        @keyframes leak-pulse {
            0% { r: 18px; opacity: 0.9; stroke-width: 2.5; }
            50% { r: 34px; opacity: 0.25; stroke-width: 1.5; }
            100% { r: 18px; opacity: 0.9; stroke-width: 2.5; }
        }
        @keyframes ring-expand {
            0% { r: 16px; opacity: 0.8; }
            100% { r: 42px; opacity: 0.0; }
        }
        .flow-dash {
            stroke-dasharray: 6, 8;
            animation: dash-flow linear infinite;
        }
        .leak-pulse-ring {
            animation: leak-pulse 1.8s infinite ease-in-out;
            transform-origin: center;
        }
        .leak-ring-expand {
            animation: ring-expand 2.2s infinite cubic-bezier(0.2, 0.8, 0.4, 1);
        }
    </style>
    ''')

    # 2. Render Zone Background Shading Panels
    for z, xl, xr, z_title in zone_bounds:
        z_risk = engine.risk.get(z, 0.0)
        is_high = z_risk >= 61
        is_warn = z_risk >= 31
        panel_fill = "rgba(220, 38, 38, 0.12)" if is_high else "rgba(245, 158, 11, 0.09)" if is_warn else "rgba(14, 29, 45, 0.6)"
        panel_stroke = "#DC2626" if is_high else "#F59E0B" if is_warn else "#1E3448"
        stroke_w = "2" if (is_high or is_warn) else "1"
        tag_color = "#DC2626" if is_high else "#F59E0B" if is_warn else "#8FA5B1"
        status_text = "CRITICAL" if is_high else "WARNING" if is_warn else "NORMAL"

        svg_parts.append(f'''
        <g class="zone-card">
            <rect x="{xl}" y="70" width="{xr - xl}" height="390" rx="8" fill="{panel_fill}" stroke="{panel_stroke}" stroke-width="{stroke_w}" />
            <text x="{(xl + xr) / 2}" y="95" fill="{tag_color}" font-size="11" font-weight="700" text-anchor="middle" letter-spacing="1">{z_title}</text>
            <rect x="{(xl + xr)/2 - 38}" y="105" width="76" height="18" rx="9" fill="rgba(0,0,0,0.4)" stroke="{panel_stroke}" stroke-width="0.8" />
            <text x="{(xl + xr) / 2}" y="117" fill="{tag_color}" font-size="9" font-weight="700" text-anchor="middle">{status_text} · {z_risk:.0f}%</text>
        </g>
        ''')

    # 3. Render Pipes (Lines, Flow Dashes, and Badges)
    affected_pipes = set()
    if leak_active and active_leak_junc:
        for p_id, s_node, e_node, p_z in pipes_def:
            if s_node == active_leak_junc or e_node == active_leak_junc:
                affected_pipes.add(p_id)

    for pipe_id, start, end, z in pipes_def:
        x1, y1 = coords[start]
        x2, y2 = coords[end]
        flow_lps = snapshot.flows_lps.get(pipe_id, 0.0)
        z_risk = engine.risk.get(z, 0.0)
        is_affected = pipe_id in affected_pipes or (z_risk >= 31 and leak_active)
        pipe_danger = is_affected and (z_risk >= 61)
        pipe_warning = is_affected and (z_risk >= 31)

        base_color = "#DC2626" if pipe_danger else "#F59E0B" if pipe_warning else "#00C8C8"
        pipe_w = 4.5 if (pipe_danger or pipe_warning) else 3.0
        marker_ref = "arrow-danger" if pipe_danger else "arrow-warn" if pipe_warning else "arrow-norm"

        # Base solid pipe
        svg_parts.append(f'''
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{base_color}" stroke-width="{pipe_w}" stroke-linecap="round" opacity="0.45" />
        ''')

        # Animated Flow Dashes & Flow Particles (Speed proportional to flow rate)
        if anim_enabled and flow_lps > 0.04:
            # Duration inversely proportional to flow velocity: higher flow = faster animation
            duration = max(0.6, min(3.5, 2.5 / (max(0.2, flow_lps))))
            svg_parts.append(f'''
            <line class="flow-dash" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{base_color}" stroke-width="{pipe_w}" stroke-linecap="round" marker-end="url(#{marker_ref})" style="animation-duration: {duration:.2f}s;" />
            ''')
            # Dynamic moving particle along the pipeline trajectory
            svg_parts.append(f'''
            <circle r="3.5" fill="#E8F1F5" filter="url(#glow-aqua)">
                <animateMotion path="M {x1} {y1} L {x2} {y2}" dur="{duration:.2f}s" repeatCount="indefinite" />
            </circle>
            <circle r="2.5" fill="{base_color}">
                <animateMotion path="M {x1} {y1} L {x2} {y2}" dur="{duration:.2f}s" begin="{duration*0.5:.2f}s" repeatCount="indefinite" />
            </circle>
            ''')
        else:
            svg_parts.append(f'''
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{base_color}" stroke-width="{pipe_w}" stroke-linecap="round" marker-end="url(#{marker_ref})" />
            ''')

        # Flow Rate Badge along pipe center
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        # Offset label so it doesn't collide with trunk
        badge_y = my - 12 if y1 == y2 else my
        badge_x = mx + 20 if x1 == x2 or abs(x2 - x1) < 80 else mx
        svg_parts.append(f'''
        <g>
            <rect x="{badge_x - 30}" y="{badge_y - 9}" width="60" height="17" rx="4" fill="#071724" stroke="{base_color}" stroke-width="0.8" opacity="0.95" />
            <text x="{badge_x}" y="{badge_y + 3}" fill="#E8F1F5" font-size="9" font-weight="bold" text-anchor="middle" font-family="monospace">{flow_lps:.2f} L/s</text>
        </g>
        ''')

    # 4. Render Reservoir Node
    rx, ry = coords["RESERVOIR"]
    svg_parts.append(f'''
    <g transform="translate({rx},{ry})">
        <circle r="30" fill="#0284C7" stroke="#38BDF8" stroke-width="3" filter="url(#glow-aqua)" />
        <text x="0" y="-4" fill="#FFFFFF" font-size="16" text-anchor="middle">💧</text>
        <text x="0" y="14" fill="#FFFFFF" font-size="9" font-weight="bold" text-anchor="middle">RESERVOIR</text>
        <rect x="-35" y="36" width="70" height="18" rx="4" fill="#0A1624" stroke="#0284C7" stroke-width="1" />
        <text x="0" y="49" fill="#38BDF8" font-size="9" font-weight="bold" text-anchor="middle">79.0 m H₂O</text>
    </g>
    ''')

    # 5. Render Junction Nodes (A1 through D3)
    node_zones = {n: z for z, ns in smartleak.ZONE_NODES.items() for n in ns}
    for node, (nx_val, ny_val) in coords.items():
        if node == "RESERVOIR":
            continue

        zone = node_zones[node]
        pressure = snapshot.pressures.get(node, 0.0)
        is_leak_target = (node == active_leak_junc and leak_active)
        is_selected = (node == selected_node)

        # Status & Node Color
        z_risk = engine.risk.get(zone, 0.0)
        if is_leak_target:
            n_color = "#DC2626" if z_risk >= 61 else "#F59E0B"
            stroke_color = "#FFFFFF"
        elif z_risk >= 61:
            n_color = "#DC2626"
            stroke_color = "#FCA5A5"
        elif z_risk >= 31:
            n_color = "#F59E0B"
            stroke_color = "#FDE68A"
        else:
            n_color = "#0E2438"
            stroke_color = "#00C8C8"

        svg_parts.append(f'<g transform="translate({nx_val},{ny_val})">')

        # Pulsing Warning Rings for Simulated Leak Target
        if is_leak_target:
            svg_parts.append(f'''
            <circle class="leak-pulse-ring" cx="0" cy="0" r="20" fill="none" stroke="{n_color}" stroke-width="2" />
            <circle class="leak-ring-expand" cx="0" cy="0" r="18" fill="none" stroke="{n_color}" stroke-width="1.5" />
            ''')

        # Highlight ring if selected
        if is_selected:
            svg_parts.append(f'''
            <circle cx="0" cy="0" r="22" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-dasharray="3,3" />
            ''')

        # Main Junction Circle
        svg_parts.append(f'''
        <circle class="node-circle" cx="0" cy="0" r="16" fill="{n_color}" stroke="{stroke_color}" stroke-width="2" />
        <text x="0" y="4" fill="#FFFFFF" font-size="10" font-weight="bold" text-anchor="middle">{node}</text>
        ''')

        # Pressure Tag below node
        svg_parts.append(f'''
        <rect x="-26" y="21" width="52" height="15" rx="3" fill="#0A1624" stroke="{stroke_color}" stroke-width="0.8" opacity="0.95" />
        <text x="0" y="32" fill="#E8F1F5" font-size="8.5" font-weight="bold" text-anchor="middle" font-family="monospace">{pressure:.1f} m</text>
        ''')

        # Warning Callout Banner for Leak Node
        if is_leak_target:
            base_p = engine.calibration.baselines.get(zone).pressure_m
            drop_m = pressure - base_p
            svg_parts.append(f'''
            <g transform="translate(0, -32)">
                <rect x="-65" y="-18" width="130" height="20" rx="4" fill="#7F1D1D" stroke="#DC2626" stroke-width="1.2" filter="url(#glow-danger)" />
                <text x="0" y="-4" fill="#FFFFFF" font-size="8.5" font-weight="bold" text-anchor="middle">⚠ SIMULATED LEAK (ΔP: {drop_m:+.2f}m)</text>
            </g>
            ''')

        svg_parts.append('</g>')

    # 6. Water Distribution Network Legend (Top Right of SVG)
    svg_parts.append('''
    <g transform="translate(25, 20)">
        <rect x="0" y="0" width="460" height="34" rx="6" fill="#0E1D2D" stroke="#263A49" stroke-width="1" />
        <circle cx="20" cy="17" r="6" fill="#0284C7" />
        <text x="32" y="21" fill="#8FA5B1" font-size="10">Reservoir</text>
        <circle cx="105" cy="17" r="6" fill="#0E2438" stroke="#00C8C8" stroke-width="1.5" />
        <text x="117" y="21" fill="#8FA5B1" font-size="10">Normal Node</text>
        <circle cx="205" cy="17" r="6" fill="#F59E0B" />
        <text x="217" y="21" fill="#8FA5B1" font-size="10">Warning</text>
        <circle cx="285" cy="17" r="6" fill="#DC2626" />
        <text x="297" y="21" fill="#8FA5B1" font-size="10">Leak Threat</text>
        <line x1="375" y1="17" x2="400" y2="17" stroke="#00C8C8" stroke-width="2.5" stroke-dasharray="4,4" />
        <text x="408" y="21" fill="#8FA5B1" font-size="10">Water Flow</text>
    </g>
    ''')

    svg_parts.append('</svg>')
    full_svg = "".join(svg_parts)

    # Render via components.html to allow full CSS animation keyframes and SVG rendering
    components.html(full_svg, height=height + 20)


# ── INTERACTIVE 3D THREE.JS WATER DISTRIBUTION NETWORK ────────────────────────
def render_3d_network(
    engine: Any,
    selected_node: str | None = None,
    anim_enabled: bool = True,
    height: int = 560,
) -> None:
    """
    Renders an interactive 3D Three.js hydraulic water distribution network
    driven by live WNTR/EPANET simulation outputs.
    Supports orbit/zoom/pan, clickable junctions, zone filters, 3D tube pipelines,
    glowing flow particles, and animated water-escape leak simulation.
    """
    snapshot = engine.snapshot
    active_leak_junc = engine.leak_junction or engine.active_leak_junction
    leak_active = (
        (engine.leak_severity > 0)
        or (max(engine.risk.values(), default=0.0) >= 31)
        or (engine._manual_leak_severity is not None)
    )
    if leak_active and not active_leak_junc:
        active_leak_junc = f"{engine.probable_zone or 'B'}2"

    # 3D XZ layout matching exact network topology (Y is computed from actual elevation)
    xz_coords = {
        "RESERVOIR": (-21.0,  0.0),
        "A1":        (-14.0,  0.0),
        "A2":        ( -9.0,  0.0),
        "A3":        (-11.5,  6.5),
        "B1":        ( -4.0,  0.0),
        "B2":        (  1.0,  0.0),
        "B3":        ( -1.5,  6.5),
        "C1":        (  6.0,  0.0),
        "C2":        ( 11.0,  0.0),
        "C3":        (  8.5,  6.5),
        "D1":        ( 16.0,  0.0),
        "D2":        ( 21.0,  0.0),
        "D3":        ( 18.5,  6.5),
    }

    junc_telemetry = {item["node"]: item for item in engine.get_junction_telemetry()}
    pipe_telemetry = engine.get_pipe_telemetry()

    # Determine affected pipes for active leak
    affected_pipes: set[str] = set()
    if leak_active and active_leak_junc:
        for p in pipe_telemetry:
            if p["start"] == active_leak_junc or p["end"] == active_leak_junc:
                affected_pipes.add(p["pipe"])

    # Build nodes payload
    nodes_payload: list[dict[str, Any]] = [{
        "id": "RESERVOIR",
        "zone": "SUPPLY",
        "x": xz_coords["RESERVOIR"][0],
        "y": 6.2,
        "z": xz_coords["RESERVOIR"][1],
        "elevation_m": 79.0,
        "pressure_m": 79.0,
        "baseline_m": 79.0,
        "dev_m": 0.0,
        "dev_pct": 0.0,
        "risk": 0.0,
        "status": "NORMAL",
        "is_leak": False,
        "is_selected": (selected_node == "RESERVOIR"),
        "connected_flows": [
            {"pipe": p["pipe"], "flow_lps": round(p["flow_lps"], 3), "direction": p["direction"]}
            for p in pipe_telemetry if p["start"] == "RESERVOIR" or p["end"] == "RESERVOIR"
        ],
    }]

    for node_id, (x_val, z_val) in xz_coords.items():
        if node_id == "RESERVOIR":
            continue
        jt = junc_telemetry[node_id]
        zone = jt["zone"]
        z_risk = float(engine.risk.get(zone, 0.0))
        is_leak = bool(leak_active and node_id == active_leak_junc)
        # Scale elevation (51m - 57m) into visible 3D Y height
        y_val = (jt["elevation_m"] - 50.0) * 0.55 + 1.8
        conn_flows = [
            {
                "pipe": p["pipe"],
                "flow_lps": round(p["flow_lps"], 3),
                "direction": p["direction"],
            }
            for p in pipe_telemetry
            if p["start"] == node_id or p["end"] == node_id
        ]
        nodes_payload.append({
            "id": node_id,
            "zone": zone,
            "x": x_val,
            "y": round(y_val, 3),
            "z": z_val,
            "elevation_m": round(jt["elevation_m"], 2),
            "pressure_m": round(jt["pressure_m"], 3),
            "baseline_m": round(jt["baseline_m"], 3),
            "dev_m": round(jt["dev_m"], 3),
            "dev_pct": round(jt["dev_pct"], 2),
            "risk": round(z_risk, 1),
            "status": jt["status"],
            "is_leak": is_leak,
            "is_selected": (selected_node == node_id),
            "connected_flows": conn_flows,
        })

    # Build pipes payload
    pipes_payload: list[dict[str, Any]] = []
    for p in pipe_telemetry:
        p_id = p["pipe"]
        z = p["zone"]
        z_risk = float(engine.risk.get(z, 0.0))
        is_affected = (p_id in affected_pipes) or (z_risk >= 31 and leak_active)
        if is_affected and (z_risk >= 61 or p_id in affected_pipes):
            level = "danger"
        elif is_affected and z_risk >= 31:
            level = "warning"
        else:
            level = "normal"

        pipes_payload.append({
            "id": p_id,
            "start": p["start"],
            "end": p["end"],
            "zone": z,
            "diameter_m": p["diameter_m"],
            "length_m": p["length_m"],
            "flow_lps": round(p["flow_lps"], 4),
            "signed_flow_lps": round(p["signed_flow_lps"], 4),
            "velocity_m_s": round(p["velocity_m_s"], 4),
            "direction": p["direction"],
            "level": level,
            "is_affected": is_affected,
        })

    # Build zones payload
    zone_meta = [
        ("A", "ZONE A · INLET & SUPPLY",     -15.8, -7.2),
        ("B", "ZONE B · CORE DISTRIBUTION",  -5.8,  2.8),
        ("C", "ZONE C · RESIDENTIAL SECTOR",  4.2, 12.8),
        ("D", "ZONE D · ELEVATED DISTRICT",  14.2, 22.8),
    ]
    zones_payload: list[dict[str, Any]] = []
    for z, title, x_min, x_max in zone_meta:
        z_risk = float(engine.risk.get(z, 0.0))
        meas = engine.measurements.get(z)
        base = engine.calibration.baselines.get(z)
        status = "CRITICAL" if z_risk >= 61 else "WARNING" if z_risk >= 31 else "NORMAL"
        zones_payload.append({
            "zone": z,
            "title": title,
            "x_min": x_min,
            "x_max": x_max,
            "z_min": -3.2,
            "z_max": 8.6,
            "risk": round(z_risk, 1),
            "status": status,
            "pressure_m": round(meas.pressure_m, 2) if meas else 0.0,
            "flow_lps": round(meas.flow_lps, 3) if meas else 0.0,
            "base_pressure_m": round(base.pressure_m, 2) if base else 0.0,
            "base_flow_lps": round(base.flow_lps, 3) if base else 0.0,
        })

    t_cfg = _theme()
    is_light = (t_cfg["name"] == "light")
    leak_zone = engine.node_zones.get(active_leak_junc, "") if active_leak_junc else ""
    payload = {
        "anim_enabled": bool(anim_enabled),
        "is_light": is_light,
        "nodes": nodes_payload,
        "pipes": pipes_payload,
        "zones": zones_payload,
        "selected_node": selected_node or active_leak_junc or "B2",
        "leak": {
            "active": bool(leak_active),
            "junction": active_leak_junc,
            "zone": leak_zone,
            "severity": round(float(engine.leak_severity), 3),
            "loss_l_min": round(float(engine.loss_estimate_l_min), 2),
            "seconds": int(engine.seconds),
        },
    }
    payload_json = json.dumps(payload)

    vp_bg = "#EAF1F8" if is_light else "#071321"
    hud_bg = "rgba(255, 255, 255, 0.90)" if is_light else "rgba(13, 27, 42, 0.88)"
    hud_border = t_cfg["border"]
    hud_text = t_cfg["text"]
    hud_muted = t_cfg["muted"]
    hud_accent = t_cfg["cyan"]

    html_3d = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100%; height: 100%; overflow: hidden;
    background: {vp_bg};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {hud_text};
    user-select: none;
  }}
  #viewport {{
    position: relative;
    width: 100%;
    height: {height}px;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {hud_border};
    background: {vp_bg};
  }}
  canvas {{ display: block; width: 100%; height: 100%; outline: none; }}

  /* Top-Left HUD: Zone Filters & Camera */
  .hud-top-left {{
    position: absolute; top: 12px; left: 12px;
    display: flex; flex-direction: column; gap: 8px;
    z-index: 10; pointer-events: none;
  }}
  .hud-bar {{
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    background: rgba(11, 18, 32, 0.85);
    backdrop-filter: blur(8px);
    border: 1px solid #263A49;
    border-radius: 8px;
    padding: 6px 10px;
    pointer-events: auto;
  }}
  .hud-title {{
    font-size: 10px; font-weight: 800; letter-spacing: 1.2px;
    color: #00C8C8; text-transform: uppercase; margin-right: 4px;
  }}
  .zone-btn {{
    background: #0E1D2D; color: #8FA5B1;
    border: 1px solid #263A49; border-radius: 5px;
    padding: 4px 9px; font-size: 10px; font-weight: 700;
    cursor: pointer; transition: all 0.2s ease;
  }}
  .zone-btn:hover {{ border-color: #00C8C8; color: #E8F1F5; }}
  .zone-btn.active {{
    background: rgba(0, 200, 200, 0.2);
    border-color: #00C8C8; color: #00C8C8;
  }}
  .zone-btn.warn {{ border-color: #F59E0B; color: #F59E0B; }}
  .zone-btn.danger {{ border-color: #DC2626; color: #FCA5A5; background: rgba(220,38,38,0.2); }}

  /* Leak Alert Pill */
  .leak-pill {{
    display: none;
    align-items: center; gap: 8px;
    background: rgba(127, 29, 29, 0.88);
    border: 1px solid #DC2626;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 11px; font-weight: 700; color: #FFFFFF;
    box-shadow: 0 0 18px rgba(220, 38, 38, 0.45);
    pointer-events: auto;
    animation: pulse-border 1.6s infinite;
  }}
  @keyframes pulse-border {{
    0%, 100% {{ box-shadow: 0 0 10px rgba(220,38,38,0.4); }}
    50% {{ box-shadow: 0 0 22px rgba(220,38,38,0.85); }}
  }}

  /* Top-Right HUD: Clickable Junction Inspector */
  .inspector-card {{
    position: absolute; top: 12px; right: 12px;
    width: 245px;
    background: rgba(11, 18, 32, 0.90);
    backdrop-filter: blur(10px);
    border: 1px solid #263A49;
    border-top: 3px solid #00C8C8;
    border-radius: 10px;
    padding: 12px 14px;
    z-index: 10;
    box-shadow: 0 8px 24px rgba(0,0,0,0.45);
  }}
  .insp-head {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px; padding-bottom: 6px;
    border-bottom: 1px solid #1E3448;
  }}
  .insp-node {{ font-size: 15px; font-weight: 800; color: #FFFFFF; }}
  .insp-badge {{
    font-size: 9px; font-weight: 700; padding: 2px 7px;
    border-radius: 10px; text-transform: uppercase; letter-spacing: 0.8px;
  }}
  .insp-row {{
    display: flex; justify-content: space-between;
    font-size: 11px; margin: 4px 0; color: #8FA5B1;
  }}
  .insp-row b {{ color: #E8F1F5; font-family: monospace; }}
  .insp-pipes {{
    margin-top: 8px; padding-top: 6px;
    border-top: 1px solid #1E3448;
    font-size: 10px; color: #8FA5B1;
  }}
  .insp-pipe-item {{
    display: flex; justify-content: space-between;
    margin-top: 3px; font-family: monospace; color: #00C8C8;
  }}

  /* Bottom-Left Legend */
  .hud-legend {{
    position: absolute; bottom: 12px; left: 12px;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    background: rgba(11, 18, 32, 0.85);
    backdrop-filter: blur(8px);
    border: 1px solid #263A49;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 10px; color: #8FA5B1;
    z-index: 10;
  }}
  .leg-item {{ display: flex; align-items: center; gap: 5px; }}
  .leg-dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}

  /* Bottom-Right Controls Hint & Sim Tag */
  .hud-bottom-right {{
    position: absolute; bottom: 12px; right: 12px;
    display: flex; flex-direction: column; align-items: flex-end; gap: 4px;
    z-index: 10; pointer-events: none;
  }}
  .sim-badge {{
    background: rgba(85, 149, 245, 0.16);
    border: 1px solid #5595F5; color: #93C5FD;
    font-size: 9px; font-weight: 700; letter-spacing: 1px;
    padding: 3px 8px; border-radius: 4px;
  }}
  .controls-hint {{
    font-size: 10px; color: #64748B;
    background: rgba(11,18,32,0.75); padding: 3px 8px; border-radius: 4px;
  }}

  #webgl-error {{
    display: none; position: absolute; inset: 0;
    background: #0B1220; color: #FCA5A5;
    align-items: center; justify-content: center; flex-direction: column;
    padding: 24px; text-align: center; z-index: 50;
  }}
</style>
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>
</head>
<body>
<div id="viewport">
  <div class="hud-top-left">
    <div class="hud-bar">
      <span class="hud-title">3D Zone Filter</span>
      <button class="zone-btn active" data-zone="ALL">ALL ZONES</button>
      <button class="zone-btn" data-zone="A" id="btn-zone-A">Zone A</button>
      <button class="zone-btn" data-zone="B" id="btn-zone-B">Zone B</button>
      <button class="zone-btn" data-zone="C" id="btn-zone-C">Zone C</button>
      <button class="zone-btn" data-zone="D" id="btn-zone-D">Zone D</button>
      <button class="zone-btn active" id="btn-toggle-labels" title="Toggle 3D Labels">🏷️ Labels</button>
      <button class="zone-btn active" id="btn-toggle-particles" title="Toggle Flow Particles">💧 Flow</button>
      <button class="zone-btn" id="btn-reset-cam" title="Reset 3D Camera">⟲ Reset View</button>
    </div>
    <div class="leak-pill" id="leak-pill">
      <span>⚠</span>
      <span id="leak-pill-text">SIMULATED LEAK ACTIVE</span>
    </div>
  </div>

  <div class="inspector-card" id="inspector-card" style="display:none;">
    <div class="insp-head">
      <span class="insp-node" id="insp-id">B2</span>
      <div style="display:flex;align-items:center;gap:6px;">
        <span class="insp-badge" id="insp-status">NORMAL</span>
        <button id="insp-close" title="Close Inspector" style="background:transparent;border:none;color:#94A3B8;font-size:15px;font-weight:700;cursor:pointer;line-height:1;padding:0 2px;">×</button>
      </div>
    </div>
    <div class="insp-row"><span>Zone / Sector</span><b id="insp-zone">Zone B</b></div>
    <div class="insp-row"><span>Elevation</span><b id="insp-elev">53.0 m</b></div>
    <div class="insp-row"><span>Observed Pressure</span><b id="insp-press">—</b></div>
    <div class="insp-row"><span>Baseline / ΔP</span><b id="insp-delta">—</b></div>
    <div class="insp-row"><span>Zone Risk Score</span><b id="insp-risk">0.0%</b></div>
    <div class="insp-pipes">
      <div style="font-weight:700;letter-spacing:0.6px;margin-bottom:2px;">CONNECTED PIPES</div>
      <div id="insp-pipes-list"></div>
    </div>
  </div>

  <div class="hud-legend">
    <div class="leg-item"><span class="leg-dot" style="background:#0284C7;"></span>Reservoir (79m)</div>
    <div class="leg-item"><span class="leg-dot" style="background:#22D3EE;"></span>Normal Flow</div>
    <div class="leg-item"><span class="leg-dot" style="background:#F59E0B;"></span>Warning (≥31%)</div>
    <div class="leg-item"><span class="leg-dot" style="background:#EF4444;"></span>Critical / Leak</div>
    <div class="leg-item"><span class="leg-dot" style="background:#64748B;"></span>Inactive</div>
    <div class="leg-item"><span class="leg-dot" style="background:#38BDF8;"></span>Escaping Water Spray</div>
  </div>

  <div class="hud-bottom-right">
    <div class="sim-badge">SIMULATED DATA · WNTR / EPANET</div>
    <div class="controls-hint">Left-Drag: Rotate · Scroll: Zoom · Right-Drag: Pan · Click Junction: Inspect</div>
  </div>

  <div id="webgl-error">
    <h3 style="margin-bottom:8px;">3D WebGL Renderer Unavailable</h3>
    <p style="font-size:12px;color:#8FA5B1;">Switch to the 2D Vector view using the toggle above, or enable WebGL/hardware acceleration in your browser.</p>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const DATA = {payload_json};
const container = document.getElementById('viewport');

try {{
  const scene = new THREE.Scene();
  const fogColor = DATA.is_light ? 0xEAF1F8 : 0x071321;
  scene.fog = new THREE.FogExp2(fogColor, 0.009);

  const width = container.clientWidth || 1000;
  const height = container.clientHeight || {height};

  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 300);
  const DEFAULT_CAM_POS = new THREE.Vector3(0, 21, 30);
  const DEFAULT_TARGET = new THREE.Vector3(1.5, 2.2, 1.8);
  camera.position.copy(DEFAULT_CAM_POS);

  const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true, powerPreference: 'high-performance' }});
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(DEFAULT_TARGET);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 8;
  controls.maxDistance = 68;
  controls.maxPolarAngle = Math.PI / 2 - 0.04;
  controls.update();

  // Lighting
  const ambient = new THREE.AmbientLight(0x94B8D0, 0.9);
  scene.add(ambient);

  const dirLight = new THREE.DirectionalLight(0xFFFFFF, 1.4);
  dirLight.position.set(14, 30, 18);
  dirLight.castShadow = true;
  scene.add(dirLight);

  const cyanFill = new THREE.PointLight(0x00C8C8, 1.2, 60);
  cyanFill.position.set(-10, 14, -8);
  scene.add(cyanFill);

  // Subtle Grid Floor
  const grid = new THREE.GridHelper(
    64, 32,
    DATA.is_light ? 0x94A3B8 : 0x1E3A52,
    DATA.is_light ? 0xCBD5E1 : 0x0E1D2D
  );
  grid.position.y = 0;
  scene.add(grid);

  const COLORS = {{
    normal: 0x00C8C8,
    normalPipe: 0x0EA5E9,
    warning: 0xF59E0B,
    danger: 0xDC2626,
    reservoir: 0x0284C7,
  }};

  function makeTextSprite(text, subtext, borderColor = '#00C8C8') {{
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = subtext ? 96 : 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(7, 17, 27, 0.88)';
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(8, 8, canvas.width - 16, canvas.height - 16, 12);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 26px Segoe UI, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(text, canvas.width / 2, subtext ? 42 : 42);
    if (subtext) {{
      ctx.fillStyle = borderColor;
      ctx.font = 'bold 20px monospace';
      ctx.fillText(subtext, canvas.width / 2, 72);
    }}
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({{ map: tex, transparent: true, depthTest: false }});
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(3.2, subtext ? 1.2 : 0.8, 1);
    return sprite;
  }}

  // 1. Zone Pads
  const zoneGroups = {{}};
  DATA.zones.forEach(z => {{
    const g = new THREE.Group();
    const w = z.x_max - z.x_min;
    const d = z.z_max - z.z_min;
    const cx = (z.x_min + z.x_max) / 2;
    const cz = (z.z_min + z.z_max) / 2;

    const hex = z.risk >= 61 ? '#DC2626' : z.risk >= 31 ? '#F59E0B' : '#00C8C8';
    const colVal = z.risk >= 61 ? COLORS.danger : z.risk >= 31 ? COLORS.warning : COLORS.normal;

    const padGeo = new THREE.BoxGeometry(w, 0.08, d);
    const padMat = new THREE.MeshStandardMaterial({{
      color: colVal,
      transparent: true,
      opacity: z.risk >= 31 ? 0.18 : 0.08,
      roughness: 0.7,
    }});
    const pad = new THREE.Mesh(padGeo, padMat);
    pad.position.set(cx, 0.04, cz);
    g.add(pad);

    const edges = new THREE.EdgesGeometry(padGeo);
    const lineMat = new THREE.LineBasicMaterial({{ color: colVal, transparent: true, opacity: 0.75 }});
    const wire = new THREE.LineSegments(edges, lineMat);
    wire.position.copy(pad.position);
    g.add(wire);

    const lbl = makeTextSprite(`ZONE ${{z.zone}}`, `${{z.status}} · ${{z.risk.toFixed(0)}}%`, hex);
    lbl.position.set(cx, 0.9, z.z_max - 0.6);
    lbl.scale.set(4.0, 1.4, 1);
    g.add(lbl);

    // Update zone button styling
    const btn = document.getElementById(`btn-zone-${{z.zone}}`);
    if (btn) {{
      if (z.risk >= 61) btn.classList.add('danger');
      else if (z.risk >= 31) btn.classList.add('warn');
    }}

    scene.add(g);
    zoneGroups[z.zone] = {{ group: g, padMat, lineMat, cx, cz }};
  }});

  // 2. Nodes (Reservoir + 12 Junctions)
  const nodeMap = {{}};
  const clickableMeshes = [];
  let leakRingMesh = null;
  let leakPulseMesh = null;
  let selectedRingMesh = null;
  let reservoirWaterMesh = null;

  DATA.nodes.forEach(n => {{
    nodeMap[n.id] = n;
    const pos = new THREE.Vector3(n.x, n.y, n.z);
    n.vec = pos;

    if (n.id === 'RESERVOIR') {{
      // Elevated Water Tower / Reservoir
      const basePillar = new THREE.Mesh(
        new THREE.CylinderGeometry(0.9, 1.3, n.y - 1.2, 16),
        new THREE.MeshStandardMaterial({{ color: 0x1E3448, metalness: 0.6, roughness: 0.4 }})
      );
      basePillar.position.set(n.x, (n.y - 1.2) / 2, n.z);
      scene.add(basePillar);

      const tank = new THREE.Mesh(
        new THREE.CylinderGeometry(2.1, 2.1, 2.6, 24),
        new THREE.MeshPhysicalMaterial({{
          color: 0x0284C7, transparent: true, opacity: 0.45,
          roughness: 0.15, metalness: 0.2,
        }})
      );
      tank.position.copy(pos);
      tank.userData = {{ nodeId: 'RESERVOIR' }};
      scene.add(tank);
      clickableMeshes.push(tank);

      reservoirWaterMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(1.9, 1.9, 1.8, 24),
        new THREE.MeshStandardMaterial({{
          color: 0x38BDF8, emissive: 0x0284C7, emissiveIntensity: 0.65,
          transparent: true, opacity: 0.85,
        }})
      );
      reservoirWaterMesh.position.set(n.x, n.y - 0.25, n.z);
      scene.add(reservoirWaterMesh);

      const rim = new THREE.Mesh(
        new THREE.TorusGeometry(2.1, 0.09, 12, 32),
        new THREE.MeshBasicMaterial({{ color: 0x38BDF8 }})
      );
      rim.rotation.x = Math.PI / 2;
      rim.position.set(n.x, n.y + 1.3, n.z);
      scene.add(rim);

      const lbl = makeTextSprite('RESERVOIR', '79.0 m H₂O', '#38BDF8');
      lbl.position.set(n.x, n.y + 2.6, n.z);
      scene.add(lbl);
    }} else {{
      const isDanger = n.is_leak || n.risk >= 61;
      const isWarn = !isDanger && n.risk >= 31;
      const nodeColor = isDanger ? COLORS.danger : isWarn ? COLORS.warning : COLORS.normal;
      const borderHex = isDanger ? '#DC2626' : isWarn ? '#F59E0B' : '#00C8C8';

      // Support column to ground
      const stem = new THREE.Mesh(
        new THREE.CylinderGeometry(0.10, 0.16, n.y, 10),
        new THREE.MeshStandardMaterial({{ color: 0x1E3448, metalness: 0.5, roughness: 0.5 }})
      );
      stem.position.set(n.x, n.y / 2, n.z);
      scene.add(stem);

      // Junction Sphere
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(isDanger ? 0.72 : 0.58, 24, 24),
        new THREE.MeshStandardMaterial({{
          color: nodeColor,
          emissive: nodeColor,
          emissiveIntensity: isDanger ? 0.85 : 0.45,
          roughness: 0.25,
          metalness: 0.35,
        }})
      );
      sphere.position.copy(pos);
      sphere.userData = {{ nodeId: n.id, zone: n.zone }};
      scene.add(sphere);
      clickableMeshes.push(sphere);
      n.mesh = sphere;

      // Ground marker ring
      const gRing = new THREE.Mesh(
        new THREE.RingGeometry(0.45, 0.62, 24),
        new THREE.MeshBasicMaterial({{ color: nodeColor, side: THREE.DoubleSide, transparent: true, opacity: 0.6 }})
      );
      gRing.rotation.x = -Math.PI / 2;
      gRing.position.set(n.x, 0.09, n.z);
      scene.add(gRing);

      // Floating Label
      const lbl = makeTextSprite(n.id, `${{n.pressure_m.toFixed(1)}} m`, borderHex);
      lbl.position.set(n.x, n.y + 1.45, n.z);
      lbl.scale.set(2.5, 0.95, 1);
      scene.add(lbl);
      n.labelSprite = lbl;

      if (n.is_leak) {{
        const pulseGeo = new THREE.RingGeometry(0.85, 1.15, 32);
        const pulseMat = new THREE.MeshBasicMaterial({{
          color: 0xDC2626, side: THREE.DoubleSide, transparent: true, opacity: 0.9,
        }});
        leakRingMesh = new THREE.Mesh(pulseGeo, pulseMat);
        leakRingMesh.rotation.x = -Math.PI / 2;
        leakRingMesh.position.set(n.x, 0.14, n.z);
        scene.add(leakRingMesh);

        leakPulseMesh = new THREE.Mesh(
          new THREE.SphereGeometry(0.95, 24, 24),
          new THREE.MeshBasicMaterial({{ color: 0xDC2626, transparent: true, opacity: 0.32, wireframe: true }})
        );
        leakPulseMesh.position.copy(pos);
        scene.add(leakPulseMesh);
      }}
    }}
  }});

  // Selection ring
  selectedRingMesh = new THREE.Mesh(
    new THREE.TorusGeometry(0.88, 0.06, 12, 32),
    new THREE.MeshBasicMaterial({{ color: 0xFFFFFF }})
  );
  selectedRingMesh.rotation.x = Math.PI / 2;
  scene.add(selectedRingMesh);

  function hideInspector() {{
    const card = document.getElementById('inspector-card');
    if (card) card.style.display = 'none';
    if (selectedRingMesh) selectedRingMesh.visible = false;
  }}

  function selectNode(nodeId) {{
    const n = nodeMap[nodeId] || nodeMap['B2'];
    if (!n) return;
    selectedRingMesh.visible = true;
    selectedRingMesh.position.set(n.x, n.y, n.z);
    const card = document.getElementById('inspector-card');
    if (card) card.style.display = 'block';

    document.getElementById('insp-id').textContent = n.id;
    const badge = document.getElementById('insp-status');
    badge.textContent = n.is_leak ? 'LEAK TARGET' : n.status;
    if (n.is_leak || n.status === 'CRITICAL') {{
      badge.style.background = 'rgba(220,38,38,0.25)';
      badge.style.color = '#FCA5A5';
      badge.style.border = '1px solid #DC2626';
    }} else if (n.status === 'WARNING') {{
      badge.style.background = 'rgba(245,158,11,0.22)';
      badge.style.color = '#FDE68A';
      badge.style.border = '1px solid #F59E0B';
    }} else {{
      badge.style.background = 'rgba(22,163,74,0.22)';
      badge.style.color = '#86EFAC';
      badge.style.border = '1px solid #16A34A';
    }}

    document.getElementById('insp-zone').textContent = n.zone === 'SUPPLY' ? 'Supply Source' : `Zone ${{n.zone}}`;
    document.getElementById('insp-elev').textContent = `${{n.elevation_m.toFixed(1)}} m`;
    document.getElementById('insp-press').textContent = `${{n.pressure_m.toFixed(2)}} m`;
    document.getElementById('insp-delta').textContent = `${{n.baseline_m.toFixed(1)}}m (${{n.dev_m >= 0 ? '+' : ''}}${{n.dev_m.toFixed(2)}}m)`;
    document.getElementById('insp-risk').textContent = `${{n.risk.toFixed(1)}}%`;

    const list = document.getElementById('insp-pipes-list');
    list.innerHTML = '';
    (n.connected_flows || []).forEach(cf => {{
      const row = document.createElement('div');
      row.className = 'insp-pipe-item';
      row.innerHTML = `<span>${{cf.pipe}}</span><b>${{cf.flow_lps.toFixed(2)}} L/s</b>`;
      list.appendChild(row);
    }});
  }}
  selectedRingMesh.visible = false;
  document.getElementById('insp-close').addEventListener('click', hideInspector);

  // 3. Pipelines (3D Tubes) + Glowing Flow Particles
  const flowParticles = [];
  const pipeMeshes = [];

  DATA.pipes.forEach(p => {{
    const sNode = nodeMap[p.start];
    const eNode = nodeMap[p.end];
    if (!sNode || !eNode) return;

    const pColor = p.level === 'danger' ? COLORS.danger
                 : p.level === 'warning' ? COLORS.warning
                 : COLORS.normalPipe;
    const radius = Math.max(0.14, Math.min(0.32, p.diameter_m * 1.25));

    const curve = new THREE.LineCurve3(sNode.vec, eNode.vec);
    const tubeGeo = new THREE.TubeGeometry(curve, 12, radius, 14, false);
    const tubeMat = new THREE.MeshPhysicalMaterial({{
      color: pColor,
      emissive: pColor,
      emissiveIntensity: p.level === 'danger' ? 0.55 : 0.25,
      transparent: true,
      opacity: p.level === 'danger' ? 0.85 : 0.55,
      roughness: 0.2,
      metalness: 0.1,
    }});
    const tube = new THREE.Mesh(tubeGeo, tubeMat);
    tube.userData = {{ zone: p.zone, level: p.level }};
    scene.add(tube);
    pipeMeshes.push(tube);

    // Glowing water-flow particles along each pipe
    if (p.flow_lps > 0.02) {{
      const count = 4;
      const reverse = p.signed_flow_lps < 0;
      const speed = Math.max(0.18, Math.min(1.4, p.flow_lps * 0.28));
      for (let i = 0; i < count; i++) {{
        const part = new THREE.Mesh(
          new THREE.SphereGeometry(radius * 0.72, 10, 10),
          new THREE.MeshBasicMaterial({{
            color: p.level === 'danger' ? 0xFCA5A5 : 0xE0F2FE,
          }})
        );
        scene.add(part);
        flowParticles.push({{
          mesh: part,
          start: reverse ? eNode.vec : sNode.vec,
          end: reverse ? sNode.vec : eNode.vec,
          t: i / count,
          speed,
          zone: p.zone,
        }});
      }}
    }}
  }});

  // 4. 3D Leak Water Spray Animation (Escaping water droplets from damaged junction/pipe)
  let leakSprayPoints = null;
  let leakDrops = [];
  if (DATA.leak.active && DATA.leak.junction && nodeMap[DATA.leak.junction]) {{
    const lNode = nodeMap[DATA.leak.junction];
    const pill = document.getElementById('leak-pill');
    const pillTxt = document.getElementById('leak-pill-text');
    pill.style.display = 'flex';
    pillTxt.textContent = `SIMULATED LEAK · Zone ${{DATA.leak.zone}} (${{DATA.leak.junction}}) · ~${{DATA.leak.loss_l_min.toFixed(1)}} L/min`;

    const dropCount = 96;
    const positions = new Float32Array(dropCount * 3);
    const colors = new Float32Array(dropCount * 3);
    for (let i = 0; i < dropCount; i++) {{
      const angle = Math.random() * Math.PI * 2;
      const spd = 0.8 + Math.random() * 1.8;
      leakDrops.push({{
        vx: Math.cos(angle) * spd,
        vy: 3.2 + Math.random() * 3.8,
        vz: Math.sin(angle) * spd,
        life: Math.random(),
      }});
      positions[i * 3] = lNode.x;
      positions[i * 3 + 1] = lNode.y;
      positions[i * 3 + 2] = lNode.z;
      colors[i * 3] = 0.22;
      colors[i * 3 + 1] = 0.85;
      colors[i * 3 + 2] = 1.0;
    }}
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({{
      size: 0.34,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }});
    leakSprayPoints = new THREE.Points(geo, mat);
    leakSprayPoints.userData = {{ origin: lNode.vec }};
    scene.add(leakSprayPoints);
  }}

  // 5. Zone Filter Logic
  let activeZoneFilter = 'ALL';
  document.querySelectorAll('.zone-btn[data-zone]').forEach(btn => {{
    btn.addEventListener('click', () => {{
      activeZoneFilter = btn.dataset.zone;
      document.querySelectorAll('.zone-btn[data-zone]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      Object.entries(zoneGroups).forEach(([z, obj]) => {{
        const match = (activeZoneFilter === 'ALL' || activeZoneFilter === z);
        obj.padMat.opacity = match ? 0.16 : 0.03;
        obj.lineMat.opacity = match ? 0.85 : 0.15;
      }});
      pipeMeshes.forEach(pm => {{
        const match = (activeZoneFilter === 'ALL' || activeZoneFilter === pm.userData.zone);
        pm.material.opacity = match ? (pm.userData.level === 'danger' ? 0.88 : 0.6) : 0.12;
      }});
      DATA.nodes.forEach(n => {{
        if (n.mesh) {{
          const match = (activeZoneFilter === 'ALL' || activeZoneFilter === n.zone);
          n.mesh.material.opacity = match ? 1.0 : 0.25;
          n.mesh.material.transparent = !match;
        }}
      }});
      if (activeZoneFilter !== 'ALL' && zoneGroups[activeZoneFilter]) {{
        const zc = zoneGroups[activeZoneFilter];
        controls.target.set(zc.cx, 2.2, 2.0);
        camera.position.set(zc.cx, 13, 18);
      }} else {{
        controls.target.copy(DEFAULT_TARGET);
        camera.position.copy(DEFAULT_CAM_POS);
      }}
    }});
  }});

  let labelsVisible = true;
  let particlesVisible = true;

  document.getElementById('btn-toggle-labels').addEventListener('click', (e) => {{
    labelsVisible = !labelsVisible;
    e.target.classList.toggle('active', labelsVisible);
    scene.traverse(obj => {{
      if (obj.isSprite) obj.visible = labelsVisible;
    }});
  }});

  document.getElementById('btn-toggle-particles').addEventListener('click', (e) => {{
    particlesVisible = !particlesVisible;
    e.target.classList.toggle('active', particlesVisible);
    flowParticles.forEach(fp => {{ fp.mesh.visible = particlesVisible; }});
  }});

  document.getElementById('btn-reset-cam').addEventListener('click', () => {{
    controls.target.copy(DEFAULT_TARGET);
    camera.position.copy(DEFAULT_CAM_POS);
    controls.update();
  }});

  // 6. Raycaster Click & Hover
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let pointerDownPos = {{ x: 0, y: 0 }};

  renderer.domElement.addEventListener('pointerdown', e => {{
    pointerDownPos = {{ x: e.clientX, y: e.clientY }};
  }});
  renderer.domElement.addEventListener('pointerup', e => {{
    const dx = e.clientX - pointerDownPos.x;
    const dy = e.clientY - pointerDownPos.y;
    if (Math.hypot(dx, dy) > 6) return; // ignore drag
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(clickableMeshes, false);
    if (hits.length > 0 && hits[0].object.userData.nodeId) {{
      selectNode(hits[0].object.userData.nodeId);
    }} else {{
      hideInspector();
    }}
  }});
  renderer.domElement.addEventListener('pointermove', e => {{
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(clickableMeshes, false);
    renderer.domElement.style.cursor = hits.length > 0 ? 'pointer' : 'grab';
  }});

  // 7. Animation Loop (Optimized: pauses when tab/iframe hidden)
  const clock = new THREE.Clock();
  function animate() {{
    requestAnimationFrame(animate);
    if (document.visibilityState === 'hidden') return;
    const dt = Math.min(0.05, clock.getDelta());
    const elapsed = clock.getElapsedTime();

    if (DATA.anim_enabled) {{
      // Flow particles
      if (particlesVisible) {{
        for (const fp of flowParticles) {{
          fp.t = (fp.t + dt * fp.speed) % 1.0;
          fp.mesh.position.lerpVectors(fp.start, fp.end, fp.t);
        }}
      }}
      // Reservoir gentle surface shimmer
      if (reservoirWaterMesh) {{
        reservoirWaterMesh.scale.y = 1.0 + Math.sin(elapsed * 2.2) * 0.03;
      }}
      // Leak pulsing beacon & ring
      if (leakRingMesh) {{
        const s = 1.0 + ((elapsed * 1.4) % 1.0) * 2.4;
        leakRingMesh.scale.set(s, s, 1);
        leakRingMesh.material.opacity = Math.max(0, 1.0 - ((elapsed * 1.4) % 1.0));
      }}
      if (leakPulseMesh) {{
        const ps = 1.0 + Math.sin(elapsed * 5.0) * 0.18;
        leakPulseMesh.scale.set(ps, ps, ps);
      }}
      // Water-escape spray fountain
      if (leakSprayPoints) {{
        const posAttr = leakSprayPoints.geometry.getAttribute('position');
        const orig = leakSprayPoints.userData.origin;
        const g = 9.8;
        for (let i = 0; i < leakDrops.length; i++) {{
          const d = leakDrops[i];
          d.life += dt * 1.25;
          if (d.life >= 1.0 || orig.y + d.vy * d.life - 0.5 * g * d.life * d.life < 0.05) {{
            d.life = 0.0;
          }}
          const t = d.life;
          posAttr.setXYZ(
            i,
            orig.x + d.vx * t,
            Math.max(0.08, orig.y + d.vy * t - 0.5 * g * t * t),
            orig.z + d.vz * t
          );
        }}
        posAttr.needsUpdate = true;
      }}
    }}

    controls.update();
    renderer.render(scene, camera);
  }}
  animate();

  window.addEventListener('resize', () => {{
    const nw = container.clientWidth || 1000;
    const nh = container.clientHeight || {height};
    camera.aspect = nw / nh;
    camera.updateProjectionMatrix();
    renderer.setSize(nw, nh);
  }});
}} catch (err) {{
  console.error('Three.js init error:', err);
  const errEl = document.getElementById('webgl-error');
  if (errEl) errEl.style.display = 'flex';
}}
</script>
</body>
</html>"""

    components.html(html_3d, height=height + 8)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        <div class="brand-header">
            <div class="brand-logo">💧</div>
            <div>
                <div class="brand-name">SmartLeak</div>
                <div class="brand-tagline">Every Drop Matters.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        engine = _engine()
        if engine:
            status = engine.network_status
            sc = "badge-danger" if status == "HIGH RISK" else \
                 "badge-warning" if status in ("WARNING", "ANOMALY DETECTED") else "badge-normal"
            st.markdown(
                f'<div style="margin-bottom:14px">'
                f'<span class="badge {sc}">{status}</span>'
                f' &nbsp; <span style="font-size:11px;color:#8FA5B1">T+{engine.seconds:03d}s / 150s</span></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="margin-bottom:14px"><span class="badge badge-warning">ENGINE OFFLINE</span></div>',
                unsafe_allow_html=True)

        pages = [
            ("📊", "Overview"),
            ("📡", "Live Monitoring"),
            ("🌐", "3D Network Map"),
            ("🔍", "Leak Detection"),
            ("📈", "Analytics"),
            ("📂", "Data Management"),
            ("🔔", "Alerts and Maintenance"),
            ("📋", "Reports"),
            ("⚙️", "Settings"),
        ]
        st.markdown(
            '<div style="font-size:9px;letter-spacing:1.5px;color:#8FA5B1;'
            'text-transform:uppercase;margin-bottom:6px">Navigation</div>',
            unsafe_allow_html=True)
        for icon, label in pages:
            is_active = (st.session_state.page == label)
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.page = label
                st.rerun()

        st.divider()
        st.markdown(
            '<div style="font-size:9px;letter-spacing:1.5px;color:#8FA5B1;'
            'text-transform:uppercase;margin-bottom:6px">Simulation Controls</div>',
            unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ Advance", use_container_width=True, type="primary"):
                e = _engine()
                if e:
                    try:
                        e.advance()
                        _record_trend(e)
                        _log_activity(e, f"Advanced simulation to T+{e.seconds:03d}s ({e.network_status}).", "INFO")
                    except Exception as err:
                        st.error(str(err))
                    st.rerun()
        with c2:
            if st.button("↺ Reset", use_container_width=True):
                e = _engine()
                if e:
                    try:
                        e.reset()
                        st.session_state.trend_time = []
                        st.session_state.trend_pressure = []
                        st.session_state.trend_flow = []
                        st.session_state.trend_risk = []
                        _record_trend(e)
                        _log_activity(e, "Simulation reset to nominal baseline (T+000s).", "NORMAL")
                    except Exception as err:
                        st.error(str(err))
                    st.rerun()

        c3, c4 = st.columns(2)
        with c3:
            if st.button("⏸ Pause" if (engine and engine.running) else "▶ Start", use_container_width=True):
                e = _engine()
                if e:
                    if e.running:
                        e.pause()
                        _log_activity(e, "Simulation paused by operator.", "INFO")
                    else:
                        e.start()
                        _log_activity(e, "Simulation stream started.", "INFO")
                    st.rerun()
        with c4:
            if st.button("▶▶ +5 Steps", use_container_width=True):
                e = _engine()
                if e:
                    for _ in range(5):
                        if e.seconds < 150:
                            e.advance(); _record_trend(e)
                    _log_activity(e, f"Fast-forwarded +5 steps to T+{e.seconds:03d}s.", "INFO")
                    st.rerun()

        # Leak Injection Controls (Any Zone A, B, C, D)
        st.markdown(
            '<div style="font-size:9px;letter-spacing:1.5px;color:#8FA5B1;'
            'text-transform:uppercase;margin-top:6px;margin-bottom:4px">Inject Leak (Any Zone)</div>',
            unsafe_allow_html=True)
        col_lz, col_btn = st.columns([1, 1])
        with col_lz:
            quick_junc = st.selectbox(
                "Leak Node",
                ["B2 (Zone B)", "A2 (Zone A)", "C2 (Zone C)", "D2 (Zone D)"],
                label_visibility="collapsed",
                key="sb_quick_leak",
            )
        with col_btn:
            if st.button("⚡ Inject", use_container_width=True):
                target_node = quick_junc.split()[0]
                e = _engine()
                if e:
                    try:
                        e.introduce_leak(target_node)
                        _record_trend(e)
                        _log_activity(e, f"Simulated leak injected at Junction {target_node}.", "WARNING")
                    except Exception as err:
                        st.error(str(err))
                    st.rerun()

        # Animation Settings Switcher
        st.divider()
        st.markdown(
            '<div style="font-size:9px;letter-spacing:1.5px;color:#8FA5B1;'
            'text-transform:uppercase;margin-bottom:6px">Visual Settings</div>',
            unsafe_allow_html=True)
        anim_toggle = st.toggle("✨ Flow & Particle Animations", value=st.session_state.anim_enabled)
        if anim_toggle != st.session_state.anim_enabled:
            st.session_state.anim_enabled = anim_toggle
            st.rerun()

        n_ds = len(st.session_state.datasets)
        sim_ok = "✓ Ready" if _engine() else "✗ Offline"
        st.markdown(
            f'<div class="info-panel" style="font-size:11px">'
            f'<b style="color:#00C8C8">{n_ds}</b> dataset(s) loaded<br>'
            f'<span style="color:#8FA5B1">Simulation: {sim_ok}</span></div>',
            unsafe_allow_html=True)


# ── PAGE 1: Overview (Operations Dashboard) ───────────────────────────────────
def _page_dashboard() -> None:
    t = _theme()
    engine = _engine()
    if engine is None:
        st.error(f"Hydraulic engine offline. {st.session_state.get('sim_error') or ''}")
        return

    # Check for active leak alert or early impending leak prediction
    highest_risk = max(engine.risk.values(), default=0.0)
    top_zone     = max(engine.risk, key=engine.risk.get) if engine.risk else "B"
    probable     = engine.probable_zone
    predicted    = getattr(engine, "predicted_leak_zone", probable)
    leak_j       = engine.leak_junction or engine.active_leak_junction or "B2"
    loss         = engine.loss_estimate_l_min

    all_pressures = list(engine.snapshot.pressures.values())
    base_pressures = list(engine.calibration.baseline_snapshot.pressures.values())
    mean_net_p = float(np.mean(all_pressures)) if all_pressures else 0.0
    base_net_p = float(np.mean(base_pressures)) if base_pressures else 0.0
    supply_flow = float(engine.snapshot.flows_lps.get("P_SUPPLY", 0.0))
    base_supply = float(engine.calibration.baseline_snapshot.flows_lps.get("P_SUPPLY", 0.0))
    alert_zones = sum(1 for z, r in engine.risk.items() if r >= 31 or engine.anomaly_flags.get(z, False))

    if highest_risk >= 31 or engine.network_status in ("HIGH RISK", "WARNING", "ANOMALY DETECTED"):
        st.markdown(f"""
        <div class="leak-alert-banner">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div>
                    <span style="font-size:13px;font-weight:800;color:{t['critical']};letter-spacing:1px;">⚠ ACTIVE LEAK DETECTED</span>
                    <span style="font-size:11px;color:{t['text']};margin-left:12px;">Suspected Location: <b>Zone {probable or predicted or top_zone} · Junction {leak_j}</b></span>
                </div>
                <div>
                    <span class="badge badge-danger">MAX RISK: {highest_risk:.1f}%</span>
                    <span style="font-size:11px;color:{t['warning']};margin-left:10px;">Est Loss: <b>~{loss:.1f} L/min</b></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif predicted is not None and engine.seconds >= 30:
        st.markdown(f"""
        <div class="leak-alert-banner" style="border-color:{t['warning']};border-left-color:{t['warning']};background:{t['warn_bg']};">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div>
                    <span style="font-size:13px;font-weight:800;color:{t['warning']};letter-spacing:1px;">🔮 EARLY PREDICTION: LEAK DEVELOPING IN ZONE {predicted}</span>
                    <span style="font-size:11px;color:{t['text']};margin-left:12px;">Mass-Balance Gradient Localized to <b>Zone {predicted}</b></span>
                </div>
                <div>
                    <span class="badge badge-warning">DEVELOPING (T+{engine.seconds:03d}s)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 6 KPI Cards: Avg Pressure, Total Supply Flow, System Health, Highest Zone Risk, Active Alerts, Reservoir Status
    zone_target = probable or predicted or top_zone
    variant_risk   = "danger" if highest_risk >= 61 else "warning" if (highest_risk >= 31 or predicted) else "success"
    variant_status = ("danger"  if engine.network_status == "HIGH RISK" else
                      "warning" if (engine.network_status in ("WARNING", "ANOMALY DETECTED") or predicted) else "success")

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(_kpi_card(
            "Average Network Pressure",
            f"{mean_net_p:.2f} m",
            f"12-Node Mean ({mean_net_p - base_net_p:+.2f} m vs baseline)",
        ), unsafe_allow_html=True)
    with k2:
        st.markdown(_kpi_card(
            "Total Supply Flow",
            f"{supply_flow:.3f} L/s",
            f"P_SUPPLY Trunk · ΔQ: {supply_flow - base_supply:+.3f} L/s",
        ), unsafe_allow_html=True)
    with k3:
        st.markdown(_kpi_card(
            "System Health",
            engine.network_status,
            f"Phase: {engine.phase} · T+{engine.seconds:03d}s / 150s",
            variant=variant_status,
        ), unsafe_allow_html=True)

    k4, k5, k6 = st.columns(3)
    with k4:
        st.markdown(_kpi_card(
            "Highest Zone Risk",
            f"{highest_risk:.1f}%",
            f"Peak: Zone {top_zone} (Suspected: {leak_j if (probable or predicted or engine.leak_severity > 0) else 'None'})",
            variant=variant_risk,
        ), unsafe_allow_html=True)
    with k5:
        st.markdown(_kpi_card(
            "Active Alerts",
            f"{alert_zones} Zone(s)",
            f"Est. Water Loss: ~{loss:.1f} L/min",
            variant="danger" if alert_zones > 0 and highest_risk >= 61 else "warning" if alert_zones > 0 else "success",
        ), unsafe_allow_html=True)
    with k6:
        st.markdown(_kpi_card(
            "Reservoir Status",
            "ONLINE · 79.0 m",
            f"Fixed Head Source · Outflow {supply_flow:.3f} L/s",
            variant="success",
        ), unsafe_allow_html=True)

    st.divider()

    # CENTRAL VISUALISATION (Left) + RIGHT PANEL (NeuralFlow AI Diagnostics, Suspicious Junctions, Risk, Alerts)
    main_col, right_col = st.columns([2.25, 1.0])

    with main_col:
        hdr_col, mode_col = st.columns([2.5, 1.4])
        with hdr_col:
            _section_header("🌐", "3D Hydraulic Digital Twin (HydroVision)")
        with mode_col:
            selected_view = st.radio(
                "View Mode",
                ["3D Interactive", "2D Vector"],
                index=0 if st.session_state.view_mode == "3D Interactive" else 1,
                horizontal=True,
                label_visibility="collapsed",
                key="dash_view_mode",
            )
            if selected_view != st.session_state.view_mode:
                st.session_state.view_mode = selected_view
                st.rerun()

        sc1, sc2, sc3, sc4, sc5 = st.columns([1.0, 1.0, 1.8, 1.1, 0.9])
        with sc1:
            if st.button("▶ Advance", key="dash_step1", type="primary", use_container_width=True):
                try:
                    engine.advance(); _record_trend(engine)
                    _log_activity(engine, f"Advanced to T+{engine.seconds:03d}s ({engine.network_status}).", "INFO")
                except Exception as err:
                    st.error(str(err))
                st.rerun()
        with sc2:
            if st.button("▶▶ +5", key="dash_step5", use_container_width=True):
                try:
                    for _ in range(5):
                        if engine.seconds < 150:
                            engine.advance(); _record_trend(engine)
                    _log_activity(engine, f"Fast-forwarded +5 steps to T+{engine.seconds:03d}s.", "INFO")
                except Exception as err:
                    st.error(str(err))
                st.rerun()
        with sc3:
            dash_junc = st.selectbox(
                "Leak Target Junction",
                ["B2 (Zone B)", "A2 (Zone A)", "C2 (Zone C)", "D2 (Zone D)",
                 "A1 (Zone A)", "A3 (Zone A)", "B1 (Zone B)", "B3 (Zone B)",
                 "C1 (Zone C)", "C3 (Zone C)", "D1 (Zone D)", "D3 (Zone D)"],
                label_visibility="collapsed",
                key="dash_leak_junc",
            )
        with sc4:
            if st.button("⚡ Inject Leak", key="dash_inject", use_container_width=True):
                try:
                    j_id = dash_junc.split()[0]
                    engine.introduce_leak(j_id)
                    _record_trend(engine)
                    _log_activity(engine, f"Simulated leak injected at Junction {j_id}.", "WARNING")
                except Exception as err:
                    st.error(str(err))
                st.rerun()
        with sc5:
            if st.button("↺ Reset", key="dash_reset", use_container_width=True):
                try:
                    engine.reset()
                    st.session_state.trend_time = []
                    st.session_state.trend_pressure = []
                    st.session_state.trend_flow = []
                    st.session_state.trend_risk = []
                    _record_trend(engine)
                    _log_activity(engine, "Network reset to nominal baseline (T+000s).", "NORMAL")
                except Exception as err:
                    st.error(str(err))
                st.rerun()

        if st.session_state.view_mode == "3D Interactive":
            render_3d_network(
                engine,
                selected_node=st.session_state.selected_node,
                anim_enabled=st.session_state.anim_enabled,
                height=520,
            )
        else:
            render_animated_2d_network(
                engine,
                selected_node=st.session_state.selected_node,
                anim_enabled=st.session_state.anim_enabled,
                height=480,
            )

    with right_col:
        _section_header("🧠", "NeuralFlow AI Diagnostics")
        diag_zone = probable or predicted or top_zone
        ml_score = engine.ml_scores.get(diag_zone, 0.0)
        ml_flag = engine.ml_flags.get(diag_zone, False)
        st.markdown(f"""
        <div class="info-panel" style="border-left-color:{t['purple']};">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-size:11px;font-weight:700;color:{t['purple']};">ISOLATION FOREST · ZONE {diag_zone}</span>
                <span class="badge {'badge-danger' if ml_flag else 'badge-purple'}">{'ANOMALY' if ml_flag else 'NOMINAL'}</span>
            </div>
            <div style="font-size:11px;color:{t['muted']};margin-bottom:6px;">
                Decision Score: <b class="mono" style="color:{t['text']}">{ml_score:.4f}</b>
            </div>
            <div style="font-size:11.5px;color:{t['text']};line-height:1.45;">
                {engine.get_zone_explanation(diag_zone)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Zone Risk Indicators
        st.markdown(f'<div class="kpi-label" style="margin-top:10px;">Zone Risk Indicators</div>', unsafe_allow_html=True)
        for z in ["A", "B", "C", "D"]:
            zr = engine.risk.get(z, 0.0)
            bar_c = t["critical"] if zr >= 61 else t["warning"] if zr >= 31 else t["success"]
            st.markdown(f"""
            <div style="margin-bottom:7px;">
                <div style="display:flex;justify-content:space-between;font-size:11px;">
                    <span><b>Zone {z}</b></span>
                    <span class="mono" style="color:{bar_c};font-weight:700;">{zr:.1f}%</span>
                </div>
                <div class="risk-progress-bar">
                    <div class="fill" style="--fill-pct:{min(100.0, max(2.0, zr)):.1f}%;background:{bar_c};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Suspicious Junctions (Top 3 by pressure drop)
        st.markdown(f'<div class="kpi-label" style="margin-top:12px;">Suspicious Junctions (ΔP)</div>', unsafe_allow_html=True)
        j_sorted = sorted(engine.get_junction_telemetry(), key=lambda x: x["dev_m"])[:3]
        for jitem in j_sorted:
            sc_col = t["critical"] if jitem["status"] == "CRITICAL" else t["warning"] if jitem["status"] == "WARNING" else t["cyan"]
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:11px;padding:4px 8px;'
                f'margin-bottom:4px;border-radius:6px;background:{t["elevated"]};border:1px solid {t["border"]};">'
                f'<span><b>{jitem["node"]}</b> (Z{jitem["zone"]})</span>'
                f'<span class="mono" style="color:{sc_col};">{jitem["dev_m"]:+.2f} m · {jitem["status"]}</span></div>',
                unsafe_allow_html=True,
            )

        # Recent Alerts
        st.markdown(f'<div class="kpi-label" style="margin-top:12px;">Recent Alerts &amp; Events</div>', unsafe_allow_html=True)
        for ev in st.session_state.get("activity_log", [])[:3]:
            st.markdown(
                f'<div style="font-size:10.5px;padding:4px 8px;margin-bottom:4px;border-radius:6px;'
                f'background:{t["surface"]};border:1px solid {t["border"]};">'
                f'<span class="mono" style="color:{t["cyan"]};">[{ev["time"]}]</span> {ev["message"]}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # BOTTOM SECTION: Pressure History, Flow History, Zone Comparisons & Simulation Events
    _section_header("📈", "Hydraulic History & Zone Comparisons (2D)")
    left, right = st.columns([1, 1])
    with left:
        # Pressure comparison across all 4 zones
        zones = ["A", "B", "C", "D"]
        curr_press = [engine.measurements[z].pressure_m for z in zones]
        base_press = [engine.calibration.baselines[z].pressure_m for z in zones]
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(name="Baseline Head (m)", x=zones, y=base_press,
                                marker_color="#263A49", opacity=0.7))
        fig_p.add_trace(go.Bar(name="Observed Head (m)", x=zones, y=curr_press,
                                marker_color=[_risk_color(engine.risk[z]) for z in zones]))
        fig_p.update_layout(
            margin=dict(l=10, r=10, t=20, b=10), barmode="group",
            yaxis=dict(title="Head Pressure (m H₂O)"),
            xaxis=dict(title="Zone"),
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=10)),
        )
        _apply_plotly_theme(fig_p, height=230)
        st.plotly_chart(fig_p, use_container_width=True, config=dict(displayModeBar=False))

    with right:
        # Flow deviation chart across pipes
        curr_flow = [engine.measurements[z].flow_lps for z in zones]
        base_flow = [engine.calibration.baselines[z].flow_lps for z in zones]
        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(name="Baseline Flow (L/s)", x=zones, y=base_flow,
                                marker_color=t["border"], opacity=0.75))
        fig_f.add_trace(go.Bar(name="Observed Flow (L/s)", x=zones, y=curr_flow,
                                marker_color=t["cyan"]))
        fig_f.update_layout(
            margin=dict(l=10, r=10, t=20, b=10), barmode="group",
            yaxis=dict(title="Flow Rate (L/s)"),
            xaxis=dict(title="Zone"),
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=10)),
        )
        _apply_plotly_theme(fig_f, height=230)
        st.plotly_chart(fig_f, use_container_width=True, config=dict(displayModeBar=False))


# ── PAGE 2: 3D / 2D Network Twin Inspector ────────────────────────────────────
def _page_network_map() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">3D & 2D Water Network Twin</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:16px">'
        'Interactive 3D Three.js & 2D SVG topological twin displaying live EPANET hydraulics and dynamic flow particles. '
        '<span class="sim-tag">SIMULATED DATA</span></p>', unsafe_allow_html=True)

    engine = _engine()
    if engine is None:
        st.error("Hydraulic engine not available.")
        return

    # Junction selector & 2D/3D view switch
    col_sel, col_mode, col_stat = st.columns([1.8, 1.4, 2.4])
    with col_sel:
        all_juncs = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "D3"]
        sel_node = st.selectbox("Inspect Junction Telemetry", all_juncs, index=all_juncs.index(st.session_state.selected_node) if st.session_state.selected_node in all_juncs else 4)
        st.session_state.selected_node = sel_node
    with col_mode:
        st.markdown('<div style="font-size:11px;color:#8FA5B1;margin-bottom:4px;">Visualization Mode</div>', unsafe_allow_html=True)
        nm_view = st.radio(
            "Network View Mode",
            ["3D Interactive", "2D Vector"],
            index=0 if st.session_state.view_mode == "3D Interactive" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="netmap_view_mode",
        )
        if nm_view != st.session_state.view_mode:
            st.session_state.view_mode = nm_view
            st.rerun()
    with col_stat:
        st.markdown(f"""
        <div style="font-size:11px;color:#8FA5B1;padding-top:22px;">
            Target Leak Junction: <b style="color:{C_DANGER}">{engine.leak_junction or 'None (Nominal)'}</b> &nbsp;·&nbsp;
            Flow Animation: <b style="color:#00C8C8">{'ACTIVE' if st.session_state.anim_enabled else 'PAUSED'}</b> &nbsp;·&nbsp;
            Sim Time: <b style="color:#E8F1F5">T+{engine.seconds:03d}s</b>
        </div>
        """, unsafe_allow_html=True)

    # Render Main 3D or 2D Network Diagram
    if st.session_state.view_mode == "3D Interactive":
        render_3d_network(
            engine,
            selected_node=st.session_state.selected_node,
            anim_enabled=st.session_state.anim_enabled,
            height=560,
        )
    else:
        render_animated_2d_network(
            engine,
            selected_node=st.session_state.selected_node,
            anim_enabled=st.session_state.anim_enabled,
            height=500,
        )

    st.divider()

    # Detailed Inspector Card for Selected Node
    _section_header("🔍", f"Junction Inspector — {sel_node}")
    telemetry_list = engine.get_junction_telemetry()
    node_data = next((item for item in telemetry_list if item["node"] == sel_node), None)
    if node_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Elevation", f"{node_data['elevation_m']:.1f} m")
        c2.metric("Observed Head", f"{node_data['pressure_m']:.3f} m")
        c3.metric("Baseline Head", f"{node_data['baseline_m']:.3f} m")
        c4.metric("Deviation (ΔP)", f"{node_data['dev_m']:+.3f} m", delta=f"{node_data['dev_pct']:+.1f}%")

    # Flow Table for Connected Pipes to this Node
    pipe_telemetry = engine.get_pipe_telemetry()
    connected_pipes = [p for p in pipe_telemetry if p["start"] == sel_node or p["end"] == sel_node]
    if connected_pipes:
        st.markdown('<div class="kpi-label" style="margin-top:12px;">Connected Pipe Hydraulics</div>', unsafe_allow_html=True)
        cdf = pd.DataFrame(connected_pipes)[["pipe", "direction", "flow_lps", "baseline_lps", "dev_lps", "velocity_m_s"]]
        cdf.columns = ["Pipe ID", "Direction", "Observed Flow (L/s)", "Baseline Flow (L/s)", "Deviation (L/s)", "Velocity (m/s)"]
        st.dataframe(cdf.style.format({
            "Observed Flow (L/s)": "{:.4f}",
            "Baseline Flow (L/s)": "{:.4f}",
            "Deviation (L/s)": "{:+.4f}",
            "Velocity (m/s)": "{:.3f}",
        }), use_container_width=True)


# ── PAGE 3: Live Monitoring ───────────────────────────────────────────────────
def _page_monitoring() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Live Monitoring</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:16px">'
        'Real-time hydraulic sensor readings from the WNTR/EPANET simulation engine. '
        '<span class="sim-tag">SIMULATED DATA</span></p>', unsafe_allow_html=True)

    engine = _engine()
    if engine is None:
        st.error("Hydraulic engine not available.")
        return

    ca, cb, _ = st.columns([2, 2, 4])
    with ca:
        if st.button("▶ Run 1 Step", type="primary", use_container_width=True):
            try:
                engine.advance(); _record_trend(engine)
            except Exception as e:
                st.error(str(e))
            st.rerun()
    with cb:
        if st.button("▶▶ Run 5 Steps", use_container_width=True):
            for _ in range(5):
                if engine.seconds < 150:
                    engine.advance(); _record_trend(engine)
            st.rerun()

    st.divider()

    _section_header("🔧", "Junction Telemetry  (12 Nodes)")
    try:
        jdf = pd.DataFrame(engine.get_junction_telemetry())
        # Add visual pulse dot indicator to junction table
        def _status_with_indicator(st_val: str) -> str:
            dot_class = "live-dot-danger" if st_val == "CRITICAL" else \
                        "live-dot-warning" if st_val == "WARNING" else \
                        "live-dot-anomaly" if st_val == "ANOMALY" else "live-dot-normal"
            return f'<span class="live-dot {dot_class}"></span>{st_val}'
        
        jdf["Status_Display"] = jdf["status"].apply(_status_with_indicator)
        display_j = jdf[["node","zone","elevation_m","pressure_m","baseline_m","dev_m","dev_pct","Status_Display"]].rename(columns={
            "node":"Junction","zone":"Zone","elevation_m":"Elev (m)",
            "pressure_m":"Pressure (m)","baseline_m":"Baseline (m)",
            "dev_m":"Δ (m)","dev_pct":"Δ (%)","Status_Display":"Status",
        })
        st.write(
            display_j.style.format({
                "Elev (m)":"{:.1f}","Pressure (m)":"{:.3f}",
                "Baseline (m)":"{:.3f}","Δ (m)":"{:+.3f}","Δ (%)":"{:+.2f}%"
            }).to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.warning(f"Junction telemetry unavailable: {exc}")

    st.divider()
    _section_header("🔩", "Pipe Hydraulics  (12 Pipes)")
    try:
        pdf = pd.DataFrame(engine.get_pipe_telemetry())
        display_p = pdf[["pipe","zone","direction","flow_lps","baseline_lps","dev_lps","dev_pct","velocity_m_s"]].rename(columns={
            "pipe":"Pipe ID","zone":"Zone","direction":"Flow Direction",
            "flow_lps":"Flow (L/s)","baseline_lps":"Baseline (L/s)",
            "dev_lps":"Δ (L/s)","dev_pct":"Δ (%)","velocity_m_s":"Velocity (m/s)",
        })
        st.dataframe(
            display_p.style.format({
                "Flow (L/s)":"{:.4f}","Baseline (L/s)":"{:.4f}",
                "Δ (L/s)":"{:+.4f}","Δ (%)":"{:+.2f}%","Velocity (m/s)":"{:.4f}",
            }),
            use_container_width=True, height=280,
        )
    except Exception as exc:
        st.warning(f"Pipe telemetry unavailable: {exc}")

    if st.session_state.trend_time:
        st.divider()
        _section_header("📈", "Live Pressure, Flow & Risk Line Charts (Historical Filter)")
        t_all = st.session_state.trend_time
        p_all = st.session_state.trend_pressure
        f_all = st.session_state.trend_flow
        r_all = st.session_state.trend_risk

        # Historical time-range filter
        if len(t_all) >= 2 and min(t_all) < max(t_all):
            t_min, t_max = int(min(t_all)), int(max(t_all))
            sel_range = st.slider(
                "Historical Time-Range Filter (seconds)",
                min_value=t_min,
                max_value=t_max,
                value=(t_min, t_max),
                key="mon_time_range",
            )
            idxs = [i for i, tv in enumerate(t_all) if sel_range[0] <= tv <= sel_range[1]]
        else:
            idxs = list(range(len(t_all)))

        t = [t_all[i] for i in idxs]
        p = [p_all[i] for i in idxs]
        f = [f_all[i] for i in idxs]
        r = [r_all[i] for i in idxs]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Zone Pressure (m H₂O)", "Net Flow Rate (L/s)", "Leak Risk Score (%)"],
        )
        fig.add_trace(go.Scatter(
            x=t, y=p, mode="lines+markers",
            line=dict(color=C_CYAN, width=2.5, shape="spline"),
            fill="tozeroy", fillcolor="rgba(0,229,255,0.10)",
            marker=dict(size=5),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=f, mode="lines+markers",
            line=dict(color=C_BLUE, width=2.5, shape="spline"),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.10)",
            marker=dict(size=5),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=t, y=r, mode="lines+markers",
            line=dict(color=C_DANGER, width=2.5, shape="spline"),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.12)",
            marker=dict(size=5),
        ), row=1, col=3)
        fig.add_hline(y=61, line_dash="dash", line_color=C_DANGER, opacity=0.55, row=1, col=3)
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=20), showlegend=False,
            transition=dict(duration=350, easing="cubic-in-out"),
        )
        _apply_plotly_theme(fig, height=270)
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

        # Zone-wise comparisons
        st.divider()
        _section_header("🧭", "Zone-Wise Live Comparison (Zones A–D)")
        z_rows = []
        for z in ["A", "B", "C", "D"]:
            m = engine.measurements[z]
            b = engine.calibration.baselines[z]
            z_rows.append({
                "Zone": f"Zone {z}",
                "Pressure (m)": round(m.pressure_m, 3),
                "Baseline P (m)": round(b.pressure_m, 3),
                "ΔP (m)": round(m.pressure_m - b.pressure_m, 3),
                "Net Flow (L/s)": round(m.flow_lps, 4),
                "ΔQ (L/s)": round(m.flow_lps - b.flow_lps, 4),
                "Risk (%)": round(engine.risk[z], 1),
            })
        st.dataframe(pd.DataFrame(z_rows), use_container_width=True, hide_index=True)


# ── PAGE 4: Leak Detection ────────────────────────────────────────────────────
def _page_leak_detection() -> None:
    t = _theme()
    st.markdown(f'<h1 style="font-size:22px;font-weight:800">NeuralFlow AI Diagnostics &amp; Leak Detection</h1>',
                unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{t["muted"]};font-size:12px;margin-bottom:16px">'
        'Scikit-Learn Isolation Forest anomaly scoring + multi-factor hydraulic risk index (not a calibrated probability). '
        '<span class="sim-tag">SIMULATED DATA</span></p>', unsafe_allow_html=True)

    engine = _engine()
    if engine is None:
        st.error("Hydraulic engine not available.")
        return

    # Manual Leak Selector for ANY Zone / Junction
    with st.expander("⚙️ Scenario Injection: Select Any Zone (A, B, C, D) & Junction", expanded=True):
        col_j, col_btn1, col_btn2 = st.columns([2, 1, 1])
        junctions = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "D3"]
        with col_j:
            sel_junc = st.selectbox("Select Target Leak Junction:", junctions, index=4)
        with col_btn1:
            if st.button(f"⚡ Inject Leak at {sel_junc}", type="primary", use_container_width=True):
                try:
                    engine.introduce_leak(sel_junc)
                    _record_trend(engine)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        with col_btn2:
            if st.button("↺ Reset to Baseline", use_container_width=True):
                engine.reset()
                st.session_state.trend_time = []
                st.session_state.trend_pressure = []
                st.session_state.trend_flow = []
                st.session_state.trend_risk = []
                st.rerun()

    st.divider()
    _section_header("🚨", "4-Zone Risk Assessment")
    priorities = engine.priorities()
    cols = st.columns(4)
    for idx, (zone, score, label) in enumerate(priorities):
        with cols[idx]:
            variant = "danger" if score >= 61 else "warning" if score >= 31 else "success"
            meas = engine.measurements.get(zone)
            base = engine.calibration.baselines.get(zone)
            p_dev = (meas.pressure_m - base.pressure_m) if meas and base else 0.0
            f_dev = (meas.flow_lps  - base.flow_lps)   if meas and base else 0.0
            ml_flag = engine.ml_flags.get(zone, False)
            st.markdown(_kpi_card(
                f"Zone {zone} Risk", f"{score:.1f}%",
                f"{label} | ML: {'⚠ ANOMALY' if ml_flag else '✓ NORMAL'}",
                variant=variant
            ), unsafe_allow_html=True)
            # Animated risk progress bar
            bar_color = C_DANGER if score >= 61 else C_WARNING if score >= 31 else C_SUCCESS
            st.markdown(
                f'<div class="risk-progress-bar">'
                f'<div class="fill" style="--fill-pct:{min(100.0, max(2.0, score)):.1f}%; background:{bar_color};"></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="font-size:10px;color:#8FA5B1;margin-top:4px;padding-left:4px">'
                f'ΔP: <b style="color:#E8F1F5">{p_dev:+.3f} m</b>'
                f'&nbsp; ΔQ: <b style="color:#E8F1F5">{f_dev:+.4f} L/s</b></div>',
                unsafe_allow_html=True)

    st.divider()
    _section_header("🤖", "Isolation Forest Model Explanation & Suspicious Junction Ranking")
    col_ml, col_jrank = st.columns([1.2, 1.0])
    with col_ml:
        for zone, score, label in priorities:
            with st.expander(f"Zone {zone}  —  Risk: {score:.1f}%  ({label})  |  IF Score: {engine.ml_scores.get(zone, 0.0):.4f}", expanded=(score >= 31)):
                exp = engine.get_zone_explanation(zone)
                contrib = engine.contributions.get(zone, {})
                st.markdown(
                    f'<div class="info-panel" style="font-size:12px">'
                    f'{exp}<br><br>'
                    f'<span class="mono" style="font-size:11px;color:{t["muted"]};">'
                    f'Factor Breakdown — Pressure: {contrib.get("pressure", 0):.1f}/35 · '
                    f'Flow: {contrib.get("flow", 0):.1f}/35 · '
                    f'Isolation Forest ML: {contrib.get("ml", 0):.1f}/20 · '
                    f'Persistence: {contrib.get("persistence", 0):.1f}/10</span></div>',
                    unsafe_allow_html=True,
                )
    with col_jrank:
        j_ranked = sorted(engine.get_junction_telemetry(), key=lambda x: x["dev_m"])
        j_df = pd.DataFrame(j_ranked)[["node", "zone", "pressure_m", "baseline_m", "dev_m", "status"]]
        j_df.columns = ["Junction", "Zone", "Pressure (m)", "Baseline (m)", "ΔP (m)", "Status"]
        st.dataframe(
            j_df.style.format({"Pressure (m)": "{:.2f}", "Baseline (m)": "{:.2f}", "ΔP (m)": "{:+.3f}"}),
            use_container_width=True, height=260, hide_index=True,
        )

    st.divider()
    _section_header("🔧", "Maintenance Work Orders")
    recs = engine.get_maintenance_recommendations()
    rdf = pd.DataFrame(recs)[["rank","work_order_id","zone","score","label","sla","action","est_loss_l_min"]]
    rdf.columns = ["Rank","Work Order","Zone","Risk","Priority","SLA","Action Plan","Est Loss (L/min)"]
    st.dataframe(
        rdf.style.format({"Risk":"{:.1f}","Est Loss (L/min)":"{:.2f}"})
           .applymap(lambda v: f"color:{C_DANGER}" if v=="CRITICAL"
                          else f"color:{C_WARNING}" if v=="HIGH"
                          else f"color:{C_SUCCESS}" if v=="LOW" else "", subset=["Priority"]),
        use_container_width=True, height=200,
    )


# ── PAGE 5: Analytics ─────────────────────────────────────────────────────────
def _page_analytics() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Analytics</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:16px">'
        'Multi-zone hydraulic performance analysis and statistical distribution. <span class="sim-tag">SIMULATED DATA</span></p>',
        unsafe_allow_html=True)

    engine = _engine()
    if engine is None:
        st.error("Hydraulic engine not available.")
        return

    zones = ["A","B","C","D"]
    curr_press  = [engine.measurements[z].pressure_m for z in zones]
    base_press  = [engine.calibration.baselines[z].pressure_m for z in zones]
    curr_flow   = [engine.measurements[z].flow_lps for z in zones]
    base_flow   = [engine.calibration.baselines[z].flow_lps for z in zones]
    risk_scores = [engine.risk[z] for z in zones]

    _section_header("📊", "Pressure & Flow Comparison — All Zones")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Zone Pressure vs Baseline (m H₂O)","Zone Flow vs Baseline (L/s)"])
    for idx, zone in enumerate(zones):
        c = _risk_color(engine.risk[zone])
        fig.add_trace(go.Scatter(
            x=["Baseline","Current"], y=[base_press[idx], curr_press[idx]],
            mode="lines+markers", name=f"Zone {zone} Pressure",
            line=dict(color=c, width=2), marker=dict(size=8),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=["Baseline","Current"], y=[base_flow[idx], curr_flow[idx]],
            mode="lines+markers", name=f"Zone {zone} Flow",
            line=dict(color=c, width=2, dash="dot"), marker=dict(size=8), showlegend=False,
        ), row=1, col=2)
    t = _theme()
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10),
                      legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
    _apply_plotly_theme(fig, height=300)
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    # Radar + scatter
    st.divider()
    _section_header("🕸️", "Zone Risk Radar Analysis")
    col_r, col_s = st.columns(2)
    with col_r:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=risk_scores + [risk_scores[0]], theta=zones + [zones[0]],
            fill="toself",
            line=dict(color=t["critical"] if max(risk_scores)>=61 else t["warning"], width=2),
            fillcolor="rgba(239,68,68,.15)" if max(risk_scores)>=61 else "rgba(245,158,11,.12)",
        ))
        fig_radar.update_layout(
            polar=dict(bgcolor=t["elevated"],
                       radialaxis=dict(visible=True, range=[0,100], gridcolor=t["border"]),
                       angularaxis=dict(gridcolor=t["border"])),
            margin=dict(l=10,r=10,t=10,b=10), showlegend=False,
        )
        _apply_plotly_theme(fig_radar, height=300)
        st.plotly_chart(fig_radar, use_container_width=True, config=dict(displayModeBar=False))
    with col_s:
        p_devs = [curr_press[i]-base_press[i] for i in range(4)]
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=p_devs, y=risk_scores, mode="markers+text",
            text=[f"Zone {z}" for z in zones], textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=16, color=risk_scores, colorscale="RdYlGn_r",
                        cmin=0, cmax=100, showscale=True,
                        colorbar=dict(title="Risk", tickfont=dict(size=9))),
        ))
        fig_sc.update_layout(
            xaxis=dict(title="Pressure Deviation (m)"),
            yaxis=dict(title="Risk Score (%)", range=[0,105]),
            margin=dict(l=10,r=10,t=10,b=40),
        )
        _apply_plotly_theme(fig_sc, height=300)
        st.plotly_chart(fig_sc, use_container_width=True, config=dict(displayModeBar=False))


# ── PAGE 6: Data Upload ───────────────────────────────────────────────────────
def _page_data_upload() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Data Upload</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:16px">'
        'Upload CSV or Excel datasets for automated quality assessment and exploration.</p>',
        unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV or Excel file",
                                 type=["csv","xlsx","xls"], label_visibility="collapsed")
    if uploaded:
        df = _read_upload(uploaded)
        if df is not None:
            exists = any(d["name"] == uploaded.name for d in st.session_state.datasets)
            if not exists:
                st.session_state.datasets.append({"name":uploaded.name,"data":df,"source":"User Upload"})
                st.success(f"✓ Loaded '{uploaded.name}'  ({df.shape[0]} rows × {df.shape[1]} cols)")
            else:
                for d in st.session_state.datasets:
                    if d["name"] == uploaded.name:
                        d["data"] = df
                st.info(f"Dataset '{uploaded.name}' updated.")

    if not st.session_state.datasets:
        st.info("No datasets uploaded yet. Upload a CSV/Excel file or check the sample files in data/samples/.")
        return

    st.divider()
    _section_header("📂", f"Loaded Datasets  ({len(st.session_state.datasets)})")
    ds_names = [ds["name"] for ds in st.session_state.datasets]
    sel_name = st.selectbox("Select dataset to preview", ds_names)
    ds = next(d for d in st.session_state.datasets if d["name"] == sel_name)
    df = ds["data"]

    q = _quality_score(df)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Rows", q["rows"])
    mc2.metric("Columns", q["cols"])
    mc3.metric("Missing Values", q["missing"])
    mc4.metric("Quality Score", f"{q['score']:.1f}/100")

    with st.expander("🔎 Column Schema & Data Types"):
        ci = pd.DataFrame({
            "Column": df.columns,
            "Dtype":  [str(df[c].dtype)        for c in df.columns],
            "Non-Null":[df[c].notna().sum()     for c in df.columns],
            "Null":   [df[c].isna().sum()       for c in df.columns],
            "Unique": [df[c].nunique()           for c in df.columns],
        })
        st.dataframe(ci, use_container_width=True)

    st.markdown('<div class="kpi-label" style="margin:10px 0 4px">Data Preview  (first 50 rows)</div>',
                unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=280)


# ── PAGE 7: Reports & Export ──────────────────────────────────────────────────
def _page_reports() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Reports & Export</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:16px">'
        'Generate and download monitoring summaries and telemetry datasets.</p>',
        unsafe_allow_html=True)

    engine = _engine()
    _section_header("📡", "Simulation Telemetry Export")
    if engine:
        col1, col2 = st.columns(2)
        with col1:
            try:
                jdf  = pd.DataFrame(engine.get_junction_telemetry())
                jcsv = jdf.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Junction Telemetry CSV", data=jcsv,
                                    file_name="smartleak_junction_telemetry.csv",
                                    mime="text/csv", use_container_width=True)
            except Exception as e:
                st.warning(f"Junction data: {e}")
        with col2:
            try:
                pdf  = pd.DataFrame(engine.get_pipe_telemetry())
                pcsv = pdf.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Pipe Hydraulics CSV", data=pcsv,
                                    file_name="smartleak_pipe_telemetry.csv",
                                    mime="text/csv", use_container_width=True)
            except Exception as e:
                st.warning(f"Pipe data: {e}")
        try:
            rdf  = pd.DataFrame(engine.get_maintenance_recommendations())
            rcsv = rdf.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Maintenance Work Orders CSV", data=rcsv,
                                file_name="smartleak_work_orders.csv",
                                mime="text/csv", use_container_width=True)
        except Exception as e:
            st.warning(f"Maintenance data: {e}")

        st.divider()
        _section_header("📋", "Executive Summary Preview")
        priorities = engine.priorities()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "SmartLeak — Hydraulic Monitoring Report",
            f"Generated: {now}",
            f"Simulation Time: T+{engine.seconds}s / 150s",
            f"Phase: {engine.phase}",
            f"Network Status: {engine.network_status}",
            "",
            "── Zone Risk Priorities ──",
        ]
        for zone, score, label in priorities:
            meas = engine.measurements.get(zone)
            base = engine.calibration.baselines.get(zone)
            if meas and base:
                p_dev = meas.pressure_m - base.pressure_m
                f_dev = meas.flow_lps   - base.flow_lps
                ml    = "YES" if engine.ml_flags.get(zone) else "NO"
                lines.append(
                    f"  Zone {zone}: {score:.1f}% [{label}]  |  "
                    f"ΔP: {p_dev:+.3f} m  |  ΔQ: {f_dev:+.4f} L/s  |  ML: {ml}")
            exp = engine.get_zone_explanation(zone)
            lines.append(f"  Explanation: {exp[:100]}...")
            lines.append("")
        lines += ["── Maintenance Work Orders ──"]
        for rec in engine.get_maintenance_recommendations():
            lines.append(f"  [{rec['rank']}] {rec['work_order_id']} | Zone {rec['zone']} | {rec['label']} | {rec['sla']}")
            lines.append(f"      Action: {rec['action']}")
            lines.append("")
        summary = "\n".join(lines)
        st.code(summary, language=None)
        st.download_button("⬇ Download Report as TXT", data=summary.encode("utf-8"),
                            file_name="smartleak_report.txt", mime="text/plain",
                            use_container_width=True)
    else:
        st.warning("Hydraulic engine not available.")

    st.divider()
    _section_header("ℹ️", "About SmartLeak")
    st.markdown("""
    <div class="info-panel">
    <b style="color:#00C8C8">SmartLeak — Intelligent Water Network Leakage Detection & Risk Monitoring</b><br><br>
    <b>Simulation Engine:</b> WNTR (Water Network Tool for Resilience) + EPANET 2.2<br>
    <b>ML Model:</b> Scikit-Learn Isolation Forest (n=200, contamination=0.14)<br>
    <b>Network:</b> 1 Reservoir · 4 Zones (A–D) · 12 Junctions · 12 Pipes<br>
    <b>Scenario:</b> Zone B / Junction B2 emitter leak (progressive, T=30–150 s)<br>
    <b>Risk scoring:</b> Pressure 35% · Flow 35% · ML 20% · Persistence 10%<br><br>
    <span style="color:#8FA5B1;font-size:11px">
    ⚠ All measurements are SIMULATED DATA from the WNTR/EPANET hydraulic solver.
    No physical sensors are connected. This system does not detect real-world leaks unless real sensor data is integrated.
    </span>
    </div>""", unsafe_allow_html=True)


# ── PAGE 7: Alerts & Maintenance ──────────────────────────────────────────────
def _page_alerts_maintenance() -> None:
    st.markdown('<h1 style="color:#EAF4FB;font-size:22px;font-weight:800">Alerts &amp; Maintenance</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:16px">'
        'Active SCADA alarm feed, zone prioritization, and automated maintenance work orders. '
        '<span class="sim-tag">SIMULATED DATA</span></p>', unsafe_allow_html=True)

    engine = _engine()
    if engine is None:
        st.error("Hydraulic engine not available.")
        return

    _section_header("🔔", "System Activity & Alarm Log")
    events = st.session_state.get("activity_log", [])
    if events:
        edf = pd.DataFrame(events)
        st.dataframe(edf, use_container_width=True, height=210, hide_index=True)
    else:
        st.info("No system events recorded yet.")

    st.divider()
    _section_header("🛠️", "Recommended Inspection & Maintenance Work Orders")
    recs = engine.get_maintenance_recommendations()
    rdf = pd.DataFrame(recs)[["rank", "work_order_id", "zone", "score", "label", "sla", "action", "est_loss_l_min"]]
    rdf.columns = ["Rank", "Work Order", "Zone", "Risk (%)", "Priority", "SLA", "Recommended Action", "Est Loss (L/min)"]
    st.dataframe(
        rdf.style.format({"Risk (%)": "{:.1f}", "Est Loss (L/min)": "{:.2f}"})
           .applymap(lambda v: f"color:{C_DANGER}" if v == "CRITICAL"
                          else f"color:{C_WARNING}" if v == "HIGH"
                          else f"color:{C_SUCCESS}" if v == "LOW" else "", subset=["Priority"]),
        use_container_width=True, height=220, hide_index=True,
    )


# ── PAGE 9: Settings ──────────────────────────────────────────────────────────
def _page_settings() -> None:
    st.markdown('<h1 style="color:#EAF4FB;font-size:22px;font-weight:800">Platform Settings</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:16px">'
        'Configure simulation step speed, 3D/2D digital twin preferences, and inspect engine status.</p>',
        unsafe_allow_html=True)

    engine = _engine()
    c1, c2 = st.columns(2)
    with c1:
        _section_header("🎛️", "Simulation & Visualization Preferences")
        default_view = st.radio(
            "Default Network Digital Twin View",
            ["3D Interactive", "2D Vector"],
            index=0 if st.session_state.view_mode == "3D Interactive" else 1,
        )
        if default_view != st.session_state.view_mode:
            st.session_state.view_mode = default_view
            st.rerun()

        anim = st.toggle("Enable Flow Particles & Pulse Animations", value=st.session_state.anim_enabled)
        if anim != st.session_state.anim_enabled:
            st.session_state.anim_enabled = anim
            st.rerun()

        if engine is not None:
            spd = st.select_slider("Simulation Step Speed", options=[1, 2, 5], value=engine.speed)
            if spd != engine.speed:
                engine.set_speed(spd)
                _log_activity(engine, f"Simulation step speed set to {spd}x.", "INFO")
                st.success(f"Simulation speed updated to {spd}x.")

    with c2:
        _section_header("🖥️", "Engine & Environment Diagnostics")
        st.markdown(f"""
        <div class="info-panel" style="font-size:12px;">
            <b>Hydraulic Engine:</b> {'✓ Online (WNTR / EPANET 2.2)' if engine else '✗ Offline'}<br>
            <b>ML Detector:</b> Isolation Forest (200 trees, contamination 0.14)<br>
            <b>Topology:</b> 1 Reservoir (79.0 m) · 4 Zones (A–D) · 12 Junctions · 12 Pipes<br>
            <b>Sim Clock:</b> T+{engine.seconds if engine else 0:03d}s / 150s<br>
            <b>Working Directory:</b> <code>{_PROJ_DIR}</code>
        </div>
        """, unsafe_allow_html=True)


# ── Main router ───────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()
    _init_state()
    _render_sidebar()
    _render_top_bar()

    page = st.session_state.page
    if   page in ("Overview", "Dashboard"):                                        _page_dashboard()
    elif page == "Live Monitoring":                                                _page_monitoring()
    elif page in ("3D Network Map", "Network Map", "3D / 2D Network", "Animated 2D Network"): _page_network_map()
    elif page == "Leak Detection":                                                 _page_leak_detection()
    elif page == "Analytics":                                                      _page_analytics()
    elif page in ("Data Management", "Data Upload"):                               _page_data_upload()
    elif page in ("Alerts and Maintenance", "Alerts & Maintenance"):               _page_alerts_maintenance()
    elif page == "Reports":                                                        _page_reports()
    elif page == "Settings":                                                       _page_settings()
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
