"""
SmartLeak — Intelligent Water Network Leakage Detection & Risk Monitoring System
Modern Streamlit Web Dashboard

Design: Industrial SaaS WaterTech analytics platform
Every Drop Matters.
"""

from __future__ import annotations

import io
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Import the existing SmartLeak engine ──────────────────────────────────────
try:
    import main as smartleak
    _ENGINE_OK = True
    _ENGINE_ERR = None
except Exception as _exc:
    smartleak = None          # type: ignore[assignment]
    _ENGINE_OK = False
    _ENGINE_ERR = str(_exc)

# ── Brand / Design Tokens ─────────────────────────────────────────────────────
C_NAVY     = "#0B1220"
C_DARK     = "#142238"
C_AQUA     = "#00C8C8"
C_AQUA_LT  = "#DDF8F5"
C_WHITE    = "#FFFFFF"
C_BG       = "#F4F7FA"
C_SUCCESS  = "#16A34A"
C_WARNING  = "#F59E0B"
C_DANGER   = "#DC2626"
C_MUTED    = "#8FA5B1"
C_PANEL    = "#0E1D2D"
C_BORDER   = "#263A49"
C_CYAN     = "#40C8D7"
C_BLUE     = "#5595F5"

PLOTLY_TEMPLATE = "plotly_dark"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartLeak — Water Network Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""
    <style>
    /* ── Root theme overrides ── */
    :root {
        --aqua: #00C8C8;
        --navy: #0B1220;
        --dark: #142238;
        --panel: #0E1D2D;
        --border: #263A49;
        --success: #16A34A;
        --warning: #F59E0B;
        --danger: #DC2626;
        --muted: #8FA5B1;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0B1220 !important;
        border-right: 1px solid #263A49;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }

    /* ── Main content area ── */
    .main .block-container {
        padding-top: 1.5rem !important;
        max-width: 100% !important;
    }

    /* ── KPI Card ── */
    .kpi-card {
        background: linear-gradient(145deg, #0E1D2D, #142238);
        border: 1px solid #263A49;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 8px;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: #00C8C8;
        border-radius: 12px 12px 0 0;
    }
    .kpi-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #8FA5B1;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #00C8C8;
        line-height: 1.1;
        margin-bottom: 4px;
        font-variant-numeric: tabular-nums;
    }
    .kpi-detail {
        font-size: 11px;
        color: #8FA5B1;
        margin-top: 2px;
    }
    .kpi-card.success .kpi-value { color: #16A34A; }
    .kpi-card.warning .kpi-value { color: #F59E0B; }
    .kpi-card.danger  .kpi-value { color: #DC2626; }
    .kpi-card.success::before { background: #16A34A; }
    .kpi-card.warning::before { background: #F59E0B; }
    .kpi-card.danger::before  { background: #DC2626; }

    /* ── Status badge ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .badge-normal  { background: rgba(22,163,74,.2);  color: #16A34A; border: 1px solid #16A34A; }
    .badge-warning { background: rgba(245,158,11,.2); color: #F59E0B; border: 1px solid #F59E0B; }
    .badge-danger  { background: rgba(220,38,38,.2);  color: #DC2626; border: 1px solid #DC2626; }
    .badge-info    { background: rgba(0,200,200,.15); color: #00C8C8; border: 1px solid #00C8C8; }

    /* ── Section header ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 0 6px;
        border-bottom: 1px solid #263A49;
        margin-bottom: 16px;
    }
    .section-header h3 {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #E8F1F5;
        text-transform: uppercase;
        margin: 0;
    }

    /* ── Info panel ── */
    .info-panel {
        background: #0B1A28;
        border: 1px solid #1E3448;
        border-left: 3px solid #00C8C8;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    /* ── Stmetric override ── */
    [data-testid="metric-container"] {
        background: #0E1D2D;
        border: 1px solid #263A49;
        border-radius: 10px;
        padding: 12px 16px !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        font-size: 10px !important;
        letter-spacing: 1px;
        color: #8FA5B1 !important;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 22px !important;
        color: #00C8C8 !important;
        font-weight: 800 !important;
    }

    /* ── Simulated data tag ── */
    .sim-tag {
        background: rgba(85,149,245,.15);
        color: #5595F5;
        border: 1px solid #5595F5;
        border-radius: 4px;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 1px 6px;
        text-transform: uppercase;
        vertical-align: middle;
    }

    /* ── Brand header ── */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 0 20px;
    }
    .brand-logo { font-size: 32px; }
    .brand-name {
        font-size: 22px;
        font-weight: 900;
        color: #00C8C8;
        letter-spacing: 2px;
        text-transform: uppercase;
        line-height: 1.1;
    }
    .brand-tagline {
        font-size: 11px;
        color: #8FA5B1;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)


# ── State Initialization ──────────────────────────────────────────────────────
def _init_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "datasets" not in st.session_state:
        st.session_state.datasets = []
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


def _record_trend(engine: Any) -> None:
    t = st.session_state.trend_time
    p = st.session_state.trend_pressure
    f = st.session_state.trend_flow
    r = st.session_state.trend_risk
    t.append(engine.seconds)
    p.append(engine.measurements["B"].pressure_m)
    f.append(engine.measurements["B"].flow_lps)
    r.append(engine.risk["B"])
    if len(t) > 60:
        st.session_state.trend_time     = t[-60:]
        st.session_state.trend_pressure = p[-60:]
        st.session_state.trend_flow     = f[-60:]
        st.session_state.trend_risk     = r[-60:]


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
            ("📊", "Dashboard"),
            ("📡", "Live Monitoring"),
            ("🔍", "Leak Detection"),
            ("🗺️", "Network Map"),
            ("📈", "Analytics"),
            ("📂", "Data Upload"),
            ("📋", "Reports"),
        ]
        st.markdown(
            '<div style="font-size:9px;letter-spacing:1.5px;color:#8FA5B1;'
            'text-transform:uppercase;margin-bottom:6px">Navigation</div>',
            unsafe_allow_html=True)
        for icon, label in pages:
            if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
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
                    except Exception as err:
                        st.error(str(err))
                    st.rerun()

        if st.button("⚡ Inject Leak (Zone B)", use_container_width=True):
            e = _engine()
            if e:
                try:
                    e.introduce_leak()
                    _record_trend(e)
                except Exception as err:
                    st.error(str(err))
                st.rerun()

        st.divider()
        n_ds = len(st.session_state.datasets)
        sim_ok = "✓ Ready" if _engine() else "✗ Offline"
        st.markdown(
            f'<div class="info-panel" style="font-size:11px">'
            f'<b style="color:#00C8C8">{n_ds}</b> dataset(s) loaded<br>'
            f'<span style="color:#8FA5B1">Simulation: {sim_ok}</span></div>',
            unsafe_allow_html=True)


# ── PAGE 1: Dashboard ─────────────────────────────────────────────────────────
def _page_dashboard() -> None:
    st.markdown(
        '<h1 style="color:#E8F1F5;font-size:22px;font-weight:800;letter-spacing:1px;margin-bottom:4px">'
        'Operations Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:20px">'
        'SmartLeak • Intelligent Water Network Leakage Detection & Risk Monitoring  '
        '<span class="sim-tag">SIMULATED DATA</span></p>',
        unsafe_allow_html=True)

    engine = _engine()
    if engine is None:
        st.error(f"Hydraulic engine offline. {st.session_state.get('sim_error') or ''}")
        return

    # KPI Row
    highest_risk = max(engine.risk.values(), default=0.0)
    probable     = engine.probable_zone
    zone_b       = engine.measurements.get("B")
    base_b       = engine.calibration.baselines.get("B")
    loss         = engine.loss_estimate_l_min

    variant_risk   = "danger" if highest_risk >= 61 else "warning" if highest_risk >= 31 else "success"
    variant_status = ("danger"  if engine.network_status == "HIGH RISK" else
                      "warning" if engine.network_status in ("WARNING", "ANOMALY DETECTED") else "success")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(_kpi_card("Network Status", engine.network_status,
                               f"Phase: {engine.phase}", variant=variant_status), unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi_card("Max Leak Risk", f"{highest_risk:.1f}%",
                               f"Zone: {probable or 'None'}", variant=variant_risk), unsafe_allow_html=True)
    with c3:
        p_val = f"{zone_b.pressure_m:.2f} m" if zone_b else "—"
        p_dev = (f"{zone_b.pressure_m - base_b.pressure_m:+.2f} m vs baseline"
                 if zone_b and base_b else "")
        st.markdown(_kpi_card("Zone B Pressure", p_val, p_dev), unsafe_allow_html=True)
    with c4:
        f_val = f"{zone_b.flow_lps:.3f} L/s" if zone_b else "—"
        f_dev = (f"{zone_b.flow_lps - base_b.flow_lps:+.3f} L/s vs baseline"
                 if zone_b and base_b else "")
        st.markdown(_kpi_card("Zone B Flow", f_val, f_dev), unsafe_allow_html=True)
    with c5:
        loss_v = "danger" if loss > 1.0 else "warning" if loss > 0.1 else "success"
        st.markdown(_kpi_card("Est. Water Loss", f"{loss:.2f} L/min",
                               "Real-time emitter estimate", variant=loss_v), unsafe_allow_html=True)

    st.divider()

    # Zone risk summary + zone pressure comparison
    left, right = st.columns([1, 2])
    with left:
        _section_header("🏗️", "Zone Risk Summary")
        for zone, score, label in engine.priorities():
            bar_color = (_risk_color(score))
            st.markdown(f"""
            <div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">
                    <span style="font-weight:700;color:#E8F1F5;font-size:13px">Zone {zone}</span>
                    <span style="font-size:11px;color:{bar_color};font-weight:700">{score:.1f}% · {label}</span>
                </div>
                <div style="background:#142238;border-radius:4px;height:6px">
                    <div style="background:{bar_color};width:{min(score,100):.0f}%;height:6px;border-radius:4px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        _section_header("📊", "Zone Pressure vs Baseline")
        zones = ["A", "B", "C", "D"]
        curr_press = [engine.measurements[z].pressure_m for z in zones]
        base_press = [engine.calibration.baselines[z].pressure_m for z in zones]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Baseline", x=zones, y=base_press,
                              marker_color="#263A49", opacity=0.6))
        fig.add_trace(go.Bar(name="Current", x=zones, y=curr_press,
                              marker_color=[_risk_color(engine.risk[z]) for z in zones]))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
            margin=dict(l=10, r=10, t=10, b=10), barmode="group", height=240,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=10)),
            yaxis=dict(title="Pressure (m H₂O)", gridcolor=C_BORDER),
            xaxis=dict(title="Zone"),
        )
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    # Risk factor decomposition
    st.divider()
    _section_header("🤖", "AI Risk Factor Decomposition — Zone B")
    contrib_b = engine.contributions.get("B", {})
    factors  = ["Pressure", "Flow", "ML (Isolation Forest)", "Persistence"]
    contrib_v = [contrib_b.get("pressure", 0), contrib_b.get("flow", 0),
                 contrib_b.get("ml", 0), contrib_b.get("persistence", 0)]
    weights  = [35, 35, 20, 10]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=factors, y=contrib_v,
        marker_color=[C_DANGER if v >= w * 0.6 else C_WARNING if v >= w * 0.2 else C_SUCCESS
                      for v, w in zip(contrib_v, weights)],
        text=[f"{v:.1f}/{w}" for v, w in zip(contrib_v, weights)],
        textposition="outside", textfont=dict(size=10, color="#E8F1F5"),
    ))
    fig2.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
        margin=dict(l=10, r=10, t=10, b=40),
        yaxis=dict(title="Risk Contribution (points)", gridcolor=C_BORDER, range=[0, 40]),
        height=200, showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True, config=dict(displayModeBar=False))

    exp = engine.get_zone_explanation("B")
    st.markdown(f'<div class="info-panel" style="font-size:12px">{exp}</div>',
                unsafe_allow_html=True)


# ── PAGE 2: Live Monitoring ───────────────────────────────────────────────────
def _page_monitoring() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Live Monitoring</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:20px">'
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
        display_j = jdf[["node","zone","elevation_m","pressure_m","baseline_m","dev_m","dev_pct","status"]].rename(columns={
            "node":"Junction","zone":"Zone","elevation_m":"Elev (m)",
            "pressure_m":"Pressure (m)","baseline_m":"Baseline (m)",
            "dev_m":"Δ (m)","dev_pct":"Δ (%)","status":"Status",
        })
        st.dataframe(
            display_j.style.applymap(
                lambda v: f"color:{C_DANGER}" if v=="CRITICAL"
                     else f"color:{C_WARNING}" if v=="WARNING"
                     else f"color:{C_CYAN}"    if v=="ANOMALY"
                     else f"color:{C_SUCCESS}" if v=="NORMAL" else "",
                subset=["Status"]
            ).format({"Elev (m)":"{:.1f}","Pressure (m)":"{:.3f}",
                      "Baseline (m)":"{:.3f}","Δ (m)":"{:+.3f}","Δ (%)":"{:+.2f}%"}),
            use_container_width=True, height=300,
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
            use_container_width=True, height=300,
        )
    except Exception as exc:
        st.warning(f"Pipe telemetry unavailable: {exc}")

    if st.session_state.trend_time:
        st.divider()
        _section_header("📊", "Zone B Sensor Trend")
        t = st.session_state.trend_time
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Pressure (m)","Flow (L/s)","Risk (%)"])
        fig.add_trace(go.Scatter(x=t, y=st.session_state.trend_pressure,
                                  mode="lines", line=dict(color=C_CYAN, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=st.session_state.trend_flow,
                                  mode="lines", line=dict(color=C_BLUE, width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=t, y=st.session_state.trend_risk,
                                  mode="lines", line=dict(color=C_DANGER, width=2)), row=1, col=3)
        fig.add_hline(y=61, line_dash="dash", line_color=C_DANGER, opacity=0.5, row=1, col=3)
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
                          margin=dict(l=10,r=10,t=40,b=20), height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))


# ── PAGE 3: Leak Detection ────────────────────────────────────────────────────
def _page_leak_detection() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Leak Detection</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:20px">'
        'Isolation Forest anomaly detection + multi-factor risk scoring. '
        '<span class="sim-tag">SIMULATED DATA</span></p>', unsafe_allow_html=True)

    engine = _engine()
    if engine is None:
        st.error("Hydraulic engine not available.")
        return

    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("⚡ Inject Leak at B2 (Deterministic Scenario)", type="primary", use_container_width=True):
            engine.introduce_leak(); _record_trend(engine); st.rerun()
    with cc2:
        if st.button("↺ Reset to Normal", use_container_width=True):
            engine.reset()
            st.session_state.trend_time = []
            st.session_state.trend_pressure = []
            st.session_state.trend_flow = []
            st.session_state.trend_risk = []
            st.rerun()

    with st.expander("⚙️ Manual Leak Trigger — Select Junction"):
        junctions = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","D3"]
        sel_junc = st.selectbox("Junction for full-strength leak", junctions, index=4)
        if st.button(f"Trigger 100% Leak at {sel_junc}"):
            try:
                engine.trigger_leak(sel_junc); _record_trend(engine); st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    _section_header("🚨", "Zone Risk Assessment")
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
            st.markdown(
                f'<div style="font-size:10px;color:#8FA5B1;margin-top:4px;padding-left:4px">'
                f'ΔP: <b style="color:#E8F1F5">{p_dev:+.3f} m</b>'
                f'&nbsp; ΔQ: <b style="color:#E8F1F5">{f_dev:+.4f} L/s</b></div>',
                unsafe_allow_html=True)

    st.divider()
    _section_header("🤖", "AI Diagnostic Explanations")
    for zone, score, label in priorities:
        with st.expander(f"Zone {zone}  —  Risk: {score:.1f}%  ({label})", expanded=(zone=="B")):
            exp = engine.get_zone_explanation(zone)
            st.markdown(f'<div class="info-panel" style="font-size:12px">{exp}</div>',
                        unsafe_allow_html=True)

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

    st.divider()
    _section_header("🧠", "Isolation Forest Decision Scores")
    mc1, mc2 = st.columns(2)
    with mc1:
        ml_data = [(z, engine.ml_scores.get(z,0.0), engine.ml_flags.get(z,False))
                   for z in ["A","B","C","D"]]
        st.dataframe(pd.DataFrame(ml_data, columns=["Zone","Decision Score","Anomaly Detected"])
                     .style.format({"Decision Score":"{:.6f}"}),
                     use_container_width=True)
    with mc2:
        st.markdown("""<div class="info-panel" style="font-size:12px">
        <b style="color:#00C8C8">Isolation Forest Model</b><br>
        Architecture: sklearn IsolationForest (n_estimators=200, contamination=0.14)<br><br>
        <b>Feature vector (6-D per zone):</b><br>
        • Live junction pressure (m H₂O)<br>
        • Live core pipe flow rate (L/s)<br>
        • Pressure change rate (ΔP/Δt)<br>
        • Flow change rate (ΔQ/Δt)<br>
        • 4-step rolling mean pressure<br>
        • 4-step rolling mean flow<br><br>
        <b>Risk weights:</b> Pressure 35% | Flow 35% | ML 20% | Persistence 10%
        </div>""", unsafe_allow_html=True)


# ── PAGE 4: Network Map ───────────────────────────────────────────────────────
def _page_network_map() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Network Map</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:20px">'
        'Interactive water distribution network topology with live hydraulic state. '
        '<span class="sim-tag">SIMULATED DATA</span></p>', unsafe_allow_html=True)

    engine = _engine()
    if engine is None:
        st.error("Hydraulic engine not available.")
        return

    positions = {
        "RESERVOIR": (-1.5, 0.0),
        "A1": (0.0, 0.0), "A2": (1.1, 0.0), "A3": (0.55, -0.9),
        "B1": (2.2, 0.0), "B2": (3.3, 0.0), "B3": (2.75, -0.9),
        "C1": (4.4, 0.0), "C2": (5.5, 0.0), "C3": (4.95, -0.9),
        "D1": (6.6, 0.0), "D2": (7.7, 0.0), "D3": (7.15, -0.9),
    }
    pipes = [
        ("P_SUPPLY",   "RESERVOIR","A1","A"),
        ("P_A_CORE",   "A1","A2","A"), ("P_A_BRANCH","A1","A3","A"),
        ("P_A_TO_B",   "A2","B1","A"),
        ("P_B_CORE",   "B1","B2","B"), ("P_B_BRANCH","B1","B3","B"),
        ("P_B_TO_C",   "B2","C1","B"),
        ("P_C_CORE",   "C1","C2","C"), ("P_C_BRANCH","C1","C3","C"),
        ("P_C_TO_D",   "C2","D1","C"),
        ("P_D_CORE",   "D1","D2","D"), ("P_D_BRANCH","D1","D3","D"),
    ]

    snapshot   = engine.snapshot
    node_zones = {n: z for z, ns in smartleak.ZONE_NODES.items() for n in ns}

    fig = go.Figure()

    # Zone shading
    zone_ranges = {"A":(-.9,1.75),"B":(1.75,3.9),"C":(3.9,6.05),"D":(6.05,8.3)}
    for zone, (xl, xr) in zone_ranges.items():
        risk = engine.risk.get(zone, 0.0)
        fill = ("rgba(220,38,38,.12)" if risk >= 61 else
                "rgba(245,158,11,.10)" if risk >= 31 else "rgba(0,200,200,.04)")
        fig.add_shape(type="rect", x0=xl, y0=-1.3, x1=xr, y1=0.5,
                      fillcolor=fill, line=dict(color="#263A49", width=0.5), layer="below")
        fig.add_annotation(x=(xl+xr)/2, y=0.42, text=f"<b>ZONE {zone}</b>",
                           showarrow=False, font=dict(size=10, color="#8FA5B1"))

    # Pipes
    for pipe_id, start, end, zone in pipes:
        x0, y0 = positions[start]
        x1, y1 = positions[end]
        flow  = snapshot.flows_lps.get(pipe_id, 0.0)
        risk  = engine.risk.get(zone, 0.0)
        color = (C_DANGER if risk >= 61 and zone == "B" else
                 C_WARNING if risk >= 31 else "#245E7A")
        width = 4 if zone == "B" and risk >= 31 else 2.5
        fig.add_trace(go.Scatter(
            x=[x0,x1,None], y=[y0,y1,None],
            mode="lines", line=dict(color=color, width=width),
            hoverinfo="skip", showlegend=False,
        ))
        mx, my = (x0+x1)/2, (y0+y1)/2
        fig.add_annotation(x=mx, y=my+0.07, text=f"{flow:.3f} L/s",
                           showarrow=False, font=dict(size=8, color="#94A3B8"),
                           bgcolor="#071724", bordercolor="#1A354B", borderwidth=0.5, opacity=0.9)

    # Nodes
    node_x, node_y, node_text, node_color, node_size, hover_text = [], [], [], [], [], []
    for node, (nx_pos, ny_pos) in positions.items():
        node_x.append(nx_pos); node_y.append(ny_pos)
        if node == "RESERVOIR":
            node_color.append("#0284C7"); node_size.append(28)
            node_text.append("💧 RES")
            hover_text.append("Reservoir<br>Head: 79.0 m")
        else:
            zone  = node_zones.get(node,"?")
            risk  = engine.risk.get(zone, 0.0)
            press = snapshot.pressures.get(node, 0.0)
            is_leak = (node == (engine.active_leak_junction or "B2") and engine.seconds >= 30)
            color = (C_DANGER if is_leak and risk >= 61 else C_WARNING if is_leak else
                     C_DANGER if risk >= 61 else C_WARNING if risk >= 31 else "#17374D")
            node_color.append(color)
            node_size.append(22 if is_leak else 18)
            node_text.append(f"{'⚠ ' if is_leak else ''}{node}")
            hover_text.append(f"<b>{node}</b> (Zone {zone})<br>Pressure: {press:.2f} m<br>Risk: {risk:.1f}%")

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(color=node_color, size=node_size, line=dict(color="#E8F1F5", width=1.5)),
        text=node_text, textposition="top center",
        textfont=dict(size=9, color="#E8F1F5"),
        hovertext=hover_text, hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
        margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(visible=False, range=[-2.0,8.5]),
        yaxis=dict(visible=False, range=[-1.5,0.7]),
        height=460, hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style="display:flex;gap:20px;padding:8px 4px;font-size:11px">
        <span>🔵 Reservoir</span>
        <span style="color:#17374D">⬛ Normal Junction</span>
        <span style="color:#F59E0B">🟡 Warning</span>
        <span style="color:#DC2626">🔴 High Risk / Leak</span>
        <span style="color:#245E7A">━ Normal Pipe</span>
        <span style="color:#DC2626">━ Affected Pipe</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div class="info-panel" style="font-size:12px">'
        f'Simulation Time: <b style="color:#00C8C8">T+{engine.seconds}s / 150s</b>'
        f' &nbsp;·&nbsp; Phase: <b style="color:#E8F1F5">{engine.phase}</b>'
        f' &nbsp;·&nbsp; Leak Junction: <b style="color:{C_DANGER}">{engine.leak_junction or "None"}</b></div>',
        unsafe_allow_html=True)


# ── PAGE 5: Analytics ─────────────────────────────────────────────────────────
def _page_analytics() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Analytics</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:20px">'
        'Multi-zone hydraulic performance analysis. <span class="sim-tag">SIMULATED DATA</span></p>',
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

    _section_header("📊", "Pressure & Flow — All Zones")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Zone Pressure vs Baseline","Zone Flow vs Baseline"])
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
    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
                      margin=dict(l=10,r=10,t=40,b=10), height=300,
                      legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    # Radar + scatter
    st.divider()
    _section_header("🕸️", "Zone Risk Analysis")
    col_r, col_s = st.columns(2)
    with col_r:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=risk_scores + [risk_scores[0]], theta=zones + [zones[0]],
            fill="toself",
            line=dict(color=C_DANGER if max(risk_scores)>=61 else C_WARNING, width=2),
            fillcolor="rgba(220,38,38,.15)" if max(risk_scores)>=61 else "rgba(245,158,11,.12)",
        ))
        fig_radar.update_layout(
            polar=dict(bgcolor=C_PANEL,
                       radialaxis=dict(visible=True, range=[0,100], gridcolor=C_BORDER),
                       angularaxis=dict(gridcolor=C_BORDER)),
            template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY,
            margin=dict(l=10,r=10,t=10,b=10), height=300, showlegend=False,
        )
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
            template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
            xaxis=dict(title="Pressure Deviation (m)", gridcolor=C_BORDER),
            yaxis=dict(title="Risk Score (%)", gridcolor=C_BORDER, range=[0,105]),
            margin=dict(l=10,r=10,t=10,b=40), height=300,
        )
        st.plotly_chart(fig_sc, use_container_width=True, config=dict(displayModeBar=False))

    # Trend if available
    if st.session_state.trend_time and len(st.session_state.trend_time) > 1:
        st.divider()
        _section_header("📈", "Historical Trend — Zone B")
        t = st.session_state.trend_time
        fig_t = make_subplots(rows=3, cols=1, shared_xaxes=True,
                               subplot_titles=["Pressure (m)","Flow (L/s)","Risk (%)"],
                               vertical_spacing=0.08)
        fig_t.add_trace(go.Scatter(x=t, y=st.session_state.trend_pressure,
                                    mode="lines", line=dict(color=C_CYAN, width=2.5),
                                    fill="tonexty", fillcolor="rgba(0,200,200,.06)"), row=1, col=1)
        base_p_b = engine.calibration.baselines["B"].pressure_m
        fig_t.add_hline(y=base_p_b, line_dash="dash", line_color=C_CYAN, opacity=0.4, row=1, col=1)
        fig_t.add_trace(go.Scatter(x=t, y=st.session_state.trend_flow,
                                    mode="lines", line=dict(color=C_BLUE, width=2.5)), row=2, col=1)
        fig_t.add_trace(go.Scatter(x=t, y=st.session_state.trend_risk,
                                    mode="lines", line=dict(color=C_DANGER, width=2.5),
                                    fill="tozeroy", fillcolor="rgba(220,38,38,.06)"), row=3, col=1)
        fig_t.add_hline(y=61, line_dash="dash", line_color=C_DANGER, opacity=0.6, row=3, col=1)
        fig_t.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
                             margin=dict(l=10,r=10,t=40,b=10), height=440, showlegend=False)
        st.plotly_chart(fig_t, use_container_width=True, config=dict(displayModeBar=False))

    # Uploaded dataset analytics
    if st.session_state.datasets:
        st.divider()
        _section_header("📂", "Uploaded Dataset Analytics")
        ds_names = [ds["name"] for ds in st.session_state.datasets]
        sel_ds = st.selectbox("Select dataset for analysis", ds_names)
        ds = next(d for d in st.session_state.datasets if d["name"] == sel_ds)
        df = ds["data"]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            st.info("No numeric columns in this dataset.")
        else:
            with st.expander("📋 Statistical Summary", expanded=True):
                st.dataframe(df[num_cols].describe().T, use_container_width=True)
            sel_col = st.selectbox("Column to visualize", num_cols)
            fig_h = px.histogram(df, x=sel_col, nbins=30, template=PLOTLY_TEMPLATE,
                                  color_discrete_sequence=[C_CYAN])
            fig_h.update_layout(paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
                                 margin=dict(l=10,r=10,t=10,b=10), height=260)
            st.plotly_chart(fig_h, use_container_width=True, config=dict(displayModeBar=False))
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                fig_c = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                   template=PLOTLY_TEMPLATE, aspect="auto")
                fig_c.update_layout(paper_bgcolor=C_NAVY, margin=dict(l=10,r=10,t=10,b=10), height=300)
                st.plotly_chart(fig_c, use_container_width=True, config=dict(displayModeBar=False))


# ── PAGE 6: Data Upload ───────────────────────────────────────────────────────
def _page_data_upload() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Data Upload</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:20px">'
        'Upload CSV or Excel datasets for quality analysis and exploration.</p>',
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
        st.info("No datasets uploaded yet.")
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

    with st.expander("🔎 Column Information"):
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
    st.dataframe(df.head(50), use_container_width=True, height=300)

    st.divider()
    _section_header("🔍", "Data Quality Report")
    score_color = C_SUCCESS if q["score"] >= 85 else C_WARNING if q["score"] >= 60 else C_DANGER
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Overall Quality Score</div>
        <div class="kpi-value" style="color:{score_color}">{q['score']:.1f} / 100</div>
        <div class="kpi-detail">Status: <b style="color:{score_color}">{q['status']}</b>
            &nbsp;|&nbsp; Completeness: {q['completeness']:.1f}%
            &nbsp;|&nbsp; Duplicates: {q['dups']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if q["issues"]:
        for issue in q["issues"]:
            st.markdown(f'<div style="color:{C_WARNING};font-size:12px;padding:2px 0">⚠ {issue}</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#16A34A;font-size:12px;margin-top:4px">✓ No major quality issues detected.</div>',
                    unsafe_allow_html=True)

    prob = df[df.isna().any(axis=1) | df.duplicated()]
    if not prob.empty:
        with st.expander(f"⚠ Show {len(prob)} Problematic Row(s)"):
            st.dataframe(prob.head(30), use_container_width=True)

    st.divider()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Dataset as CSV", data=csv_bytes,
                        file_name=f"{sel_name}_export.csv",
                        mime="text/csv", use_container_width=True)
    if st.button(f"🗑 Remove '{sel_name}' from session", use_container_width=True):
        st.session_state.datasets = [d for d in st.session_state.datasets if d["name"] != sel_name]
        st.rerun()


# ── PAGE 7: Reports ───────────────────────────────────────────────────────────
def _page_reports() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Reports & Export</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:20px">'
        'Generate and download monitoring summaries and telemetry data.</p>',
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

    if st.session_state.datasets:
        st.divider()
        _section_header("📂", "Uploaded Dataset Exports")
        for ds in st.session_state.datasets:
            df = ds["data"]
            q  = _quality_score(df)
            csv_b = df.to_csv(index=False).encode("utf-8")
            ca, cb = st.columns([3,1])
            with ca:
                st.markdown(
                    f'<div style="font-size:12px;color:#E8F1F5;padding:4px 0">'
                    f'<b>{ds["name"]}</b>  —  {df.shape[0]} rows × {df.shape[1]} cols  |  '
                    f'Quality: <span style="color:{_risk_color(q["score"])}">{q["score"]:.1f}/100</span></div>',
                    unsafe_allow_html=True)
            with cb:
                st.download_button("⬇ CSV", data=csv_b,
                                    file_name=f"export_{ds['name']}.csv",
                                    mime="text/csv", key=f"dl_{ds['name']}")

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


# ── Main router ───────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()
    _init_state()
    _render_sidebar()

    page = st.session_state.page
    if   page == "Dashboard":      _page_dashboard()
    elif page == "Live Monitoring": _page_monitoring()
    elif page == "Leak Detection":  _page_leak_detection()
    elif page == "Network Map":     _page_network_map()
    elif page == "Analytics":       _page_analytics()
    elif page == "Data Upload":     _page_data_upload()
    elif page == "Reports":         _page_reports()
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
