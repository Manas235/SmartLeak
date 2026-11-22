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
from datetime import datetime
from pathlib import Path
from typing import Any

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

# ── Global CSS with 2D Transitions & Pulse Keyframes ──────────────────────────
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

    /* ── Respect reduced motion preferences ── */
    @media (prefers-reduced-motion: reduce) {
        *, ::before, ::after {
            animation-delay: -1ms !important;
            animation-duration: 1ms !important;
            animation-iteration-count: 1 !important;
            background-attachment: initial !important;
            scroll-behavior: auto !important;
            transition-duration: 0s !important;
            transition-delay: 0s !important;
        }
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
        padding-top: 1.2rem !important;
        max-width: 100% !important;
    }

    /* ── KPI Card with Smooth 2D Transitions ── */
    .kpi-card {
        background: linear-gradient(145deg, #0E1D2D, #142238);
        border: 1px solid #263A49;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 8px;
        position: relative;
        overflow: hidden;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #00C8C8;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: #00C8C8;
        border-radius: 12px 12px 0 0;
        transition: background 0.3s ease;
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
        font-size: 26px;
        font-weight: 800;
        color: #00C8C8;
        line-height: 1.15;
        margin-bottom: 4px;
        font-variant-numeric: tabular-nums;
        transition: color 0.3s ease;
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

    /* ── Pulsing leak banner ── */
    @keyframes subtle-pulse {
        0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
        70% { box-shadow: 0 0 0 8px rgba(220, 38, 38, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
    }
    .leak-alert-banner {
        background: linear-gradient(90deg, rgba(220, 38, 38, 0.15), rgba(245, 158, 11, 0.1));
        border: 1px solid #DC2626;
        border-left: 5px solid #DC2626;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 10px 0 16px 0;
        animation: subtle-pulse 2s infinite ease-in-out;
    }

    /* ── Status badge ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.2s ease;
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
        margin-bottom: 14px;
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

    /* ── Simulated data tag ── */
    .sim-tag {
        background: rgba(85,149,245,.15);
        color: #5595F5;
        border: 1px solid #5595F5;
        border-radius: 4px;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 2px 7px;
        text-transform: uppercase;
        vertical-align: middle;
    }

    /* ── Brand header ── */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 0 16px;
    }
    .brand-logo { font-size: 30px; }
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
    if "anim_enabled" not in st.session_state:
        st.session_state.anim_enabled = True
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = "B2"
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
    active_leak_junc = engine.leak_junction or "B2"
    leak_active = (engine.seconds >= 30) or (engine.risk.get("B", 0) >= 31) or (engine._manual_leak_severity is not None)

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

        # Animated Flow Dashes (Speed proportional to flow rate)
        if anim_enabled and flow_lps > 0.04:
            # Duration inversely proportional to flow velocity: higher flow = faster animation
            duration = max(0.6, min(3.5, 2.5 / (max(0.2, flow_lps))))
            svg_parts.append(f'''
            <line class="flow-dash" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{base_color}" stroke-width="{pipe_w}" stroke-linecap="round" marker-end="url(#{marker_ref})" style="animation-duration: {duration:.2f}s;" />
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
            ("🗺️", "Animated 2D Network"),
            ("📡", "Live Monitoring"),
            ("🔍", "Leak Detection"),
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
            if st.button("▶ Step +1", use_container_width=True, type="primary"):
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
                    except Exception as err:
                        st.error(str(err))
                    st.rerun()

        # Animation Settings Switcher
        st.divider()
        st.markdown(
            '<div style="font-size:9px;letter-spacing:1.5px;color:#8FA5B1;'
            'text-transform:uppercase;margin-bottom:6px">Visual Settings</div>',
            unsafe_allow_html=True)
        anim_toggle = st.toggle("✨ 2D Flow Animations", value=st.session_state.anim_enabled)
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


# ── PAGE 1: Operations Dashboard ──────────────────────────────────────────────
def _page_dashboard() -> None:
    st.markdown(
        '<h1 style="color:#E8F1F5;font-size:22px;font-weight:800;letter-spacing:1px;margin-bottom:4px">'
        'Operations Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:16px">'
        'SmartLeak • Intelligent Water Network Leakage Detection & Risk Monitoring  '
        '<span class="sim-tag">SIMULATED DATA</span></p>',
        unsafe_allow_html=True)

    engine = _engine()
    if engine is None:
        st.error(f"Hydraulic engine offline. {st.session_state.get('sim_error') or ''}")
        return

    # Check for active leak alert or early impending leak prediction
    highest_risk = max(engine.risk.values(), default=0.0)
    probable     = engine.probable_zone
    predicted    = getattr(engine, "predicted_leak_zone", probable)
    leak_j       = engine.leak_junction or "B2"
    loss         = engine.loss_estimate_l_min

    if highest_risk >= 31 or engine.network_status in ("HIGH RISK", "WARNING", "ANOMALY DETECTED"):
        st.markdown(f"""
        <div class="leak-alert-banner">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="font-size:13px;font-weight:800;color:#DC2626;letter-spacing:1px;">⚠ ACTIVE LEAK DETECTED</span>
                    <span style="font-size:11px;color:#E8F1F5;margin-left:12px;">Auto-Detected Location: <b>Zone {probable or predicted or 'B'} · Junction {leak_j}</b></span>
                </div>
                <div>
                    <span class="badge badge-danger">SEVERITY: {highest_risk:.1f}%</span>
                    <span style="font-size:11px;color:#F59E0B;margin-left:10px;">Est Loss: <b>~{loss:.1f} L/min</b></span>
                </div>
            </div>
            <div style="font-size:11px;color:#8FA5B1;margin-top:4px;">
                Diagnostics: {engine.get_zone_explanation(probable or predicted or 'B')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif predicted is not None and engine.seconds >= 30:
        st.markdown(f"""
        <div class="leak-alert-banner" style="border-color:#F59E0B;border-left-color:#F59E0B;background:rgba(245,158,11,0.12);">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="font-size:13px;font-weight:800;color:#F59E0B;letter-spacing:1px;">🔮 EARLY PREDICTION: LEAK DEVELOPING IN ZONE {predicted}</span>
                    <span style="font-size:11px;color:#E8F1F5;margin-left:12px;">Pre-Anomaly Hydraulic Mass-Balance Gradient Localized to <b>Zone {predicted}</b></span>
                </div>
                <div>
                    <span class="badge badge-warning">DEVELOPING (T+{engine.seconds}s)</span>
                </div>
            </div>
            <div style="font-size:11px;color:#8FA5B1;margin-top:4px;">
                Predictive telemetry indicates early net inflow variance in District Metered Area <b>Zone {predicted}</b> before full risk threshold breach.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4 KPI Cards (Updated with smooth transition styling)
    zone_target = probable or predicted or "B"
    meas_z = engine.measurements.get(zone_target)
    base_z = engine.calibration.baselines.get(zone_target)

    variant_risk   = "danger" if highest_risk >= 61 else "warning" if (highest_risk >= 31 or predicted) else "success"
    variant_status = ("danger"  if engine.network_status == "HIGH RISK" else
                      "warning" if (engine.network_status in ("WARNING", "ANOMALY DETECTED") or predicted) else "success")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_kpi_card("Network Status", engine.network_status,
                               f"Phase: {engine.phase}", variant=variant_status), unsafe_allow_html=True)
    with c2:
        pred_label = f"Auto-Detected: Zone {zone_target}" if (probable or predicted) else "Nominal (No Leak)"
        st.markdown(_kpi_card("Impending Leak Zone", f"Zone {zone_target}" if (probable or predicted) else "None",
                               pred_label, variant=variant_risk), unsafe_allow_html=True)
    with c3:
        p_val = f"{meas_z.pressure_m:.2f} m" if meas_z else "—"
        p_dev = f"{meas_z.pressure_m - base_z.pressure_m:+.2f} m vs base" if meas_z and base_z else ""
        st.markdown(_kpi_card(f"Zone {zone_target} Pressure", p_val, p_dev), unsafe_allow_html=True)
    with c4:
        f_val = f"{meas_z.flow_lps:.3f} L/s" if meas_z else "—"
        f_dev = f"{meas_z.flow_lps - base_z.flow_lps:+.3f} L/s vs base" if meas_z and base_z else ""
        st.markdown(_kpi_card(f"Zone {zone_target} Flow", f_val, f_dev), unsafe_allow_html=True)

    st.divider()

    # Centerpiece: Live Animated 2D Water Network Diagram
    _section_header("🌐", "Live Animated 2D Water Distribution Network")
    render_animated_2d_network(
        engine,
        selected_node=st.session_state.selected_node,
        anim_enabled=st.session_state.anim_enabled,
        height=480,
    )

    st.divider()

    # Synchronized Sensor Waveforms
    _section_header("📈", "Synchronized Pressure & Flow Waveforms")
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
            template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
            margin=dict(l=10, r=10, t=20, b=10), barmode="group", height=230,
            yaxis=dict(title="Head Pressure (m H₂O)", gridcolor=C_BORDER),
            xaxis=dict(title="Zone"),
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=10)),
        )
        st.plotly_chart(fig_p, use_container_width=True, config=dict(displayModeBar=False))

    with right:
        # Flow deviation chart across pipes
        curr_flow = [engine.measurements[z].flow_lps for z in zones]
        base_flow = [engine.calibration.baselines[z].flow_lps for z in zones]
        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(name="Baseline Flow (L/s)", x=zones, y=base_flow,
                                marker_color="#263A49", opacity=0.7))
        fig_f.add_trace(go.Bar(name="Observed Flow (L/s)", x=zones, y=curr_flow,
                                marker_color=C_BLUE))
        fig_f.update_layout(
            template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
            margin=dict(l=10, r=10, t=20, b=10), barmode="group", height=230,
            yaxis=dict(title="Flow Rate (L/s)", gridcolor=C_BORDER),
            xaxis=dict(title="Zone"),
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=10)),
        )
        st.plotly_chart(fig_f, use_container_width=True, config=dict(displayModeBar=False))


# ── PAGE 2: Animated 2D Network Inspector ─────────────────────────────────────
def _page_network_map() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Animated 2D Water Network</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:16px">'
        'Interactive 2D topological twin displaying live EPANET hydraulics and dynamic vector flow indicators. '
        '<span class="sim-tag">SIMULATED DATA</span></p>', unsafe_allow_html=True)

    engine = _engine()
    if engine is None:
        st.error("Hydraulic engine not available.")
        return

    # Junction selector to inspect telemetry
    col_sel, col_stat = st.columns([2, 3])
    with col_sel:
        all_juncs = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "D3"]
        sel_node = st.selectbox("Inspect Junction Telemetry", all_juncs, index=all_juncs.index(st.session_state.selected_node) if st.session_state.selected_node in all_juncs else 4)
        st.session_state.selected_node = sel_node
    with col_stat:
        st.markdown(f"""
        <div style="font-size:11px;color:#8FA5B1;padding-top:14px;">
            Target Leak Junction: <b style="color:{C_DANGER}">{engine.leak_junction or 'None (Nominal)'}</b> &nbsp;·&nbsp;
            Flow Animation: <b style="color:#00C8C8">{'ACTIVE' if st.session_state.anim_enabled else 'PAUSED'}</b> &nbsp;·&nbsp;
            Sim Time: <b style="color:#E8F1F5">T+{engine.seconds:03d}s</b>
        </div>
        """, unsafe_allow_html=True)

    # Render Main 2D Network Vector Diagram
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
            use_container_width=True, height=280,
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
        _section_header("📊", "Transient Sensor Trend Buffer")
        t = st.session_state.trend_time
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Pressure (m H₂O)", "Core Flow (L/s)", "Risk Score (%)"])
        fig.add_trace(go.Scatter(x=t, y=st.session_state.trend_pressure,
                                  mode="lines", line=dict(color=C_CYAN, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=st.session_state.trend_flow,
                                  mode="lines", line=dict(color=C_BLUE, width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=t, y=st.session_state.trend_risk,
                                  mode="lines", line=dict(color=C_DANGER, width=2)), row=1, col=3)
        fig.add_hline(y=61, line_dash="dash", line_color=C_DANGER, opacity=0.5, row=1, col=3)
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
                          margin=dict(l=10,r=10,t=40,b=20), height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))


# ── PAGE 4: Leak Detection ────────────────────────────────────────────────────
def _page_leak_detection() -> None:
    st.markdown('<h1 style="color:#E8F1F5;font-size:22px;font-weight:800">Leak Detection</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8FA5B1;font-size:12px;margin-bottom:16px">'
        'Isolation Forest anomaly detection + multi-factor risk scoring. '
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
            st.markdown(
                f'<div style="font-size:10px;color:#8FA5B1;margin-top:4px;padding-left:4px">'
                f'ΔP: <b style="color:#E8F1F5">{p_dev:+.3f} m</b>'
                f'&nbsp; ΔQ: <b style="color:#E8F1F5">{f_dev:+.4f} L/s</b></div>',
                unsafe_allow_html=True)

    st.divider()
    _section_header("🤖", "AI Diagnostic Explanations")
    for zone, score, label in priorities:
        with st.expander(f"Zone {zone}  —  Risk: {score:.1f}%  ({label})", expanded=(score >= 31)):
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
    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=C_NAVY, plot_bgcolor=C_PANEL,
                      margin=dict(l=10,r=10,t=40,b=10), height=300,
                      legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
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


# ── Main router ───────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()
    _init_state()
    _render_sidebar()

    page = st.session_state.page
    if   page == "Dashboard":             _page_dashboard()
    elif page == "Animated 2D Network":   _page_network_map()
    elif page == "Live Monitoring":       _page_monitoring()
    elif page == "Leak Detection":        _page_leak_detection()
    elif page == "Analytics":             _page_analytics()
    elif page == "Data Upload":           _page_data_upload()
    elif page == "Reports":               _page_reports()
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
