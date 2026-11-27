#!/usr/bin/env python3
"""
SmartLeak — Full-Screen 3D Three.js Hydraulic Digital Twin Web Server
Connects a 100% 3D Three.js WebGL interface directly to the real WNTR/EPANET
and Isolation Forest engine in main.py via a local ThreadingHTTPServer API.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

_PROJ_DIR = str(Path(__file__).resolve().parent)
if _PROJ_DIR not in sys.path:
    sys.path.insert(0, _PROJ_DIR)

import main as smartleak

# ── Shared Thread-Safe Simulation State ───────────────────────────────────────
_STATE_LOCK = threading.Lock()
_CALIBRATION = smartleak.HydraulicCalibration()
_SIM = smartleak.SmartLeakSimulation(_CALIBRATION)
_TREND: dict[str, list[Any]] = {"t": [], "p": [], "f": [], "r": []}
_ACTIVITY: list[dict[str, str]] = []


def _log_event(msg: str, level: str = "INFO") -> None:
    _ACTIVITY.insert(0, {
        "time": f"T+{_SIM.seconds:03d}s",
        "level": level,
        "message": msg,
    })
    if len(_ACTIVITY) > 30:
        del _ACTIVITY[30:]


def _record_trend() -> None:
    z = _SIM.probable_zone or getattr(_SIM, "predicted_leak_zone", None) or max(_SIM.risk, key=_SIM.risk.get)
    _TREND["t"].append(int(_SIM.seconds))
    _TREND["p"].append(round(float(_SIM.measurements[z].pressure_m), 3))
    _TREND["f"].append(round(float(_SIM.measurements[z].flow_lps), 4))
    _TREND["r"].append(round(float(_SIM.risk[z]), 2))
    if len(_TREND["t"]) > 60:
        for k in _TREND:
            _TREND[k] = _TREND[k][-60:]


_record_trend()
_log_event("WNTR/EPANET 3D Digital Twin initialized. All 4 zones nominal.", "NORMAL")


def _build_state_payload() -> dict[str, Any]:
    snapshot = _SIM.snapshot
    active_leak_junc = _SIM.leak_junction or _SIM.active_leak_junction
    leak_active = (
        (_SIM.leak_severity > 0)
        or (max(_SIM.risk.values(), default=0.0) >= 31)
        or (_SIM._manual_leak_severity is not None)
    )
    if leak_active and not active_leak_junc:
        det_z = _SIM.probable_zone or getattr(_SIM, "predicted_leak_zone", None)
        active_leak_junc = f"{det_z}2" if det_z else None

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

    junc_telemetry = {item["node"]: item for item in _SIM.get_junction_telemetry()}
    pipe_telemetry = _SIM.get_pipe_telemetry()

    affected_pipes: set[str] = set()
    if leak_active and active_leak_junc:
        for p in pipe_telemetry:
            if p["start"] == active_leak_junc or p["end"] == active_leak_junc:
                affected_pipes.add(p["pipe"])

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
        z_risk = float(_SIM.risk.get(zone, 0.0))
        y_val = (jt["elevation_m"] - 50.0) * 0.55 + 1.8
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
            "is_leak": bool(leak_active and node_id == active_leak_junc),
            "connected_flows": [
                {"pipe": p["pipe"], "flow_lps": round(p["flow_lps"], 3), "direction": p["direction"]}
                for p in pipe_telemetry if p["start"] == node_id or p["end"] == node_id
            ],
        })

    pipes_payload: list[dict[str, Any]] = []
    for p in pipe_telemetry:
        p_id = p["pipe"]
        z = p["zone"]
        z_risk = float(_SIM.risk.get(z, 0.0))
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
        })

    zone_meta = [
        ("A", "INLET & SUPPLY",     -15.8, -7.2),
        ("B", "CORE DISTRIBUTION",  -5.8,  2.8),
        ("C", "RESIDENTIAL SECTOR",  4.2, 12.8),
        ("D", "ELEVATED DISTRICT",  14.2, 22.8),
    ]
    zones_payload: list[dict[str, Any]] = []
    for z, title, x_min, x_max in zone_meta:
        zr = float(_SIM.risk.get(z, 0.0))
        meas = _SIM.measurements[z]
        base = _CALIBRATION.baselines[z]
        zones_payload.append({
            "zone": z,
            "title": title,
            "x_min": x_min,
            "x_max": x_max,
            "z_min": -3.2,
            "z_max": 8.6,
            "risk": round(zr, 1),
            "status": "CRITICAL" if zr >= 61 else "WARNING" if zr >= 31 else "NORMAL",
            "pressure_m": round(meas.pressure_m, 2),
            "base_pressure_m": round(base.pressure_m, 2),
            "dev_p": round(meas.pressure_m - base.pressure_m, 2),
            "flow_lps": round(meas.flow_lps, 3),
            "base_flow_lps": round(base.flow_lps, 3),
            "dev_q": round(meas.flow_lps - base.flow_lps, 3),
            "ml_score": round(float(_SIM.ml_scores.get(z, 0.0)), 4),
            "ml_flag": bool(_SIM.ml_flags.get(z, False)),
            "explanation": _SIM.get_zone_explanation(z),
        })

    all_p = list(snapshot.pressures.values())
    base_p = list(_CALIBRATION.baseline_snapshot.pressures.values())
    mean_p = float(np.mean(all_p)) if all_p else 0.0
    mean_base_p = float(np.mean(base_p)) if base_p else 0.0
    supply_q = float(snapshot.flows_lps.get("P_SUPPLY", 0.0))
    base_supply_q = float(_CALIBRATION.baseline_snapshot.flows_lps.get("P_SUPPLY", 0.0))
    highest_risk = max(_SIM.risk.values(), default=0.0)
    top_zone = max(_SIM.risk, key=_SIM.risk.get) if _SIM.risk else "B"
    probable = _SIM.probable_zone
    predicted = getattr(_SIM, "predicted_leak_zone", probable)
    leak_zone = _SIM.node_zones.get(active_leak_junc, "") if active_leak_junc else ""

    return {
        "seconds": int(_SIM.seconds),
        "running": bool(_SIM.running),
        "phase": _SIM.phase,
        "network_status": _SIM.network_status,
        "mean_pressure_m": round(mean_p, 2),
        "mean_pressure_dev_m": round(mean_p - mean_base_p, 2),
        "supply_flow_lps": round(supply_q, 3),
        "supply_flow_dev_lps": round(supply_q - base_supply_q, 3),
        "highest_risk": round(float(highest_risk), 1),
        "top_zone": top_zone,
        "probable_zone": probable,
        "predicted_zone": predicted,
        "loss_l_min": round(float(_SIM.loss_estimate_l_min), 1),
        "alert_count": sum(1 for z, r in _SIM.risk.items() if r >= 31 or _SIM.anomaly_flags.get(z, False)),
        "leak": {
            "active": bool(leak_active),
            "junction": active_leak_junc,
            "zone": leak_zone,
            "severity": round(float(_SIM.leak_severity), 3),
        },
        "nodes": nodes_payload,
        "pipes": pipes_payload,
        "zones": zones_payload,
        "trend": _TREND,
        "activity": _ACTIVITY[:8],
        "recommendations": _SIM.get_maintenance_recommendations(),
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>SMARTLEAK — Full 3D Hydraulic Digital Twin</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #08111d;
    --surface: rgba(17, 29, 44, 0.84);
    --elevated: rgba(27, 44, 65, 0.88);
    --border: rgba(56, 189, 248, 0.22);
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #06b6d4;
    --success: #2fb344;
    --warning: #f59f00;
    --danger: #d63939;
    --purple: #a78bfa;
  }
  body.light {
    --bg: #e9f0f7;
    --surface: rgba(255, 255, 255, 0.88);
    --elevated: rgba(241, 245, 249, 0.92);
    --border: rgba(15, 23, 42, 0.14);
    --text: #0f172a;
    --muted: #475569;
    --accent: #0891b2;
    --success: #15803d;
    --warning: #d97706;
    --danger: #dc2626;
    --purple: #7c3aed;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 100vw; height: 100vh; overflow: hidden;
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    user-select: none;
  }
  #canvas-container {
    position: fixed; inset: 0; z-index: 1;
  }
  canvas { display: block; width: 100%; height: 100%; outline: none; }

  /* 3D HUD Layer */
  .hud {
    position: fixed; inset: 0; z-index: 10; pointer-events: none;
    perspective: 1400px;
  }
  .panel {
    background: var(--surface);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    pointer-events: auto;
    transition: transform 0.25s ease, border-color 0.2s ease;
  }

  /* Top 3D Nav Bar */
  .topbar {
    position: absolute; top: 14px; left: 14px; right: 14px;
    padding: 10px 16px;
    display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  }
  .brand { display: flex; align-items: center; gap: 10px; }
  .brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px; font-weight: 700; letter-spacing: 1.5px; color: var(--accent);
  }
  .brand-sub { font-size: 11px; color: var(--muted); }
  .nav-pills { display: flex; gap: 6px; flex-wrap: wrap; }
  .btn {
    background: var(--elevated); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 11px; font-size: 11px; font-weight: 600;
    cursor: pointer; transition: all 0.16s ease; font-family: 'Inter', sans-serif;
  }
  .btn:hover { border-color: var(--accent); transform: translateY(-1px); }
  .btn.active, .btn.primary {
    background: var(--accent); color: #06131f; border-color: var(--accent); font-weight: 700;
  }
  .btn.danger { background: rgba(214, 57, 57, 0.22); border-color: var(--danger); color: var(--danger); }
  .btn.danger:hover { background: var(--danger); color: #fff; }
  .status-pill {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700;
    padding: 4px 10px; border-radius: 999px;
    background: var(--elevated); border: 1px solid var(--border);
  }

  /* Left 3D Control Dock */
  .left-dock {
    position: absolute; top: 82px; left: 14px; width: 252px;
    padding: 12px 14px;
    transform: rotateY(4deg);
    transform-origin: left center;
    max-height: calc(100vh - 200px); overflow-y: auto;
  }
  .left-dock:hover { transform: rotateY(0deg); }
  .sec-title {
    font-size: 10px; font-weight: 700; letter-spacing: 1.3px;
    color: var(--muted); text-transform: uppercase; margin: 10px 0 6px;
  }
  .sec-title:first-child { margin-top: 0; }
  .ctrl-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  select {
    width: 100%; background: var(--elevated); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 8px; font-size: 11px; margin-bottom: 6px;
  }

  /* Right 3D Telemetry / AI Panel */
  .right-dock {
    position: absolute; top: 82px; right: 14px; width: 310px;
    padding: 12px 14px;
    transform: rotateY(-4deg);
    transform-origin: right center;
    max-height: calc(100vh - 200px); overflow-y: auto;
  }
  .right-dock:hover { transform: rotateY(0deg); }
  .zone-row {
    background: var(--elevated); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 10px; margin-bottom: 6px;
  }
  .zone-top { display: flex; justify-content: space-between; font-size: 11px; font-weight: 700; }
  .bar-bg { height: 5px; background: rgba(148,163,184,0.18); border-radius: 3px; margin: 5px 0; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.35s ease; }
  .mono { font-family: 'IBM Plex Mono', monospace; }

  /* Bottom 3D KPI Strip */
  .bottom-strip {
    position: absolute; bottom: 14px; left: 14px; right: 14px;
    padding: 10px 14px;
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px;
  }
  .kpi {
    background: var(--elevated); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px;
  }
  .kpi-lbl { font-size: 9px; font-weight: 700; letter-spacing: 1px; color: var(--muted); text-transform: uppercase; }
  .kpi-val { font-family: 'IBM Plex Mono', monospace; font-size: 16px; font-weight: 700; color: var(--accent); margin: 2px 0; }
  .kpi-sub { font-size: 10px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* Dismissible 3D Junction Inspector */
  .inspector {
    position: absolute; top: 82px; left: 280px; width: 245px;
    padding: 12px 14px; display: none;
    border-top: 3px solid var(--accent);
    z-index: 200;
    pointer-events: all;
  }
  .insp-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
  .insp-row { display: flex; justify-content: space-between; font-size: 11px; margin: 4px 0; color: var(--muted); }
  .insp-row b { color: var(--text); font-family: 'IBM Plex Mono', monospace; }
  #insp-close {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    color: #ef4444;
    font-size: 15px;
    line-height: 1;
    width: 24px; height: 24px;
    border-radius: 6px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.15s;
    flex-shrink: 0;
  }
  #insp-close:hover { background: rgba(239,68,68,0.35); }

  /* Leak Alert Banner */
  .leak-banner {
    position: absolute; top: 76px; left: 50%; transform: translateX(-50%);
    background: rgba(127, 29, 29, 0.92); color: #fff;
    border: 1px solid #ef4444; border-radius: 8px;
    padding: 7px 16px; font-size: 12px; font-weight: 700;
    display: none; align-items: center; gap: 10px;
    box-shadow: 0 0 24px rgba(239, 68, 68, 0.55);
  }
  @media (max-width: 1100px) {
    .bottom-strip { grid-template-columns: repeat(3, 1fr); }
    .left-dock, .right-dock { width: 225px; }
  }
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<div id="canvas-container"></div>

<div class="hud">
  <!-- TOP BAR -->
  <div class="panel topbar">
    <div class="brand">
      <span style="font-size:22px;">💧</span>
      <div>
        <div class="brand-title">SMARTLEAK <span style="font-size:10px;color:var(--purple);">3D DIGITAL TWIN</span></div>
        <div class="brand-sub">Every Drop Matters. · WNTR / EPANET Simulated Telemetry</div>
      </div>
    </div>
    <div class="nav-pills">
      <button class="btn active" data-cam="overview">🌐 3D Overview</button>
      <button class="btn" data-tab="zones">🚨 Zone Risk</button>
      <button class="btn" data-tab="ai">🧠 NeuralFlow AI</button>
      <button class="btn" data-tab="telemetry">📡 12-Node Telemetry</button>
      <button class="btn" data-tab="events">🔔 Activity &amp; Export</button>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <span class="status-pill" id="hud-clock">T+000s / 150s</span>
      <span class="status-pill" id="hud-status">NORMAL</span>
      <button class="btn" id="btn-theme">☀️ Light</button>
    </div>
  </div>

  <!-- LEAK ALERT -->
  <div class="panel leak-banner" id="leak-banner">
    <span>⚠</span>
    <span id="leak-banner-txt">SIMULATED LEAK DETECTED</span>
  </div>

  <!-- LEFT DOCK: SIMULATION & CAMERA -->
  <div class="panel left-dock">
    <div class="sec-title">Simulation Controls</div>
    <div class="ctrl-grid">
      <button class="btn primary" id="btn-advance">▶ Advance</button>
      <button class="btn" id="btn-step5">▶▶ +5 Steps</button>
      <button class="btn" id="btn-play">▶ Auto-Run</button>
      <button class="btn" id="btn-reset">↺ Reset</button>
    </div>

    <div class="sec-title">Auto-Detect / Inject Leak</div>
    <select id="sel-leak-junc">
      <option value="AUTO" selected>🎲 Random Zone (Auto-Detect)</option>
      <option value="A1">A1 (Zone A)</option>
      <option value="A2">A2 (Zone A)</option>
      <option value="A3">A3 (Zone A)</option>
      <option value="B1">B1 (Zone B)</option>
      <option value="B2">B2 (Zone B)</option>
      <option value="B3">B3 (Zone B)</option>
      <option value="C1">C1 (Zone C)</option>
      <option value="C2">C2 (Zone C)</option>
      <option value="C3">C3 (Zone C)</option>
      <option value="D1">D1 (Zone D)</option>
      <option value="D2">D2 (Zone D)</option>
      <option value="D3">D3 (Zone D)</option>
    </select>
    <button class="btn danger" id="btn-inject" style="width:100%;">⚡ Inject Simulated Leak</button>

    <div class="sec-title">3D Camera Fly-To</div>
    <div class="ctrl-grid">
      <button class="btn" data-cam="overview">Overview</button>
      <button class="btn" data-cam="RESERVOIR">Reservoir</button>
      <button class="btn" data-cam="A">Zone A</button>
      <button class="btn" data-cam="B">Zone B</button>
      <button class="btn" data-cam="C">Zone C</button>
      <button class="btn" data-cam="D">Zone D</button>
    </div>

    <div class="sec-title">3D Scene Options</div>
    <div class="ctrl-grid">
      <button class="btn active" id="btn-labels">🏷️ Labels</button>
      <button class="btn active" id="btn-particles">💧 Flow</button>
      <button class="btn" id="btn-orbit" style="grid-column: span 2;">🔄 Auto-Orbit: OFF</button>
    </div>
  </div>

  <!-- DISMISSIBLE 3D JUNCTION INSPECTOR -->
  <div class="panel inspector" id="inspector">
    <div class="insp-head">
      <b id="insp-id" style="font-size:15px;">B2</b>
      <div style="display:flex;align-items:center;gap:6px;">
        <span class="status-pill" id="insp-status" style="font-size:9px;">NORMAL</span>
        <button id="insp-close" title="Close (Esc)">&#10005;</button>
      </div>
    </div>
    <div class="insp-row"><span>Zone</span><b id="insp-zone">Zone B</b></div>
    <div class="insp-row"><span>Elevation</span><b id="insp-elev">53.0 m</b></div>
    <div class="insp-row"><span>Pressure</span><b id="insp-press">—</b></div>
    <div class="insp-row"><span>Baseline / ΔP</span><b id="insp-delta">—</b></div>
    <div class="insp-row"><span>Zone Risk</span><b id="insp-risk">0.0%</b></div>
    <div class="sec-title" style="margin-top:8px;">Connected Pipes</div>
    <div id="insp-pipes" class="mono" style="font-size:10px;color:var(--accent);"></div>
  </div>

  <!-- RIGHT DOCK: ZONE RISK / NEURALFLOW AI / TELEMETRY / EVENTS -->
  <div class="panel right-dock" id="right-dock">
    <div class="sec-title" id="right-title">Zone Risk Assessment</div>
    <div id="right-content"></div>
  </div>

  <!-- BOTTOM KPI STRIP -->
  <div class="panel bottom-strip">
    <div class="kpi"><div class="kpi-lbl">Avg Pressure</div><div class="kpi-val" id="kpi-p">—</div><div class="kpi-sub" id="kpi-p-sub">12-node mean</div></div>
    <div class="kpi"><div class="kpi-lbl">Supply Flow</div><div class="kpi-val" id="kpi-q">—</div><div class="kpi-sub" id="kpi-q-sub">P_SUPPLY trunk</div></div>
    <div class="kpi"><div class="kpi-lbl">System Health</div><div class="kpi-val" id="kpi-health">NORMAL</div><div class="kpi-sub" id="kpi-phase">NORMAL NETWORK</div></div>
    <div class="kpi"><div class="kpi-lbl">Highest Risk</div><div class="kpi-val" id="kpi-risk">0.0%</div><div class="kpi-sub" id="kpi-risk-sub">Zone B</div></div>
    <div class="kpi"><div class="kpi-lbl">Suspected Leak</div><div class="kpi-val" id="kpi-leak">None</div><div class="kpi-sub" id="kpi-loss">Est loss: 0.0 L/min</div></div>
    <div class="kpi"><div class="kpi-lbl">Reservoir</div><div class="kpi-val" style="color:var(--success);">79.0 m</div><div class="kpi-sub">Fixed head online</div></div>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x08111d, 0.0085);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 350);
const DEFAULT_POS = new THREE.Vector3(0, 22, 32);
const DEFAULT_TGT = new THREE.Vector3(1.5, 2.2, 1.8);
camera.position.copy(DEFAULT_POS);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(DEFAULT_TGT);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 8;
controls.maxDistance = 75;
controls.maxPolarAngle = Math.PI / 2 - 0.03;

// Lighting
scene.add(new THREE.AmbientLight(0x9ec5e0, 0.95));
const sun = new THREE.DirectionalLight(0xffffff, 1.35);
sun.position.set(16, 32, 20);
scene.add(sun);
const cyanLight = new THREE.PointLight(0x06b6d4, 1.2, 70);
cyanLight.position.set(-12, 14, -8);
scene.add(cyanLight);

let grid = new THREE.GridHelper(80, 40, 0x1e3a52, 0x0e1d2d);
scene.add(grid);

// Ambient 3D floating backdrop particles for full-world depth
const starGeo = new THREE.BufferGeometry();
const starPos = new Float32Array(240 * 3);
for (let i = 0; i < 240; i++) {
  starPos[i * 3] = (Math.random() - 0.5) * 90;
  starPos[i * 3 + 1] = 1 + Math.random() * 24;
  starPos[i * 3 + 2] = (Math.random() - 0.5) * 70;
}
starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0x38bdf8, size: 0.16, transparent: true, opacity: 0.45 }));
scene.add(stars);

const networkGroup = new THREE.Group();
scene.add(networkGroup);

let CURRENT_STATE = null;
let ACTIVE_TAB = 'zones';
let LABELS_VISIBLE = true;
let PARTICLES_VISIBLE = true;
let AUTO_ORBIT = false;
let IS_LIGHT = false;
let autoRunTimer = null;

let camAnimTargetPos = null;
let camAnimTargetLook = null;

let clickableMeshes = [];
let flowParticles = [];
let leakSprayPoints = null;
let leakDrops = [];
let leakRingMesh = null;
let leakPulseMesh = null;
let reservoirWaterMesh = null;
let selectedRingMesh = new THREE.Mesh(
  new THREE.TorusGeometry(0.92, 0.06, 12, 32),
  new THREE.MeshBasicMaterial({ color: 0xffffff })
);
selectedRingMesh.rotation.x = Math.PI / 2;
selectedRingMesh.visible = false;
scene.add(selectedRingMesh);

let nodeMap = {};

function makeLabel(title, sub, borderHex) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = sub ? 92 : 64;
  const ctx = c.getContext('2d');
  ctx.fillStyle = IS_LIGHT ? 'rgba(255,255,255,0.92)' : 'rgba(8,17,29,0.88)';
  ctx.strokeStyle = borderHex;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.roundRect(8, 8, c.width - 16, c.height - 16, 12);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = IS_LIGHT ? '#0f172a' : '#ffffff';
  ctx.font = 'bold 26px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(title, 128, 42);
  if (sub) {
    ctx.fillStyle = borderHex;
    ctx.font = 'bold 20px monospace';
    ctx.fillText(sub, 128, 70);
  }
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(c), transparent: true, depthTest: false,
  }));
  sp.visible = LABELS_VISIBLE;
  return sp;
}

function rebuild3DScene(state) {
  CURRENT_STATE = state;
  while (networkGroup.children.length > 0) {
    networkGroup.remove(networkGroup.children[0]);
  }
  clickableMeshes = [];
  flowParticles = [];
  leakSprayPoints = null;
  leakDrops = [];
  leakRingMesh = null;
  leakPulseMesh = null;
  reservoirWaterMesh = null;
  nodeMap = {};

  // Zones
  state.zones.forEach(z => {
    const w = z.x_max - z.x_min, d = z.z_max - z.z_min;
    const cx = (z.x_min + z.x_max) / 2, cz = (z.z_min + z.z_max) / 2;
    const col = z.risk >= 61 ? 0xd63939 : z.risk >= 31 ? 0xf59f00 : 0x06b6d4;
    const hex = z.risk >= 61 ? '#d63939' : z.risk >= 31 ? '#f59f00' : '#06b6d4';

    const geo = new THREE.BoxGeometry(w, 0.08, d);
    const pad = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color: col, transparent: true, opacity: z.risk >= 31 ? 0.2 : 0.08,
    }));
    pad.position.set(cx, 0.04, cz);
    networkGroup.add(pad);

    const wire = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.75 })
    );
    wire.position.copy(pad.position);
    networkGroup.add(wire);

    const lbl = makeLabel(`ZONE ${z.zone}`, `${z.status} · ${z.risk.toFixed(0)}%`, hex);
    lbl.position.set(cx, 0.9, z.z_max - 0.5);
    lbl.scale.set(4.0, 1.4, 1);
    networkGroup.add(lbl);
  });

  // Nodes
  state.nodes.forEach(n => {
    nodeMap[n.id] = n;
    const pos = new THREE.Vector3(n.x, n.y, n.z);
    n.vec = pos;

    if (n.id === 'RESERVOIR') {
      const pillar = new THREE.Mesh(
        new THREE.CylinderGeometry(0.9, 1.3, n.y - 1.2, 16),
        new THREE.MeshStandardMaterial({ color: 0x1e3448, metalness: 0.6, roughness: 0.4 })
      );
      pillar.position.set(n.x, (n.y - 1.2) / 2, n.z);
      networkGroup.add(pillar);

      const tank = new THREE.Mesh(
        new THREE.CylinderGeometry(2.1, 2.1, 2.6, 24),
        new THREE.MeshPhysicalMaterial({ color: 0x0284c7, transparent: true, opacity: 0.45, roughness: 0.15 })
      );
      tank.position.copy(pos);
      tank.userData = { nodeId: 'RESERVOIR' };
      networkGroup.add(tank);
      clickableMeshes.push(tank);

      reservoirWaterMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(1.9, 1.9, 1.8, 24),
        new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0284c7, emissiveIntensity: 0.6, transparent: true, opacity: 0.85 })
      );
      reservoirWaterMesh.position.set(n.x, n.y - 0.25, n.z);
      networkGroup.add(reservoirWaterMesh);

      const lbl = makeLabel('RESERVOIR', '79.0 m H₂O', '#38bdf8');
      lbl.position.set(n.x, n.y + 2.5, n.z);
      lbl.scale.set(3.2, 1.2, 1);
      networkGroup.add(lbl);
    } else {
      const isDanger = n.is_leak || n.risk >= 61;
      const isWarn = !isDanger && n.risk >= 31;
      const col = isDanger ? 0xd63939 : isWarn ? 0xf59f00 : 0x06b6d4;
      const hex = isDanger ? '#d63939' : isWarn ? '#f59f00' : '#06b6d4';

      const stem = new THREE.Mesh(
        new THREE.CylinderGeometry(0.1, 0.16, n.y, 10),
        new THREE.MeshStandardMaterial({ color: 0x1e3448, metalness: 0.5 })
      );
      stem.position.set(n.x, n.y / 2, n.z);
      networkGroup.add(stem);

      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(isDanger ? 0.72 : 0.58, 24, 24),
        new THREE.MeshStandardMaterial({
          color: col, emissive: col, emissiveIntensity: isDanger ? 0.85 : 0.4, roughness: 0.25, metalness: 0.3,
        })
      );
      sphere.position.copy(pos);
      sphere.userData = { nodeId: n.id };
      networkGroup.add(sphere);
      clickableMeshes.push(sphere);

      const lbl = makeLabel(n.id, `${n.pressure_m.toFixed(1)} m`, hex);
      lbl.position.set(n.x, n.y + 1.45, n.z);
      lbl.scale.set(2.5, 0.95, 1);
      networkGroup.add(lbl);

      if (n.is_leak) {
        leakRingMesh = new THREE.Mesh(
          new THREE.RingGeometry(0.85, 1.15, 32),
          new THREE.MeshBasicMaterial({ color: 0xd63939, side: THREE.DoubleSide, transparent: true, opacity: 0.9 })
        );
        leakRingMesh.rotation.x = -Math.PI / 2;
        leakRingMesh.position.set(n.x, 0.14, n.z);
        networkGroup.add(leakRingMesh);

        leakPulseMesh = new THREE.Mesh(
          new THREE.SphereGeometry(0.95, 20, 20),
          new THREE.MeshBasicMaterial({ color: 0xd63939, transparent: true, opacity: 0.35, wireframe: true })
        );
        leakPulseMesh.position.copy(pos);
        networkGroup.add(leakPulseMesh);
      }
    }
  });

  // Pipes + Flow Particles
  state.pipes.forEach(p => {
    const s = nodeMap[p.start], e = nodeMap[p.end];
    if (!s || !e) return;
    const pCol = p.level === 'danger' ? 0xd63939 : p.level === 'warning' ? 0xf59f00 : 0x0ea5e9;
    const rad = Math.max(0.14, Math.min(0.32, p.diameter_m * 1.25));
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(new THREE.LineCurve3(s.vec, e.vec), 10, rad, 12, false),
      new THREE.MeshPhysicalMaterial({
        color: pCol, emissive: pCol, emissiveIntensity: p.level === 'danger' ? 0.55 : 0.22,
        transparent: true, opacity: p.level === 'danger' ? 0.88 : 0.55, roughness: 0.2,
      })
    );
    networkGroup.add(tube);

    if (p.flow_lps > 0.02) {
      const rev = p.signed_flow_lps < 0;
      const spd = Math.max(0.18, Math.min(1.4, p.flow_lps * 0.28));
      for (let i = 0; i < 4; i++) {
        const pt = new THREE.Mesh(
          new THREE.SphereGeometry(rad * 0.72, 8, 8),
          new THREE.MeshBasicMaterial({ color: p.level === 'danger' ? 0xfca5a5 : 0xe0f2fe })
        );
        pt.visible = PARTICLES_VISIBLE;
        networkGroup.add(pt);
        flowParticles.push({
          mesh: pt, start: rev ? e.vec : s.vec, end: rev ? s.vec : e.vec, t: i / 4, speed: spd,
        });
      }
    }
  });

  // 3D Water-Escape Spray
  if (state.leak.active && state.leak.junction && nodeMap[state.leak.junction]) {
    const ln = nodeMap[state.leak.junction];
    const cnt = 96;
    const posArr = new Float32Array(cnt * 3);
    const colArr = new Float32Array(cnt * 3);
    for (let i = 0; i < cnt; i++) {
      const a = Math.random() * Math.PI * 2;
      const sp = 0.8 + Math.random() * 1.8;
      leakDrops.push({ vx: Math.cos(a) * sp, vy: 3.2 + Math.random() * 3.8, vz: Math.sin(a) * sp, life: Math.random() });
      posArr[i * 3] = ln.x; posArr[i * 3 + 1] = ln.y; posArr[i * 3 + 2] = ln.z;
      colArr[i * 3] = 0.22; colArr[i * 3 + 1] = 0.85; colArr[i * 3 + 2] = 1.0;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
    leakSprayPoints = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.35, vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    leakSprayPoints.userData = { origin: ln.vec };
    networkGroup.add(leakSprayPoints);
  }

  updateHUD(state);
}

function updateHUD(state) {
  document.getElementById('hud-clock').textContent = `T+${String(state.seconds).padStart(3, '0')}s / 150s`;
  const stEl = document.getElementById('hud-status');
  stEl.textContent = state.network_status;
  stEl.style.color = state.network_status === 'HIGH RISK' ? 'var(--danger)' :
                     state.network_status === 'NORMAL' ? 'var(--success)' : 'var(--warning)';

  document.getElementById('kpi-p').textContent = `${state.mean_pressure_m.toFixed(2)} m`;
  document.getElementById('kpi-p-sub').textContent = `${state.mean_pressure_dev_m >= 0 ? '+' : ''}${state.mean_pressure_dev_m.toFixed(2)} m vs base`;
  document.getElementById('kpi-q').textContent = `${state.supply_flow_lps.toFixed(3)} L/s`;
  document.getElementById('kpi-q-sub').textContent = `ΔQ ${state.supply_flow_dev_lps >= 0 ? '+' : ''}${state.supply_flow_dev_lps.toFixed(3)} L/s`;
  document.getElementById('kpi-health').textContent = state.network_status;
  document.getElementById('kpi-health').style.color = stEl.style.color;
  document.getElementById('kpi-phase').textContent = state.phase;
  document.getElementById('kpi-risk').textContent = `${state.highest_risk.toFixed(1)}%`;
  document.getElementById('kpi-risk').style.color = state.highest_risk >= 61 ? 'var(--danger)' : state.highest_risk >= 31 ? 'var(--warning)' : 'var(--success)';
  document.getElementById('kpi-risk-sub').textContent = `Peak: Zone ${state.top_zone}`;

  const susZ = state.probable_zone || state.predicted_zone;
  document.getElementById('kpi-leak').textContent = susZ ? `Zone ${susZ} · ${state.leak.junction || ''}` : 'None';
  document.getElementById('kpi-loss').textContent = `Est loss: ~${state.loss_l_min.toFixed(1)} L/min`;

  const banner = document.getElementById('leak-banner');
  if (state.leak.active || state.highest_risk >= 31) {
    banner.style.display = 'flex';
    document.getElementById('leak-banner-txt').textContent =
      `SIMULATED LEAK · Zone ${susZ || state.top_zone} (${state.leak.junction || ''}) · Risk ${state.highest_risk.toFixed(1)}% · ~${state.loss_l_min.toFixed(1)} L/min`;
  } else {
    banner.style.display = 'none';
  }

  renderRightPanel(state);
}

function renderRightPanel(state) {
  const titleEl = document.getElementById('right-title');
  const box = document.getElementById('right-content');
  if (ACTIVE_TAB === 'zones') {
    titleEl.textContent = 'Zone-Wise Risk Assessment';
    box.innerHTML = state.zones.map(z => {
      const col = z.risk >= 61 ? 'var(--danger)' : z.risk >= 31 ? 'var(--warning)' : 'var(--success)';
      return `
        <div class="zone-row">
          <div class="zone-top"><span>Zone ${z.zone} · ${z.title}</span><span class="mono" style="color:${col}">${z.risk.toFixed(1)}%</span></div>
          <div class="bar-bg"><div class="bar-fill" style="width:${Math.max(2, z.risk)}%;background:${col};"></div></div>
          <div class="mono" style="font-size:10px;color:var(--muted);display:flex;justify-content:space-between;">
            <span>P: ${z.pressure_m}m (${z.dev_p >= 0 ? '+' : ''}${z.dev_p})</span>
            <span>Q: ${z.flow_lps} L/s</span>
          </div>
        </div>`;
    }).join('');
  } else if (ACTIVE_TAB === 'ai') {
    titleEl.textContent = 'NeuralFlow Isolation Forest AI';
    box.innerHTML = state.zones.map(z => `
      <div class="zone-row">
        <div class="zone-top">
          <span>Zone ${z.zone} (${z.ml_flag ? '⚠ ANOMALY' : '✓ NOMINAL'})</span>
          <span class="mono" style="color:var(--purple);">IF: ${z.ml_score.toFixed(4)}</span>
        </div>
        <div style="font-size:10.5px;color:var(--muted);margin-top:4px;line-height:1.35;">${z.explanation}</div>
      </div>`).join('');
  } else if (ACTIVE_TAB === 'telemetry') {
    titleEl.textContent = '12-Junction Live Telemetry';
    box.innerHTML = state.nodes.filter(n => n.id !== 'RESERVOIR').map(n => {
      const col = n.is_leak || n.status === 'CRITICAL' ? 'var(--danger)' : n.status === 'WARNING' ? 'var(--warning)' : 'var(--accent)';
      return `
        <div class="zone-row" style="padding:5px 8px;display:flex;justify-content:space-between;align-items:center;font-size:11px;">
          <span><b>${n.id}</b> <span style="color:var(--muted);">(Z${n.zone})</span></span>
          <span class="mono" style="color:${col};">${n.pressure_m.toFixed(2)}m (${n.dev_m >= 0 ? '+' : ''}${n.dev_m.toFixed(2)})</span>
        </div>`;
    }).join('');
  } else {
    titleEl.textContent = 'Activity Log & Recommendations';
    const actHtml = state.activity.map(a =>
      `<div class="zone-row" style="font-size:10.5px;"><span class="mono" style="color:var(--accent);">[${a.time}]</span> ${a.message}</div>`
    ).join('');
    box.innerHTML = actHtml + `<button class="btn primary" style="width:100%;margin-top:6px;" onclick="window.open('/api/export','_blank')">⬇ Download JSON Telemetry</button>`;
  }
}

async function apiCall(endpoint, body = null) {
  const res = await fetch(endpoint, {
    method: body !== null ? 'POST' : 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: body !== null ? JSON.stringify(body) : null,
  });
  const data = await res.json();
  rebuild3DScene(data);
}

// Controls
document.getElementById('btn-advance').onclick = () => apiCall('/api/advance', {});
document.getElementById('btn-step5').onclick = () => apiCall('/api/step5', {});
document.getElementById('btn-reset').onclick = () => {
  if (autoRunTimer) { clearInterval(autoRunTimer); autoRunTimer = null; document.getElementById('btn-play').textContent = '▶ Auto-Run'; }
  apiCall('/api/reset', {});
};
document.getElementById('btn-inject').onclick = () => {
  const j = document.getElementById('sel-leak-junc').value;
  apiCall('/api/inject', { junction: j });
};
document.getElementById('btn-play').onclick = () => {
  const btn = document.getElementById('btn-play');
  if (autoRunTimer) {
    clearInterval(autoRunTimer); autoRunTimer = null; btn.textContent = '▶ Auto-Run';
  } else {
    btn.textContent = '⏸ Pause';
    autoRunTimer = setInterval(() => {
      if (CURRENT_STATE && CURRENT_STATE.seconds < 150) apiCall('/api/advance', {});
      else { clearInterval(autoRunTimer); autoRunTimer = null; btn.textContent = '▶ Auto-Run'; }
    }, 900);
  }
};

document.querySelectorAll('[data-tab]').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('[data-tab]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    ACTIVE_TAB = btn.dataset.tab;
    if (CURRENT_STATE) renderRightPanel(CURRENT_STATE);
  };
});

document.querySelectorAll('[data-cam]').forEach(btn => {
  btn.onclick = () => {
    const target = btn.dataset.cam;
    if (target === 'overview') {
      camAnimTargetPos = DEFAULT_POS.clone();
      camAnimTargetLook = DEFAULT_TGT.clone();
    } else if (target === 'RESERVOIR') {
      camAnimTargetPos = new THREE.Vector3(-21, 14, 16);
      camAnimTargetLook = new THREE.Vector3(-21, 5, 0);
    } else if (CURRENT_STATE) {
      const z = CURRENT_STATE.zones.find(item => item.zone === target);
      if (z) {
        const cx = (z.x_min + z.x_max) / 2;
        camAnimTargetPos = new THREE.Vector3(cx, 14, 18);
        camAnimTargetLook = new THREE.Vector3(cx, 2.5, 2.0);
      }
    }
  };
});

document.getElementById('btn-labels').onclick = (e) => {
  LABELS_VISIBLE = !LABELS_VISIBLE;
  e.target.classList.toggle('active', LABELS_VISIBLE);
  networkGroup.traverse(o => { if (o.isSprite) o.visible = LABELS_VISIBLE; });
};
document.getElementById('btn-particles').onclick = (e) => {
  PARTICLES_VISIBLE = !PARTICLES_VISIBLE;
  e.target.classList.toggle('active', PARTICLES_VISIBLE);
  flowParticles.forEach(fp => { fp.mesh.visible = PARTICLES_VISIBLE; });
};
document.getElementById('btn-orbit').onclick = (e) => {
  AUTO_ORBIT = !AUTO_ORBIT;
  controls.autoRotate = AUTO_ORBIT;
  controls.autoRotateSpeed = 0.85;
  e.target.textContent = `🔄 Auto-Orbit: ${AUTO_ORBIT ? 'ON' : 'OFF'}`;
  e.target.classList.toggle('active', AUTO_ORBIT);
};
document.getElementById('btn-theme').onclick = (e) => {
  IS_LIGHT = !IS_LIGHT;
  document.body.classList.toggle('light', IS_LIGHT);
  e.target.textContent = IS_LIGHT ? '🌙 Dark' : '☀️ Light';
  scene.fog.color.setHex(IS_LIGHT ? 0xe9f0f7 : 0x08111d);
  scene.remove(grid);
  grid = new THREE.GridHelper(80, 40, IS_LIGHT ? 0x94a3b8 : 0x1e3a52, IS_LIGHT ? 0xcbd5e1 : 0x0e1d2d);
  scene.add(grid);
  if (CURRENT_STATE) rebuild3DScene(CURRENT_STATE);
};

// Clickable 3D Junctions
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let downPos = { x: 0, y: 0 };
renderer.domElement.addEventListener('pointerdown', e => { downPos = { x: e.clientX, y: e.clientY }; });
renderer.domElement.addEventListener('pointerup', e => {
  if (Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y) > 6) return;
  pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(clickableMeshes, false);
  if (hits.length > 0 && hits[0].object.userData.nodeId) {
    const n = nodeMap[hits[0].object.userData.nodeId];
    if (n) {
      selectedRingMesh.visible = true;
      selectedRingMesh.position.set(n.x, n.y, n.z);
      document.getElementById('inspector').style.display = 'block';
      document.getElementById('insp-id').textContent = n.id;
      document.getElementById('insp-status').textContent = n.is_leak ? 'LEAK TARGET' : n.status;
      document.getElementById('insp-zone').textContent = n.zone === 'SUPPLY' ? 'Supply' : `Zone ${n.zone}`;
      document.getElementById('insp-elev').textContent = `${n.elevation_m.toFixed(1)} m`;
      document.getElementById('insp-press').textContent = `${n.pressure_m.toFixed(2)} m`;
      document.getElementById('insp-delta').textContent = `${n.baseline_m.toFixed(1)}m (${n.dev_m >= 0 ? '+' : ''}${n.dev_m.toFixed(2)})`;
      document.getElementById('insp-risk').textContent = `${n.risk.toFixed(1)}%`;
      document.getElementById('insp-pipes').innerHTML = (n.connected_flows || [])
        .map(cf => `<div style="display:flex;justify-content:space-between;margin-top:2px;"><span>${cf.pipe}</span><b>${cf.flow_lps.toFixed(2)} L/s</b></div>`).join('');
    }
  } else {
    document.getElementById('inspector').style.display = 'none';
    selectedRingMesh.visible = false;
  }
});
document.getElementById('insp-close').onclick = (e) => {
  e.stopPropagation();
  document.getElementById('inspector').style.display = 'none';
  selectedRingMesh.visible = false;
};
// Stop pointer events on the inspector from reaching the canvas
document.getElementById('inspector').addEventListener('pointerdown', e => e.stopPropagation());
document.getElementById('inspector').addEventListener('pointerup',   e => e.stopPropagation());
// Escape key dismisses inspector
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && document.getElementById('inspector').style.display === 'block') {
    document.getElementById('inspector').style.display = 'none';
    selectedRingMesh.visible = false;
  }
});

// Main 3D Render Loop
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  if (document.visibilityState === 'hidden') return;
  const dt = Math.min(0.05, clock.getDelta());
  const elapsed = clock.getElapsedTime();

  if (camAnimTargetPos && camAnimTargetLook) {
    camera.position.lerp(camAnimTargetPos, 0.08);
    controls.target.lerp(camAnimTargetLook, 0.08);
    if (camera.position.distanceTo(camAnimTargetPos) < 0.15) {
      camAnimTargetPos = null; camAnimTargetLook = null;
    }
  }

  if (PARTICLES_VISIBLE) {
    for (const fp of flowParticles) {
      fp.t = (fp.t + dt * fp.speed) % 1.0;
      fp.mesh.position.lerpVectors(fp.start, fp.end, fp.t);
    }
  }
  if (reservoirWaterMesh) reservoirWaterMesh.scale.y = 1.0 + Math.sin(elapsed * 2.2) * 0.03;
  if (leakRingMesh) {
    const s = 1.0 + ((elapsed * 1.4) % 1.0) * 2.4;
    leakRingMesh.scale.set(s, s, 1);
    leakRingMesh.material.opacity = Math.max(0, 1.0 - ((elapsed * 1.4) % 1.0));
  }
  if (leakPulseMesh) {
    const ps = 1.0 + Math.sin(elapsed * 5.0) * 0.18;
    leakPulseMesh.scale.set(ps, ps, ps);
  }
  if (leakSprayPoints) {
    const attr = leakSprayPoints.geometry.getAttribute('position');
    const orig = leakSprayPoints.userData.origin;
    for (let i = 0; i < leakDrops.length; i++) {
      const d = leakDrops[i];
      d.life += dt * 1.25;
      if (d.life >= 1.0 || orig.y + d.vy * d.life - 4.9 * d.life * d.life < 0.05) d.life = 0.0;
      const t = d.life;
      attr.setXYZ(i, orig.x + d.vx * t, Math.max(0.08, orig.y + d.vy * t - 4.9 * t * t), orig.z + d.vz * t);
    }
    attr.needsUpdate = true;
  }

  stars.rotation.y += dt * 0.015;
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Load initial state from Python WNTR engine
apiCall('/api/state');
</script>
</body>
</html>
"""


class SmartLeak3DHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif path == "/api/state":
            with _STATE_LOCK:
                self._send_json(_build_state_payload())
        elif path == "/api/export":
            with _STATE_LOCK:
                raw = json.dumps(_build_state_payload(), indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", 'attachment; filename="smartleak_3d_telemetry.json"')
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        with _STATE_LOCK:
            if path == "/api/advance":
                _SIM.advance()
                _record_trend()
                _log_event(f"Advanced to T+{_SIM.seconds:03d}s ({_SIM.network_status}).", "INFO")
            elif path == "/api/step5":
                for _ in range(5):
                    if _SIM.seconds < 150:
                        _SIM.advance()
                        _record_trend()
                _log_event(f"Fast-forwarded +5 steps to T+{_SIM.seconds:03d}s.", "INFO")
            elif path == "/api/inject":
                junc = str(body.get("junction", "AUTO"))
                if junc == "AUTO":
                    import random
                    junc = random.choice(list(_SIM.node_zones.keys()))
                _SIM.introduce_leak(junc)
                _record_trend()
                det = _SIM.probable_zone or getattr(_SIM, "predicted_leak_zone", None) or _SIM.node_zones[junc]
                _log_event(f"Leak triggered at {junc} — Auto-detected in Zone {det}.", "WARNING")
            elif path == "/api/reset":
                _SIM.reset()
                for k in _TREND:
                    _TREND[k].clear()
                _record_trend()
                _log_event("Network reset to nominal baseline (T+000s).", "NORMAL")
            else:
                self.send_error(404)
                return
            payload = _build_state_payload()
        self._send_json(payload)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def main() -> None:
    port = int(os.environ.get("SMARTLEAK_3D_PORT", "8502"))
    server = ThreadingHTTPServer(("0.0.0.0", port), SmartLeak3DHandler)
    print(f"SmartLeak Full-3D Three.js Digital Twin running at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
