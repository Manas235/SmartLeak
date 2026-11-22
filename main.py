#!/usr/bin/env python3
"""SmartLeak Industrial SCADA & Water Utility Intelligence Digital Twin Platform.

Driven by WNTR/EPANET pressure-dependent hydraulics and Isolation Forest ML.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import wntr

# -----------------------------------------------------------------------------
# CORE SIMULATION CONSTANTS & HYDRAULIC CONFIGURATION
# -----------------------------------------------------------------------------
APP_TITLE = "SMARTLEAK  |  INDUSTRIAL WATER UTILITY SCADA & DIGITAL TWIN"
DEMO_SECONDS = 150
LEAK_START_SECONDS = 30
MAX_EMITTER_COEFFICIENT = 0.00028
PRESSURE_UNIT = "m"
FLOW_UNIT = "L/s"

RISK_WEIGHTS = {
    "pressure": 35.0,
    "flow": 35.0,
    "ml": 20.0,
    "persistence": 10.0,
}

ZONE_NODES: dict[str, tuple[str, ...]] = {
    "A": ("A1", "A2", "A3"),
    "B": ("B1", "B2", "B3"),
    "C": ("C1", "C2", "C3"),
    "D": ("D1", "D2", "D3"),
}
ZONE_PRESSURE_NODES = {
    "A": ("A1", "A2", "A3"),
    "B": ("B1", "B2", "B3"),
    "C": ("C1", "C2", "C3"),
    "D": ("D1", "D2", "D3"),
}
ZONE_FLOW_PIPES = {
    "A": "P_A_BRANCH",
    "B": "P_B_CORE",
    "C": "P_C_BRANCH",
    "D": "P_D_BRANCH",
}
ZONE_COLORS = {
    "NORMAL": "#21845b",
    "WARNING": "#c28b1c",
    "ANOMALY DETECTED": "#c28b1c",
    "HIGH RISK": "#d74b50",
}
NORMAL_PROFILE = (0.94, 0.97, 1.00, 1.03, 1.00, 0.97, 0.94, 1.00)
HYDRAULIC_STEP_SECONDS = 3600
LEAK_LEVELS = tuple(np.linspace(0.0, 1.0, 6))
PRESSURE_TOLERANCE_M = 2.8
FLOW_TOLERANCE_LPS = 1.3
HYDRAULIC_SOLVER_LOCK = threading.Lock()


@dataclass(frozen=True)
class HydraulicSnapshot:
    """Pressure and flow measurements returned by one WNTR simulation step."""

    pressures: dict[str, float]
    flows_lps: dict[str, float]
    signed_flows_lps: dict[str, float]


@dataclass(frozen=True)
class ZoneMeasurement:
    pressure_m: float
    flow_lps: float


def build_network(
    leak_fraction: float,
    demand_pattern: tuple[float, ...],
    leak_junction: str = "B2",
) -> wntr.network.WaterNetworkModel:
    """Construct the same 12-junction network for each hydraulic scenario."""
    network = wntr.network.WaterNetworkModel()
    network.options.hydraulic.demand_model = "PDD"
    network.options.hydraulic.minimum_pressure = 0.0
    network.options.hydraulic.required_pressure = 20.0
    network.options.hydraulic.pressure_exponent = 0.5
    network.options.time.hydraulic_timestep = HYDRAULIC_STEP_SECONDS
    network.options.time.report_timestep = HYDRAULIC_STEP_SECONDS
    network.options.time.pattern_timestep = HYDRAULIC_STEP_SECONDS
    network.options.time.duration = max(
        0, len(demand_pattern) - 1
    ) * HYDRAULIC_STEP_SECONDS
    network.add_pattern("CLASSROOM_DEMAND", list(demand_pattern))
    network.add_reservoir("RESERVOIR", base_head=79.0)

    elevations = {
        "A1": 51.0, "A2": 52.0, "A3": 51.5,
        "B1": 52.0, "B2": 53.0, "B3": 52.5,
        "C1": 53.0, "C2": 54.0, "C3": 53.5,
        "D1": 56.0, "D2": 57.0, "D3": 56.5,
    }
    if leak_junction not in elevations:
        raise ValueError(f"Unsupported leak junction {leak_junction!r}.")
    for junction, elevation in elevations.items():
        network.add_junction(
            junction,
            base_demand=0.00016,
            demand_pattern="CLASSROOM_DEMAND",
            elevation=elevation,
        )

    if leak_fraction > 0:
        network.get_node(leak_junction).emitter_coefficient = (
            MAX_EMITTER_COEFFICIENT * leak_fraction
        )

    # Trunk A->B->C->D allows Zone B leak to affect both local & downstream pressures
    pipes = (
        ("P_SUPPLY", "RESERVOIR", "A1", 300, 0.25),
        ("P_A_CORE", "A1", "A2", 550, 0.17),
        ("P_A_BRANCH", "A1", "A3", 420, 0.12),
        ("P_A_TO_B", "A2", "B1", 1300, 0.10),
        ("P_B_CORE", "B1", "B2", 1100, 0.10),
        ("P_B_BRANCH", "B1", "B3", 420, 0.12),
        ("P_B_TO_C", "B2", "C1", 1300, 0.10),
        ("P_C_CORE", "C1", "C2", 1100, 0.10),
        ("P_C_BRANCH", "C1", "C3", 420, 0.12),
        ("P_C_TO_D", "C2", "D1", 1300, 0.10),
        ("P_D_CORE", "D1", "D2", 1100, 0.10),
        ("P_D_BRANCH", "D1", "D3", 420, 0.12),
    )
    for name, start, end, length, diameter in pipes:
        network.add_pipe(
            name,
            start,
            end,
            length=length,
            diameter=diameter,
            roughness=130,
            minor_loss=0,
        )
    return network


def run_hydraulic_profile(
    leak_fraction: float,
    demand_pattern: tuple[float, ...],
    leak_junction: str = "B2",
) -> list[HydraulicSnapshot]:
    """Run EPANET hydraulics and convert its Pandas results into snapshots."""
    network = build_network(leak_fraction, demand_pattern, leak_junction)
    with tempfile.TemporaryDirectory(prefix="smartleak_epanet_") as work_dir:
        with HYDRAULIC_SOLVER_LOCK:
            result = wntr.sim.EpanetSimulator(network).run_sim(
                file_prefix=os.path.join(work_dir, "hydraulic")
            )
    pressure_frame: pd.DataFrame = result.node["pressure"]
    flow_frame: pd.DataFrame = result.link["flowrate"]

    snapshots: list[HydraulicSnapshot] = []
    for time_index in pressure_frame.index:
        pressures = {
            name: float(pressure_frame.loc[time_index, name])
            for name in pressure_frame.columns
            if name != "RESERVOIR"
        }
        signed_flows_lps = {
            name: float(flow_frame.loc[time_index, name]) * 1000.0
            for name in flow_frame.columns
        }
        flows_lps = {
            name: abs(flow) for name, flow in signed_flows_lps.items()
        }
        snapshots.append(HydraulicSnapshot(
            pressures,
            flows_lps,
            signed_flows_lps,
        ))
    if not snapshots:
        raise RuntimeError("WNTR returned no hydraulic measurements.")
    return snapshots


ZONE_INFLOW_PIPE = {
    "A": "P_SUPPLY",
    "B": "P_A_TO_B",
    "C": "P_B_TO_C",
    "D": "P_C_TO_D",
}
ZONE_OUTFLOW_PIPE = {
    "A": "P_A_TO_B",
    "B": "P_B_TO_C",
    "C": "P_C_TO_D",
    "D": None,
}


def zone_measurement(
    snapshot: HydraulicSnapshot,
    zone: str,
) -> ZoneMeasurement:
    """Calculates DMA-compliant mean zone pressure and net zone inflow.
    
    Conservation of mass ensures any leak in Zone A, B, C, or D creates
    an unambiguous surge in net inflow for that specific DMA zone.
    """
    p_mean = float(np.mean([
        snapshot.pressures[node] for node in ZONE_PRESSURE_NODES[zone]
    ]))
    inp = ZONE_INFLOW_PIPE[zone]
    outp = ZONE_OUTFLOW_PIPE[zone]
    q_net = snapshot.flows_lps[inp] - (snapshot.flows_lps[outp] if outp else 0.0)
    return ZoneMeasurement(
        pressure_m=p_mean,
        flow_lps=q_net,
    )


def features_for_history(
    history: list[ZoneMeasurement],
) -> np.ndarray:
    """Pressure, flow, changes, and short rolling means for the forest."""
    current = history[-1]
    previous = history[-2] if len(history) > 1 else current
    recent = history[-4:]
    return np.asarray([[
        current.pressure_m,
        current.flow_lps,
        current.pressure_m - previous.pressure_m,
        current.flow_lps - previous.flow_lps,
        float(np.mean([sample.pressure_m for sample in recent])),
        float(np.mean([sample.flow_lps for sample in recent])),
    ]], dtype=float)


class HydraulicCalibration:
    """Precompute repeatable WNTR cases and fit one normal-only forest per zone."""

    def __init__(self) -> None:
        self.normal_profile = run_hydraulic_profile(0.0, NORMAL_PROFILE)
        self.reference_index = min(
            range(len(NORMAL_PROFILE)),
            key=lambda index: abs(NORMAL_PROFILE[index] - 1.0),
        )
        self.baseline_snapshot = self.normal_profile[self.reference_index]
        self.scenarios: list[tuple[float, HydraulicSnapshot]] = []
        for strength in LEAK_LEVELS:
            profile = run_hydraulic_profile(
                float(strength),
                (1.0, 1.0),
            )
            self.scenarios.append((float(strength), profile[0]))

        self.baselines: dict[str, ZoneMeasurement] = {}
        self.pressure_limits: dict[str, float] = {}
        self.flow_limits: dict[str, float] = {}
        self.ml_scales: dict[str, float] = {}
        self.models: dict[str, IsolationForest] = {}

        for zone in ZONE_NODES:
            normal_series = [
                zone_measurement(snapshot, zone)
                for snapshot in self.normal_profile
            ]
            self.baselines[zone] = zone_measurement(
                self.baseline_snapshot,
                zone,
            )
            pressure_values = np.asarray(
                [sample.pressure_m for sample in normal_series],
                dtype=float,
            )
            flow_values = np.asarray(
                [sample.flow_lps for sample in normal_series],
                dtype=float,
            )
            self.pressure_limits[zone] = max(
                PRESSURE_TOLERANCE_M,
                float(np.std(pressure_values)) * 2.5,
            )
            self.flow_limits[zone] = max(
                FLOW_TOLERANCE_LPS,
                float(np.std(flow_values)) * 2.5,
            )
            training_rows = np.vstack([
                features_for_history(normal_series[:index + 1])[0]
                for index in range(len(normal_series))
            ])
            model = IsolationForest(
                n_estimators=200,
                contamination=0.14,
                random_state=42,
            )
            model.fit(training_rows)
            self.models[zone] = model
            self.ml_scales[zone] = max(
                0.012,
                float(np.std(model.decision_function(training_rows))) * 0.4,
            )

        self._validate_hydraulics()

    def _validate_hydraulics(self) -> None:
        base = dict(self.scenarios)[0.0]
        full = dict(self.scenarios)[1.0]
        if full.flows_lps[ZONE_FLOW_PIPES["B"]] <= base.flows_lps[
            ZONE_FLOW_PIPES["B"]
        ]:
            raise RuntimeError(
                "The WNTR leak scenario did not increase Zone B pipe flow."
            )
        if full.pressures["B2"] >= base.pressures["B2"]:
            raise RuntimeError(
                "The WNTR leak scenario did not reduce pressure at junction B2."
            )

class SmartLeakSimulation:
    """Deterministic clock, per-step WNTR readings, and explainable risk."""

    def __init__(self, calibration: HydraulicCalibration) -> None:
        self.calibration = calibration
        self._step_lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._step_lock:
            baseline = run_hydraulic_profile(0.0, (1.0,))[0]
            self.seconds = 0
            self.step_number = 0
            self.speed = 1
            self.running = False
            self.snapshot = baseline
            self.active_leak_junction: str | None = None
            self._manual_leak_severity: float | None = None
            self.measurements: dict[str, ZoneMeasurement] = {}
            self.risk: dict[str, float] = {}
            self.contributions: dict[str, dict[str, float]] = {}
            self.ml_flags: dict[str, bool] = {}
            self.ml_scores: dict[str, float] = {}
            self.anomaly_flags: dict[str, bool] = {}
            self.histories: dict[str, deque[ZoneMeasurement]] = {
                zone: deque(maxlen=12) for zone in ZONE_NODES
            }
            self.persistence: dict[str, deque[bool]] = {
                zone: deque(maxlen=8) for zone in ZONE_NODES
            }
            self.node_zones = {
                node: zone for zone, nodes in ZONE_NODES.items() for node in nodes
            }
            self._analyze_current_snapshot()

    @property
    def leak_severity(self) -> float:
        if self._manual_leak_severity is not None:
            return self._manual_leak_severity
        return self.leak_severity_at(self.seconds)

    @property
    def leak_junction(self) -> str | None:
        if self.leak_severity <= 0:
            return None
        return self.active_leak_junction or "B2"

    @staticmethod
    def leak_severity_at(seconds: int) -> float:
        if seconds <= LEAK_START_SECONDS:
            return 0.0
        if seconds < 60:
            return 0.04 * (seconds - 30) / 30
        if seconds < 90:
            return 0.04 + 0.11 * (seconds - 60) / 30
        if seconds < 120:
            return 0.15 + 0.47 * (seconds - 90) / 30
        return 0.62 + 0.38 * (seconds - 120) / 30

    @property
    def phase(self) -> str:
        if self._manual_leak_severity is not None:
            return "ACTIVE LEAK SCENARIO"
        if self.seconds < LEAK_START_SECONDS:
            return "NORMAL NETWORK"
        if self.seconds < 60:
            return "LEAK DEVELOPING"
        if self.seconds < 90:
            return "PRESSURE / FLOW ANOMALY"
        if self.seconds < 120:
            return "ISOLATION FOREST ANALYSIS"
        return "HIGH-RISK REVIEW"

    @property
    def network_status(self) -> str:
        highest = max(self.risk.values(), default=0.0)
        if highest >= 61:
            return "HIGH RISK"
        if highest >= 31:
            return "WARNING"
        if any(self.anomaly_flags.values()):
            return "ANOMALY DETECTED"
        return "NORMAL"

    @property
    def probable_zone(self) -> str | None:
        if max(self.risk.values(), default=0.0) < 31:
            return None
        return max(self.risk, key=self.risk.get)

    @property
    def predicted_leak_zone(self) -> str | None:
        """Early predictive detection of the impending leak zone using hydraulic mass balance gradients.
        
        Even before the threshold for full leak confirmation (risk >= 31%) is crossed,
        this detects the earliest anomalous gradient to identify where a leak is developing.
        """
        if self.probable_zone is not None:
            return self.probable_zone
        if self.seconds > LEAK_START_SECONDS or self._manual_leak_severity is not None:
            # Measure net DMA inflow deviations across all zones
            net_devs = {}
            for zone in ZONE_NODES:
                inp = ZONE_INFLOW_PIPE[zone]
                outp = ZONE_OUTFLOW_PIPE[zone]
                net_flow = self.snapshot.flows_lps[inp] - (self.snapshot.flows_lps[outp] if outp else 0.0)
                base_flow = self.calibration.baselines[zone].flow_lps
                dev = net_flow - base_flow
                if dev > 0.0005:  # Measurable positive mass deviation threshold
                    net_devs[zone] = dev
            if net_devs:
                return max(net_devs, key=net_devs.get)
        return None

    def set_speed(self, speed: int) -> None:
        if speed not in (1, 2, 5):
            raise ValueError("Simulation speed must be 1x, 2x, or 5x.")
        self.speed = speed

    def start(self) -> None:
        if self.seconds < DEMO_SECONDS:
            self.running = True

    def pause(self) -> None:
        self.running = False

    def introduce_leak(self, junction: str | None = None) -> None:
        with self._step_lock:
            if junction is not None:
                if junction not in self.node_zones:
                    raise ValueError(f"Unsupported leak junction {junction!r}.")
                self.active_leak_junction = junction
            leak_target = self.active_leak_junction or "B2"
            next_seconds = max(self.seconds, LEAK_START_SECONDS + 1)
            snapshot = run_hydraulic_profile(
                self.leak_severity_at(next_seconds),
                (1.0,),
                leak_target,
            )[0]
            self.seconds = next_seconds
            self.step_number += 1
            self.snapshot = snapshot
            self._analyze_current_snapshot()
            self.start()

    def trigger_leak(self, junction: str) -> None:
        """Start a clean, full-strength single-junction dashboard scenario."""
        if junction not in self.node_zones:
            raise ValueError(f"Unsupported leak junction {junction!r}.")
        self.reset()
        with self._step_lock:
            self.active_leak_junction = junction
            self._manual_leak_severity = 1.0
            self.seconds = LEAK_START_SECONDS + 1
            self.step_number = 1
            self.snapshot = run_hydraulic_profile(
                self._manual_leak_severity,
                (1.0,),
                junction,
            )[0]
            self._analyze_current_snapshot()
            self.start()

    def tick(self) -> bool:
        with self._step_lock:
            if not self.running:
                return False
            return self._advance_one_step()

    def advance(self) -> bool:
        """Execute one hydraulic step without changing the running/paused state."""
        with self._step_lock:
            if self.seconds >= DEMO_SECONDS:
                return False
            return self._advance_one_step()

    def _advance_one_step(self) -> bool:
        next_seconds = min(DEMO_SECONDS, self.seconds + self.speed)
        leak_severity = (
            self._manual_leak_severity
            if self._manual_leak_severity is not None
            else self.leak_severity_at(next_seconds)
        )
        snapshot = run_hydraulic_profile(
            leak_severity,
            (1.0,),
            self.active_leak_junction or "B2",
        )[0]
        self.seconds = next_seconds
        self.step_number += 1
        self.snapshot = snapshot
        self._analyze_current_snapshot()
        if self.seconds >= DEMO_SECONDS:
            self.running = False
        return True

    def _analyze_current_snapshot(self) -> None:
        for zone in ZONE_NODES:
            measurement = zone_measurement(self.snapshot, zone)
            history = self.histories[zone]
            history.append(measurement)
            feature_row = features_for_history(list(history))
            ml_decision = float(
                self.calibration.models[zone].decision_function(feature_row)[0]
            )
            self.ml_scores[zone] = ml_decision
            ml_strength = float(np.clip(
                max(0.0, -ml_decision) / self.calibration.ml_scales[zone],
                0.0,
                1.0,
            ))
            ml_flag = ml_strength >= 0.5
            baseline = self.calibration.baselines[zone]
            pressure_drop = max(0.0, baseline.pressure_m - measurement.pressure_m)
            flow_deviation = abs(measurement.flow_lps - baseline.flow_lps)
            pressure_strength = min(
                1.0,
                pressure_drop / self.calibration.pressure_limits[zone],
            )
            flow_strength = min(
                1.0,
                flow_deviation / self.calibration.flow_limits[zone],
            )
            abnormal = (
                pressure_strength >= 0.20
                or flow_strength >= 0.20
                or ml_flag
            )
            persistence = self.persistence[zone]
            persistence.append(abnormal)
            current_contributions = {
                "pressure": RISK_WEIGHTS["pressure"] * pressure_strength,
                "flow": RISK_WEIGHTS["flow"] * flow_strength,
                "ml": RISK_WEIGHTS["ml"] * ml_strength,
                "persistence": (
                    RISK_WEIGHTS["persistence"]
                    * (sum(persistence) / len(persistence))
                    if persistence else 0.0
                ),
            }
            self.measurements[zone] = measurement
            self.contributions[zone] = current_contributions
            self.risk[zone] = float(np.clip(
                sum(current_contributions.values()),
                0.0,
                100.0,
            ))
            self.ml_flags[zone] = ml_flag
            self.anomaly_flags[zone] = abnormal

    def priorities(self) -> list[tuple[str, float, str]]:
        ordered = sorted(self.risk.items(), key=lambda item: item[1], reverse=True)
        priorities = []
        for zone, score in ordered:
            if score >= 85:
                label = "CRITICAL"
            elif score >= 61:
                label = "HIGH"
            elif score >= 31:
                label = "MEDIUM"
            else:
                label = "LOW"
            priorities.append((zone, score, label))
        return priorities

    # --- Engineering Diagnostics Helpers ---
    def get_junction_telemetry(self) -> list[dict[str, Any]]:
        telemetry = []
        snapshot = self.snapshot
        baseline_snapshot = self.calibration.baseline_snapshot
        elevations = {
            "A1": 51.0, "A2": 52.0, "A3": 51.5,
            "B1": 52.0, "B2": 53.0, "B3": 52.5,
            "C1": 53.0, "C2": 54.0, "C3": 53.5,
            "D1": 56.0, "D2": 57.0, "D3": 56.5,
        }
        for node in ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "D3"):
            zone = self.node_zones[node]
            p_curr = snapshot.pressures[node]
            p_base = baseline_snapshot.pressures[node]
            dev = p_curr - p_base
            dev_pct = (dev / p_base * 100.0) if p_base else 0.0
            elev = elevations[node]
            status = "NORMAL"
            critical_nodes = (
                (self.active_leak_junction,)
                if self.active_leak_junction is not None
                else ("B2", "B1")
            )
            if self.risk[zone] >= 61 and node in critical_nodes:
                status = "CRITICAL"
            elif self.risk[zone] >= 31:
                status = "WARNING"
            elif self.anomaly_flags[zone]:
                status = "ANOMALY"

            telemetry.append({
                "node": node,
                "zone": zone,
                "elevation_m": elev,
                "pressure_m": p_curr,
                "baseline_m": p_base,
                "dev_m": dev,
                "dev_pct": dev_pct,
                "head_m": p_curr + elev,
                "status": status,
            })
        return telemetry

    def get_pipe_telemetry(self) -> list[dict[str, Any]]:
        pipes_def = (
            ("P_SUPPLY", "RESERVOIR", "A1", 300, 0.25, "A"),
            ("P_A_CORE", "A1", "A2", 550, 0.17, "A"),
            ("P_A_BRANCH", "A1", "A3", 420, 0.12, "A"),
            ("P_A_TO_B", "A2", "B1", 1300, 0.10, "A"),
            ("P_B_CORE", "B1", "B2", 1100, 0.10, "B"),
            ("P_B_BRANCH", "B1", "B3", 420, 0.12, "B"),
            ("P_B_TO_C", "B2", "C1", 1300, 0.10, "B"),
            ("P_C_CORE", "C1", "C2", 1100, 0.10, "C"),
            ("P_C_BRANCH", "C1", "C3", 420, 0.12, "C"),
            ("P_C_TO_D", "C2", "D1", 1300, 0.10, "C"),
            ("P_D_CORE", "D1", "D2", 1100, 0.10, "D"),
            ("P_D_BRANCH", "D1", "D3", 420, 0.12, "D"),
        )
        snapshot = self.snapshot
        baseline_snapshot = self.calibration.baseline_snapshot
        telemetry = []
        for pipe, start, end, length, diameter, zone in pipes_def:
            q_curr = snapshot.flows_lps[pipe]
            signed_q_curr = snapshot.signed_flows_lps[pipe]
            q_base = baseline_snapshot.flows_lps[pipe]
            dev = q_curr - q_base
            dev_pct = (dev / q_base * 100.0) if q_base else 0.0
            area = math.pi * ((diameter / 2.0) ** 2)
            velocity = (q_curr / 1000.0) / area if area > 0 else 0.0
            telemetry.append({
                "pipe": pipe,
                "start": start,
                "end": end,
                "zone": zone,
                "length_m": length,
                "diameter_m": diameter,
                "flow_lps": q_curr,
                "signed_flow_lps": signed_q_curr,
                "direction": (
                    f"{start} → {end}"
                    if signed_q_curr > 0
                    else f"{end} → {start}"
                    if signed_q_curr < 0
                    else "No flow"
                ),
                "baseline_lps": q_base,
                "dev_lps": dev,
                "dev_pct": dev_pct,
                "velocity_m_s": velocity,
            })
        return telemetry

    def get_zone_explanation(self, zone: str) -> str:
        meas = self.measurements[zone]
        base = self.calibration.baselines[zone]
        risk = self.risk[zone]
        contrib = self.contributions[zone]
        ml_flag = self.ml_flags[zone]
        p_diff = meas.pressure_m - base.pressure_m
        q_diff = meas.flow_lps - base.flow_lps

        if risk < 31 and not self.anomaly_flags[zone]:
            return (
                f"Zone {zone} operating within nominal hydraulic tolerances. "
                f"Pressure deviation ({p_diff:+.2f} m), flow deviation ({q_diff:+.2f} L/s) "
                f"are within baseline variance limits."
            )

        reasons = []
        if contrib["pressure"] >= RISK_WEIGHTS["pressure"] * 0.2:
            reasons.append(
                f"significant head drop of {abs(p_diff):.2f} m "
                f"({abs(p_diff)/base.pressure_m*100:.1f}% below baseline)"
            )
        if contrib["flow"] >= RISK_WEIGHTS["flow"] * 0.2:
            reasons.append(f"abnormal core flow surge of {q_diff:+.2f} L/s")
        if ml_flag:
            reasons.append(
                f"Isolation Forest feature anomaly detected "
                f"(score {self.ml_scores.get(zone, 0.0):.4f}, contrib +{contrib['ml']:.1f}/20)"
            )
        if contrib["persistence"] >= 5.0:
            pers_count = sum(self.persistence[zone])
            total_count = len(self.persistence[zone])
            reasons.append(
                f"persistent anomaly in {pers_count}/{total_count} evaluation windows"
            )

        reason_str = "; ".join(reasons) if reasons else "minor hydraulic variance"
        return (
            f"Zone {zone} flagged with {risk:.0f}/100 total risk score. Primary evidence: {reason_str}. "
            f"EPANET PDD simulation confirms emitter-induced hydraulic gradient alteration."
        )

    def get_maintenance_recommendations(self) -> list[dict[str, Any]]:
        priorities = self.priorities()
        recs = []
        for rank, (zone, score, label) in enumerate(priorities, start=1):
            target_junction = (
                self.active_leak_junction
                if self.active_leak_junction is not None
                and self.node_zones[self.active_leak_junction] == zone
                else f"{zone}2"
            )
            if label == "CRITICAL":
                sla = "IMMEDIATE (within 1 hr)"
                action = (
                    f"Deploy emergency repair crew to Zone {zone}. "
                    f"Isolate pipe P_{zone}_CORE and inspect Junction "
                    f"{target_junction} emitter valve."
                )
                loss_rate_lps = max(
                    0.0,
                    self.measurements[zone].flow_lps - self.calibration.baselines[zone].flow_lps,
                )
                loss_l_min = loss_rate_lps * 60.0
            elif label == "HIGH":
                sla = "HIGH PRIORITY (within 4 hrs)"
                action = (
                    f"Inspect Zone {zone} trunk connections. "
                    f"Verify pressure transducer PT-{zone}1 calibration and valve seals."
                )
                loss_l_min = max(
                    0.0,
                    self.measurements[zone].flow_lps - self.calibration.baselines[zone].flow_lps,
                ) * 60.0
            elif label == "MEDIUM":
                sla = "MEDIUM PRIORITY (within 24 hrs)"
                action = f"Perform routine acoustic leak correlation along Zone {zone} pipeline corridor."
                loss_l_min = 0.0
            else:
                sla = "ROUTINE (Scheduled)"
                action = f"Zone {zone} operating normally. Standard preventive maintenance schedule."
                loss_l_min = 0.0

            recs.append({
                "rank": rank,
                "zone": zone,
                "score": score,
                "label": label,
                "sla": sla,
                "action": action,
                "est_loss_l_min": loss_l_min,
                "work_order_id": f"WO-2026-Z{zone}-{int(score)}",
            })
        return recs

    @property
    def loss_estimate_l_min(self) -> float:
        """Estimated real-time water loss in liters per minute for the critical zone."""
        zone = self.probable_zone or "B"
        base_flow = self.calibration.baselines[zone].flow_lps
        curr_flow = self.measurements[zone].flow_lps
        return max(0.0, curr_flow - base_flow) * 60.0


# -----------------------------------------------------------------------------
# ADVANCED SCADA DEKSTOP APPLICATION (TKINTER + MATPLOTLIB)
# -----------------------------------------------------------------------------
class SmartLeakApp:
    """Professional SCADA water-utility intelligence desktop platform."""

    BG = "#07111B"
    PANEL = "#0E1D2D"
    PANEL_ALT = "#16283B"
    BORDER = "#263A49"
    TEXT = "#E8F1F5"
    MUTED = "#8FA5B1"
    CYAN = "#40C8D7"
    BLUE = "#5595F5"
    GREEN = "#39C98A"
    AMBER = "#F3BB4F"
    RED = "#F16C79"
    TRACK = "#243846"

    STATUS_COLORS = {
        "NORMAL": GREEN,
        "WARNING": AMBER,
        "ANOMALY DETECTED": CYAN,
        "HIGH RISK": RED,
    }

    def __init__(self, engine: SmartLeakSimulation) -> None:
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        self.engine = engine
        self.calibration = engine.calibration
        self.graph = self._build_graph()
        self.positions = {
            "RESERVOIR": (-0.9, 0.0),
            "A1": (0.1, 0.0), "A2": (1.1, 0.0), "A3": (0.6, -0.72),
            "B1": (2.1, 0.0), "B2": (3.1, 0.0), "B3": (2.6, -0.72),
            "C1": (4.1, 0.0), "C2": (5.1, 0.0), "C3": (4.6, -0.72),
            "D1": (6.1, 0.0), "D2": (7.1, 0.0), "D3": (6.6, -0.72),
        }
        self.node_zones = {
            node: zone for zone, nodes in ZONE_NODES.items() for node in nodes
        }
        self.selected_node: str = "B2"
        self.active_view_name: str = "Overview"

        self.time_history: list[int] = []
        self.pressure_history: list[float] = []
        self.flow_history: list[float] = []
        self.risk_history: list[float] = []
        self.full_history: list[dict[str, Any]] = []
        self.raw_event_log: list[tuple[str, int, str, str]] = []

        self._logged_events: set[str] = set()

        # Animation states & settings
        self.anim_enabled: bool = True
        self.flow_particle_phase: float = 0.0
        self.leak_pulse_phase: float = 0.0
        self.current_gauge_angle: float = math.pi
        self.current_zone_bars: dict[str, float] = {z: 0.0 for z in ("A", "B", "C", "D")}
        self.current_contrib_bars: list[float] = [0.0, 0.0, 0.0, 0.0]
        self._last_sec_tick: float = 0.0
        self._last_wall_time: float = 0.0
        self._after_id = None
        self._anim_timer_id = None

        # Network artist caches for high-performance in-place updates
        self._ov_net_artists: dict[str, Any] = {}
        self._twin_artists: dict[str, Any] = {}
        self._ov_diag_artists: dict[str, Any] = {}

        self.root = tk.Tk()
        self.root.title("SmartLeak  |  Industrial SCADA & Water Network Digital Twin")
        self.root.configure(bg=self.BG)
        self.root.geometry("1480x920")
        self.root.minsize(1280, 800)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

        self._build_header()

        self.body_frame = tk.Frame(self.root, bg=self.BG)
        self.body_frame.pack(fill=tk.BOTH, expand=True)

        self._build_sidebar(self.body_frame)

        self.main_container = tk.Frame(self.body_frame, bg=self.BG)
        self.main_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Build functional views
        self.views: dict[str, tk.Frame] = {}
        self._build_view_overview()
        self._build_view_digital_twin()
        self._build_view_telemetry()
        self._build_view_ai_diagnostics()
        self._build_view_maintenance()
        self._build_view_timeline()
        self._build_view_reports()

        self._bind_keyboard_shortcuts()
        self._append_trend()

        self._log_event("Industrial SCADA monitoring platform initialized.", "NORMAL")
        self._log_event("EPANET hydraulic solver & Isolation Forest ready.", "INFO")

        self.switch_view("Overview")
        self.refresh()
        self._after_id = None
        self._start_animation_loop()

    @staticmethod
    def _build_graph() -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_node("RESERVOIR")
        pipes = (
            ("P_SUPPLY", "RESERVOIR", "A1", "A"),
            ("P_A_CORE", "A1", "A2", "A"),
            ("P_A_BRANCH", "A1", "A3", "A"),
            ("P_A_TO_B", "A2", "B1", "A"),
            ("P_B_CORE", "B1", "B2", "B"),
            ("P_B_BRANCH", "B1", "B3", "B"),
            ("P_B_TO_C", "B2", "C1", "B"),
            ("P_C_CORE", "C1", "C2", "C"),
            ("P_C_BRANCH", "C1", "C3", "C"),
            ("P_C_TO_D", "C2", "D1", "C"),
            ("P_D_CORE", "D1", "D2", "D"),
            ("P_D_BRANCH", "D1", "D3", "D"),
        )
        for pipe, start, end, zone in pipes:
            graph.add_edge(start, end, pipe=pipe, zone=zone)
        return graph

    def _build_header(self) -> None:
        header = tk.Frame(
            self.root, bg="#0A1624", height=50,
            highlightbackground=self.BORDER, highlightthickness=1
        )
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        title_frame = tk.Frame(header, bg="#0A1624")
        title_frame.pack(side=tk.LEFT, padx=14, pady=4)

        lbl_title = tk.Label(
            title_frame, text="SMARTLEAK SCADA", bg="#0A1624", fg=self.TEXT,
            font=("Segoe UI", 13, "bold")
        )
        lbl_title.pack(side=tk.LEFT, padx=(0, 8))

        lbl_sub = tk.Label(
            title_frame, text="Water Network Intelligence & Digital Twin", bg="#0A1624", fg=self.MUTED,
            font=("Segoe UI", 9)
        )
        lbl_sub.pack(side=tk.LEFT)

        status_center = tk.Frame(header, bg="#0A1624")
        status_center.pack(side=tk.LEFT, expand=True)

        self.status_canvas = tk.Canvas(
            status_center, width=12, height=12, bg="#0A1624", highlightthickness=0
        )
        self.status_canvas.pack(side=tk.LEFT, padx=(0, 6))
        self.status_led = self.status_canvas.create_oval(1, 1, 11, 11, fill=self.GREEN, outline="")

        self.status_header_lbl = tk.Label(
            status_center, text="SYSTEM NORMAL", bg="#0A1624", fg=self.GREEN,
            font=("Segoe UI", 9, "bold")
        )
        self.status_header_lbl.pack(side=tk.LEFT)

        right_frame = tk.Frame(header, bg="#0A1624")
        right_frame.pack(side=tk.RIGHT, padx=14)

        self.btn_anim_toggle = tk.Button(
            right_frame, text="✨ FX: ON", bg="#193347", fg=self.CYAN,
            activebackground=self.BORDER, activeforeground=self.TEXT,
            font=("Segoe UI", 8, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._toggle_animations
        )
        self.btn_anim_toggle.pack(side=tk.RIGHT, padx=4, ipady=2, ipadx=6)

        self.btn_header_export_csv = tk.Button(
            right_frame, text="📥 CSV", bg=self.PANEL_ALT, fg=self.CYAN,
            activebackground=self.BORDER, activeforeground=self.TEXT,
            font=("Segoe UI", 8, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._quick_export_csv
        )
        self.btn_header_export_csv.pack(side=tk.RIGHT, padx=4, ipady=2, ipadx=6)

        self.btn_header_export_pdf = tk.Button(
            right_frame, text="📄 PDF Report", bg=self.PANEL_ALT, fg=self.AMBER,
            activebackground=self.BORDER, activeforeground=self.TEXT,
            font=("Segoe UI", 8, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._quick_export_pdf
        )
        self.btn_header_export_pdf.pack(side=tk.RIGHT, padx=4, ipady=2, ipadx=6)

        self.time_lbl = tk.Label(
            right_frame, text="T+000s / 150s", bg="#0A1624", fg=self.CYAN,
            font=("Consolas", 10, "bold")
        )
        self.time_lbl.pack(side=tk.RIGHT, padx=(12, 12))

    def _build_sidebar(self, parent_frame: tk.Frame) -> None:
        self.sidebar = tk.Frame(
            parent_frame, bg=self.PANEL,
            highlightbackground=self.BORDER, highlightthickness=1,
            width=195,
        )
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 2), pady=(4, 6))
        self.sidebar.pack_propagate(False)

        # View Navigation Buttons
        view_lbl = tk.Label(
            self.sidebar, text="NAVIGATION VIEWS", bg=self.PANEL, fg=self.MUTED,
            font=("Segoe UI", 8, "bold"), anchor="w"
        )
        view_lbl.pack(fill=tk.X, padx=12, pady=(10, 4))

        self.view_buttons: dict[str, tk.Button] = {}
        view_items = (
            ("Overview", "📊  Operations Overview"),
            ("Digital Twin", "🌐  Network Digital Twin"),
            ("Telemetry", "📈  Live Telemetry"),
            ("AI Diagnostics", "🤖  AI Diagnostics"),
            ("Maintenance", "🔧  Maintenance & Assets"),
            ("Timeline", "📋  Event Timeline"),
            ("Reports", "📄  Reports & Export"),
        )
        for key, label in view_items:
            btn = tk.Button(
                self.sidebar, text=label,
                bg=self.PANEL_ALT if key == self.active_view_name else self.PANEL,
                fg=self.CYAN if key == self.active_view_name else self.TEXT,
                activebackground=self.PANEL_ALT, activeforeground=self.CYAN,
                font=("Segoe UI", 9, "bold"), bd=0, relief=tk.FLAT, anchor="w", cursor="hand2",
                command=lambda k=key: self.switch_view(k)
            )
            btn.pack(fill=tk.X, padx=8, pady=2, ipady=5, ipadx=8)
            self.view_buttons[key] = btn

        sep = tk.Frame(self.sidebar, bg=self.BORDER, height=1)
        sep.pack(fill=tk.X, padx=10, pady=8)

        # Controls Section
        ctrl_lbl = tk.Label(
            self.sidebar, text="SIMULATION CONTROLS", bg=self.PANEL, fg=self.MUTED,
            font=("Segoe UI", 8, "bold"), anchor="w"
        )
        ctrl_lbl.pack(fill=tk.X, padx=12, pady=(2, 4))

        def _bind_hover(btn: tk.Button, normal_bg: str, hover_bg: str) -> None:
            btn.bind("<Enter>", lambda e, b=btn, h=hover_bg: b.config(bg=h))
            btn.bind("<Leave>", lambda e, b=btn, n=normal_bg: b.config(bg=n))

        self.btn_start = tk.Button(
            self.sidebar, text="▶  START SIMULATION",
            bg=self.GREEN, fg=self.BG, activebackground="#2fb579", activeforeground=self.BG,
            font=("Segoe UI", 9, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._start
        )
        self.btn_start.pack(fill=tk.X, padx=10, pady=2, ipady=4)
        _bind_hover(self.btn_start, self.GREEN, "#2fb579")

        self.btn_pause = tk.Button(
            self.sidebar, text="⏸  PAUSE",
            bg=self.PANEL_ALT, fg=self.TEXT, activebackground="#233a52", activeforeground=self.TEXT,
            font=("Segoe UI", 9, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._pause
        )
        self.btn_pause.pack(fill=tk.X, padx=10, pady=2, ipady=4)
        _bind_hover(self.btn_pause, self.PANEL_ALT, "#233a52")

        self.btn_reset = tk.Button(
            self.sidebar, text="↺  RESET",
            bg=self.PANEL_ALT, fg=self.TEXT, activebackground="#233a52", activeforeground=self.TEXT,
            font=("Segoe UI", 9, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._reset
        )
        self.btn_reset.pack(fill=tk.X, padx=10, pady=2, ipady=4)
        _bind_hover(self.btn_reset, self.PANEL_ALT, "#233a52")

        self.btn_leak = tk.Button(
            self.sidebar, text="⚡  INTRODUCE LEAK",
            bg="#6D5225", fg="#FFE8A3", activebackground="#87662e", activeforeground="#FFF",
            font=("Segoe UI", 9, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._introduce_leak
        )
        self.btn_leak.pack(fill=tk.X, padx=10, pady=(2, 6), ipady=4)
        _bind_hover(self.btn_leak, "#6D5225", "#87662e")

        speed_lbl = tk.Label(
            self.sidebar, text="SIMULATION SPEED", bg=self.PANEL, fg=self.MUTED,
            font=("Segoe UI", 8, "bold"), anchor="w"
        )
        speed_lbl.pack(fill=tk.X, padx=12, pady=(4, 2))

        speed_frame = tk.Frame(self.sidebar, bg=self.PANEL)
        speed_frame.pack(fill=tk.X, padx=10, pady=2)

        self.speed_btns = {}
        for sp in (1, 2, 5):
            btn = tk.Button(
                speed_frame, text=f"{sp}x",
                bg="#23566A" if sp == self.engine.speed else self.PANEL_ALT,
                fg=self.CYAN if sp == self.engine.speed else self.TEXT,
                activebackground="#2a6982", activeforeground=self.CYAN,
                font=("Segoe UI", 9, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
                command=lambda s=sp: self._set_speed(s)
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, ipady=3)
            self.speed_btns[sp] = btn

        self.sidebar_status = tk.Label(
            self.sidebar, text="PAUSED", bg=self.PANEL_ALT, fg=self.CYAN,
            font=("Segoe UI", 8, "bold"), anchor="center", wraplength=170
        )
        self.sidebar_status.pack(fill=tk.X, padx=10, pady=(8, 4), ipady=4)

        self.btn_sidebar_anim = tk.Button(
            self.sidebar, text="✨  TOGGLE ANIMATIONS (ON)",
            bg="#162D3E", fg=self.CYAN, activebackground="#1e3b52", activeforeground=self.CYAN,
            font=("Segoe UI", 8, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._toggle_animations
        )
        self.btn_sidebar_anim.pack(fill=tk.X, padx=10, pady=(2, 6), ipady=3)

        # Shortcuts Box
        sc_frame = tk.Frame(self.sidebar, bg="#0D1E30", highlightbackground="#33526E", highlightthickness=1)
        sc_frame.pack(fill=tk.X, padx=10, pady=(6, 8), side=tk.BOTTOM)
        sc_title = tk.Label(sc_frame, text="KEYBOARD SHORTCUTS", bg="#0D1E30", fg=self.CYAN, font=("Segoe UI", 8, "bold"))
        sc_title.pack(anchor="w", padx=10, pady=(6, 3))
        sc_text = tk.Label(
            sc_frame,
            text="Space : Start / Pause\nR : Reset  |  L : Inject Leak\n1, 2, 5 : Speed  |  A : FX",
            bg="#0D1E30", fg="#FFFFFF", font=("Consolas", 9, "bold"), justify=tk.LEFT, anchor="w",
            padx=2, pady=2
        )
        sc_text.pack(anchor="w", padx=8, pady=(0, 6))

    def switch_view(self, name: str) -> None:
        self.active_view_name = name
        for view_name, frame in self.views.items():
            if view_name == name:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()

        for view_key, btn in self.view_buttons.items():
            if view_key == name:
                btn.config(bg=self.PANEL_ALT, fg=self.CYAN)
            else:
                btn.config(bg=self.PANEL, fg=self.TEXT)

        self.refresh()

    # -------------------------------------------------------------------------
    # VIEW 1: OPERATIONS OVERVIEW
    # -------------------------------------------------------------------------
    def _build_view_overview(self) -> None:
        view = tk.Frame(self.main_container, bg=self.BG)
        self.views["Overview"] = view

        # ---------------------------------------------------------------------
        # TOP KPI BANNER (5 High-Density Industrial SCADA Metric Cards)
        # ---------------------------------------------------------------------
        self.kpi_cards_frame = tk.Frame(view, bg=self.BG)
        self.kpi_cards_frame.pack(side=tk.TOP, fill=tk.X, pady=(2, 6))

        self.kpi_widgets = []
        for i in range(5):
            card = tk.Frame(
                self.kpi_cards_frame, bg=self.PANEL,
                highlightbackground=self.BORDER, highlightthickness=1, bd=0,
            )
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

            title_lbl = tk.Label(card, text="", bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 8, "bold"), anchor="w")
            title_lbl.pack(fill=tk.X, padx=10, pady=(6, 1))

            val_lbl = tk.Label(card, text="", bg=self.PANEL, fg=self.CYAN, font=("Segoe UI", 14, "bold"), anchor="w")
            val_lbl.pack(fill=tk.X, padx=10, pady=(0, 1))

            det_lbl = tk.Label(card, text="", bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 8), anchor="w")
            det_lbl.pack(fill=tk.X, padx=10, pady=(0, 6))

            self.kpi_widgets.append((title_lbl, val_lbl, det_lbl, card))

        # ---------------------------------------------------------------------
        # MAIN SCADA COCKPIT: Center Column (Visual Centerpiece) & Right Column (Analytics)
        # ---------------------------------------------------------------------
        cockpit = tk.Frame(view, bg=self.BG)
        cockpit.pack(fill=tk.BOTH, expand=True)

        # Center Column: Dedicated Network Visualizer (Top) + Multi-Waveform Analytics (Bottom)
        center_col = tk.Frame(cockpit, bg=self.BG)
        center_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Upper Center: Primary Network Digital Twin Visualization
        net_container = tk.Frame(
            center_col, bg=self.PANEL,
            highlightbackground=self.BORDER, highlightthickness=1
        )
        net_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 4))

        net_header = tk.Frame(net_container, bg=self.PANEL_ALT, height=28)
        net_header.pack(fill=tk.X)
        net_header.pack_propagate(False)

        tk.Label(
            net_header, text="WATER DISTRIBUTION NETWORK DIGITAL TWIN  |  LIVE HYDRAULIC TOPOLOGY",
            bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=10)

        self.net_header_status = tk.Label(
            net_header, text="EPANET 2.2 ENGINE  ·  PRESSURE-DEPENDENT DEMAND",
            bg=self.PANEL_ALT, fg=self.CYAN, font=("Segoe UI", 8, "bold")
        )
        self.net_header_status.pack(side=tk.RIGHT, padx=10)

        self.fig_ov_network = plt.figure(figsize=(10.0, 5.0), facecolor=self.BG)
        self.ax_ov_network = self.fig_ov_network.add_axes([0.015, 0.02, 0.97, 0.96])
        self._style_panel(self.ax_ov_network)

        self.canvas_ov_network = FigureCanvasTkAgg(self.fig_ov_network, master=net_container)
        self.canvas_ov_network.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Lower Center: 3 Synchronized Telemetry Waveform Charts
        charts_container = tk.Frame(
            center_col, bg=self.PANEL,
            highlightbackground=self.BORDER, highlightthickness=1, height=240
        )
        charts_container.pack(side=tk.BOTTOM, fill=tk.X)
        charts_container.pack_propagate(False)

        charts_header = tk.Frame(charts_container, bg=self.PANEL_ALT, height=26)
        charts_header.pack(fill=tk.X)
        charts_header.pack_propagate(False)

        tk.Label(
            charts_header, text="SYNCHRONIZED SENSOR TELEMETRY & TRANSIENT WAVEFORMS",
            bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=10)

        tk.Label(
            charts_header, text="TARGET ZONE: ZONE B  ·  150-SECOND BUFFER",
            bg=self.PANEL_ALT, fg=self.MUTED, font=("Segoe UI", 8)
        ).pack(side=tk.RIGHT, padx=10)

        self.fig_ov_charts = plt.figure(figsize=(10.0, 2.6), facecolor=self.BG)
        chart_w = 0.28
        chart_gap = 0.045
        self.ax_ov_pressure = self.fig_ov_charts.add_axes([0.05, 0.18, chart_w, 0.68])
        self.ax_ov_flow = self.fig_ov_charts.add_axes([0.05 + chart_w + chart_gap, 0.18, chart_w, 0.68])
        self.ax_ov_risk = self.fig_ov_charts.add_axes([0.05 + 2 * (chart_w + chart_gap), 0.18, chart_w, 0.68])

        self._style_chart(self.ax_ov_pressure, "ZONE B PRESSURE (m H2O)", "Pressure (m)", self.CYAN)
        self._style_chart(self.ax_ov_flow, "ZONE B PIPE FLOW (L/s)", "Flow (L/s)", self.BLUE)
        self._style_chart(self.ax_ov_risk, "LEAK THREAT PROBABILITY (%)", "Risk Score", self.RED)

        self.ov_pressure_line, = self.ax_ov_pressure.plot([], [], color=self.CYAN, linewidth=2.2, label="Observed")
        self.ov_flow_line, = self.ax_ov_flow.plot([], [], color=self.BLUE, linewidth=2.2, label="Observed")
        self.ov_risk_line, = self.ax_ov_risk.plot([], [], color=self.RED, linewidth=2.2, label="Calculated")

        self.ax_ov_pressure.axhline(self.engine.calibration.baselines["B"].pressure_m, color="#38BDF8", linestyle="--", linewidth=1.2, alpha=0.65)
        self.ax_ov_flow.axhline(self.engine.calibration.baselines["B"].flow_lps, color="#60A5FA", linestyle="--", linewidth=1.2, alpha=0.65)
        self.ax_ov_risk.axhline(61, color=self.RED, linestyle="--", linewidth=1.2, alpha=0.8)
        self.ax_ov_risk.set_ylim(0, 100)

        self.canvas_ov_charts = FigureCanvasTkAgg(self.fig_ov_charts, master=charts_container)
        self.canvas_ov_charts.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---------------------------------------------------------------------
        # Right Column: High-Density SCADA Diagnostics & Alarm Console (380px)
        # ---------------------------------------------------------------------
        right_col = tk.Frame(
            cockpit, bg=self.PANEL,
            highlightbackground=self.BORDER, highlightthickness=1,
            width=320,
        )
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(4, 0))
        right_col.pack_propagate(False)

        diag_header = tk.Frame(right_col, bg=self.PANEL_ALT, height=28)
        diag_header.pack(fill=tk.X)
        diag_header.pack_propagate(False)

        tk.Label(
            diag_header, text="SCADA DIAGNOSTICS & THREAT CONSOLE",
            bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=10)

        # Matplotlib Analytics Console (Arc Gauge + Zone Threat Bars + Factor Decomposition)
        self.fig_ov_diag = plt.figure(figsize=(3.2, 3.8), facecolor=self.PANEL)
        self.ax_ov_gauge = self.fig_ov_diag.add_axes([0.06, 0.58, 0.88, 0.38])
        self.ax_ov_zones = self.fig_ov_diag.add_axes([0.18, 0.31, 0.76, 0.21])
        self.ax_ov_contrib = self.fig_ov_diag.add_axes([0.18, 0.05, 0.76, 0.21])

        self.canvas_ov_diag = FigureCanvasTkAgg(self.fig_ov_diag, master=right_col)
        self.canvas_ov_diag.get_tk_widget().pack(fill=tk.X, padx=4, pady=2)

        # Lower Dispatch & Recommendation Card
        self.alert_card_frame = tk.Frame(
            right_col, bg="#091420",
            highlightbackground=self.BORDER, highlightthickness=1
        )
        self.alert_card_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 6))

        self.alert_banner_lbl = tk.Label(
            self.alert_card_frame, text="SYSTEM STATUS: NOMINAL",
            bg="#0D2E26", fg=self.GREEN, font=("Segoe UI", 9, "bold"),
            anchor="center", pady=5
        )
        self.alert_banner_lbl.pack(fill=tk.X)

        det_box = tk.Frame(self.alert_card_frame, bg="#091420")
        det_box.pack(fill=tk.X, padx=8, pady=4)

        self.alert_zone_lbl = tk.Label(
            det_box, text="PROBABLE LOCATION: NONE", bg="#091420", fg=self.TEXT,
            font=("Segoe UI", 8, "bold"), anchor="w"
        )
        self.alert_zone_lbl.pack(fill=tk.X, pady=1)

        self.alert_action_lbl = tk.Label(
            det_box, text="ACTION: Normal continuous scanning", bg="#091420", fg=self.MUTED,
            font=("Segoe UI", 8), anchor="w", wraplength=290, justify=tk.LEFT
        )
        self.alert_action_lbl.pack(fill=tk.X, pady=1)

        self.alert_priority_lbl = tk.Label(
            det_box, text="DISPATCH PRIORITY: Nominal", bg="#091420", fg=self.CYAN,
            font=("Segoe UI", 8, "bold"), anchor="w", wraplength=290, justify=tk.LEFT
        )
        self.alert_priority_lbl.pack(fill=tk.X, pady=1)

        # Live Event Log (Compact Ticker)
        ticker_head = tk.Frame(self.alert_card_frame, bg=self.PANEL_ALT, height=20)
        ticker_head.pack(fill=tk.X, pady=(4, 0))
        tk.Label(
            ticker_head, text="LIVE SCADA EVENT FEED",
            bg=self.PANEL_ALT, fg=self.MUTED, font=("Segoe UI", 7, "bold")
        ).pack(side=tk.LEFT, padx=6)

        self.alarm_text = tk.Text(
            self.alert_card_frame, bg="#060E17", fg=self.TEXT,
            font=("Consolas", 7), bd=0, highlightthickness=0, wrap=tk.WORD, height=4
        )
        self.alarm_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.alarm_text.tag_config("INFO", foreground=self.CYAN)
        self.alarm_text.tag_config("ANOMALY", foreground=self.AMBER)
        self.alarm_text.tag_config("WARNING", foreground=self.AMBER)
        self.alarm_text.tag_config("HIGH_RISK", foreground=self.RED)
        self.alarm_text.tag_config("NORMAL", foreground=self.GREEN)
        self.alarm_text.tag_config("MUTED", foreground=self.MUTED)

    # -------------------------------------------------------------------------
    # VIEW 2: NETWORK DIGITAL TWIN
    # -------------------------------------------------------------------------
    def _build_view_digital_twin(self) -> None:
        view = tk.Frame(self.main_container, bg=self.BG)
        self.views["Digital Twin"] = view

        # Top Bar
        top_bar = tk.Frame(view, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1, height=36)
        top_bar.pack(fill=tk.X, pady=(0, 4))
        top_bar.pack_propagate(False)

        title = tk.Label(top_bar, text="NETWORK DIGITAL TWIN INSPECTOR", bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 10, "bold"))
        title.pack(side=tk.LEFT, padx=10)

        hint = tk.Label(top_bar, text="Click any junction on the map to inspect telemetry & connected pipes", bg=self.PANEL, fg=self.CYAN, font=("Segoe UI", 8, "italic"))
        hint.pack(side=tk.LEFT, padx=15)

        # Content Split: Map (Left) & Inspector (Right)
        dt_split = tk.Frame(view, bg=self.BG)
        dt_split.pack(fill=tk.BOTH, expand=True)

        map_frame = tk.Frame(dt_split, bg=self.BG)
        map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_twin = plt.figure(figsize=(9.0, 7.0), facecolor=self.BG)
        self.ax_twin_map = self.fig_twin.add_axes([0.02, 0.04, 0.96, 0.92])
        self._style_panel(self.ax_twin_map)

        self.canvas_twin = FigureCanvasTkAgg(self.fig_twin, master=map_frame)
        self.canvas_twin.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_twin.mpl_connect("button_press_event", self._on_twin_click)

        # Inspector Panel (Right)
        self.dt_inspector = tk.Frame(
            dt_split, bg=self.PANEL,
            highlightbackground=self.BORDER, highlightthickness=1,
            width=320,
        )
        self.dt_inspector.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(4, 0))
        self.dt_inspector.pack_propagate(False)

        insp_head = tk.Frame(self.dt_inspector, bg=self.PANEL_ALT, height=32)
        insp_head.pack(fill=tk.X)
        insp_head.pack_propagate(False)

        insp_title = tk.Label(insp_head, text="JUNCTION INSPECTOR", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        insp_title.pack(side=tk.LEFT, padx=10, pady=4)

        self.insp_content = tk.Frame(self.dt_inspector, bg=self.PANEL)
        self.insp_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _on_twin_click(self, event) -> None:
        if event.inaxes != self.ax_twin_map or event.xdata is None or event.ydata is None:
            return
        cx, cy = event.xdata, event.ydata
        best_node = None
        best_dist = float("inf")
        for node, (nx_x, nx_y) in self.positions.items():
            d = math.hypot(cx - nx_x, cy - nx_y)
            if d < best_dist:
                best_dist = d
                best_node = node
        if best_node and best_dist < 0.6:
            self.selected_node = best_node
            self._draw_digital_twin_map()
            self._update_twin_inspector()

    # -------------------------------------------------------------------------
    # VIEW 3: LIVE TELEMETRY
    # -------------------------------------------------------------------------
    def _build_view_telemetry(self) -> None:
        view = tk.Frame(self.main_container, bg=self.BG)
        self.views["Telemetry"] = view

        # Top Section: Two Tables Side-by-Side
        tables_frame = tk.Frame(view, bg=self.BG)
        tables_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        # Junctions Table
        junc_box = tk.Frame(tables_frame, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        junc_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))

        junc_lbl = tk.Label(junc_box, text="JUNCTION TELEMETRY (12 NODES)", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        junc_lbl.pack(fill=tk.X, ipady=4)

        self.junc_tree = ttk.Treeview(
            junc_box, columns=("node", "zone", "elev", "base_p", "curr_p", "dev_p", "status"),
            show="headings", height=8
        )
        self.junc_tree.heading("node", text="Junction")
        self.junc_tree.heading("zone", text="Zone")
        self.junc_tree.heading("elev", text="Elev (m)")
        self.junc_tree.heading("base_p", text="Baseline (m)")
        self.junc_tree.heading("curr_p", text="Pressure (m)")
        self.junc_tree.heading("dev_p", text="Dev (m)")
        self.junc_tree.heading("status", text="Status")

        for col, w in (("node", 70), ("zone", 50), ("elev", 70), ("base_p", 85), ("curr_p", 90), ("dev_p", 75), ("status", 90)):
            self.junc_tree.column(col, width=w, anchor="center")

        self.junc_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Pipes Table
        pipe_box = tk.Frame(tables_frame, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        pipe_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))

        pipe_lbl = tk.Label(pipe_box, text="PIPE HYDRAULICS (12 PIPES)", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        pipe_lbl.pack(fill=tk.X, ipady=4)

        self.pipe_tree = ttk.Treeview(
            pipe_box, columns=("pipe", "from_to", "zone", "base_q", "curr_q", "dev_q", "vel"),
            show="headings", height=8
        )
        self.pipe_tree.heading("pipe", text="Pipe ID")
        self.pipe_tree.heading("from_to", text="From-To")
        self.pipe_tree.heading("zone", text="Zone")
        self.pipe_tree.heading("base_q", text="Baseline (L/s)")
        self.pipe_tree.heading("curr_q", text="Flow (L/s)")
        self.pipe_tree.heading("dev_q", text="Dev (L/s)")
        self.pipe_tree.heading("vel", text="Vel (m/s)")

        for col, w in (("pipe", 90), ("from_to", 80), ("zone", 50), ("base_q", 85), ("curr_q", 85), ("dev_q", 80), ("vel", 75)):
            self.pipe_tree.column(col, width=w, anchor="center")

        self.pipe_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Bottom Section: 4-Zone Waveforms
        wave_frame = tk.Frame(view, bg=self.BG, height=220)
        wave_frame.pack(fill=tk.BOTH, expand=True)

        self.fig_telem = plt.figure(figsize=(9.5, 2.8), facecolor=self.BG)
        self.ax_tel_a = self.fig_telem.add_subplot(141)
        self.ax_tel_b = self.fig_telem.add_subplot(142)
        self.ax_tel_c = self.fig_telem.add_subplot(143)
        self.ax_tel_d = self.fig_telem.add_subplot(144)

        self.canvas_telem = FigureCanvasTkAgg(self.fig_telem, master=wave_frame)
        self.canvas_telem.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------------------
    # VIEW 4: AI DIAGNOSTICS
    # -------------------------------------------------------------------------
    def _build_view_ai_diagnostics(self) -> None:
        view = tk.Frame(self.main_container, bg=self.BG)
        self.views["AI Diagnostics"] = view

        # Split Top (ML Params & Persistence) and Bottom (Risk Weights & Explanation)
        top_split = tk.Frame(view, bg=self.BG)
        top_split.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        # ML Model Card
        ml_card = tk.Frame(top_split, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        ml_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))

        ml_title = tk.Label(ml_card, text="ISOLATION FOREST MACHINE LEARNING ENGINE", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        ml_title.pack(fill=tk.X, ipady=4)

        ml_info = (
            "Model Architecture: Scikit-Learn IsolationForest (n_estimators=200, contamination=0.14)\n"
            "Evaluated Feature Vector (6 Dimensions per zone):\n"
            "  - Live Junction Pressure (m H2O)\n"
            "  - Live Core Pipe Flow Rate (L/s)\n"
            "  - Pressure Change rate (dP/dt)\n"
            "  - Flow Change rate (dQ/dt)\n"
            "  - 4-Step Rolling Mean Pressure\n"
            "  - 4-Step Rolling Mean Flow Rate"
        )
        lbl_ml_info = tk.Label(ml_card, text=ml_info, bg=self.PANEL, fg=self.TEXT, font=("Consolas", 8), justify=tk.LEFT, anchor="w")
        lbl_ml_info.pack(fill=tk.X, padx=10, pady=6)

        # Decision Scores
        self.ml_scores_lbl = tk.Label(ml_card, text="", bg="#091420", fg=self.CYAN, font=("Consolas", 8), justify=tk.LEFT, anchor="w")
        self.ml_scores_lbl.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        # Persistence Matrix Card
        pers_card = tk.Frame(top_split, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        pers_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))

        pers_title = tk.Label(pers_card, text="8-TIMESTEP ANOMALY PERSISTENCE EVIDENCE", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        pers_title.pack(fill=tk.X, ipady=4)

        self.pers_matrix_frame = tk.Frame(pers_card, bg=self.PANEL)
        self.pers_matrix_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bottom Split (Radar/Bar Chart & Natural Language AI Explanation)
        bot_split = tk.Frame(view, bg=self.BG)
        bot_split.pack(fill=tk.BOTH, expand=True)

        chart_card = tk.Frame(bot_split, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        chart_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))

        chart_title = tk.Label(chart_card, text="4-FACTOR RISK WEIGHT CONTRIBUTIONS", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        chart_title.pack(fill=tk.X, ipady=4)

        self.fig_ai = plt.figure(figsize=(4.5, 3.2), facecolor=self.PANEL)
        self.ax_ai_bars = self.fig_ai.add_axes([0.15, 0.15, 0.80, 0.75])
        self._style_panel(self.ax_ai_bars)

        self.canvas_ai = FigureCanvasTkAgg(self.fig_ai, master=chart_card)
        self.canvas_ai.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        exp_card = tk.Frame(bot_split, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        exp_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))

        exp_title = tk.Label(exp_card, text="ENGINEERING DIAGNOSTIC EXPLANATION TRACE", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        exp_title.pack(fill=tk.X, ipady=4)

        self.ai_exp_text = tk.Text(
            exp_card, bg="#091420", fg=self.TEXT,
            font=("Consolas", 9), bd=0, highlightthickness=0, wrap=tk.WORD, state=tk.DISABLED
        )
        self.ai_exp_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # -------------------------------------------------------------------------
    # VIEW 5: MAINTENANCE & ASSETS
    # -------------------------------------------------------------------------
    def _build_view_maintenance(self) -> None:
        view = tk.Frame(self.main_container, bg=self.BG)
        self.views["Maintenance"] = view

        # Top: Work Orders Table
        wo_box = tk.Frame(view, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        wo_box.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        wo_title = tk.Label(wo_box, text="RANKED SCADA WORK ORDERS & MAINTENANCE RECOMMENDATIONS", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        wo_title.pack(fill=tk.X, ipady=4)

        self.wo_tree = ttk.Treeview(
            wo_box, columns=("rank", "wo_id", "zone", "risk", "priority", "sla", "action", "loss"),
            show="headings", height=6
        )
        self.wo_tree.heading("rank", text="Rank")
        self.wo_tree.heading("wo_id", text="Work Order ID")
        self.wo_tree.heading("zone", text="Zone")
        self.wo_tree.heading("risk", text="Risk Score")
        self.wo_tree.heading("priority", text="Priority")
        self.wo_tree.heading("sla", text="SLA Response Time")
        self.wo_tree.heading("action", text="Action Plan & Pipeline Procedure")
        self.wo_tree.heading("loss", text="Est Loss (L/min)")

        for col, w in (("rank", 50), ("wo_id", 110), ("zone", 50), ("risk", 75), ("priority", 85), ("sla", 150), ("action", 350), ("loss", 95)):
            self.wo_tree.column(col, width=w, anchor="w" if col == "action" else "center")

        self.wo_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Bottom Split: Asset Health & Field Isolation Protocol
        bot_maint = tk.Frame(view, bg=self.BG)
        bot_maint.pack(fill=tk.BOTH, expand=True)

        ahi_card = tk.Frame(bot_maint, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        ahi_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))

        ahi_title = tk.Label(ahi_card, text="ZONE ASSET HEALTH INDEX (AHI)", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        ahi_title.pack(fill=tk.X, ipady=4)

        self.ahi_lbl = tk.Label(ahi_card, text="", bg="#091420", fg=self.TEXT, font=("Consolas", 9), justify=tk.LEFT, anchor="w")
        self.ahi_lbl.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        iso_card = tk.Frame(bot_maint, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        iso_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))

        iso_title = tk.Label(iso_card, text="EMERGENCY LEAK ISOLATION PROCEDURE", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        iso_title.pack(fill=tk.X, ipady=4)

        protocol_text = (
            "STANDARD OPERATING PROCEDURE (SOP-W-042):\n"
            "1. Confirm telemetry drop at Junction B2 and flow surge on Pipe P_B_CORE.\n"
            "2. Dispatch field crew with Work Order ID WO-2026-ZB-90.\n"
            "3. Manually or remotely close Isolation Valve P_A_TO_B to halt downstream loss.\n"
            "4. Verify pressure stabilization at downstream Junction C1 & D1.\n"
            "5. Execute acoustic ground microphone correlation at Junction B2.\n"
            "6. Replace pipe fitting / seal & perform hydraulic line pressure test."
        )
        lbl_proto = tk.Label(iso_card, text=protocol_text, bg="#091420", fg=self.TEXT, font=("Consolas", 8), justify=tk.LEFT, anchor="w")
        lbl_proto.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # -------------------------------------------------------------------------
    # VIEW 6: EVENT TIMELINE
    # -------------------------------------------------------------------------
    def _build_view_timeline(self) -> None:
        view = tk.Frame(self.main_container, bg=self.BG)
        self.views["Timeline"] = view

        # Header Filter Bar
        flt_bar = tk.Frame(view, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1, height=36)
        flt_bar.pack(fill=tk.X, pady=(0, 4))
        flt_bar.pack_propagate(False)

        flt_lbl = tk.Label(flt_bar, text="EVENT & ALARM TIMELINE LOG", bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        flt_lbl.pack(side=tk.LEFT, padx=10)

        btn_export_log = tk.Button(
            flt_bar, text="📥 Export Log CSV", bg=self.PANEL_ALT, fg=self.CYAN,
            font=("Segoe UI", 8, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._quick_export_events_csv
        )
        btn_export_log.pack(side=tk.RIGHT, padx=10, ipady=2, ipadx=6)

        # Timeline Table
        tbl_box = tk.Frame(view, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        tbl_box.pack(fill=tk.BOTH, expand=True)

        self.timeline_tree = ttk.Treeview(
            tbl_box, columns=("time", "sec", "severity", "msg"),
            show="headings", height=15
        )
        self.timeline_tree.heading("time", text="Timestamp")
        self.timeline_tree.heading("sec", text="Sim Time")
        self.timeline_tree.heading("severity", text="Severity")
        self.timeline_tree.heading("msg", text="SCADA Event & Diagnostic Message")

        for col, w in (("time", 140), ("sec", 80), ("severity", 110), ("msg", 600)):
            self.timeline_tree.column(col, width=w, anchor="w" if col == "msg" else "center")

        self.timeline_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # -------------------------------------------------------------------------
    # VIEW 7: REPORTS & EXPORT
    # -------------------------------------------------------------------------
    def _build_view_reports(self) -> None:
        view = tk.Frame(self.main_container, bg=self.BG)
        self.views["Reports"] = view

        # Top Buttons Card
        actions_card = tk.Frame(view, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        actions_card.pack(fill=tk.X, pady=(0, 6))

        act_title = tk.Label(actions_card, text="INTELLIGENCE REPORT GENERATION & DATA EXPORT", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        act_title.pack(fill=tk.X, ipady=4)

        btn_frame = tk.Frame(actions_card, bg=self.PANEL)
        btn_frame.pack(fill=tk.X, padx=12, pady=10)

        b1 = tk.Button(
            btn_frame, text="📄 Generate PDF Engineering Report",
            bg="#23566A", fg=self.TEXT, font=("Segoe UI", 9, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._quick_export_pdf
        )
        b1.pack(side=tk.LEFT, padx=6, ipady=6, ipadx=12)

        b2 = tk.Button(
            btn_frame, text="📥 Export Telemetry CSV",
            bg=self.PANEL_ALT, fg=self.CYAN, font=("Segoe UI", 9, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._quick_export_csv
        )
        b2.pack(side=tk.LEFT, padx=6, ipady=6, ipadx=12)

        b3 = tk.Button(
            btn_frame, text="📋 Export Event History CSV",
            bg=self.PANEL_ALT, fg=self.AMBER, font=("Segoe UI", 9, "bold"), bd=0, relief=tk.FLAT, cursor="hand2",
            command=self._quick_export_events_csv
        )
        b3.pack(side=tk.LEFT, padx=6, ipady=6, ipadx=12)

        # Preview Text Box
        prev_card = tk.Frame(view, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        prev_card.pack(fill=tk.BOTH, expand=True)

        prev_title = tk.Label(prev_card, text="SCADA EXECUTIVE SUMMARY PREVIEW", bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 9, "bold"))
        prev_title.pack(fill=tk.X, ipady=4)

        self.report_preview_text = tk.Text(
            prev_card, bg="#091420", fg=self.TEXT,
            font=("Consolas", 8), bd=0, highlightthickness=0, wrap=tk.WORD, state=tk.DISABLED
        )
        self.report_preview_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # -------------------------------------------------------------------------
    # REFRESH & DRAWING METHODS
    # -------------------------------------------------------------------------
    def refresh(self) -> None:
        status = self.engine.network_status
        status_color = self._phase_color(status)
        self.status_header_lbl.config(text=f"SYSTEM {status}", fg=status_color)
        self.status_canvas.itemconfig(self.status_led, fill=status_color)

        sec = self.engine.seconds
        mins = sec // 60
        secs = sec % 60
        self.time_lbl.config(text=f"T+{sec:03d}s ({mins:02d}:{secs:02d}) / {DEMO_SECONDS}s")

        state_text = "RUNNING" if self.engine.running else "PAUSED"
        if sec >= DEMO_SECONDS:
            state_text = "COMPLETE"
        self.sidebar_status.config(
            text=f"{state_text}\n{self.engine.speed}x SPEED\n{self.engine.phase}",
            fg=self.GREEN if self.engine.running else self.AMBER if sec < DEMO_SECONDS else self.CYAN
        )

        for speed, button in self.speed_btns.items():
            selected = speed == self.engine.speed
            button.config(
                bg="#23566A" if selected else self.PANEL_ALT,
                fg=self.CYAN if selected else self.TEXT
            )

        self._record_history_snapshot()
        self._update_kpi_cards()

        # Update active view graphics
        if self.active_view_name == "Overview":
            self._draw_overview_network()
            self._draw_overview_diagnostics()
            self._update_overview_alert_card()
            self._update_chart(self.ax_ov_pressure, self.ov_pressure_line, self.pressure_history)
            self._update_chart(self.ax_ov_flow, self.ov_flow_line, self.flow_history)
            self._update_chart(self.ax_ov_risk, self.ov_risk_line, self.risk_history)
            self.canvas_ov_network.draw_idle()
            self.canvas_ov_charts.draw_idle()
            self.canvas_ov_diag.draw_idle()
        elif self.active_view_name == "Digital Twin":
            self._draw_digital_twin_map()
            self._update_twin_inspector()
            self.canvas_twin.draw_idle()
        elif self.active_view_name == "Telemetry":
            self._update_telemetry_tables()
            self._draw_telemetry_waveforms()
            self.canvas_telem.draw_idle()
        elif self.active_view_name == "AI Diagnostics":
            self._update_ai_diagnostics_view()
            self.canvas_ai.draw_idle()
        elif self.active_view_name == "Maintenance":
            self._update_maintenance_view()
        elif self.active_view_name == "Timeline":
            self._update_timeline_view()
        elif self.active_view_name == "Reports":
            self._update_report_preview()

    def _record_history_snapshot(self) -> None:
        rec = {
            "second": self.engine.seconds,
            "phase": self.engine.phase,
            "status": self.engine.network_status,
            "probable_zone": self.engine.probable_zone or "NONE",
        }
        for z in ("A", "B", "C", "D"):
            rec[f"risk_{z}"] = self.engine.risk[z]
            rec[f"press_{z}"] = self.engine.measurements[z].pressure_m
            rec[f"flow_{z}"] = self.engine.measurements[z].flow_lps
            rec[f"ml_{z}"] = self.engine.ml_flags[z]
        self.full_history.append(rec)

    def _update_kpi_cards(self) -> None:
        status = self.engine.network_status
        highest_risk = max(self.engine.risk.values(), default=0.0)
        probable_zone = self.engine.probable_zone
        zone_b_p = self.engine.measurements["B"].pressure_m
        zone_b_f = self.engine.measurements["B"].flow_lps
        p_dev = zone_b_p - self.engine.calibration.baselines["B"].pressure_m
        f_dev = zone_b_f - self.engine.calibration.baselines["B"].flow_lps

        metrics = (
            ("SYSTEM STATUS", status, f"Network state ({status})", self._phase_color(status)),
            ("PROBABLE LEAK ZONE", f"ZONE {probable_zone}" if probable_zone else "NONE (NOMINAL)", "Sensor anomaly localization", self._phase_color(
                self._zone_state(probable_zone) if probable_zone else "NORMAL"
            )),
            ("MAX LEAK THREAT", f"{highest_risk:.1f}%", f"Target: Zone {probable_zone or 'B'}", self._phase_color(
                "HIGH RISK" if highest_risk >= 61 else "WARNING" if highest_risk >= 31 else "NORMAL"
            )),
            ("ZONE B PRESSURE", f"{zone_b_p:.2f} m", f"{p_dev:+.2f} m vs baseline", self.CYAN),
            ("ACTIVE WORK ORDER", f"WO-2026-Z{probable_zone or 'B'}-{int(highest_risk)}", f"Est loss: {self.engine.loss_estimate_l_min:.1f} L/min", self.AMBER if highest_risk >= 31 else self.MUTED),
        )
        for (title_lbl, val_lbl, det_lbl, card), (label, value, detail, color) in zip(self.kpi_widgets, metrics):
            title_lbl.config(text=label)
            val_lbl.config(text=value, fg=color)
            det_lbl.config(text=detail)

    def _draw_overview_network(self) -> None:
        axis = self.ax_ov_network
        axis.clear()
        self._style_panel(axis)
        axis.set_xticks([])
        axis.set_yticks([])

        snapshot = self.engine.snapshot
        leak_active = (self.engine.seconds >= LEAK_START_SECONDS) or (self.engine.risk["B"] >= 31)

        # 1. Zone Background Sectors & Header Tags with Gradual Risk Highlights
        zone_bounds = (
            ("A", -0.45, 1.70, "ZONE A  ·  TRANSMISSION INLET"),
            ("B", 1.70, 3.80, "ZONE B  ·  DISTRIBUTION GRID (LEAK TARGET)"),
            ("C", 3.80, 5.90, "ZONE C  ·  RESIDENTIAL DISTRICT"),
            ("D", 5.90, 8.00, "ZONE D  ·  ELEVATED SECTOR"),
        )
        for zone, left, right, title in zone_bounds:
            state = self._zone_state(zone)
            risk_val = self.engine.risk[zone]
            is_anomaly = state in ("WARNING", "HIGH RISK", "ANOMALY DETECTED")

            if state == "HIGH RISK":
                # Gradual dynamic highlight based on actual risk severity
                pulse_factor = 0.5 + 0.5 * math.sin(self.leak_pulse_phase) if (self.anim_enabled and zone == "B") else 0.5
                alpha = 0.40 + 0.35 * (risk_val / 100.0) + (0.15 * pulse_factor if self.anim_enabled else 0.0)
                bg_col = "#38141C"
                edge_col = self.RED
                edge_w = 2.0
            elif state in ("WARNING", "ANOMALY DETECTED"):
                bg_col = "#2D2012"
                edge_col = self.AMBER
                edge_w = 1.6
                alpha = 0.35 + 0.25 * (risk_val / 100.0)
            else:
                bg_col = "#0B1A28"
                edge_col = self.BORDER
                edge_w = 0.9
                alpha = 0.50

            rect = Rectangle(
                (left, -1.02), right - left, 1.70,
                facecolor=bg_col, edgecolor=edge_col,
                linewidth=edge_w, alpha=min(0.95, alpha), zorder=0
            )
            axis.add_patch(rect)

            header_col = self._phase_color(state) if is_anomaly else self.MUTED
            axis.text(
                (left + right) / 2.0, 0.61, f"{title} [{state}]",
                ha="center", va="center", fontsize=8.2, fontweight="bold",
                color=header_col, zorder=2
            )

        # 2. Pipe Links
        edge_colors = []
        edge_widths = []
        for _, _, data in self.graph.edges(data=True):
            zone = data["zone"]
            state = self._zone_state(zone)
            if zone == "B" and leak_active:
                edge_colors.append(self.RED if self.engine.risk["B"] >= 61 else self.AMBER)
                edge_widths.append(4.2)
            elif state != "NORMAL":
                edge_colors.append(self._phase_color(state))
                edge_widths.append(3.2)
            else:
                edge_colors.append("#245E7A")
                edge_widths.append(2.4)

        nx.draw_networkx_edges(
            self.graph, self.positions, ax=axis,
            edge_color=edge_colors, width=edge_widths,
            arrows=True, arrowstyle="-|>", arrowsize=18,
            min_source_margin=24, min_target_margin=26,
        )

        # 3. Dynamic Water-Flow Particles (along actual network pipe connections and correct direction)
        if self.anim_enabled:
            particle_xs = []
            particle_ys = []
            particle_colors = []
            particle_sizes = []
            for u, v, data in self.graph.edges(data=True):
                p_id = data["pipe"]
                flow = snapshot.flows_lps.get(p_id, 0.0)
                if flow <= 0.05:
                    continue
                p1 = np.array(self.positions[u], dtype=float)
                p2 = np.array(self.positions[v], dtype=float)
                pipe_vec = p2 - p1
                pipe_len = float(np.linalg.norm(pipe_vec))
                if pipe_len < 1e-4:
                    continue

                # Margins so particles travel cleanly within visible pipe segment
                m_start = 0.22 / pipe_len
                m_end = 1.0 - (0.24 / pipe_len)
                if m_start >= m_end:
                    continue

                zone = data["zone"]
                is_b_leak = (zone == "B" and leak_active)
                base_color = self.AMBER if is_b_leak and self.engine.risk["B"] < 61 else self.RED if is_b_leak else "#6EE7B7" if zone == "A" else self.CYAN

                # Up to 3 animated particles per active pipe
                num_particles = 3 if flow > 1.2 else 2
                for i in range(num_particles):
                    fraction = (self.flow_particle_phase + i / num_particles) % 1.0
                    pos_ratio = m_start + fraction * (m_end - m_start)
                    pos = p1 + pos_ratio * pipe_vec
                    particle_xs.append(pos[0])
                    particle_ys.append(pos[1])
                    particle_colors.append(base_color)
                    particle_sizes.append(28 if is_b_leak else 18)

            if particle_xs:
                axis.scatter(
                    particle_xs, particle_ys, s=particle_sizes, c=particle_colors,
                    marker="o", alpha=0.92, zorder=5, edgecolors="#FFFFFF", linewidths=0.5
                )

        # 4. Pipe Flow Rate Callouts
        edge_labels = {
            (start, end): f"{snapshot.flows_lps[data['pipe']]:.2f} L/s"
            for start, end, data in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.graph, self.positions, edge_labels=edge_labels, ax=axis,
            font_size=7.5, font_color="#E2F1F8", font_weight="bold",
            label_pos=0.48, rotate=False,
            bbox={"boxstyle": "round,pad=0.22", "fc": "#071724", "ec": "#1A354B", "lw": 0.8, "alpha": 0.95},
        )

        # 5. Nodes Styling & Pressure Displays
        node_colors = []
        node_sizes = []
        node_labels = {}
        for node in self.graph.nodes:
            if node == "RESERVOIR":
                node_colors.append("#0284C7")
                node_sizes.append(2200)
                node_labels[node] = "RESERVOIR\n79.0 m"
            else:
                zone = self.node_zones[node]
                state = self._zone_state(zone)
                p = snapshot.pressures[node]
                if node == "B2" and leak_active:
                    node_colors.append(self.RED if self.engine.risk["B"] >= 61 else self.AMBER)
                    node_sizes.append(2100)
                elif state == "HIGH RISK":
                    node_colors.append(self.RED)
                    node_sizes.append(1700)
                elif state in ("WARNING", "ANOMALY DETECTED"):
                    node_colors.append(self.AMBER if state == "WARNING" else self.CYAN)
                    node_sizes.append(1700)
                else:
                    node_colors.append("#17374D")
                    node_sizes.append(1600)
                node_labels[node] = f"{node}\n{p:.1f} m"

        nx.draw_networkx_nodes(
            self.graph, self.positions, ax=axis,
            node_color=node_colors, node_size=node_sizes,
            edgecolors=self.TEXT, linewidths=1.8,
        )
        nx.draw_networkx_labels(
            self.graph, self.positions, labels=node_labels, ax=axis,
            font_size=8.2, font_color="#FFFFFF", font_weight="bold",
        )

        # 6. Pulsing Warning Rings around Junction B2 (Simulated Leak)
        if leak_active:
            b2_x, b2_y = self.positions["B2"]
            r_col = self.RED if self.engine.risk["B"] >= 61 else self.AMBER
            pulse_mult = 1.0 + 0.18 * math.sin(self.leak_pulse_phase) if self.anim_enabled else 1.0

            pulse_ring1 = Circle(
                (b2_x, b2_y), 0.28 * pulse_mult,
                fill=False, edgecolor=r_col, linewidth=2.4, linestyle="--", zorder=4
            )
            pulse_ring2 = Circle(
                (b2_x, b2_y), 0.42 * pulse_mult,
                fill=False, edgecolor=r_col, linewidth=1.4, linestyle=":", zorder=4
            )
            axis.add_patch(pulse_ring1)
            axis.add_patch(pulse_ring2)

            p_b2 = snapshot.pressures["B2"]
            base_b2 = self.calibration.baselines["B"].pressure_m
            drop_b2 = p_b2 - base_b2
            loss_rate = self.engine.loss_estimate_l_min
            axis.text(
                b2_x, b2_y + 0.38,
                f"⚠ LEAK EMITTER DETECTED (B2)\nΔP: {drop_b2:+.2f} m  ·  Loss: ~{loss_rate:.1f} L/min",
                ha="center", va="bottom", fontsize=8.2, fontweight="bold",
                color="#FFFFFF",
                bbox={"boxstyle": "round,pad=0.3", "fc": "#7F1D1D" if self.engine.risk["B"] >= 61 else "#78350F", "ec": r_col, "lw": 1.4},
                zorder=7
            )

        axis.set_xlim(-1.45, 8.10)
        axis.set_ylim(-1.08, 0.75)
        axis.axis("off")

    def _draw_overview_diagnostics(self) -> None:
        # 1. ARC RISK GAUGE WITH SMOOTH NEEDLE INTERPOLATION
        ax_g = self.ax_ov_gauge
        ax_g.clear()
        self._style_panel(ax_g)
        ax_g.set_xticks([])
        ax_g.set_yticks([])

        target_risk = float(max(self.engine.risk.values(), default=0.0))
        target_angle = math.pi - (target_risk / 100.0) * math.pi

        if self.anim_enabled and self.engine.running:
            # Smooth exponential moving settlement
            self.current_gauge_angle += (target_angle - self.current_gauge_angle) * 0.35
        else:
            self.current_gauge_angle = target_angle

        # Display risk reflecting current needle angle
        display_risk = max(0.0, min(100.0, (math.pi - self.current_gauge_angle) / math.pi * 100.0))
        sev_color = self.RED if display_risk >= 61 else self.AMBER if display_risk >= 31 else self.GREEN
        sev_label = "HIGH RISK" if display_risk >= 61 else "WARNING" if display_risk >= 31 else "NORMAL"

        r_inner, r_outer = 0.65, 0.95

        # Sector bands
        t_green = np.linspace(np.pi, np.pi * 0.70, 30)
        ax_g.fill_between(np.cos(t_green) * r_outer, np.sin(t_green) * r_inner, np.sin(t_green) * r_outer, color="#065F46", alpha=0.9)

        t_amber = np.linspace(np.pi * 0.70, np.pi * 0.40, 30)
        ax_g.fill_between(np.cos(t_amber) * r_outer, np.sin(t_amber) * r_inner, np.sin(t_amber) * r_outer, color="#92400E", alpha=0.9)

        t_red = np.linspace(np.pi * 0.40, 0, 30)
        ax_g.fill_between(np.cos(t_red) * r_outer, np.sin(t_red) * r_inner, np.sin(t_red) * r_outer, color="#991B1B", alpha=0.9)

        # Needle drawing
        needle_len = 0.82
        nx_val = math.cos(self.current_gauge_angle) * needle_len
        ny_val = math.sin(self.current_gauge_angle) * needle_len
        ax_g.plot([0, nx_val], [0, ny_val], color=self.TEXT, linewidth=2.8, zorder=5)
        ax_g.plot(0, 0, "o", color=self.CYAN, markersize=8, zorder=6)

        # Digital Readout
        ax_g.text(0, -0.02, f"{display_risk:.1f}%", ha="center", va="top", fontsize=15, fontweight="bold", color=sev_color)
        ax_g.text(0, -0.32, f"LEAK THREAT: {sev_label}", ha="center", va="top", fontsize=8.5, fontweight="bold", color=self.TEXT)
        ax_g.set_xlim(-1.15, 1.15)
        ax_g.set_ylim(-0.45, 1.05)
        ax_g.axis("off")

        # 2. 4-ZONE THREAT COMPARISON BARS (Smooth Transition)
        ax_z = self.ax_ov_zones
        ax_z.clear()
        self._style_panel(ax_z)
        zones = ("D", "C", "B", "A")
        y_pos = np.arange(len(zones))

        displayed_risks = []
        for z in zones:
            target_r = float(self.engine.risk[z])
            if self.anim_enabled and self.engine.running:
                self.current_zone_bars[z] += (target_r - self.current_zone_bars.get(z, 0.0)) * 0.40
            else:
                self.current_zone_bars[z] = target_r
            displayed_risks.append(self.current_zone_bars[z])

        bar_colors = [self.RED if r >= 61 else self.AMBER if r >= 31 else self.GREEN for r in displayed_risks]

        ax_z.barh(y_pos, [100]*4, height=0.55, color="#16293B", edgecolor="none")
        ax_z.barh(y_pos, displayed_risks, height=0.55, color=bar_colors, edgecolor="none")
        ax_z.set_yticks(y_pos)
        ax_z.set_yticklabels([f"Zone {z}" for z in zones], fontsize=7.8, color=self.TEXT, fontweight="bold")
        ax_z.set_xlim(0, 100)
        ax_z.set_title("ZONE THREAT COMPARISON (0-100%)", loc="left", fontsize=7.8, color=self.MUTED, fontweight="bold", pad=4)
        ax_z.tick_params(colors=self.MUTED, labelsize=7.0, length=2)
        ax_z.grid(axis="x", color=self.BORDER, linestyle="--", linewidth=0.6, alpha=0.7)

        for i, (r, z) in enumerate(zip(displayed_risks, zones)):
            p_val = self.engine.measurements[z].pressure_m
            ax_z.text(min(r + 3, 97), i, f"{r:.0f}% ({p_val:.1f}m)", va="center", fontsize=7.2, color=self.TEXT, fontweight="bold")

        # 3. EXPLAINABLE AI RISK CONTRIBUTIONS (Smooth Transition)
        ax_c = self.ax_ov_contrib
        ax_c.clear()
        self._style_panel(ax_c)
        focus_zone = self.engine.probable_zone or "B"
        contribs = self.engine.contributions[focus_zone]
        factors = ("Persistence", "ML Anomaly", "Flow Surge", "Pressure Drop")
        weights = (10.0, 20.0, 35.0, 35.0)
        targets = (contribs["persistence"], contribs["ml"], contribs["flow"], contribs["pressure"])

        for idx, t_val in enumerate(targets):
            if self.anim_enabled and self.engine.running:
                self.current_contrib_bars[idx] += (t_val - self.current_contrib_bars[idx]) * 0.40
            else:
                self.current_contrib_bars[idx] = float(t_val)

        actuals = self.current_contrib_bars
        y_c = np.arange(len(factors))

        ax_c.barh(y_c, weights, height=0.55, color="#16293B", edgecolor="none")
        c_colors = [
            self.RED if a >= w * 0.7 else self.AMBER if a >= w * 0.3 else self.CYAN
            for a, w in zip(actuals, weights)
        ]
        ax_c.barh(y_c, actuals, height=0.55, color=c_colors, edgecolor="none")
        ax_c.set_yticks(y_c)
        ax_c.set_yticklabels(factors, fontsize=7.5, color=self.TEXT)
        ax_c.set_xlim(0, 38)
        ax_c.set_title(f"EXPLAINABLE AI DECOMPOSITION (ZONE {focus_zone})", loc="left", fontsize=7.8, color=self.MUTED, fontweight="bold", pad=4)
        ax_c.tick_params(colors=self.MUTED, labelsize=7.0, length=2)
        ax_c.grid(axis="x", color=self.BORDER, linestyle="--", linewidth=0.6, alpha=0.7)

        for i, (a, w) in enumerate(zip(actuals, weights)):
            ax_c.text(min(a + 1.2, 36), i, f"+{a:.1f}/{w:.0f}", va="center", fontsize=7.2, color=self.TEXT, fontweight="bold", family="monospace")

    def _update_overview_alert_card(self) -> None:
        max_risk = max(self.engine.risk.values(), default=0.0)
        anomaly = any(self.engine.anomaly_flags.values())
        probable_zone = self.engine.probable_zone

        if max_risk >= 61:
            self.alert_banner_lbl.config(
                text=f"🚨 CRITICAL ALARM: ZONE {probable_zone or 'B'} PIPE RUPTURE DETECTED",
                bg="#7F1D1D", fg="#FCA5A5"
            )
            self.alert_zone_lbl.config(
                text=f"PROBABLE LOCATION: ZONE {probable_zone or 'B'} (JUNCTION B2)",
                fg="#FCA5A5"
            )
            self.alert_action_lbl.config(
                text=f"ACTION: Immediate dispatch · Isolate Valve P_A_TO_B · Est Loss: {self.engine.loss_estimate_l_min:.1f} L/min",
                fg=self.TEXT
            )
        elif max_risk >= 31 or anomaly:
            self.alert_banner_lbl.config(
                text=f"⚠ ANOMALY DETECTED: ELEVATED THREAT IN ZONE {probable_zone or 'B'}",
                bg="#78350F", fg="#FDE68A"
            )
            self.alert_zone_lbl.config(
                text=f"PROBABLE LOCATION: ZONE {probable_zone or 'B'}",
                fg="#FDE68A"
            )
            self.alert_action_lbl.config(
                text="ACTION: Continuous transient tracking; prepare isolation crew.",
                fg=self.TEXT
            )
        else:
            self.alert_banner_lbl.config(
                text="● SYSTEM STATUS: NOMINAL (ALL ZONES STABLE)",
                bg="#064E3B", fg="#6EE7B7"
            )
            self.alert_zone_lbl.config(
                text="PROBABLE LOCATION: NONE (NOMINAL CONDITIONS)",
                fg=self.MUTED
            )
            self.alert_action_lbl.config(
                text="ACTION: Routine hydraulic SCADA monitoring. All 12 nodes within limits.",
                fg=self.MUTED
            )

        priorities = "  →  ".join(f"{zone} ({label})" for zone, _score, label in self.engine.priorities())
        self.alert_priority_lbl.config(
            text=f"MAINTENANCE DISPATCH QUEUE:\n{priorities}"
        )

    def _draw_digital_twin_map(self) -> None:
        axis = self.ax_twin_map
        axis.clear()
        self._style_panel(axis)
        axis.set_xticks([])
        axis.set_yticks([])

        axis.text(0.02, 0.96, "WATER NETWORK DIGITAL TWIN  |  FULL SYSTEM SCHEMATIC", transform=axis.transAxes, fontsize=11, fontweight="bold", color=self.TEXT, va="top")
        axis.text(0.98, 0.96, "SELECT NODE FOR DETAILED HYDRAULIC INSPECTION", transform=axis.transAxes, fontsize=8, color=self.CYAN, ha="right", va="top")

        # Draw Zone Background Lanes
        zone_bounds = (("A", -0.35, 1.65), ("B", 1.65, 3.65), ("C", 3.65, 5.65), ("D", 5.65, 7.55))
        lane_colors = ("#122738", "#152a3a", "#132b3d", "#142838")
        for (zone, left, right), lane_color in zip(zone_bounds, lane_colors):
            state = self._zone_state(zone)
            fill_color = self._phase_color(state) if state != "NORMAL" else lane_color
            alpha = 0.20 if state != "NORMAL" else 0.40
            axis.axvspan(left, right, ymin=0.08, ymax=0.90, facecolor=fill_color, alpha=alpha, zorder=0)
            axis.axvline(right, ymin=0.08, ymax=0.90, color=self.BORDER, linewidth=1.0, linestyle="--", zorder=1)
            axis.text((left + right) / 2, 0.85, f"ZONE {zone} ({state})", ha="center", va="center", fontsize=9, fontweight="bold", color=self._phase_color(state), zorder=1)

        snapshot = self.engine.snapshot
        edge_colors = []
        edge_widths = []
        for _, _, data in self.graph.edges(data=True):
            zone = data["zone"]
            state = self._zone_state(zone)
            edge_colors.append(self._phase_color(state))
            edge_widths.append(3.5 if zone == self.engine.probable_zone else 1.8)

        node_colors = []
        node_labels = {}
        for node in self.graph.nodes:
            if node == "RESERVOIR":
                node_colors.append("#23536C")
                node_labels[node] = "RES\n79.0m"
            else:
                zone = self.node_zones[node]
                state = self._zone_state(zone)
                node_colors.append(self._phase_color(state) if state != "NORMAL" else "#2C5266")
                node_labels[node] = f"{node}\n{snapshot.pressures[node]:.1f}m"

        nx.draw_networkx_edges(
            self.graph, self.positions, ax=axis, edge_color=edge_colors, width=edge_widths,
            arrows=True, arrowstyle="-|>", arrowsize=18, connectionstyle="arc3,rad=0.0",
            min_source_margin=22, min_target_margin=24,
        )

        # Flow Particles in Digital Twin View
        if self.anim_enabled:
            dt_px, dt_py, dt_pc, dt_ps = [], [], [], []
            for u, v, data in self.graph.edges(data=True):
                p_id = data["pipe"]
                flow = snapshot.flows_lps.get(p_id, 0.0)
                if flow <= 0.05:
                    continue
                p1 = np.array(self.positions[u], dtype=float)
                p2 = np.array(self.positions[v], dtype=float)
                pvec = p2 - p1
                plen = float(np.linalg.norm(pvec))
                if plen < 1e-4:
                    continue
                m_start = 0.22 / plen
                m_end = 1.0 - (0.24 / plen)
                if m_start >= m_end:
                    continue
                zone = data["zone"]
                is_b_leak = (zone == "B" and (self.engine.leak_severity > 0 or self.engine.risk["B"] >= 31))
                pcol = self.RED if (is_b_leak and self.engine.risk["B"] >= 61) else self.AMBER if is_b_leak else self.CYAN
                num_p = 3 if flow > 1.2 else 2
                for i in range(num_p):
                    frac = (self.flow_particle_phase + i / num_p) % 1.0
                    pos = p1 + (m_start + frac * (m_end - m_start)) * pvec
                    dt_px.append(pos[0])
                    dt_py.append(pos[1])
                    dt_pc.append(pcol)
                    dt_ps.append(26 if is_b_leak else 16)
            if dt_px:
                axis.scatter(dt_px, dt_py, s=dt_ps, c=dt_pc, marker="o", alpha=0.92, zorder=5, edgecolors="#FFF", linewidths=0.5)

        nx.draw_networkx_nodes(
            self.graph, self.positions, ax=axis, node_color=node_colors, node_size=1600,
            edgecolors=self.BORDER, linewidths=1.8,
        )
        nx.draw_networkx_labels(
            self.graph, self.positions, labels=node_labels, ax=axis,
            font_size=8.0, font_color="#FFFFFF", font_weight="bold",
        )
        edge_labels = {
            (start, end): f"{snapshot.flows_lps[data['pipe']]:.2f} L/s"
            for start, end, data in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.graph, self.positions, edge_labels=edge_labels, ax=axis,
            font_size=7.2, font_color=self.TEXT, label_pos=0.48, rotate=False,
            bbox={"boxstyle": "round,pad=0.15", "fc": "#060D15", "ec": self.BORDER, "alpha": 0.95},
        )

        # Highlight Selected Node
        if self.selected_node and self.selected_node in self.positions:
            sx, sy = self.positions[self.selected_node]
            axis.add_patch(Circle((sx, sy), 0.28, facecolor="none", edgecolor=self.CYAN, linewidth=2.5, linestyle="-", zorder=10))

        # Highlight Pulsing Leak Site at Junction B2 when leak active
        if self.engine.leak_severity > 0 or self.engine.risk["B"] >= 31:
            lx, ly = self.positions["B2"]
            r_col = self.RED if self.engine.risk["B"] >= 61 else self.AMBER
            pulse_mult = 1.0 + 0.18 * math.sin(self.leak_pulse_phase) if self.anim_enabled else 1.0
            axis.add_patch(Circle((lx, ly), 0.40 * pulse_mult, facecolor="none", edgecolor=r_col, linewidth=2.2, linestyle="--", zorder=9))
            axis.text(lx, ly + 0.38, "SIMULATED LEAK SITE (B2)", color="#FFFFFF", fontsize=8, fontweight="bold", ha="center", va="bottom", zorder=11, bbox={"boxstyle": "square,pad=0.2", "fc": "#7F1D1D" if self.engine.risk["B"] >= 61 else "#78350F", "ec": r_col})

        axis.set_xlim(-1.30, 7.62)
        axis.set_ylim(-1.05, 0.75)
        axis.axis("off")

    def _update_twin_inspector(self) -> None:
        for widget in self.insp_content.winfo_children():
            widget.destroy()

        node = self.selected_node
        snapshot = self.engine.snapshot
        baseline_snapshot = self.calibration.baseline_snapshot

        if node == "RESERVOIR":
            title_text = "NODE: RESERVOIR (HEAD SOURCE)"
            p_curr = 79.0
            p_base = 79.0
            zone = "MAIN SOURCE"
            elev = 0.0
            status = "NOMINAL"
        else:
            zone = self.node_zones[node]
            p_curr = snapshot.pressures[node]
            p_base = baseline_snapshot.pressures[node]
            elev = {"A1":51,"A2":52,"A3":51.5,"B1":52,"B2":53,"B3":52.5,"C1":53,"C2":54,"C3":53.5,"D1":56,"D2":57,"D3":56.5}[node]
            status = self._zone_state(zone)
            title_text = f"JUNCTION {node} (ZONE {zone})"

        lbl = tk.Label(self.insp_content, text=title_text, bg=self.PANEL, fg=self.CYAN, font=("Segoe UI", 10, "bold"), anchor="w")
        lbl.pack(fill=tk.X, pady=(0, 6))

        dev = p_curr - p_base
        dev_pct = (dev / p_base * 100.0) if p_base else 0.0

        details = (
            ("Zone Assignment", f"Zone {zone}"),
            ("Elevation", f"{elev:.1f} m"),
            ("Base Demand", "0.00016 m3/s"),
            ("Hydraulic Head", f"{p_curr + elev:.2f} m"),
            ("Live Pressure", f"{p_curr:.2f} m H2O"),
            ("Baseline Pressure", f"{p_base:.2f} m H2O"),
            ("Pressure Drop (Delta)", f"{dev:+.2f} m ({dev_pct:+.1f}%)"),
            ("Anomaly Status", status),
        )
        for k, v in details:
            row = tk.Frame(self.insp_content, bg=self.PANEL)
            row.pack(fill=tk.X, pady=2)
            k_lbl = tk.Label(row, text=k, bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 8), anchor="w")
            k_lbl.pack(side=tk.LEFT)
            v_lbl = tk.Label(row, text=v, bg=self.PANEL, fg=self.RED if "Drop" in k and dev < -0.5 else self.TEXT, font=("Segoe UI", 8, "bold"), anchor="e")
            v_lbl.pack(side=tk.RIGHT)

        sep = tk.Frame(self.insp_content, bg=self.BORDER, height=1)
        sep.pack(fill=tk.X, pady=8)

        conn_lbl = tk.Label(self.insp_content, text="CONNECTED PIPES & FLOWS", bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 8, "bold"), anchor="w")
        conn_lbl.pack(fill=tk.X, pady=(0, 4))

        for u, v, data in self.graph.edges(data=True):
            if u == node or v == node:
                pipe_id = data["pipe"]
                flow = snapshot.flows_lps[pipe_id]
                base_flow = baseline_snapshot.flows_lps[pipe_id]
                flow_dev = flow - base_flow
                p_row = tk.Frame(self.insp_content, bg="#091420", highlightbackground=self.BORDER, highlightthickness=1)
                p_row.pack(fill=tk.X, pady=2, ipady=2)
                p_text = f"{pipe_id} ({u}->{v})\nFlow: {flow:.2f} L/s ({flow_dev:+.2f} L/s vs base)"
                plbl = tk.Label(p_row, text=p_text, bg="#091420", fg=self.TEXT, font=("Consolas", 8), justify=tk.LEFT, anchor="w")
                plbl.pack(fill=tk.X, padx=6)

    def _update_telemetry_tables(self) -> None:
        for item in self.junc_tree.get_children():
            self.junc_tree.delete(item)
        for item in self.pipe_tree.get_children():
            self.pipe_tree.delete(item)

        junc_data = self.engine.get_junction_telemetry()
        for j in junc_data:
            self.junc_tree.insert("", tk.END, values=(
                j["node"], j["zone"], f"{j['elevation_m']:.1f}", f"{j['baseline_m']:.2f}",
                f"{j['pressure_m']:.2f}", f"{j['dev_m']:+.2f}", j["status"]
            ))

        pipe_data = self.engine.get_pipe_telemetry()
        for p in pipe_data:
            self.pipe_tree.insert("", tk.END, values=(
                p["pipe"], f"{p['start']}->{p['end']}", p["zone"], f"{p['baseline_lps']:.2f}",
                f"{p['flow_lps']:.2f}", f"{p['dev_lps']:+.2f}", f"{p['velocity_m_s']:.2f}"
            ))

    def _draw_telemetry_waveforms(self) -> None:
        fig = self.fig_telem
        fig.clear()
        axes = [fig.add_subplot(141 + i) for i in range(4)]
        zones = ("A", "B", "C", "D")

        for ax, zone in zip(axes, zones):
            self._style_panel(ax)
            ax.set_title(f"ZONE {zone} PRESSURE & FLOW", fontsize=8, color=self.TEXT, fontweight="bold", pad=4)
            ax.set_xlim(0, DEMO_SECONDS)
            ax.set_xticks((0, 50, 100, 150))
            ax.grid(True, color=self.BORDER, linewidth=0.6, alpha=0.7)
            ax.tick_params(colors=self.MUTED, labelsize=6.5)

            # Draw zone risk history or pressure if recorded
            p_val = self.engine.measurements[zone].pressure_m
            f_val = self.engine.measurements[zone].flow_lps
            ax.text(0.05, 0.82, f"P: {p_val:.2f} m\nQ: {f_val:.2f} L/s", transform=ax.transAxes, fontsize=7.5, color=self.CYAN if zone != "B" or self.engine.risk["B"] < 31 else self.RED, fontweight="bold")

    def _update_ai_diagnostics_view(self) -> None:
        # Update ML Scores Text
        txt = "ZONE ISOLATION FOREST ANOMALY EVALUATION:\n"
        for zone in ("A", "B", "C", "D"):
            score = self.engine.ml_scores.get(zone, 0.0)
            flagged = self.engine.ml_flags[zone]
            txt += f"  Zone {zone}: Score = {score:+.4f}  |  ML Flag = {'[ ANOMALY DETECTED ]' if flagged else '[ NORMAL ]'}\n"
        self.ml_scores_lbl.config(text=txt)

        # Update Persistence Matrix Widgets
        for w in self.pers_matrix_frame.winfo_children():
            w.destroy()

        for idx, zone in enumerate(("A", "B", "C", "D")):
            row_frame = tk.Frame(self.pers_matrix_frame, bg=self.PANEL)
            row_frame.pack(fill=tk.X, pady=2)

            z_lbl = tk.Label(row_frame, text=f"Zone {zone}:", bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 8, "bold"), width=8, anchor="w")
            z_lbl.pack(side=tk.LEFT)

            pers_list = list(self.engine.persistence[zone])
            # fill up to 8 items
            while len(pers_list) < 8:
                pers_list.insert(0, False)

            for is_anom in pers_list[-8:]:
                box = tk.Label(
                    row_frame, text="ALERT" if is_anom else "OK",
                    bg=self.RED if is_anom else self.GREEN, fg=self.BG,
                    font=("Segoe UI", 7, "bold"), width=5, relief=tk.FLAT
                )
                box.pack(side=tk.LEFT, padx=2)

        # Draw AI Contribution Bars
        ax = self.ax_ai_bars
        ax.clear()
        self._style_panel(ax)
        zone = self.engine.probable_zone or "B"
        contribs = self.engine.contributions[zone]

        keys = ["Pressure", "Flow", "ML Score", "Persistence"]
        vals = [contribs["pressure"], contribs["flow"], contribs["ml"], contribs["persistence"]]
        maxs = [RISK_WEIGHTS["pressure"], RISK_WEIGHTS["flow"], RISK_WEIGHTS["ml"], RISK_WEIGHTS["persistence"]]

        y_pos = np.arange(len(keys))
        ax.barh(y_pos, vals, align="center", color=[self.CYAN, self.BLUE, self.AMBER, self.RED], alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keys, fontsize=7.5, color=self.TEXT)
        ax.set_xlim(0, 40)
        ax.grid(True, color=self.BORDER, linewidth=0.6, alpha=0.7)
        ax.tick_params(colors=self.MUTED, labelsize=7.0)
        ax.set_title(f"RISK DECOMPOSITION - ZONE {zone}", fontsize=8, color=self.TEXT, fontweight="bold")

        # Update Natural Language Explanation
        exp_str = self.engine.get_zone_explanation(zone)
        self.ai_exp_text.config(state=tk.NORMAL)
        self.ai_exp_text.delete("1.0", tk.END)
        self.ai_exp_text.insert(tk.END, exp_str)
        self.ai_exp_text.config(state=tk.DISABLED)

    def _update_maintenance_view(self) -> None:
        for item in self.wo_tree.get_children():
            self.wo_tree.delete(item)

        recs = self.engine.get_maintenance_recommendations()
        for r in recs:
            self.wo_tree.insert("", tk.END, values=(
                r["rank"], r["work_order_id"], r["zone"], f"{r['score']:.0f}/100",
                r["label"], r["sla"], r["action"], f"{r['est_loss_l_min']:.1f}"
            ))

        ahi_text = "ZONE ASSET HEALTH INDEX (AHI) SUMMARY:\n"
        for z in ("A", "B", "C", "D"):
            risk = self.engine.risk[z]
            ahi = max(0.0, 100.0 - risk)
            status = "CRITICAL" if ahi < 40 else "DEGRADED" if ahi < 70 else "EXCELLENT"
            ahi_text += f"  Zone {z} AHI: {ahi:.0f}%  [{status}]\n"
        self.ahi_lbl.config(text=ahi_text)

    def _update_timeline_view(self) -> None:
        for item in self.timeline_tree.get_children():
            self.timeline_tree.delete(item)

        for time_str, sec, severity, msg in reversed(self.raw_event_log):
            self.timeline_tree.insert("", tk.END, values=(time_str, f"T+{sec:03d}s", severity, msg))

    def _update_report_preview(self) -> None:
        report = (
            "======================================================================\n"
            "   SMARTLEAK INDUSTRIAL SCADA & WATER NETWORK EXECUTIVE REPORT\n"
            "======================================================================\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Sim Time: T+{self.engine.seconds:03d}s\n"
            f"Network Operating Status: SYSTEM {self.engine.network_status}\n"
            f"Probable Leak Localization: ZONE {self.engine.probable_zone or 'NONE'}\n\n"
            "1. HYDRAULIC NETWORK TELEMETRY SUMMARY:\n"
            f"   - Zone B Head Pressure: {self.engine.measurements['B'].pressure_m:.2f} m H2O\n"
            f"   - Zone B Core Pipe Flow: {self.engine.measurements['B'].flow_lps:.2f} L/s\n"
            f"   - Baseline Pressure: {self.engine.calibration.baselines['B'].pressure_m:.2f} m H2O\n\n"
            "2. ISOLATION FOREST ML ANOMALY DETECTION:\n"
            f"   - Zone B Anomaly Flag: {'ACTIVE' if self.engine.ml_flags['B'] else 'INACTIVE'}\n"
            f"   - ML Decision Score: {self.engine.ml_scores.get('B', 0.0):+.4f}\n\n"
            "3. RANKED MAINTENANCE WORK ORDERS:\n"
        )
        for r in self.engine.get_maintenance_recommendations():
            report += f"   [Rank {r['rank']}] Work Order {r['work_order_id']} (Zone {r['zone']}, Risk {r['score']:.0f}/100):\n"
            report += f"            Action: {r['action']}\n"
            report += f"            SLA: {r['sla']} | Est Loss: {r['est_loss_l_min']:.1f} L/min\n"

        self.report_preview_text.config(state=tk.NORMAL)
        self.report_preview_text.delete("1.0", tk.END)
        self.report_preview_text.insert(tk.END, report)
        self.report_preview_text.config(state=tk.DISABLED)

    # -------------------------------------------------------------------------
    # EXPORT HELPERS (CSV & PDF)
    # -------------------------------------------------------------------------
    def _quick_export_csv(self) -> None:
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Spreadsheet", "*.csv")],
            initialfile="SmartLeak_Telemetry_Export.csv",
            title="Export Simulation Telemetry CSV"
        )
        if not filepath:
            return
        try:
            if not self.full_history:
                self._record_history_snapshot()
            fieldnames = list(self.full_history[0].keys())
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.full_history)
            messagebox.showinfo("Export Success", f"Successfully exported simulation telemetry to:\n{filepath}")
            self._log_event(f"Exported telemetry CSV to {os.path.basename(filepath)}", "INFO")
        except Exception as err:
            messagebox.showerror("Export Error", f"Could not export CSV: {err}")

    def _quick_export_events_csv(self) -> None:
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Spreadsheet", "*.csv")],
            initialfile="SmartLeak_Alarm_Events.csv",
            title="Export Event History CSV"
        )
        if not filepath:
            return
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Sim_Second", "Severity", "Message"])
                for item in self.raw_event_log:
                    writer.writerow(item)
            messagebox.showinfo("Export Success", f"Successfully exported event log to:\n{filepath}")
            self._log_event(f"Exported event log CSV to {os.path.basename(filepath)}", "INFO")
        except Exception as err:
            messagebox.showerror("Export Error", f"Could not export CSV: {err}")

    def _quick_export_pdf(self) -> None:
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Document", "*.pdf")],
            initialfile="SmartLeak_SCADA_Report.pdf",
            title="Export Comprehensive PDF Engineering Report"
        )
        if not filepath:
            return

        try:
            with PdfPages(filepath) as pdf:
                # Page 1: Executive Summary & SCADA Status
                fig1 = plt.figure(figsize=(8.5, 11), facecolor="#FFFFFF")
                fig1.text(0.1, 0.92, "SMARTLEAK WATER UTILITY SCADA REPORT", fontsize=16, fontweight="bold", color="#0A1624")
                fig1.text(0.1, 0.89, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Sim Time: T+{self.engine.seconds:03d}s", fontsize=9, color="#555555")

                status_txt = (
                    f"System Operating Status: {self.engine.network_status}\n"
                    f"Localized Probable Leak Zone: ZONE {self.engine.probable_zone or 'NONE'}\n"
                    f"Zone B Pressure: {self.engine.measurements['B'].pressure_m:.2f} m H2O (Baseline: {self.engine.calibration.baselines['B'].pressure_m:.2f} m)\n"
                    f"Zone B Core Pipe Flow: {self.engine.measurements['B'].flow_lps:.2f} L/s\n"
                )
                fig1.text(0.1, 0.78, "1. EXECUTIVE SYSTEM SUMMARY", fontsize=12, fontweight="bold", color="#23566A")
                fig1.text(0.1, 0.68, status_txt, fontsize=10, family="monospace")

                fig1.text(0.1, 0.60, "2. RANKED MAINTENANCE WORK ORDERS", fontsize=12, fontweight="bold", color="#23566A")
                wo_y = 0.54
                for r in self.engine.get_maintenance_recommendations():
                    wo_line = f"Rank {r['rank']} | {r['work_order_id']} | Zone {r['zone']} | Risk: {r['score']:.0f}/100\n  Action: {r['action']}\n  SLA: {r['sla']}\n"
                    fig1.text(0.1, wo_y, wo_line, fontsize=8.5, family="monospace")
                    wo_y -= 0.10

                pdf.savefig(fig1)
                plt.close(fig1)

                # Page 2: Digital Twin Schematic Snapshot
                fig2 = plt.figure(figsize=(8.5, 11), facecolor="#FFFFFF")
                ax_pdf = fig2.add_axes([0.1, 0.35, 0.8, 0.55])
                ax_pdf.set_title("WATER NETWORK DIGITAL TWIN SCHEMATIC", fontsize=12, fontweight="bold")
                snapshot = self.engine.snapshot
                node_colors = ["#23536C" if n == "RESERVOIR" else "#d74b50" if n in ("B1", "B2") and self.engine.risk["B"] >= 61 else "#39C98A" for n in self.graph.nodes]
                node_labels = {n: f"{n}\n{snapshot.pressures[n]:.1f}m" if n != "RESERVOIR" else "RES" for n in self.graph.nodes}
                nx.draw_networkx(self.graph, self.positions, ax=ax_pdf, node_color=node_colors, labels=node_labels, font_size=8, font_color="#FFF")
                ax_pdf.axis("off")
                fig2.text(0.1, 0.20, "EPANET Hydraulic Model Simulation Snapshot (Pressure-Dependent Demands)", fontsize=10, style="italic")
                pdf.savefig(fig2)
                plt.close(fig2)

            messagebox.showinfo("Report Success", f"Successfully generated PDF report:\n{filepath}")
            self._log_event(f"Generated PDF engineering report at {os.path.basename(filepath)}", "INFO")
        except Exception as err:
            messagebox.showerror("Report Error", f"Could not generate PDF report: {err}")

    # -------------------------------------------------------------------------
    # APP EVENT LOGGING & SHORTCUT BINDINGS
    # -------------------------------------------------------------------------
    def _bind_keyboard_shortcuts(self) -> None:
        self.root.bind("<space>", lambda e: self._toggle_start_pause())
        self.root.bind("<r>", lambda e: self._reset())
        self.root.bind("<R>", lambda e: self._reset())
        self.root.bind("<l>", lambda e: self._introduce_leak())
        self.root.bind("<L>", lambda e: self._introduce_leak())
        self.root.bind("<a>", lambda e: self._toggle_animations())
        self.root.bind("<A>", lambda e: self._toggle_animations())
        self.root.bind("1", lambda e: self._set_speed(1))
        self.root.bind("2", lambda e: self._set_speed(2))
        self.root.bind("5", lambda e: self._set_speed(5))

    def _toggle_animations(self) -> None:
        self.anim_enabled = not self.anim_enabled
        status_txt = "ON" if self.anim_enabled else "OFF"
        col = self.CYAN if self.anim_enabled else self.MUTED
        bg_col = "#193347" if self.anim_enabled else self.PANEL_ALT
        if hasattr(self, "btn_anim_toggle"):
            self.btn_anim_toggle.config(text=f"✨ FX: {status_txt}", fg=col, bg=bg_col)
        if hasattr(self, "btn_sidebar_anim"):
            self.btn_sidebar_anim.config(text=f"✨  TOGGLE ANIMATIONS ({status_txt})", fg=col)
        self._log_event(f"Visual flow animations set to {status_txt}.", "INFO")
        self.refresh()

    def _toggle_start_pause(self) -> None:
        if self.engine.running:
            self._pause()
        else:
            self._start()

    def _log_event(self, msg: str, tag: str = "INFO") -> None:
        sec = self.engine.seconds
        mins = sec // 60
        secs = sec % 60
        timestamp = f"[{mins:02d}:{secs:02d} | T+{sec:03d}s]"
        clock_str = datetime.now().strftime("%H:%M:%S")
        entry = f"{timestamp} {msg}\n"

        self.raw_event_log.append((clock_str, sec, tag, msg))

        self.alarm_text.config(state=tk.NORMAL)
        self.alarm_text.insert(tk.END, entry, tag)
        self.alarm_text.see(tk.END)
        self.alarm_text.config(state=tk.DISABLED)

    def _check_and_log_events(self) -> None:
        sec = self.engine.seconds
        if sec == LEAK_START_SECONDS and "leak_started" not in self._logged_events:
            self._logged_events.add("leak_started")
            self._log_event("Progressive leak introduced at Zone B (Junction B2).", "ANOMALY")

        for zone in ("A", "B", "C", "D"):
            ml_key = f"ml_{zone}"
            if self.engine.ml_flags[zone] and ml_key not in self._logged_events:
                self._logged_events.add(ml_key)
                self._log_event(f"Isolation Forest ML anomaly detected in Zone {zone}!", "ANOMALY")

            risk = self.engine.risk[zone]
            hr_key = f"high_risk_{zone}"
            if risk >= 61 and hr_key not in self._logged_events:
                self._logged_events.add(hr_key)
                self._log_event(f"HIGH RISK THRESHOLD REACHED in Zone {zone} ({risk:.1f}%)!", "HIGH_RISK")

        if sec >= DEMO_SECONDS and "demo_complete" not in self._logged_events:
            self._logged_events.add("demo_complete")
            priorities = " -> ".join(f"{zone}" for zone, _s, _l in self.engine.priorities())
            self._log_event(f"Simulation run complete. Maintenance order: {priorities}", "NORMAL")

    def _start(self) -> None:
        if not self.engine.running:
            self.engine.start()
            self._log_event("Simulation STARTED.", "NORMAL")
            self.refresh()

    def _pause(self) -> None:
        if self.engine.running:
            self.engine.pause()
            self._log_event("Simulation PAUSED.", "MUTED")
            # Settle gauge needle and bars to exact current simulation values immediately on pause
            max_risk = float(max(self.engine.risk.values(), default=0.0))
            self.current_gauge_angle = math.pi - (max_risk / 100.0) * math.pi
            for z in ("A", "B", "C", "D"):
                self.current_zone_bars[z] = float(self.engine.risk[z])
            focus_zone = self.engine.probable_zone or "B"
            contribs = self.engine.contributions[focus_zone]
            self.current_contrib_bars = [
                float(contribs["persistence"]), float(contribs["ml"]),
                float(contribs["flow"]), float(contribs["pressure"])
            ]
            self.refresh()

    def _reset(self) -> None:
        self.engine.reset()
        self._logged_events.clear()
        self.time_history.clear()
        self.pressure_history.clear()
        self.flow_history.clear()
        self.risk_history.clear()
        self.full_history.clear()
        # Reset visual animation states and settled values
        self.flow_particle_phase = 0.0
        self.leak_pulse_phase = 0.0
        self.current_gauge_angle = math.pi
        self.current_zone_bars = {z: 0.0 for z in ("A", "B", "C", "D")}
        self.current_contrib_bars = [0.0, 0.0, 0.0, 0.0]
        self._append_trend()
        self._log_event("Simulation RESET to initial state.", "MUTED")
        self.refresh()

    def _introduce_leak(self) -> None:
        self.engine.introduce_leak()
        self._append_trend()
        self._log_event("Manual leak injection triggered at t=30s.", "ANOMALY")
        self.refresh()

    def _set_speed(self, speed: int) -> None:
        self.engine.set_speed(speed)
        self._log_event(f"Simulation speed set to {speed}x.", "INFO")
        self.refresh()

    def _start_animation_loop(self) -> None:
        if self._anim_timer_id is not None:
            try:
                self.root.after_cancel(self._anim_timer_id)
            except Exception:
                pass
            self._anim_timer_id = None
        self._last_sec_tick = float(self.engine.seconds)
        import time
        self._last_wall_time = time.time()
        self._on_anim_frame()

    def _on_anim_frame(self) -> None:
        """High-efficiency decoupled animation & simulation pump."""
        import time
        now = time.time()
        dt_wall = min(0.1, max(0.001, now - self._last_wall_time))
        self._last_wall_time = now

        need_full_refresh = False

        if self.engine.running:
            # Advance simulation step in sync with chosen speed
            self._last_sec_tick += dt_wall * float(self.engine.speed)
            steps_to_take = int(self._last_sec_tick - self.engine.seconds)
            if steps_to_take > 0:
                for _ in range(steps_to_take):
                    if self.engine.tick():
                        self._append_trend()
                        self._check_and_log_events()
                need_full_refresh = True

            # Advance visual phases proportionally to simulation speed
            speed_mult = float(self.engine.speed)
            self.flow_particle_phase = (self.flow_particle_phase + dt_wall * 1.5 * speed_mult) % 1.0
            self.leak_pulse_phase = (self.leak_pulse_phase + dt_wall * 4.0 * speed_mult) % (2.0 * math.pi)

        if need_full_refresh:
            self.refresh()
        elif self.anim_enabled and self.engine.running:
            # Intermediate micro-animation frame: update particle and needle graphics smoothly
            if self.active_view_name == "Overview":
                self._draw_overview_network()
                self._draw_overview_diagnostics()
                self.canvas_ov_network.draw_idle()
                self.canvas_ov_diag.draw_idle()
            elif self.active_view_name == "Digital Twin":
                self._draw_digital_twin_map()
                self.canvas_twin.draw_idle()

        if not self.root.winfo_exists():
            return

        # Schedule next animation frame at ~40ms (25fps, light CPU load, no flickering)
        self._anim_timer_id = self.root.after(40, self._on_anim_frame)

    def _on_close(self) -> None:
        if self._anim_timer_id is not None:
            try:
                self.root.after_cancel(self._anim_timer_id)
            except Exception:
                pass
            self._anim_timer_id = None
        self.root.destroy()

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    @staticmethod
    def _style_panel(axis: Axes) -> None:
        axis.set_facecolor(SmartLeakApp.PANEL)
        axis.patch.set_edgecolor(SmartLeakApp.BORDER)
        axis.patch.set_linewidth(0.9)
        for spine in axis.spines.values():
            spine.set_color(SmartLeakApp.BORDER)
            spine.set_linewidth(0.8)

    def _style_chart(
        self,
        axis: Axes,
        title: str,
        ylabel: str,
        color: str,
    ) -> None:
        self._style_panel(axis)
        axis.set_title(
            title,
            loc="left",
            color=self.TEXT,
            fontsize=8.5,
            fontweight="bold",
            pad=6,
        )
        axis.set_xlabel("Simulation time (s)", fontsize=7.0, color=self.MUTED)
        axis.set_ylabel(ylabel, fontsize=7.0, color=self.MUTED)
        axis.set_xlim(0, DEMO_SECONDS)
        axis.set_xticks((0, 30, 60, 90, 120, 150))
        axis.grid(True, color=self.BORDER, linewidth=0.7, alpha=0.8)
        axis.tick_params(colors=self.MUTED, labelsize=7.0, length=3)
        axis.margins(x=0.02, y=0.15)

    def _append_trend(self) -> None:
        zone_b = self.engine.measurements["B"]
        self.time_history.append(self.engine.seconds)
        self.pressure_history.append(zone_b.pressure_m)
        self.flow_history.append(zone_b.flow_lps)
        self.risk_history.append(self.engine.risk["B"])

    @staticmethod
    def _phase_color(status: str) -> str:
        return SmartLeakApp.STATUS_COLORS.get(
            status,
            SmartLeakApp.MUTED,
        )

    def _zone_state(self, zone: str) -> str:
        score = self.engine.risk[zone]
        if score >= 61:
            return "HIGH RISK"
        if score >= 31:
            return "WARNING"
        if self.engine.anomaly_flags[zone]:
            return "ANOMALY DETECTED"
        return "NORMAL"

    def _update_chart(
        self,
        axis: Axes,
        line: Any,
        values: list[float],
    ) -> None:
        line.set_data(self.time_history, values)
        axis.set_xlim(0, DEMO_SECONDS)
        axis.relim()
        axis.autoscale_view(scalex=False, scaley=True)


SmartLeakWindow = SmartLeakApp


# -----------------------------------------------------------------------------
# SELF TEST & ENTRY POINTS
# -----------------------------------------------------------------------------
def run_self_test() -> None:
    """Exercise WNTR, scoring, controls, reset, and three deterministic runs."""
    print("SmartLeak self-test: calculating WNTR network scenarios...")
    calibration = HydraulicCalibration()
    baseline = dict(calibration.scenarios)[0.0]
    full_leak = dict(calibration.scenarios)[1.0]
    b_base = zone_measurement(baseline, "B")
    b_leak = zone_measurement(full_leak, "B")
    assert b_leak.flow_lps > b_base.flow_lps
    assert full_leak.pressures["B2"] < baseline.pressures["B2"]
    assert len(full_leak.pressures) == 12

    engine = SmartLeakSimulation(calibration)
    engine.start()
    assert engine.tick() and engine.seconds == 1
    engine.pause()
    assert not engine.tick() and engine.seconds == 1
    engine.set_speed(2)
    engine.start()
    assert engine.tick() and engine.seconds == 3
    engine.reset()
    assert engine.seconds == 0 and not engine.running and engine.speed == 1
    engine.introduce_leak()
    assert engine.seconds == LEAK_START_SECONDS + 1 and engine.running
    assert engine.leak_severity > 0

    final_runs: list[tuple[tuple[str, float], ...]] = []
    for run_number in range(1, 4):
        engine.reset()
        engine.start()
        high_risk_at_120 = False
        ml_detected_by_120 = False
        while engine.running:
            engine.tick()
            if engine.seconds == 90:
                ml_detected_by_120 = engine.ml_flags["B"]
            if engine.seconds == 120:
                high_risk_at_120 = engine.risk["B"] >= 61
        assert ml_detected_by_120, (
            f"Run {run_number}: Isolation Forest did not flag Zone B by 90 seconds."
        )
        assert high_risk_at_120, (
            f"Run {run_number}: Zone B did not reach high risk by 120 seconds."
        )
        zone_b_risk = engine.risk["B"]
        assert zone_b_risk >= 61, (
            f"Run {run_number}: expected high Zone B risk, got {zone_b_risk:.1f}%."
        )
        assert engine.ml_flags["B"], (
            f"Run {run_number}: Isolation Forest did not flag Zone B."
        )
        assert engine.probable_zone == "B", (
            f"Run {run_number}: expected Zone B to be the probable leak zone."
        )
        priorities = engine.priorities()
        assert [(zone, label) for zone, _score, label in priorities] == [
            ("B", "CRITICAL"),
            ("D", "HIGH"),
            ("C", "MEDIUM"),
            ("A", "LOW"),
        ], (
            f"Run {run_number}: unexpected risk-based maintenance priority "
            f"{priorities!r}."
        )
        final_runs.append(tuple(
            (zone, round(engine.risk[zone], 6))
            for zone in sorted(engine.risk)
        ))
        print(
            f"  Run {run_number}: {engine.seconds}s, "
            f"Zone B pressure {engine.measurements['B'].pressure_m:.2f} m, "
            f"flow {engine.measurements['B'].flow_lps:.2f} L/s, "
            f"risk {zone_b_risk:.1f}%"
        )
    assert final_runs[0] == final_runs[1] == final_runs[2], (
        "Reset did not reproduce the same end-of-demo results."
    )
    print("PASS: hydraulic leak response, pause, speed, reset, ML, and repeatability.")


def show_application() -> None:
    _load_desktop_dependencies()
    print("Calculating the deterministic WNTR / EPANET hydraulic scenarios...")
    calibration = HydraulicCalibration()
    engine = SmartLeakSimulation(calibration)
    app = SmartLeakApp(engine)
    print("SmartLeak SCADA platform is ready.")
    app.run()


def _load_desktop_dependencies() -> None:
    """Load GUI-only packages when the desktop application is launched."""
    global tk, ttk, filedialog, messagebox
    global plt, Axes, PdfPages, Line2D, FigureCanvasTkAgg
    global Circle, Polygon, Rectangle

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon, Rectangle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the SmartLeak water-network desktop simulation."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run hydraulic and deterministic-control checks without a GUI",
    )
    arguments = parser.parse_args()
    try:
        if arguments.self_test:
            run_self_test()
        else:
            show_application()
    except (AssertionError, RuntimeError, ValueError) as error:
        print(f"SmartLeak could not start: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
