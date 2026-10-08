#!/usr/bin/env python3
"""SmartLeak deterministic water-network leak detection classroom demo."""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import wntr
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from sklearn.ensemble import IsolationForest


APP_TITLE = "SMARTLEAK  |  INTELLIGENT WATER NETWORK SIMULATION"
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
    "A": ("A1",),
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


@dataclass(frozen=True)
class HydraulicSnapshot:
    """Pressure and flow measurements returned by one WNTR simulation step."""

    pressures: dict[str, float]
    flows_lps: dict[str, float]


@dataclass(frozen=True)
class ZoneMeasurement:
    pressure_m: float
    flow_lps: float


def build_network(
    leak_fraction: float,
    demand_pattern: tuple[float, ...],
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
    for junction, elevation in elevations.items():
        network.add_junction(
            junction,
            base_demand=0.00016,
            demand_pattern="CLASSROOM_DEMAND",
            elevation=elevation,
        )

    if leak_fraction > 0:
        network.get_node("B2").emitter_coefficient = (
            MAX_EMITTER_COEFFICIENT * leak_fraction
        )

    # The A→B→C→D trunk lets a Zone B leak affect both its local readings
    # and downstream pressure. Each zone also has a two-junction branch.
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
) -> list[HydraulicSnapshot]:
    """Run EPANET hydraulics and convert its Pandas results into snapshots."""
    network = build_network(leak_fraction, demand_pattern)
    result = wntr.sim.EpanetSimulator(network).run_sim()
    pressure_frame: pd.DataFrame = result.node["pressure"]
    flow_frame: pd.DataFrame = result.link["flowrate"]

    snapshots: list[HydraulicSnapshot] = []
    for time_index in pressure_frame.index:
        pressures = {
            name: float(pressure_frame.loc[time_index, name])
            for name in pressure_frame.columns
            if name != "RESERVOIR"
        }
        flows_lps = {
            name: abs(float(flow_frame.loc[time_index, name])) * 1000.0
            for name in flow_frame.columns
        }
        snapshots.append(HydraulicSnapshot(pressures, flows_lps))
    if not snapshots:
        raise RuntimeError("WNTR returned no hydraulic measurements.")
    return snapshots


def zone_measurement(
    snapshot: HydraulicSnapshot,
    zone: str,
) -> ZoneMeasurement:
    return ZoneMeasurement(
        pressure_m=float(np.mean([
            snapshot.pressures[node] for node in ZONE_PRESSURE_NODES[zone]
        ])),
        flow_lps=snapshot.flows_lps[ZONE_FLOW_PIPES[zone]],
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
            # Tolerances account for ordinary demand-profile movement before
            # converting a change into a 0–1 weighted risk contribution.
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

    def snapshot_at(self, severity: float) -> HydraulicSnapshot:
        """Interpolate between WNTR-computed leakage cases."""
        severity = float(np.clip(severity, 0.0, 1.0))
        position = severity * (len(self.scenarios) - 1)
        lower_index = int(math.floor(position))
        upper_index = min(lower_index + 1, len(self.scenarios) - 1)
        weight = position - lower_index
        lower = self.scenarios[lower_index][1]
        upper = self.scenarios[upper_index][1]
        pressures = {
            node: lower.pressures[node]
            + weight * (upper.pressures[node] - lower.pressures[node])
            for node in lower.pressures
        }
        flows_lps = {
            pipe: lower.flows_lps[pipe]
            + weight * (upper.flows_lps[pipe] - lower.flows_lps[pipe])
            for pipe in lower.flows_lps
        }
        return HydraulicSnapshot(pressures, flows_lps)


class SmartLeakSimulation:
    """Deterministic clock, live hydraulic readings, and explainable risk."""

    def __init__(self, calibration: HydraulicCalibration) -> None:
        self.calibration = calibration
        self.reset()

    def reset(self) -> None:
        self.seconds = 0
        self.speed = 1
        self.running = False
        self.snapshot = self.calibration.baseline_snapshot
        self.measurements: dict[str, ZoneMeasurement] = {}
        self.risk: dict[str, float] = {}
        self.contributions: dict[str, dict[str, float]] = {}
        self.ml_flags: dict[str, bool] = {}
        self.anomaly_flags: dict[str, bool] = {}
        self.histories: dict[str, deque[ZoneMeasurement]] = {
            zone: deque(maxlen=12) for zone in ZONE_NODES
        }
        self.persistence: dict[str, deque[bool]] = {
            zone: deque(maxlen=8) for zone in ZONE_NODES
        }
        self._analyze_current_snapshot()

    @property
    def leak_severity(self) -> float:
        if self.seconds <= LEAK_START_SECONDS:
            return 0.0
        if self.seconds < 60:
            return 0.04 * (self.seconds - 30) / 30
        if self.seconds < 90:
            return 0.04 + 0.11 * (self.seconds - 60) / 30
        if self.seconds < 120:
            return 0.15 + 0.47 * (self.seconds - 90) / 30
        return 0.62 + 0.38 * (self.seconds - 120) / 30

    @property
    def phase(self) -> str:
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

    def set_speed(self, speed: int) -> None:
        if speed not in (1, 2, 5):
            raise ValueError("Simulation speed must be 1×, 2×, or 5×.")
        self.speed = speed

    def start(self) -> None:
        if self.seconds < DEMO_SECONDS:
            self.running = True

    def pause(self) -> None:
        self.running = False

    def introduce_leak(self) -> None:
        self.seconds = max(self.seconds, LEAK_START_SECONDS)
        self.snapshot = self.calibration.snapshot_at(self.leak_severity)
        self._analyze_current_snapshot()
        self.start()

    def tick(self) -> bool:
        if not self.running:
            return False
        self.seconds = min(DEMO_SECONDS, self.seconds + self.speed)
        self.snapshot = self.calibration.snapshot_at(self.leak_severity)
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


class SmartLeakWindow:
    """Single interactive Matplotlib window for live classroom presentation."""

    BG = "#07111B"
    PANEL = "#101E2A"
    PANEL_ALT = "#142634"
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
        self.engine = engine
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
        self.time_history: list[int] = []
        self.pressure_history: list[float] = []
        self.flow_history: list[float] = []
        self.risk_history: list[float] = []
        self._make_figure()
        self._append_trend()
        self.timer = self.figure.canvas.new_timer(interval=1000)
        self.timer.add_callback(self._on_timer)
        self.timer.start()
        self.refresh()

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

    def _make_figure(self) -> None:
        self.figure = plt.figure(
            figsize=(16, 10),
            facecolor=self.BG,
            num=APP_TITLE,
        )
        self.figure.canvas.manager.set_window_title("SmartLeak Classroom Demo")
        self.ax_header = self.figure.add_axes([0.035, 0.915, 0.93, 0.07])
        self.ax_header.set_facecolor(self.BG)
        self.ax_header.axis("off")

        card_gap = 0.012
        card_width = (0.93 - card_gap * 4) / 5
        self.kpi_axes = [
            self.figure.add_axes([
                0.035 + index * (card_width + card_gap),
                0.795,
                card_width,
                0.105,
            ])
            for index in range(5)
        ]
        self.ax_network = self.figure.add_axes([0.035, 0.415, 0.60, 0.36])
        self.ax_analysis = self.figure.add_axes([0.652, 0.415, 0.313, 0.36])
        self.ax_alert = self.figure.add_axes([0.035, 0.295, 0.93, 0.10])
        for axis in (self.ax_network, self.ax_analysis, self.ax_alert):
            self._style_panel(axis)

        chart_gap = 0.018
        chart_width = (0.93 - chart_gap * 2) / 3
        chart_axes = [
            self.figure.add_axes([
                0.035 + index * (chart_width + chart_gap),
                0.075,
                chart_width,
                0.195,
            ])
            for index in range(3)
        ]
        self.ax_pressure, self.ax_flow, self.ax_risk = chart_axes
        self._style_chart(
            self.ax_pressure,
            "PRESSURE  ·  ZONE B",
            "Pressure (m)",
            self.CYAN,
        )
        self._style_chart(
            self.ax_flow,
            "PIPE FLOW  ·  ZONE B",
            "Flow (L/s)",
            self.BLUE,
        )
        self._style_chart(
            self.ax_risk,
            "LEAK RISK  ·  ZONE B",
            "Risk score",
            self.RED,
        )
        self.pressure_line, = self.ax_pressure.plot(
            [], [], color=self.CYAN, linewidth=2.2, solid_capstyle="round",
        )
        self.flow_line, = self.ax_flow.plot(
            [], [], color=self.BLUE, linewidth=2.2, solid_capstyle="round",
        )
        self.risk_line, = self.ax_risk.plot(
            [], [], color=self.RED, linewidth=2.2, solid_capstyle="round",
        )
        self.ax_pressure.axhline(
            self.engine.calibration.baselines["B"].pressure_m,
            color=self.MUTED,
            linestyle=(0, (4, 4)),
            linewidth=0.9,
            alpha=0.65,
        )
        self.ax_flow.axhline(
            self.engine.calibration.baselines["B"].flow_lps,
            color=self.MUTED,
            linestyle=(0, (4, 4)),
            linewidth=0.9,
            alpha=0.65,
        )
        self.ax_risk.axhline(
            61,
            color=self.AMBER,
            linestyle=(0, (4, 4)),
            linewidth=0.9,
            alpha=0.75,
        )
        self.ax_risk.set_ylim(0, 100)
        self._make_controls()

    @staticmethod
    def _style_panel(axis: Axes) -> None:
        axis.set_facecolor(SmartLeakWindow.PANEL)
        axis.patch.set_edgecolor(SmartLeakWindow.BORDER)
        axis.patch.set_linewidth(0.9)
        for spine in axis.spines.values():
            spine.set_color(SmartLeakWindow.BORDER)
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
            fontsize=9,
            fontweight="bold",
            pad=8,
        )
        axis.set_xlabel("Simulation time (s)", fontsize=7.5, color=self.MUTED)
        axis.set_ylabel(ylabel, fontsize=7.5, color=self.MUTED)
        axis.set_xlim(0, DEMO_SECONDS)
        axis.set_xticks((0, 30, 60, 90, 120, 150))
        axis.grid(True, color=self.BORDER, linewidth=0.7, alpha=0.8)
        axis.tick_params(colors=self.MUTED, labelsize=7.5, length=3)
        axis.margins(x=0.02, y=0.15)

    def _make_controls(self) -> None:
        self.ax_controls = self.figure.add_axes([0.035, 0.012, 0.93, 0.05])
        self._style_panel(self.ax_controls)
        self.ax_controls.set_xticks([])
        self.ax_controls.set_yticks([])
        self.ax_controls.text(
            0.99,
            0.5,
            "",
            transform=self.ax_controls.transAxes,
            fontsize=8.5,
            color=self.MUTED,
            ha="right",
            va="center",
        )
        self.control_status = self.ax_controls.texts[0]

        control_specs = [
            (0.008, 0.15, 0.155, "START SIMULATION", self._start, self.GREEN),
            (0.170, 0.15, 0.092, "PAUSE", self._pause, self.PANEL_ALT),
            (0.269, 0.15, 0.092, "RESET", self._reset, self.PANEL_ALT),
            (
                0.370, 0.15, 0.158, "INTRODUCE LEAK",
                self._introduce_leak, "#6D5225",
            ),
            (0.548, 0.15, 0.075, "1× SPEED", lambda _: self._set_speed(1), self.PANEL_ALT),
            (0.630, 0.15, 0.075, "2× SPEED", lambda _: self._set_speed(2), self.PANEL_ALT),
            (0.712, 0.15, 0.075, "5× SPEED", lambda _: self._set_speed(5), self.PANEL_ALT),
        ]
        self.buttons: list[Button] = []
        self.speed_buttons: dict[int, Button] = {}
        control_position = self.ax_controls.get_position()
        for x, y, width, label, callback, color in control_specs:
            axis = self.figure.add_axes([
                control_position.x0 + x * control_position.width,
                control_position.y0 + y * control_position.height,
                width * control_position.width,
                0.70 * control_position.height,
            ])
            button = Button(
                axis,
                label,
                color=color,
                hovercolor="#294454",
            )
            button.ax.set_facecolor(color)
            button.ax.patch.set_edgecolor(self.BORDER)
            button.ax.patch.set_linewidth(0.7)
            button.label.set_color(self.BG if label == "START SIMULATION" else self.TEXT)
            button.label.set_fontsize(8.2)
            button.on_clicked(callback)
            self.buttons.append(button)
            if label.endswith("SPEED"):
                self.speed_buttons[int(label[0])] = button

    def _start(self, _event: Any) -> None:
        self.engine.start()
        self.refresh()

    def _pause(self, _event: Any) -> None:
        self.engine.pause()
        self.refresh()

    def _reset(self, _event: Any) -> None:
        self.engine.reset()
        self.time_history.clear()
        self.pressure_history.clear()
        self.flow_history.clear()
        self.risk_history.clear()
        self._append_trend()
        self.refresh()

    def _introduce_leak(self, _event: Any) -> None:
        self.engine.introduce_leak()
        self._append_trend()
        self.refresh()

    def _set_speed(self, speed: int) -> None:
        self.engine.set_speed(speed)
        self.refresh()

    def _on_timer(self) -> bool:
        if self.engine.tick():
            self._append_trend()
            self.refresh()
        return True

    def _append_trend(self) -> None:
        zone_b = self.engine.measurements["B"]
        self.time_history.append(self.engine.seconds)
        self.pressure_history.append(zone_b.pressure_m)
        self.flow_history.append(zone_b.flow_lps)
        self.risk_history.append(self.engine.risk["B"])

    @staticmethod
    def _phase_color(status: str) -> str:
        return SmartLeakWindow.STATUS_COLORS.get(
            status,
            SmartLeakWindow.MUTED,
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

    def _draw_kpis(self) -> None:
        mean_pressure = float(np.mean(list(self.engine.snapshot.pressures.values())))
        total_flow = self.engine.snapshot.flows_lps["P_SUPPLY"]
        highest_risk = max(self.engine.risk.values(), default=0.0)
        probable_zone = self.engine.probable_zone
        status = self.engine.network_status
        metrics = (
            ("SYSTEM STATUS", status, "Live network state", self._phase_color(status)),
            ("NETWORK PRESSURE", f"{mean_pressure:.1f} m", "Mean · 12 junctions", self.CYAN),
            ("TOTAL FLOW", f"{total_flow:.2f} L/s", "Reservoir supply", self.BLUE),
            ("LEAK RISK", f"{highest_risk:.0f} / 100", "Highest zone score", self._phase_color(
                "HIGH RISK" if highest_risk >= 61 else
                "WARNING" if highest_risk >= 31 else "NORMAL"
            )),
            (
                "PROBABLE LEAK ZONE",
                f"ZONE {probable_zone}" if probable_zone else "PENDING",
                "Inspection estimate only",
                self._phase_color(
                    self._zone_state(probable_zone) if probable_zone else "NORMAL"
                ),
            ),
        )
        for axis, (label, value, detail, accent) in zip(self.kpi_axes, metrics):
            axis.clear()
            self._style_panel(axis)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.text(
                0.055, 0.80, label,
                transform=axis.transAxes,
                fontsize=7.7,
                fontweight="bold",
                color=self.MUTED,
                va="center",
            )
            value_size = 13 if len(value) > 14 else 16
            axis.text(
                0.055, 0.47, value,
                transform=axis.transAxes,
                fontsize=value_size,
                fontweight="bold",
                color=accent,
                va="center",
            )
            axis.text(
                0.055, 0.15, detail,
                transform=axis.transAxes,
                fontsize=7.4,
                color=self.MUTED,
                va="center",
            )

    def _draw_network(self) -> None:
        axis = self.ax_network
        axis.clear()
        self._style_panel(axis)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(
            0.025, 0.96, "LIVE WATER NETWORK",
            transform=axis.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color=self.TEXT,
            va="top",
        )
        axis.text(
            0.975, 0.96, "JUNCTION LABELS: PRESSURE (m)  ·  PIPE LABELS: FLOW (L/s)",
            transform=axis.transAxes,
            fontsize=6.8,
            color=self.MUTED,
            ha="right",
            va="top",
        )

        zone_bounds = (
            ("A", -0.35, 1.65),
            ("B", 1.65, 3.65),
            ("C", 3.65, 5.65),
            ("D", 5.65, 7.55),
        )
        lane_colors = ("#153044", "#183243", "#163447", "#173044")
        for (zone, left, right), lane_color in zip(zone_bounds, lane_colors):
            axis.axvspan(
                left, right, ymin=0.12, ymax=0.86,
                facecolor=lane_color, alpha=0.63, zorder=0,
            )
            axis.axvline(
                right, ymin=0.12, ymax=0.86,
                color=self.BORDER, linewidth=0.8,
                linestyle=(0, (3, 4)), zorder=1,
            )
            axis.text(
                (left + right) / 2, 0.49, f"ZONE {zone}",
                ha="center",
                va="center",
                fontsize=7.6,
                fontweight="bold",
                color=self._phase_color(self._zone_state(zone)),
                alpha=0.95,
                zorder=1,
            )

        snapshot = self.engine.snapshot
        edge_colors = []
        edge_widths = []
        for _, _, data in self.graph.edges(data=True):
            zone = data["zone"]
            state = self._zone_state(zone)
            edge_colors.append(self._phase_color(state))
            is_highlighted = zone == self.engine.probable_zone and state in (
                "WARNING", "HIGH RISK", "ANOMALY DETECTED",
            )
            edge_widths.append(3.2 if is_highlighted else 1.5)
        node_colors = []
        node_labels = {}
        for node in self.graph.nodes:
            if node == "RESERVOIR":
                node_colors.append("#23536C")
                node_labels[node] = "RES\n79.0m"
            else:
                zone = self.node_zones[node]
                state = self._zone_state(zone)
                node_colors.append(
                    self._phase_color(state)
                    if state != "NORMAL"
                    else "#31576A"
                )
                node_labels[node] = f"{node}\n{snapshot.pressures[node]:.1f} m"
        nx.draw_networkx_edges(
            self.graph,
            self.positions,
            ax=axis,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            connectionstyle="arc3,rad=0.0",
            min_source_margin=15,
            min_target_margin=17,
        )
        nx.draw_networkx_nodes(
            self.graph,
            self.positions,
            ax=axis,
            node_color=node_colors,
            node_size=780,
            edgecolors=self.BORDER,
            linewidths=1.3,
        )
        nx.draw_networkx_labels(
            self.graph,
            self.positions,
            labels=node_labels,
            ax=axis,
            font_size=7.1,
            font_color="#F7FBFD",
            font_weight="bold",
        )
        edge_labels = {
            (start, end): f"{snapshot.flows_lps[data['pipe']]:.2f}"
            for start, end, data in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.graph,
            self.positions,
            edge_labels=edge_labels,
            ax=axis,
            font_size=6.8,
            font_color=self.TEXT,
            label_pos=0.48,
            rotate=False,
            bbox={
                "boxstyle": "round,pad=0.12",
                "fc": self.BG,
                "ec": "none",
                "alpha": 0.95,
            },
        )
        axis.set_xlim(-1.30, 7.62)
        axis.set_ylim(-1.03, 0.72)
        axis.axis("off")

    def _draw_analysis(self) -> None:
        axis = self.ax_analysis
        axis.clear()
        self._style_panel(axis)
        axis.set_xticks([])
        axis.set_yticks([])
        zone = self.engine.probable_zone or max(
            self.engine.risk,
            key=self.engine.risk.get,
        )
        score = self.engine.risk[zone]
        parts = self.engine.contributions[zone]
        axis.text(
            0.035, 0.95, "LIVE ANALYSIS",
            transform=axis.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color=self.TEXT,
            va="top",
        )
        axis.text(
            0.965, 0.95, f"ZONE {zone}  ·  {self._zone_state(zone)}",
            transform=axis.transAxes,
            fontsize=7.8,
            color=self._phase_color(self._zone_state(zone)),
            fontweight="bold",
            ha="right",
            va="top",
        )

        analysis_rows = (
            ("Pressure anomaly", "pressure", RISK_WEIGHTS["pressure"],
             parts["pressure"] >= RISK_WEIGHTS["pressure"] * 0.20),
            ("Flow anomaly", "flow", RISK_WEIGHTS["flow"],
             parts["flow"] >= RISK_WEIGHTS["flow"] * 0.20),
            ("ML anomaly", "ml", RISK_WEIGHTS["ml"],
             self.engine.ml_flags[zone]),
            ("Persistence", "persistence", RISK_WEIGHTS["persistence"],
             parts["persistence"] >= 0.5),
        )
        row_positions = (0.73, 0.575, 0.42, 0.265)
        for (label, key, weight, active), y in zip(
            analysis_rows, row_positions,
        ):
            value = parts[key]
            bar_color = (
                self._phase_color("HIGH RISK")
                if active and score >= 61
                else self._phase_color("WARNING")
                if active and score >= 31
                else self.CYAN
                if active
                else self.GREEN
            )
            axis.text(
                0.035, y, label,
                transform=axis.transAxes,
                fontsize=8.2,
                color=self.TEXT,
                va="center",
            )
            axis.text(
                0.72, y, f"+{value:.1f} / {weight:.0f}",
                transform=axis.transAxes,
                fontsize=7.8,
                color=self.MUTED,
                family="monospace",
                ha="right",
                va="center",
            )
            axis.text(
                0.965, y,
                "ACTIVE" if active else "NORMAL",
                transform=axis.transAxes,
                fontsize=7.2,
                fontweight="bold",
                color=bar_color,
                ha="right",
                va="center",
            )
            bar_y = y - 0.067
            axis.add_patch(Rectangle(
                (0.035, bar_y),
                0.93,
                0.026,
                transform=axis.transAxes,
                facecolor=self.TRACK,
                edgecolor="none",
                zorder=2,
            ))
            axis.add_patch(Rectangle(
                (0.035, bar_y),
                0.93 * min(1.0, value / weight),
                0.026,
                transform=axis.transAxes,
                facecolor=bar_color,
                edgecolor="none",
                zorder=3,
            ))

        axis.text(
            0.035, 0.075, "TOTAL LEAK RISK",
            transform=axis.transAxes,
            fontsize=8.0,
            color=self.MUTED,
            fontweight="bold",
            va="center",
        )
        axis.text(
            0.965, 0.075, f"{score:.0f} / 100",
            transform=axis.transAxes,
            fontsize=12.5,
            color=self._phase_color(
                "HIGH RISK" if score >= 61 else
                "WARNING" if score >= 31 else "NORMAL"
            ),
            fontweight="bold",
            ha="right",
            va="center",
        )
        axis.add_patch(Rectangle(
            (0.035, 0.025),
            0.93,
            0.022,
            transform=axis.transAxes,
            facecolor=self.TRACK,
            edgecolor="none",
            zorder=2,
        ))
        axis.add_patch(Rectangle(
            (0.035, 0.025),
            0.93 * min(1.0, score / 100),
            0.022,
            transform=axis.transAxes,
            facecolor=self._phase_color(
                "HIGH RISK" if score >= 61 else
                "WARNING" if score >= 31 else "NORMAL"
            ),
            edgecolor="none",
            zorder=3,
        ))

    def _draw_alert(self) -> None:
        axis = self.ax_alert
        axis.clear()
        self._style_panel(axis)
        axis.set_xticks([])
        axis.set_yticks([])
        max_risk = max(self.engine.risk.values(), default=0.0)
        anomaly_detected = any(self.engine.anomaly_flags.values())
        probable_zone = self.engine.probable_zone
        if max_risk >= 61:
            severity = "HIGH"
            severity_color = self.RED
        elif max_risk >= 31:
            severity = "MODERATE"
            severity_color = self.AMBER
        elif anomaly_detected:
            severity = "EARLY SIGNAL"
            severity_color = self.CYAN
        else:
            severity = "NORMAL"
            severity_color = self.GREEN

        heading = "ANOMALY DETECTED" if anomaly_detected else "SYSTEM NORMAL"
        zone_text = f"ZONE {probable_zone}" if probable_zone else "PENDING"
        if max_risk >= 61 and probable_zone:
            recommendation = f"Inspect the Zone {probable_zone} pipeline; verify in field."
            note = "Probable zone only; not a confirmed physical leak."
        elif anomaly_detected:
            recommendation = "Continue monitoring; inspect if risk continues to rise."
            note = "The anomaly score is a simulation signal, not a confirmed leak."
        else:
            recommendation = "No inspection recommended while readings remain normal."
            note = "Monitoring all four zones."
        if self.engine.seconds >= DEMO_SECONDS:
            priorities = "  >  ".join(
                f"{label} {zone}"
                for zone, _score, label in self.engine.priorities()
            )
            note = f"MAINTENANCE PRIORITY  {priorities}"

        axis.text(
            0.025, 0.68, heading,
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=severity_color,
            va="center",
        )
        axis.text(
            0.025, 0.28, f"Severity  ·  {severity}  ·  {max_risk:.0f}/100",
            transform=axis.transAxes,
            fontsize=8.2,
            color=self.TEXT,
            va="center",
        )
        axis.text(
            0.385, 0.68, "PROBABLE LEAK ZONE",
            transform=axis.transAxes,
            fontsize=7.6,
            fontweight="bold",
            color=self.MUTED,
            va="center",
        )
        axis.text(
            0.385, 0.28, zone_text,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            color=severity_color if probable_zone else self.MUTED,
            va="center",
        )
        axis.text(
            0.655, 0.68, "RECOMMENDED ACTION",
            transform=axis.transAxes,
            fontsize=7.6,
            fontweight="bold",
            color=self.MUTED,
            va="center",
        )
        axis.text(
            0.655, 0.40, recommendation,
            transform=axis.transAxes,
            fontsize=8.2,
            fontweight="bold",
            color=self.TEXT,
            va="center",
        )
        if self.engine.seconds >= DEMO_SECONDS:
            priorities = "  →  ".join(
                f"{label} ZONE {zone}"
                for zone, _score, label in self.engine.priorities()
            )
            axis.text(
                0.025, 0.075, f"MAINTENANCE PRIORITY  ·  {priorities}",
                transform=axis.transAxes,
                fontsize=8.8,
                fontweight="bold",
                color=self.CYAN,
                va="center",
            )
        else:
            axis.text(
                0.655, 0.15, note,
                transform=axis.transAxes,
                fontsize=7.8,
                color=self.MUTED,
                va="center",
            )

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

    def refresh(self) -> None:
        self.ax_header.clear()
        self.ax_header.set_facecolor(self.BG)
        self.ax_header.axis("off")
        self.ax_header.text(
            0.0, 0.88, "SMARTLEAK",
            transform=self.ax_header.transAxes,
            fontsize=21,
            fontweight="bold",
            color=self.TEXT,
            va="top",
        )
        self.ax_header.text(
            0.002, 0.10, "Intelligent Water Network Monitoring",
            transform=self.ax_header.transAxes,
            fontsize=10.5,
            color=self.MUTED,
            va="bottom",
        )
        self.ax_header.text(
            1.0, 0.68,
            "WNTR / EPANET     ·     ISOLATION FOREST     ·     DETERMINISTIC DEMO",
            transform=self.ax_header.transAxes,
            fontsize=8.3,
            color=self.CYAN,
            ha="right",
            va="top",
        )
        self._draw_kpis()
        self._draw_network()
        self._draw_analysis()
        self._draw_alert()
        self._update_chart(self.ax_pressure, self.pressure_line, self.pressure_history)
        self._update_chart(self.ax_flow, self.flow_line, self.flow_history)
        self._update_chart(self.ax_risk, self.risk_line, self.risk_history)

        state = "RUNNING" if self.engine.running else "PAUSED"
        if self.engine.seconds >= DEMO_SECONDS:
            state = "DEMO COMPLETE"
        self.control_status.set_text(
            f"{state}   ·   {self.engine.phase}   ·   "
            f"{self.engine.seconds:03d}/{DEMO_SECONDS}s   ·   "
            f"{self.engine.speed}× SPEED"
        )
        for speed, button in self.speed_buttons.items():
            selected = speed == self.engine.speed
            button.ax.set_facecolor("#23566A" if selected else self.PANEL_ALT)
            button.label.set_color(self.CYAN if selected else self.TEXT)
        self.figure.canvas.draw_idle()


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
    assert engine.seconds == LEAK_START_SECONDS and engine.running

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
    backend = matplotlib.get_backend().lower()
    if backend == "agg" or "inline" in backend:
        raise RuntimeError(
            "No interactive Matplotlib desktop backend is available. "
            "Run this application on a computer with a graphical desktop "
            "and a Tk or Qt Matplotlib backend. Use `python main.py --self-test` "
            "for a headless functional check."
        )
    print("Calculating the deterministic WNTR / EPANET hydraulic scenarios...")
    calibration = HydraulicCalibration()
    engine = SmartLeakSimulation(calibration)
    SmartLeakWindow(engine)
    print("SmartLeak is ready. Press START SIMULATION in the window.")
    plt.show()


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
