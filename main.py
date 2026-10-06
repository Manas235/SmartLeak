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
            facecolor="#f3f6f7",
            num=APP_TITLE,
        )
        self.figure.canvas.manager.set_window_title("SmartLeak Classroom Demo")
        self.ax_header = self.figure.add_axes([0.035, 0.90, 0.93, 0.075])
        self.ax_network = self.figure.add_axes([0.04, 0.48, 0.47, 0.39])
        self.ax_readings = self.figure.add_axes([0.53, 0.48, 0.43, 0.39])
        self.ax_pressure = self.figure.add_axes([0.06, 0.15, 0.27, 0.25])
        self.ax_flow = self.figure.add_axes([0.365, 0.15, 0.27, 0.25])
        self.ax_risk = self.figure.add_axes([0.67, 0.15, 0.27, 0.25])
        self.ax_controls = self.figure.add_axes([0.06, 0.045, 0.88, 0.065])
        for axis in (
            self.ax_header, self.ax_network, self.ax_readings, self.ax_controls,
        ):
            axis.set_facecolor("#ffffff")
        self.ax_header.axis("off")
        self.ax_readings.axis("off")
        self.ax_controls.axis("off")
        self._style_chart(self.ax_pressure, "PRESSURE VS TIME", "Pressure (m)")
        self._style_chart(self.ax_flow, "ZONE B PIPE FLOW VS TIME", "Flow (L/s)")
        self._style_chart(self.ax_risk, "LEAK RISK VS TIME", "Risk (%)")
        self.pressure_line, = self.ax_pressure.plot(
            [], [], color="#208d88", linewidth=2.3,
        )
        self.flow_line, = self.ax_flow.plot(
            [], [], color="#dd8f45", linewidth=2.3,
        )
        self.risk_line, = self.ax_risk.plot(
            [], [], color="#d74b50", linewidth=2.3,
        )
        self.ax_risk.set_ylim(0, 100)
        self._make_controls()

    @staticmethod
    def _style_chart(axis: Axes, title: str, ylabel: str) -> None:
        axis.set_facecolor("#ffffff")
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=10)
        axis.set_xlabel("Simulation time (s)", fontsize=8)
        axis.set_ylabel(ylabel, fontsize=8)
        axis.grid(True, color="#e8edef", linewidth=0.8)
        axis.tick_params(labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#dfe6e8")

    def _make_controls(self) -> None:
        control_specs = [
            (0.01, 0.50, 0.13, "START SIMULATION", self._start, "#18785d"),
            (0.15, 0.50, 0.09, "PAUSE", self._pause, "#546975"),
            (0.25, 0.50, 0.09, "RESET", self._reset, "#546975"),
            (0.36, 0.50, 0.15, "INTRODUCE LEAK", self._introduce_leak, "#b77821"),
            (0.58, 0.50, 0.10, "1× SPEED", lambda _: self._set_speed(1), "#1d7774"),
            (0.69, 0.50, 0.10, "2× SPEED", lambda _: self._set_speed(2), "#1d7774"),
            (0.80, 0.50, 0.10, "5× SPEED", lambda _: self._set_speed(5), "#1d7774"),
        ]
        self.buttons: list[Button] = []
        control_position = self.ax_controls.get_position()
        for x, y, width, label, callback, color in control_specs:
            axis = self.figure.add_axes([
                control_position.x0 + x * control_position.width,
                control_position.y0 + y * control_position.height,
                width * control_position.width,
                0.46 * control_position.height,
            ])
            button = Button(
                axis,
                label,
                color=color,
                hovercolor="#253c49",
            )
            button.label.set_color("#ffffff")
            button.label.set_fontsize(8)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.control_status = self.figure.text(
            0.075, 0.052, "", fontsize=8, color="#49606a", va="bottom",
        )

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
        return ZONE_COLORS.get(status, "#21845b")

    def _draw_network(self) -> None:
        axis = self.ax_network
        axis.clear()
        axis.set_facecolor("#ffffff")
        axis.set_title(
            "LIVE NETWORK  ·  pressure at junctions (m)  ·  pipes (L/s)",
            loc="left",
            fontsize=9,
            fontweight="bold",
        )
        snapshot = self.engine.snapshot
        edge_colors = []
        edge_widths = []
        for _, _, data in self.graph.edges(data=True):
            zone = data["zone"]
            score = self.engine.risk[zone]
            status = "HIGH RISK" if score >= 61 else "WARNING" if score >= 31 else "NORMAL"
            edge_colors.append(self._phase_color(status))
            edge_widths.append(
                3.0 if zone == self.engine.probable_zone and score >= 61 else 1.8
            )
        node_colors = []
        node_labels = {}
        for node in self.graph.nodes:
            if node == "RESERVOIR":
                node_colors.append("#245e77")
                node_labels[node] = "RES"
            else:
                zone = self.node_zones[node]
                score = self.engine.risk[zone]
                status = "HIGH RISK" if score >= 61 else "WARNING" if score >= 31 else "NORMAL"
                node_colors.append(self._phase_color(status))
                node_labels[node] = f"{node}\n{snapshot.pressures[node]:.1f}"
        nx.draw_networkx_edges(
            self.graph,
            self.positions,
            ax=axis,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=11,
            connectionstyle="arc3,rad=0.0",
        )
        nx.draw_networkx_nodes(
            self.graph,
            self.positions,
            ax=axis,
            node_color=node_colors,
            node_size=690,
            edgecolors="#ffffff",
            linewidths=1.5,
        )
        nx.draw_networkx_labels(
            self.graph,
            self.positions,
            labels=node_labels,
            ax=axis,
            font_size=7,
            font_color="#ffffff",
            font_weight="bold",
        )
        edge_labels = {
            (start, end): (
                f"{data['pipe'].replace('P_', '')}\n"
                f"{snapshot.flows_lps[data['pipe']]:.2f}"
            )
            for start, end, data in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.graph,
            self.positions,
            edge_labels=edge_labels,
            ax=axis,
            font_size=5.5,
            label_pos=0.48,
            rotate=False,
            bbox={
                "boxstyle": "round,pad=0.08",
                "fc": "#ffffff",
                "ec": "none",
                "alpha": 0.88,
            },
        )
        axis.set_xlim(-1.45, 7.65)
        axis.set_ylim(-1.08, 0.75)
        axis.axis("off")

    def _draw_readings(self) -> None:
        axis = self.ax_readings
        axis.clear()
        axis.axis("off")
        axis.set_facecolor("#ffffff")
        status = self.engine.network_status
        axis.text(
            0.0, 0.98, f"NETWORK STATUS  ·  {status}",
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            color=self._phase_color(status),
            va="top",
        )
        axis.text(
            0.0, 0.90,
            f"SIM TIME  {self.engine.seconds:03d}s / {DEMO_SECONDS}s"
            f"     PHASE  {self.engine.phase}",
            transform=axis.transAxes,
            fontsize=8,
            color="#49606a",
            va="top",
        )

        zone_x = (0.00, 0.25, 0.50, 0.75)
        for (zone, _nodes), x in zip(ZONE_NODES.items(), zone_x):
            score = self.engine.risk[zone]
            state = "HIGH RISK" if score >= 61 else "WARNING" if score >= 31 else "NORMAL"
            axis.text(
                x, 0.79,
                f"ZONE {zone}\n{state}\nRisk {score:.0f}%",
                transform=axis.transAxes,
                fontsize=8,
                color=self._phase_color(state),
                fontweight="bold",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": "#f7fafb",
                    "edgecolor": self._phase_color(state),
                    "linewidth": 1.0,
                },
            )

        axis.text(
            0.0, 0.57, f"JUNCTION PRESSURE ({PRESSURE_UNIT})",
            transform=axis.transAxes, fontsize=8, fontweight="bold",
            color="#29414b",
        )
        pressure_rows = []
        for zone, nodes in ZONE_NODES.items():
            values = "   ".join(
                f"{node} {self.engine.snapshot.pressures[node]:.1f}"
                for node in nodes
            )
            pressure_rows.append(f"{zone}:  {values}")
        axis.text(
            0.0, 0.53, "\n".join(pressure_rows),
            transform=axis.transAxes,
            fontsize=7,
            color="#43565e",
            family="monospace",
            va="top",
            linespacing=1.5,
        )

        axis.text(
            0.0, 0.34, f"PIPE FLOW ({FLOW_UNIT})",
            transform=axis.transAxes, fontsize=8, fontweight="bold",
            color="#29414b",
        )
        flow_rows = []
        for zone in ZONE_NODES:
            core = ZONE_FLOW_PIPES[zone]
            value = self.engine.snapshot.flows_lps[core]
            flow_rows.append(f"ZONE {zone}  {core:<12} {value:>5.2f}")
        axis.text(
            0.0, 0.30, "\n".join(flow_rows),
            transform=axis.transAxes,
            fontsize=7,
            color="#43565e",
            family="monospace",
            va="top",
            linespacing=1.4,
        )

        zone = self.engine.probable_zone or "B"
        parts = self.engine.contributions[zone]
        axis.text(
            0.49, 0.34, f"RISK BREAKDOWN  ·  ZONE {zone}",
            transform=axis.transAxes, fontsize=8, fontweight="bold",
            color="#29414b",
        )
        contribution_text = (
            f"Pressure anomaly  +{parts['pressure']:.0f} / 35\n"
            f"Flow anomaly       +{parts['flow']:.0f} / 35\n"
            f"ML anomaly         +{parts['ml']:.0f} / 20\n"
            f"Persistence        +{parts['persistence']:.0f} / 10\n"
            f"TOTAL RISK          {self.engine.risk[zone]:.0f}%"
        )
        axis.text(
            0.49, 0.30, contribution_text,
            transform=axis.transAxes,
            fontsize=7,
            color="#43565e",
            family="monospace",
            va="top",
            linespacing=1.4,
        )

        if self.engine.risk["B"] >= 61:
            b = self.engine.measurements["B"]
            why = []
            if self.engine.contributions["B"]["pressure"] > 0:
                why.append("pressure deviation")
            if self.engine.contributions["B"]["flow"] > 0:
                why.append("abnormal flow")
            if self.engine.ml_flags["B"]:
                why.append("ML anomaly")
            if self.engine.contributions["B"]["persistence"] > 0:
                why.append("persistent behavior")
            alert = (
                "HIGH-RISK ANOMALY DETECTED  ·  ZONE B\n"
                f"Pressure {b.pressure_m:.2f} m  ·  Flow {b.flow_lps:.2f} L/s"
                f"  ·  Risk {self.engine.risk['B']:.0f}%\n"
                f"WHY FLAGGED  ✓ {'  ·  ✓ '.join(why)}\n"
                "Recommended: inspect the affected Zone B pipeline section."
            )
            if self.engine.seconds >= DEMO_SECONDS:
                priorities = "  ·  ".join(
                    f"{label} Zone {zone}"
                    for zone, _score, label in self.engine.priorities()
                )
                alert += f"\nMAINTENANCE PRIORITY  {priorities}"
            axis.text(
                0.49, 0.075, alert,
                transform=axis.transAxes,
                fontsize=6.5,
                color="#a7363d",
                fontweight="bold",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#fff1f1",
                    "edgecolor": "#e9b2b4",
                },
            )
        elif self.engine.probable_zone:
            axis.text(
                0.49, 0.09,
                f"PROBABLE LEAK ZONE: ZONE {self.engine.probable_zone}\n"
                "Inspect the affected pipeline; this is not a confirmed leak.",
                transform=axis.transAxes,
                fontsize=7,
                color="#a16a16",
                fontweight="bold",
                va="top",
            )
        else:
            axis.text(
                0.49, 0.10,
                "All zones are within the normal operating range.",
                transform=axis.transAxes,
                fontsize=7,
                color="#21845b",
                va="top",
            )

    def _update_chart(
        self,
        axis: Axes,
        line: Any,
        values: list[float],
    ) -> None:
        line.set_data(self.time_history, values)
        if self.time_history:
            axis.set_xlim(0, max(30, self.time_history[-1] + 2))
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)

    def refresh(self) -> None:
        self.ax_header.clear()
        self.ax_header.axis("off")
        self.ax_header.text(
            0.0, 0.74, "SMARTLEAK",
            transform=self.ax_header.transAxes,
            fontsize=20,
            fontweight="bold",
            color="#173844",
            va="top",
        )
        self.ax_header.text(
            0.0, 0.18, "INTELLIGENT WATER NETWORK LEAK DETECTION SIMULATION",
            transform=self.ax_header.transAxes,
            fontsize=9,
            color="#536a73",
            va="bottom",
        )
        self.ax_header.text(
            1.0, 0.66,
            "DETERMINISTIC  ·  WNTR / EPANET  ·  ISOLATION FOREST",
            transform=self.ax_header.transAxes,
            fontsize=8,
            color="#536a73",
            ha="right",
            va="top",
        )
        self._draw_network()
        self._draw_readings()
        self._update_chart(self.ax_pressure, self.pressure_line, self.pressure_history)
        self._update_chart(self.ax_flow, self.flow_line, self.flow_history)
        self._update_chart(self.ax_risk, self.risk_line, self.risk_history)

        state = "RUNNING" if self.engine.running else "PAUSED"
        if self.engine.seconds >= DEMO_SECONDS:
            state = "DEMO COMPLETE"
        self.control_status.set_text(
            f"{state}     ·     SPEED {self.engine.speed}×"
            "     ·     RESET restores the same 150-second sequence"
        )
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
