"""Utility helpers for the 3‑D Three.js visualisation.

This module is deliberately lightweight – it only contains Python‑side helpers
that are used by the ``server_3d.py`` HTTP API and by the JavaScript client via
string interpolation.  All heavy lifting (animation, ray‑casting, colour
updates) stays in the client; the Python code just supplies colour maps,
severity thresholds and a tiny wrapper to format pipe data as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

# ---------------------------------------------------------------------------
# colour definitions – match the design system (dark theme) used elsewhere.
# ---------------------------------------------------------------------------
COLOR_NORMAL: str = "#22D3EE"  # cyan – normal pipe
COLOR_WARNING: str = "#F59E0B"  # amber – flow anomaly / warning
COLOR_CRITICAL: str = "#EF4444"  # red – leak suspected

# Severity thresholds (based on IsolationForest anomaly scores, 0‑1).
# These are simple quartiles that can be tweaked later.
SEVERITY_THRESHOLDS = {
    "LOW": 0.25,
    "MEDIUM": 0.5,
    "HIGH": 0.75,
    "CRITICAL": 0.9,
}


def severity_from_score(score: float) -> Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    """Map a raw IsolationForest anomaly score (0‑1) to a severity label.

    The score is expected to be normalised – the engine already outputs a
    value between 0 and 1.  The function returns the highest severity whose
    threshold is *not* exceeded.
    """
    if score >= SEVERITY_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if score >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def color_for_severity(severity: str) -> str:
    """Return the HEX colour string for a given severity label."""
    return {
        "LOW": COLOR_NORMAL,
        "MEDIUM": COLOR_WARNING,
        "HIGH": COLOR_WARNING,
        "CRITICAL": COLOR_CRITICAL,
    }.get(severity, COLOR_NORMAL)


@dataclass(frozen=True)
class PipeInfo:
    """Simple container for pipe telemetry used by the API endpoint.

    ``flow_lps`` – absolute flow in L/s (float)
    ``pressure_m`` – pressure at the *start* node in metres (float)
    ``status`` – one of ``NORMAL``, ``WARNING`` or ``CRITICAL``
    """

    id: str
    flow_lps: float
    pressure_m: float
    status: Literal["NORMAL", "WARNING", "CRITICAL"]

    def to_dict(self) -> Mapping[str, object]:
        return {
            "id": self.id,
            "flow_lps": round(self.flow_lps, 3),
            "pressure_m": round(self.pressure_m, 3),
            "status": self.status,
            "color": color_for_severity(self.status),
        }


# Helper used by ``server_3d.py`` to build the list of clickable pipes.
def pipe_status_from_flow(flow: float, warning_thresh: float = 0.8, critical_thresh: float = 1.2) -> Literal["NORMAL", "WARNING", "CRITICAL"]:
    """Classify a pipe based on its absolute flow relative to the design flow.

    The design flow is *not* stored globally, so callers provide the thresholds.
    The defaults roughly correspond to 80 % (warning) and 120 % (critical) of the
    baseline flow for the network.
    """
    if flow > critical_thresh:
        return "CRITICAL"
    if flow > warning_thresh:
        return "WARNING"
    return "NORMAL"

# End of file
