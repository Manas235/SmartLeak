# SmartLeak — Water Network Leak Detection Simulation

A deterministic, single-window Python classroom simulation. WNTR and its EPANET-compatible simulator calculate pressure and flow for a small four-zone distribution network. A leak emitter is progressively opened at a Zone B junction; rule-based checks and an Isolation Forest assess the resulting measurements. This is a simulation, not a physical sensor system or a confirmed leak locator.

## Requirements

- Python 3.12 or newer
- A desktop session with a Matplotlib graphical window backend (for example, Tk or Qt)

The desktop application does not use a web server, external APIs, a database, or IoT hardware. Once the Python packages are installed, the demo runs offline. A browser-only or headless session cannot show the interactive desktop window.

## Install and run

```bash
pip install -r requirements.txt
python main.py
```

## Browser dashboard

The Streamlit dashboard reuses the same deterministic WNTR/EPANET engine and
Isolation Forest as the desktop app. It is a hydraulic simulation, not a
physical sensor connection. Its controls advance the simulation only when
requested or while **Start** is active; rerunning the page does not duplicate
simulation steps or telemetry.

```bash
pip install -r requirements.txt
streamlit run web_app.py
```

The **Network Digital Twin** tab shows the reservoir, all 12 junctions, and the
12 pipes from the WNTR model. Animated markers follow the actual pipe paths;
their direction follows signed EPANET link flow and their speed scales with
simulated flow magnitude. EPANET's `result.link["flowrate"]` values are in
m³/s in the WNTR results and are converted to L/s by multiplying by 1,000.
The engine retains flow magnitudes for existing scoring and exposes signed
flows for direction. Each requested or automatic simulation step runs
WNTR/EPANET using the existing network and the deterministic leak emitter
coefficient for that simulation time. The solver's current pressure and flow
tables feed the existing risk, anomaly, and persistence logic. Automatic
advancement uses a configurable real-time interval; pausing keeps the latest
solver result. Browser animation frames only move particles between hydraulic
updates and never invoke the solver.

**Auto Detect** evaluates current EPANET junction pressures and incident pipe
flows alongside the existing per-zone risk and Isolation Forest/rule-based
flags. It ranks junction evidence within zones that meet the existing engine's
anomaly criteria; the table orders by existing zone risk first, then by
calibrated junction-pressure and incident-pipe flow deviations. If the
highest-risk zone conflicts with the strongest local hydraulic evidence in
another zone, the result is reported as uncertain rather than asserting a
location. Those relative evidence values are not new risk scores or
probabilities. When the existing engine finds no significant
anomaly, the dashboard reports that instead of naming a suspect. If the
measurements cannot distinguish a junction, it reports uncertain localization.
The assessment is refreshed after each completed hydraulic solve and is also
available on demand with **Auto Detect**.

**Auto Leak Scenario** chooses a valid WNTR junction reproducibly, rotating
through Zones A, B, C, and D, and then through the actual junctions in each
zone. The known injected scenario target is shown separately from the
measurement-based suspicion; an injected leak is not itself proof that the
detector identified it. Optional manual target controls and **Trigger Leak**
remain available for inspecting a particular junction. Every new scenario
clears the prior emitter and analysis history before solving EPANET with the
existing maximum emitter coefficient. Only one leak emitter is supported at a
time. **Reset to normal baseline** clears it and returns the dashboard to the
normal hydraulic baseline. The desktop application's deterministic Zone B /
Junction B2 scenario remains unchanged.

To test autonomous detection, start from **Reset to normal baseline** and press
**Auto Detect**; the normal network should report no significant anomaly. Press
**Auto Leak Scenario** four times to exercise one valid junction in each zone.
For each run, compare the separately displayed confirmed scenario target with
the independent suspected location, junction pressure/pipe-flow evidence,
existing zone risk and Isolation Forest results, map highlights, and timeline.
The real hydraulic readings determine whether an alert threshold is crossed;
the dashboard does not force alerts or guarantee exact junction localization.
In validation, the existing zone-risk engine ranked Zone B highest for the
demonstration's C1 and D1 injections, while local hydraulic evidence was
strongest in Zones C and D. The dashboard therefore reports uncertain
localization in those cases rather than claiming the injected location was
independently detected. Improving that inherited zone-level risk model is
outside this dashboard change.

The tab also shows net zone flow calculated from the pipes crossing each
zone's boundary: A = P_SUPPLY − P_A_TO_B; B = P_A_TO_B − P_B_TO_C;
C = P_B_TO_C − P_C_TO_D; D = P_C_TO_D (the terminal-zone inlet). These
values represent the zone's net simulated consumption, including the active
emitter discharge when open.
Flow values are simulated data, not physical sensor telemetry. The live map
uses Streamlit's inline component API and adds no package dependency.

To deploy on Streamlit Community Cloud, push this project to GitHub, select
`web_app.py` as the app file, and choose Python 3.13 in the advanced settings.
The project includes both `requirements.txt` and `uv.lock`; keep them in sync
with `pyproject.toml` when changing Python dependencies.

To check hydraulic calculations, anomaly scoring, controls, and repeatability without opening the graphical window:

```bash
python main.py --self-test
```

## Classroom demo

1. Press **RESET**, then **START**. The 1× demonstration takes 150 seconds (2 minutes 30 seconds).
2. The network stays normal for 30 seconds. A leak then develops progressively in Zone B; the same sequence plays after each reset.
3. Use **PAUSE** to stop the clock, **START** to resume, and **1× / 2× / 5×** to change its rate. **INTRODUCE LEAK** skips to the leak phase when you want to demonstrate it immediately.
4. Review the live network, junction pressures, pipe flows, three trend graphs, explainable risk contributions, and maintenance priority list.
5. A high-risk result means **probable leak zone** and recommends inspection. It does not claim that a leak was physically confirmed or repaired.

## How detection works

- WNTR simulates normal calibration demand profiles and deterministic Zone B emitter cases to establish baselines and Isolation Forest training. During a dashboard run, every completed simulation step solves the current network again with the current emitter coefficient at the selected junction (or the timed Zone B / Junction B2 scenario); dashboard pressure and pipe-flow readings are the resulting EPANET outputs, not interpolated values.
- An Isolation Forest is trained on normal demand-profile measurements. Its anomaly flag is one 20% contribution to the score, not proof of a leak.
- Rule-based pressure and flow deviations contribute up to 35 points each. Persistent abnormal readings contribute up to 10 points. The total is clamped to 0–100: **LOW** (0–30), **MEDIUM** (31–60), or **HIGH** (61–100).
- The network contains one reservoir, twelve junctions across four named zones, and a main route with zone branches. A tank is omitted because it is not needed for this steady-state demonstration.

## Project files

```text
main.py           Simulation, analysis, and Matplotlib desktop window
web_app.py        Browser dashboard backed by the existing simulation engine
.streamlit/       Streamlit dark industrial theme
requirements.txt  Python dependencies
pyproject.toml    Python dependency constraints
uv.lock           Locked deployment dependency resolution
README.md         Installation and presentation instructions
```
