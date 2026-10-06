# SmartLeak — Water Network Leak Detection Simulation

A deterministic, single-window Python classroom simulation. WNTR and its EPANET-compatible simulator calculate pressure and flow for a small four-zone distribution network. A leak emitter is progressively opened at a Zone B junction; rule-based checks and an Isolation Forest assess the resulting measurements. This is a simulation, not a physical sensor system or a confirmed leak locator.

## Requirements

- Python 3.10 or newer
- A desktop session with a Matplotlib graphical window backend (for example, Tk or Qt)

The application does not use a web server, external APIs, a database, or IoT hardware. Once the Python packages are installed, the demo runs offline. A browser-only or headless session cannot show the interactive desktop window.

## Install and run

```bash
pip install -r requirements.txt
python main.py
```

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

- WNTR simulates the normal network and a set of progressively larger Zone B emitter leaks. Displayed pressure and pipe-flow observations are interpolated between those hydraulic simulation results as the demo clock advances.
- An Isolation Forest is trained on normal demand-profile measurements. Its anomaly flag is one 20% contribution to the score, not proof of a leak.
- Rule-based pressure and flow deviations contribute up to 35 points each. Persistent abnormal readings contribute up to 10 points. The total is clamped to 0–100: **LOW** (0–30), **MEDIUM** (31–60), or **HIGH** (61–100).
- The network contains one reservoir, twelve junctions across four named zones, and a main route with zone branches. A tank is omitted because it is not needed for this steady-state demonstration.

## Project files

```text
main.py           Simulation, analysis, and Matplotlib desktop window
requirements.txt  Python dependencies
README.md         Installation and presentation instructions
```
