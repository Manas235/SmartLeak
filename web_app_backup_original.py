"""SmartLeak Streamlit dashboard for hydraulic simulation and basic data quality workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import main as smartleak
except Exception as exc:  # pragma: no cover - depends on runtime environment.
    smartleak = None
    import_error = exc
else:
    import_error = None


st.set_page_config(
    page_title="SmartLeak Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _initialize_state() -> None:
    if "datasets" not in st.session_state:
        st.session_state.datasets = []
    if "simulation" not in st.session_state:
        if smartleak is not None:
            try:
                calibration = smartleak.HydraulicCalibration()
                st.session_state.simulation = smartleak.SmartLeakSimulation(calibration)
            except Exception as exc:  # pragma: no cover - depends on WNTR runtime.
                st.session_state.simulation = None
                st.session_state.simulation_error = str(exc)
        else:
            st.session_state.simulation = None
            st.session_state.simulation_error = str(import_error)


def _read_uploaded_data(uploaded_file: Any) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    try:
        name = str(uploaded_file.name).lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        st.warning("Unsupported file type. Upload a CSV or Excel workbook.")
        return None
    except Exception as exc:  # pragma: no cover - report to user.
        st.error(f"Could not read '{uploaded_file.name}': {exc}")
        return None


def _quality_score(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "rows": 0,
            "columns": 0,
            "missing": 0,
            "duplicates": 0,
            "complete_rows": 0,
            "completeness": 0.0,
            "quality_score": 0.0,
            "status": "No data",
        }

    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    complete_rows = int(df.dropna().shape[0])
    total_cells = rows * cols
    completeness = (1 - (missing / total_cells)) * 100.0 if total_cells else 0.0
    duplicate_penalty = (duplicates / rows) * 100.0 if rows else 0.0
    valid_numeric = 0
    numeric_columns = 0
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns += 1
            valid_numeric += int(pd.to_numeric(df[column], errors="coerce").notna().sum())
    numeric_coverage = (valid_numeric / max(1, rows * max(1, numeric_columns))) * 100.0

    quality = (
        completeness * 0.60
        + (100.0 - duplicate_penalty) * 0.20
        + numeric_coverage * 0.20
    )
    quality = max(0.0, min(100.0, quality))
    if quality >= 85:
        status = "Good"
    elif quality >= 60:
        status = "Needs attention"
    else:
        status = "Poor"

    return {
        "rows": rows,
        "columns": cols,
        "missing": missing,
        "duplicates": duplicates,
        "complete_rows": complete_rows,
        "completeness": round(completeness, 2),
        "quality_score": round(quality, 2),
        "status": status,
    }


def _render_overview() -> None:
    st.title("SmartLeak — Water Network Intelligence Dashboard")
    st.caption("Deterministic WNTR/EPANET simulation plus lightweight data-quality workflow.")

    if smartleak is None:
        st.error("The SmartLeak engine could not be imported. Check that main.py exists and all required dependencies are installed.")
        st.code(str(import_error))
        return

    dataset_count = len(st.session_state.datasets)
    total_records = sum(len(df) for df in [item["data"] for item in st.session_state.datasets])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Datasets", dataset_count)
    col2.metric("Records", total_records)
    col3.metric("Hydraulic engine", "Active")
    col4.metric("Status", "Ready")

    st.subheader("Project summary")
    st.markdown(
        """
        This dashboard reuses the existing SmartLeak hydraulic engine from `main.py`, so the
        network topology, pressure and flow calculations, and leak-analysis logic remain unchanged.
        The Streamlit app adds a safe interface for project overview, dataset uploads, preview,
        and a basic quality-analysis layer without inventing real-world measurements.
        """
    )

    if st.session_state.get("simulation") is not None:
        engine = st.session_state.simulation
        try:
            zone_scores = engine.priorities()
        except Exception:
            zone_scores = []
        if zone_scores:
            st.subheader("Current zone risk summary")
            st.dataframe(pd.DataFrame(zone_scores, columns=["Zone", "Risk", "Label"]), use_container_width=True)


def _render_upload_page() -> None:
    st.title("Data Upload and Preview")
    uploaded_file = st.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        df = _read_uploaded_data(uploaded_file)
        if df is not None:
            dataset_record = {
                "name": uploaded_file.name,
                "data": df,
                "source": "Uploaded file",
            }
            existing = [item for item in st.session_state.datasets if item["name"] == uploaded_file.name]
            if not existing:
                st.session_state.datasets.append(dataset_record)
            else:
                existing[0]["data"] = df
            st.success(f"Loaded dataset: {uploaded_file.name}")

    if st.session_state.datasets:
        names = [item["name"] for item in st.session_state.datasets]
        selected_name = st.selectbox("Choose dataset to preview", names)
        selected = next(item for item in st.session_state.datasets if item["name"] == selected_name)
        st.dataframe(selected["data"].head(50), use_container_width=True)
        st.caption(f"Shape: {selected['data'].shape[0]} rows × {selected['data'].shape[1]} columns")
    else:
        st.info("No dataset has been uploaded yet.")


def _render_quality_page() -> None:
    st.title("Basic Data Quality Analysis")
    if not st.session_state.datasets:
        st.info("Upload a CSV or Excel file to begin quality analysis.")
        return

    names = [item["name"] for item in st.session_state.datasets]
    selected_name = st.selectbox("Dataset for quality analysis", names)
    selected = next(item for item in st.session_state.datasets if item["name"] == selected_name)
    df = selected["data"]
    metrics = _quality_score(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", metrics["rows"])
    col2.metric("Columns", metrics["columns"])
    col3.metric("Missing values", metrics["missing"])
    col4.metric("Quality score", f"{metrics['quality_score']:.2f}/100")

    st.subheader("Quality summary")
    st.write(
        f"Completeness: {metrics['completeness']:.2f}% | "
        f"Duplicate rows: {metrics['duplicates']} | "
        f"Complete rows: {metrics['complete_rows']} | "
        f"Status: {metrics['status']}"
    )

    quality_df = pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Missing values",
                "Duplicate rows",
                "Complete rows",
                "Completeness (%)",
                "Quality score",
                "Status",
            ],
            "Value": [
                metrics["rows"],
                metrics["columns"],
                metrics["missing"],
                metrics["duplicates"],
                metrics["complete_rows"],
                metrics["completeness"],
                metrics["quality_score"],
                metrics["status"],
            ],
        }
    )
    st.dataframe(quality_df, use_container_width=True)

    st.subheader("Problematic rows")
    problem_rows = df[df.isna().any(axis=1) | df.duplicated()].head(20)
    if problem_rows.empty:
        st.info("No missing or duplicate rows detected in the selected dataset.")
    else:
        st.dataframe(problem_rows, use_container_width=True)


def _render_hydraulic_page() -> None:
    st.title("Hydraulic Simulation")
    if smartleak is None:
        st.error("Unable to load the hydraulic engine; import failed.")
        return

    if st.session_state.get("simulation") is None:
        st.error(st.session_state.get("simulation_error", "The SmartLeak simulation could not initialize."))
        return

    engine = st.session_state.simulation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Advance 1 step"):
            try:
                engine.advance()
                st.success("Hydraulic step advanced.")
            except Exception as exc:
                st.error(f"Simulation step failed: {exc}")
    with col2:
        if st.button("Reset simulation"):
            try:
                st.session_state.simulation = smartleak.SmartLeakSimulation(smartleak.HydraulicCalibration())
                st.success("Simulation reset.")
            except Exception as exc:
                st.error(f"Simulation reset failed: {exc}")

    if hasattr(engine, "risk"):
        risk_df = pd.DataFrame(
            [(zone, score) for zone, score in getattr(engine, "risk", {}).items()],
            columns=["Zone", "Risk score"],
        )
        st.subheader("Zone risk")
        st.dataframe(risk_df, use_container_width=True)

    try:
        junction_df = pd.DataFrame(engine.get_junction_telemetry())
        pipe_df = pd.DataFrame(engine.get_pipe_telemetry())
    except Exception as exc:
        st.warning(f"Telemetry could not be read: {exc}")
        return

    st.subheader("Junction telemetry")
    st.dataframe(junction_df, use_container_width=True)
    st.subheader("Pipe telemetry")
    st.dataframe(pipe_df, use_container_width=True)


def main() -> None:
    _initialize_state()

    pages = ["Overview", "Data Upload", "Data Quality", "Hydraulic Simulation"]
    page = st.sidebar.radio("Navigation", pages)

    if page == "Overview":
        _render_overview()
    elif page == "Data Upload":
        _render_upload_page()
    elif page == "Data Quality":
        _render_quality_page()
    else:
        _render_hydraulic_page()


if __name__ == "__main__":
    main()
