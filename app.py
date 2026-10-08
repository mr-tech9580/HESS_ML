from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.hess import HESS
from src.teacher import teacher_action


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = ROOT / "models"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HESS-ML | Energy Management",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 2rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(128, 128, 128, 0.25);
    background: linear-gradient(
        135deg,
        rgba(40, 80, 120, 0.25),
        rgba(30, 30, 50, 0.35)
    );
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.2rem;
    opacity: 0.85;
}

.metric-card {
    padding: 1.2rem;
    border-radius: 15px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    text-align: center;
    min-height: 125px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.75;
    margin-top: 0.3rem;
}

.section-title {
    font-size: 1.7rem;
    font-weight: 650;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# FILES
# ============================================================

comparison_file = (
    RESULTS_DIR /
    "controller_comparison.csv"
)

evaluation_file = (
    METRICS_DIR /
    "controller_evaluation.json"
)

model_metrics_file = (
    METRICS_DIR /
    "model_metrics.json"
)

demand_model_file = (
    MODELS_DIR /
    "demand_predictor.pkl"
)

student_model_file = (
    MODELS_DIR /
    "student_policy.pkl"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not comparison_file.exists():

    st.error(
        "Controller evaluation results were not found."
    )

    st.code(
        "python -m src.evaluate"
    )

    st.stop()


if not evaluation_file.exists():

    st.error(
        "Controller evaluation JSON was not found."
    )

    st.stop()


# ============================================================
# LOAD RESULTS
# ============================================================

comparison = pd.read_csv(
    comparison_file,
    index_col=0
)

with open(
    evaluation_file,
    "r"
) as file:

    evaluation = json.load(file)


display_names = {
    "ml": "ML Controller",
    "rule": "Rule-Based Controller",
    "battery_only": "Battery-Priority Baseline"
}


comparison_display = comparison.rename(
    index=display_names
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## HESS-ML")

    st.markdown(
        """
        **Machine Learning Based Hybrid
        Energy Storage Management**
        """
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Live HESS Controller",
            "Controller Comparison",
            "HESS Simulation",
            "ML Models",
            "Model Explainability"
        ]
    )

    st.divider()

    st.markdown(
        """
        ### Technology Stack

        **Language**
        - Python

        **Machine Learning**
        - Scikit-learn
        - Random Forest

        **Data**
        - NumPy
        - Pandas

        **Visualization**
        - Plotly
        - Streamlit
        """
    )

    st.divider()

    st.caption(
        "HESS-ML | EV Energy Management"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="hero-title">
ML-Based Hybrid Energy Storage System
</div>

<div class="hero-subtitle">
Intelligent Energy Management for Electric Vehicles
</div>

<br>

<div>
Machine-learning-based power allocation between
a battery and supercapacitor using a
teacher-student control architecture.
</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Key Performance Results</div>',
        unsafe_allow_html=True
    )

    improvements = evaluation[
        "ml_vs_battery_priority"
    ]

    col1, col2, col3, col4 = st.columns(4)

    metrics = [

        (
            col1,
            "Battery Energy",
            improvements[
                "battery_energy_reduction_percent"
            ]
        ),

        (
            col2,
            "Peak Battery Power",
            improvements[
                "peak_battery_power_reduction_percent"
            ]
        ),

        (
            col3,
            "RMS Battery Power",
            improvements[
                "battery_rms_power_reduction_percent"
            ]
        ),

        (
            col4,
            "Estimated Loss",
            improvements[
                "mean_battery_loss_reduction_percent"
            ]
        )
    ]

    for column, label, value in metrics:

        with column:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-value">
{value:.2f}%
</div>

<div class="metric-label">
{label} Reduction
</div>

</div>
""",
                unsafe_allow_html=True
            )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Project Architecture</div>',
        unsafe_allow_html=True
    )

    architecture = """
Driving Cycle
      |
      v
Power Demand Dataset
      |
      v
+---------------------------+
| Random Forest Predictor   |
| Short-term Demand Model   |
+---------------------------+
      |
      v
Teacher Control Policy
      |
      v
Policy Dataset
      |
      v
+---------------------------+
| Random Forest Student     |
| HESS Energy Controller    |
+---------------------------+
      |
      v
Battery + Supercapacitor
      |
      v
Controller Evaluation
"""

    st.code(
        architecture,
        language="text"
    )

    st.markdown(
        '<div class="section-title">Result Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
The ML controller is evaluated against a
rule-based teacher and a battery-priority baseline.

The simulation indicates substantially lower
battery-side energy usage, peak battery power,
RMS battery power and estimated resistive losses
compared with the battery-priority strategy.

The ML policy closely follows the rule-based teacher
because it was trained using teacher-generated
control actions.
"""
    )

    st.markdown(
        '<div class="section-title">ML vs Baseline</div>',
        unsafe_allow_html=True
    )

    baseline = comparison.loc[
        "battery_only"
    ]

    ml = comparison.loc[
        "ml"
    ]

    baseline_data = pd.DataFrame(
        {
            "Metric": [
                "Battery Energy (kWh)",
                "Peak Battery Power (kW)",
                "RMS Battery Power (kW)",
                "Mean Battery Loss (W)"
            ],

            "ML Controller": [
                ml["battery_energy_kwh"],
                ml["peak_battery_power_kw"],
                ml["battery_rms_power_kw"],
                ml["mean_battery_loss_w"]
            ],

            "Battery-Priority Baseline": [
                baseline["battery_energy_kwh"],
                baseline["peak_battery_power_kw"],
                baseline["battery_rms_power_kw"],
                baseline["mean_battery_loss_w"]
            ]
        }
    )

    st.dataframe(
        baseline_data.round(3),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 2 — LIVE HESS CONTROLLER
# ============================================================

elif page == "Live HESS Controller":

    st.markdown(
        '<div class="section-title">Live HESS Controller</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
Enter the current vehicle demand and energy-storage
conditions. The trained Random Forest student policy
will predict the battery power allocation.
"""
    )

    st.info(
        """
The ML model predicts the battery-side power allocation.
The remaining demand is assigned to the supercapacitor,
subject to the physical constraints implemented in the
HESS simulator.
"""
    )

    st.subheader(
        "Enter HESS Conditions"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        demand_kw = st.number_input(
            "Power Demand (kW)",
            min_value=5.0,
            max_value=150.0,
            value=80.0,
            step=1.0
        )

    with col2:

        battery_soc = st.number_input(
            "Battery SOC (%)",
            min_value=10.0,
            max_value=95.0,
            value=70.0,
            step=1.0
        )

    with col3:

        supercap_soc = st.number_input(
            "Supercapacitor SOC (%)",
            min_value=10.0,
            max_value=95.0,
            value=60.0,
            step=1.0
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    predict_button = st.button(
        "Predict Power Allocation",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        battery_soc_fraction = (
            battery_soc / 100.0
        )

        supercap_soc_fraction = (
            supercap_soc / 100.0
        )

        if not student_model_file.exists():

            st.error(
                "Student policy model was not found."
            )

            st.stop()

        model_bundle = joblib.load(
            student_model_file
        )

        model = model_bundle["model"]

        model_features = model_bundle[
            "features"
        ]

        input_data = pd.DataFrame(
            [
                [
                    demand_kw,
                    battery_soc_fraction,
                    supercap_soc_fraction
                ]
            ],
            columns=model_features
        )

        raw_battery_power = float(
            model.predict(
                input_data
            )[0]
        )

        raw_battery_power = max(
            0.0,
            raw_battery_power
        )

        hess = HESS()

        hess.battery.soc = (
            battery_soc_fraction
        )

        hess.supercap.soc = (
            supercap_soc_fraction
        )

        result = hess.step(
            demand_kw,
            raw_battery_power
        )

        battery_power = result[
            "battery_power_kw"
        ]

        supercap_power = result[
            "supercap_power_kw"
        ]

        delivered_power = result[
            "delivered_power_kw"
        ]

        unmet_power = result[
            "unmet_power_kw"
        ]

        battery_soc_after = result[
            "battery_soc"
        ]

        supercap_soc_after = result[
            "supercap_soc"
        ]

        st.divider()

        st.subheader(
            "ML Power Allocation"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Vehicle Demand",
                f"{demand_kw:.2f} kW"
            )

        with col2:

            st.metric(
                "Battery Power",
                f"{battery_power:.2f} kW"
            )

        with col3:

            st.metric(
                "Supercapacitor Power",
                f"{supercap_power:.2f} kW"
            )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        if demand_kw > 0:

            battery_percentage = (
                battery_power /
                demand_kw
            ) * 100

            supercap_percentage = (
                supercap_power /
                demand_kw
            ) * 100

        else:

            battery_percentage = 0.0
            supercap_percentage = 0.0

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Battery Contribution",
                f"{battery_percentage:.1f}%"
            )

        with col2:

            st.metric(
                "Supercapacitor Contribution",
                f"{supercap_percentage:.1f}%"
            )

        st.subheader(
            "Power Split"
        )

        split_data = pd.DataFrame(
            {
                "Storage": [
                    "Battery",
                    "Supercapacitor"
                ],

                "Power": [
                    battery_power,
                    supercap_power
                ]
            }
        )

        fig = px.bar(
            split_data,
            x="Storage",
            y="Power",
            text="Power",
            title="ML-Based Power Allocation"
        )

        fig.update_traces(
            texttemplate="%{text:.2f} kW",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Power (kW)",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Storage State After This Timestep"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Battery SOC",
                f"{battery_soc_after * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Supercapacitor SOC",
                f"{supercap_soc_after * 100:.2f}%"
            )

        if unmet_power > 0:

            st.warning(
                f"""
The requested demand could not be fully supplied
under the current storage constraints.

Unmet power: {unmet_power:.2f} kW
"""
            )

        else:

            st.success(
                f"""
Full demand satisfied.

Delivered power: {delivered_power:.2f} kW
"""
            )

        st.subheader(
            "Controller Decision"
        )

        st.markdown(
            f"""
The Random Forest student policy received:

- **Power demand:** {demand_kw:.2f} kW
- **Battery SOC:** {battery_soc:.1f}%
- **Supercapacitor SOC:** {supercap_soc:.1f}%

The model initially predicted approximately
**{raw_battery_power:.2f} kW** of battery power.

After applying the HESS physical constraints:

- **Battery:** {battery_power:.2f} kW
- **Supercapacitor:** {supercap_power:.2f} kW

The controller delivered **{delivered_power:.2f} kW**
out of the requested **{demand_kw:.2f} kW**.
"""
        )


# ============================================================
# PAGE 3 — CONTROLLER COMPARISON
# ============================================================

elif page == "Controller Comparison":

    st.markdown(
        '<div class="section-title">Controller Performance</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        comparison_display.round(4),
        use_container_width=True
    )

    chart_data = (
        comparison_display
        .reset_index()
    )

    fig = px.bar(
        chart_data,
        x="strategy",
        y="battery_energy_kwh",
        title="Battery Energy Consumption",
        labels={
            "strategy": "Controller",
            "battery_energy_kwh":
                "Battery Energy (kWh)"
        },
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Battery Energy (kWh)",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.bar(
        chart_data,
        x="strategy",
        y="peak_battery_power_kw",
        title="Peak Battery Power",
        labels={
            "strategy": "Controller",
            "peak_battery_power_kw":
                "Peak Power (kW)"
        },
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Peak Power (kW)",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.bar(
        chart_data,
        x="strategy",
        y="battery_rms_power_kw",
        title="RMS Battery Power",
        labels={
            "strategy": "Controller",
            "battery_rms_power_kw":
                "RMS Power (kW)"
        },
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="RMS Power (kW)",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.bar(
        chart_data,
        x="strategy",
        y="mean_battery_loss_w",
        title="Estimated Battery Resistive Loss",
        labels={
            "strategy": "Controller",
            "mean_battery_loss_w":
                "Mean Loss (W)"
        },
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Mean Loss (W)",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PAGE 4 — HESS SIMULATION
# ============================================================

elif page == "HESS Simulation":

    st.markdown(
        '<div class="section-title">HESS Simulation</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
Explore how each controller distributes vehicle
power demand between the battery and supercapacitor.
"""
    )

    strategy_options = {

        "ML Controller":
            "ml",

        "Rule-Based Controller":
            "rule",

        "Battery-Priority Baseline":
            "battery_only"
    }

    selected_name = st.selectbox(
        "Select Controller",
        list(
            strategy_options.keys()
        )
    )

    selected_strategy = strategy_options[
        selected_name
    ]

    simulation_file = (
        RESULTS_DIR /
        f"{selected_strategy}_simulation.csv"
    )

    if not simulation_file.exists():

        st.warning(
            "Simulation file not found."
        )

        st.stop()

    simulation = pd.read_csv(
        simulation_file
    )

    selected_summary = comparison.loc[
        selected_strategy
    ]

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Battery Energy",
            f"{selected_summary['battery_energy_kwh']:.2f} kWh"
        )

    with col2:

        st.metric(
            "Peak Battery Power",
            f"{selected_summary['peak_battery_power_kw']:.2f} kW"
        )

    with col3:

        st.metric(
            "Final Battery SOC",
            f"{selected_summary['final_battery_soc'] * 100:.2f}%"
        )

    with col4:

        st.metric(
            "Final Supercap SOC",
            f"{selected_summary['final_supercap_soc'] * 100:.2f}%"
        )

    st.subheader(
        "Power Allocation"
    )

    power_fig = go.Figure()

    power_fig.add_trace(
        go.Scatter(
            x=simulation["time_s"],
            y=simulation["demand_kw"],
            mode="lines",
            name="Vehicle Demand"
        )
    )

    power_fig.add_trace(
        go.Scatter(
            x=simulation["time_s"],
            y=simulation["battery_power_kw"],
            mode="lines",
            name="Battery Power"
        )
    )

    power_fig.add_trace(
        go.Scatter(
            x=simulation["time_s"],
            y=simulation["supercap_power_kw"],
            mode="lines",
            name="Supercapacitor Power"
        )
    )

    power_fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Power (kW)",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        power_fig,
        use_container_width=True
    )

    st.subheader(
        "Energy Storage State of Charge"
    )

    soc_fig = go.Figure()

    soc_fig.add_trace(
        go.Scatter(
            x=simulation["time_s"],
            y=simulation["battery_soc"] * 100,
            mode="lines",
            name="Battery SOC"
        )
    )

    soc_fig.add_trace(
        go.Scatter(
            x=simulation["time_s"],
            y=simulation["supercap_soc"] * 100,
            mode="lines",
            name="Supercapacitor SOC"
        )
    )

    soc_fig.add_hline(
        y=10,
        line_dash="dash",
        annotation_text="Minimum SOC"
    )

    soc_fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="SOC (%)",
        hovermode="x unified",
        height=500,
        yaxis=dict(
            range=[0, 100]
        )
    )

    st.plotly_chart(
        soc_fig,
        use_container_width=True
    )


# ============================================================
# PAGE 5 — ML MODELS
# ============================================================

elif page == "ML Models":

    st.markdown(
        '<div class="section-title">Machine Learning Models</div>',
        unsafe_allow_html=True
    )

    if model_metrics_file.exists():

        with open(
            model_metrics_file,
            "r"
        ) as file:

            model_metrics = json.load(
                file
            )

        demand = model_metrics[
            "demand_predictor"
        ]

        policy = model_metrics[
            "student_policy"
        ]

        st.subheader(
            "1. Short-Term Power Demand Predictor"
        )

        st.write(
            """
A Random Forest Regressor predicts vehicle power
demand using the previous six power readings,
vehicle speed and acceleration.
"""
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "MAE",
                f"{demand['MAE']:.4f} kW"
            )

        with col2:

            st.metric(
                "RMSE",
                f"{demand['RMSE']:.4f} kW"
            )

        with col3:

            st.metric(
                "R²",
                f"{demand['R2']:.4f}"
            )

        st.divider()

        st.subheader(
            "2. ML Energy-Management Policy"
        )

        st.write(
            """
A Random Forest Regressor learns to predict the
battery power allocation generated by the constrained
teacher policy.
"""
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "MAE",
                f"{policy['MAE']:.4f} kW"
            )

        with col2:

            st.metric(
                "RMSE",
                f"{policy['RMSE']:.4f} kW"
            )

        with col3:

            st.metric(
                "R²",
                f"{policy['R2']:.4f}"
            )

        st.divider()

        st.subheader(
            "Model Features"
        )

        feature_data = pd.DataFrame(
            {
                "Model": [

                    "Demand Predictor",
                    "Demand Predictor",
                    "Demand Predictor",

                    "Student Policy",
                    "Student Policy",
                    "Student Policy"
                ],

                "Feature": [

                    "Previous 6 power readings",
                    "Vehicle speed",
                    "Vehicle acceleration",

                    "Power demand",
                    "Battery SOC",
                    "Supercapacitor SOC"
                ]
            }
        )

        st.dataframe(
            feature_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 6 — MODEL EXPLAINABILITY
# ============================================================

elif page == "Model Explainability":

    st.markdown(
        '<div class="section-title">Model Explainability</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
This section explains which input features influence the
Random Forest models and compares the trained student
policy with the teacher-generated control policy.
"""
    )

    # ========================================================
    # LOAD STUDENT MODEL
    # ========================================================

    if not student_model_file.exists():

        st.error(
            "Student policy model was not found."
        )

        st.stop()

    student_bundle = joblib.load(
        student_model_file
    )

    student_model = student_bundle[
        "model"
    ]

    student_features = student_bundle[
        "features"
    ]

    # ========================================================
    # STUDENT POLICY FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "1. Student Policy Feature Importance"
    )

    student_importance = pd.DataFrame(
        {
            "Feature": student_features,
            "Importance": (
                student_model
                .feature_importances_
            )
        }
    )

    student_importance = (
        student_importance
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    fig = px.bar(
        student_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Student Policy Feature Importance",
        text="Importance"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Relative Importance",
        yaxis_title=None,
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        student_importance.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
Feature importance shows how much each feature contributes
to the Random Forest's decision-making across the training
trees. It does not mean that a feature causes the output,
and it should not be interpreted as a causal relationship.
"""
    )

    st.divider()

    # ========================================================
    # DEMAND MODEL FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "2. Demand Predictor Feature Importance"
    )

    if demand_model_file.exists():

        demand_bundle = joblib.load(
            demand_model_file
        )

        demand_model = demand_bundle[
            "model"
        ]

        demand_features = demand_bundle[
            "features"
        ]

        demand_importance = pd.DataFrame(
            {
                "Feature": demand_features,

                "Importance": (
                    demand_model
                    .feature_importances_
                )
            }
        )

        demand_importance = (
            demand_importance
            .sort_values(
                "Importance",
                ascending=False
            )
        )

        fig = px.bar(
            demand_importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Demand Predictor Feature Importance",
            text="Importance"
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Relative Importance",
            yaxis_title=None,
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            demand_importance.round(4),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Demand predictor model was not found."
        )

    st.divider()

    # ========================================================
    # TEACHER VS STUDENT
    # ========================================================

    st.subheader(
        "3. Teacher vs Student Policy"
    )

    st.write(
        """
The student model was trained using control actions
generated by the constrained teacher policy.

This section lets you compare both decisions using
the same HESS conditions.
"""
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        explain_demand = st.number_input(
            "Power Demand (kW)",
            min_value=5.0,
            max_value=150.0,
            value=80.0,
            step=1.0,
            key="explain_demand"
        )

    with col2:

        explain_battery_soc = st.number_input(
            "Battery SOC (%)",
            min_value=10.0,
            max_value=95.0,
            value=70.0,
            step=1.0,
            key="explain_battery_soc"
        )

    with col3:

        explain_supercap_soc = st.number_input(
            "Supercapacitor SOC (%)",
            min_value=10.0,
            max_value=95.0,
            value=60.0,
            step=1.0,
            key="explain_supercap_soc"
        )

    compare_button = st.button(
        "Compare Teacher and Student",
        type="primary",
        use_container_width=True
    )

    if compare_button:

        battery_soc_fraction = (
            explain_battery_soc / 100.0
        )

        supercap_soc_fraction = (
            explain_supercap_soc / 100.0
        )

        # --------------------------------------------
        # TEACHER
        # --------------------------------------------

        teacher_prediction = teacher_action(
            explain_demand,
            battery_soc_fraction,
            supercap_soc_fraction
        )

        # --------------------------------------------
        # STUDENT
        # --------------------------------------------

        student_input = pd.DataFrame(
            [
                [
                    explain_demand,
                    battery_soc_fraction,
                    supercap_soc_fraction
                ]
            ],
            columns=student_features
        )

        student_prediction = float(
            student_model.predict(
                student_input
            )[0]
        )

        student_prediction = max(
            0.0,
            min(
                student_prediction,
                explain_demand
            )
        )

        difference = (
            student_prediction -
            teacher_prediction
        )

        absolute_difference = abs(
            difference
        )

        # --------------------------------------------
        # RESULTS
        # --------------------------------------------

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Teacher Battery Power",
                f"{teacher_prediction:.2f} kW"
            )

        with col2:

            st.metric(
                "Student Battery Power",
                f"{student_prediction:.2f} kW"
            )

        with col3:

            st.metric(
                "Absolute Difference",
                f"{absolute_difference:.2f} kW"
            )

        # --------------------------------------------
        # COMPARISON CHART
        # --------------------------------------------

        comparison_data = pd.DataFrame(
            {
                "Controller": [
                    "Teacher",
                    "Student"
                ],

                "Battery Power": [
                    teacher_prediction,
                    student_prediction
                ]
            }
        )

        fig = px.bar(
            comparison_data,
            x="Controller",
            y="Battery Power",
            text="Battery Power",
            title="Teacher vs Student Battery Power"
        )

        fig.update_traces(
            texttemplate="%{text:.2f} kW",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Battery Power (kW)",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------------------------
        # INTERPRETATION
        # --------------------------------------------

        if absolute_difference < 1.0:

            st.success(
                f"""
The student prediction is very close to the teacher
decision. The difference is only
{absolute_difference:.2f} kW.
"""
            )

        elif absolute_difference < 3.0:

            st.info(
                f"""
The student model is reasonably close to the teacher
decision. The difference is
{absolute_difference:.2f} kW.
"""
            )

        else:

            st.warning(
                f"""
The student and teacher decisions differ by
{absolute_difference:.2f} kW for this input.
This is useful for investigating regions where the
student model does not perfectly imitate the teacher.
"""
            )

        st.markdown(
            f"""
### Current Decision

For a demand of **{explain_demand:.2f} kW**:

- Battery SOC: **{explain_battery_soc:.1f}%**
- Supercapacitor SOC: **{explain_supercap_soc:.1f}%**
- Teacher battery allocation: **{teacher_prediction:.2f} kW**
- Student battery allocation: **{student_prediction:.2f} kW**
- Student minus teacher: **{difference:.2f} kW**

The student model is therefore acting as a learned
approximation of the teacher policy.
"""
        )

    st.divider()

    # ========================================================
    # TEACHER-STUDENT ARCHITECTURE
    # ========================================================

    st.subheader(
        "4. Teacher-Student Architecture"
    )

    teacher_student_architecture = """
HESS Conditions
      |
      +----------------------+
      |                      |
      v                      v
Teacher Policy         Training Dataset
      |                      |
      |                      v
      |                Random Forest
      |                      |
      +--------------------->+
                             |
                             v
                      Student Policy
                             |
                             v
                    Battery Power Allocation
"""

    st.code(
        teacher_student_architecture,
        language="text"
    )

    st.markdown(
        """
### Why use a teacher-student approach?

The teacher policy evaluates candidate battery-power
allocations while considering HESS constraints, estimated
losses and SOC conditions.

Those teacher-generated decisions become labeled training
data.

The Random Forest student then learns the relationship
between HESS state and the teacher's selected battery
power.

During inference, the trained student can directly predict
the battery allocation without performing the full
candidate-search process used by the teacher.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
HESS-ML | Educational Machine Learning Simulation |
Battery + Supercapacitor Energy Management
"""
)