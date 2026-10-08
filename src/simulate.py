from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .driving import generate_driving_cycle
from .hess import HESS
from .teacher import teacher_action


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation(
    strategy="ml",
    n_steps=900,
    seed=123
):

    # --------------------------------------------------------
    # Generate driving cycle
    # --------------------------------------------------------

    driving_data = generate_driving_cycle(
        n_steps=n_steps,
        seed=seed
    )

    # --------------------------------------------------------
    # Create HESS system
    # --------------------------------------------------------

    hess = HESS()

    # --------------------------------------------------------
    # Load ML policy
    # --------------------------------------------------------

    model = None
    model_features = None

    if strategy == "ml":

        model_bundle = joblib.load(
            ROOT /
            "models" /
            "student_policy.pkl"
        )

        model = model_bundle["model"]

        # These are the exact feature names used
        # while training the model.

        model_features = model_bundle["features"]

        # IMPORTANT:
        # Prediction is performed one step at a time.
        # Parallel prediction for a single sample creates
        # unnecessary overhead.

        model.n_jobs = 1

    # --------------------------------------------------------
    # Store simulation results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Simulation loop
    # --------------------------------------------------------

    for _, row in driving_data.iterrows():

        # ----------------------------------------------------
        # Vehicle power demand
        # ----------------------------------------------------

        demand = float(
            row["power_demand_kw"]
        )

        # ----------------------------------------------------
        # BATTERY ONLY
        # ----------------------------------------------------

        if strategy == "battery_only":

            battery_power = min(
                demand,
                hess.battery.max_power_kw
            )

        # ----------------------------------------------------
        # RULE-BASED / TEACHER
        # ----------------------------------------------------

        elif strategy == "rule":

            battery_power = teacher_action(

                demand,

                hess.battery.soc,

                hess.supercap.soc
            )

        # ----------------------------------------------------
        # MACHINE LEARNING POLICY
        # ----------------------------------------------------

        elif strategy == "ml":

            # Create a one-row DataFrame using the SAME
            # feature names used during model training.

            X = pd.DataFrame(

                [[

                    demand,

                    hess.battery.soc,

                    hess.supercap.soc

                ]],

                columns=model_features
            )

            # Predict battery power.

            battery_power = float(
                model.predict(X)[0]
            )

        else:

            raise ValueError(
                "Unknown strategy. "
                "Use 'ml', 'rule', or 'battery_only'."
            )

        # ----------------------------------------------------
        # Safety limits
        # ----------------------------------------------------

        battery_power = max(
            0.0,
            min(
                battery_power,
                hess.battery.max_power_kw
            )
        )

        # ----------------------------------------------------
        # Update HESS
        # ----------------------------------------------------

        result = hess.step(

            demand,

            battery_power
        )

        # ----------------------------------------------------
        # Add driving information
        # ----------------------------------------------------

        result.update({

            "time_s":
                row["time_s"],

            "speed_kmh":
                row["speed_kmh"],

            "strategy":
                strategy
        })

        results.append(
            result
        )

    # --------------------------------------------------------
    # Convert results to DataFrame
    # --------------------------------------------------------

    return pd.DataFrame(
        results
    )


# ============================================================
# CALCULATE SUMMARY
# ============================================================

def calculate_summary(
    dataframe
):

    return {

        # Total requested energy
        "energy_demand_kwh":
            float(
                dataframe[
                    "demand_kw"
                ].sum() / 3600
            ),

        # Energy supplied by battery
        "battery_energy_kwh":
            float(
                dataframe[
                    "battery_power_kw"
                ].sum() / 3600
            ),

        # Maximum battery power
        "peak_battery_power_kw":
            float(
                dataframe[
                    "battery_power_kw"
                ].max()
            ),

        # RMS battery power
        "battery_rms_power_kw":
            float(
                np.sqrt(
                    np.mean(
                        dataframe[
                            "battery_power_kw"
                        ] ** 2
                    )
                )
            ),

        # Average battery loss
        "mean_battery_loss_w":
            float(
                dataframe[
                    "battery_loss_w"
                ].mean()
            ),

        # Final battery SOC
        "final_battery_soc":
            float(
                dataframe[
                    "battery_soc"
                ].iloc[-1]
            ),

        # Final supercapacitor SOC
        "final_supercap_soc":
            float(
                dataframe[
                    "supercap_soc"
                ].iloc[-1]
            )
    }