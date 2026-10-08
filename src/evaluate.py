from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from .simulate import run_simulation, calculate_summary


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_improvement(baseline, proposed):
    """
    Calculate percentage improvement.

    Positive value means the proposed controller
    is lower than the baseline.
    """

    if baseline == 0:
        return 0.0

    return ((baseline - proposed) / baseline) * 100.0


def run_evaluation():

    strategies = [
        "ml",
        "rule",
        "battery_only"
    ]

    summaries = {}

    simulation_data = {}

    print("================================")
    print("       HESS CONTROLLER EVALUATION")
    print("================================")

    for strategy in strategies:

        print(f"\nRunning: {strategy}")

        dataframe = run_simulation(
            strategy=strategy,
            n_steps=900,
            seed=123
        )

        summary = calculate_summary(dataframe)

        summaries[strategy] = summary
        simulation_data[strategy] = dataframe

        print(summary)

        output_path = (
            RESULTS_DIR /
            f"{strategy}_simulation.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False
        )

    # ---------------------------------------------------------
    # Create comparison table
    # ---------------------------------------------------------

    comparison = pd.DataFrame(summaries).T

    comparison.index.name = "strategy"

    comparison_path = (
        RESULTS_DIR /
        "controller_comparison.csv"
    )

    comparison.to_csv(
        comparison_path
    )

    # ---------------------------------------------------------
    # Calculate ML improvement over baseline
    # ---------------------------------------------------------

    baseline = summaries["battery_only"]
    ml = summaries["ml"]

    improvements = {

        "battery_energy_reduction_percent":
            calculate_improvement(
                baseline["battery_energy_kwh"],
                ml["battery_energy_kwh"]
            ),

        "peak_battery_power_reduction_percent":
            calculate_improvement(
                baseline["peak_battery_power_kw"],
                ml["peak_battery_power_kw"]
            ),

        "battery_rms_power_reduction_percent":
            calculate_improvement(
                baseline["battery_rms_power_kw"],
                ml["battery_rms_power_kw"]
            ),

        "mean_battery_loss_reduction_percent":
            calculate_improvement(
                baseline["mean_battery_loss_w"],
                ml["mean_battery_loss_w"]
            )
    }

    # ---------------------------------------------------------
    # Save metrics JSON
    # ---------------------------------------------------------

    evaluation_results = {
        "controllers": summaries,
        "ml_vs_battery_priority": improvements
    }

    metrics_path = (
        METRICS_DIR /
        "controller_evaluation.json"
    )

    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            evaluation_results,
            file,
            indent=4
        )

    # ---------------------------------------------------------
    # Plot 1: Battery Energy
    # ---------------------------------------------------------

    labels = [
        "ML",
        "Rule-Based",
        "Battery Priority"
    ]

    strategies_display = [
        "ml",
        "rule",
        "battery_only"
    ]

    battery_energy = [
        summaries[s]["battery_energy_kwh"]
        for s in strategies_display
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        labels,
        battery_energy
    )

    plt.ylabel("Battery Energy (kWh)")
    plt.title("Battery Energy Consumption Comparison")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "battery_energy_comparison.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Peak Battery Power
    # ---------------------------------------------------------

    peak_power = [
        summaries[s]["peak_battery_power_kw"]
        for s in strategies_display
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        labels,
        peak_power
    )

    plt.ylabel("Peak Battery Power (kW)")
    plt.title("Peak Battery Power Comparison")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "peak_battery_power_comparison.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Plot 3: RMS Battery Power
    # ---------------------------------------------------------

    rms_power = [
        summaries[s]["battery_rms_power_kw"]
        for s in strategies_display
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        labels,
        rms_power
    )

    plt.ylabel("RMS Battery Power (kW)")
    plt.title("RMS Battery Power Comparison")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "rms_battery_power_comparison.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Plot 4: Battery Loss
    # ---------------------------------------------------------

    losses = [
        summaries[s]["mean_battery_loss_w"]
        for s in strategies_display
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        labels,
        losses
    )

    plt.ylabel("Mean Battery Loss (W)")
    plt.title("Estimated Battery Resistive Loss Comparison")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "battery_loss_comparison.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Print final results
    # ---------------------------------------------------------

    print("\n================================")
    print("       FINAL COMPARISON")
    print("================================")

    print("\n")
    print(comparison)

    print("\n================================")
    print("     ML vs BATTERY BASELINE")
    print("================================")

    for key, value in improvements.items():

        print(
            f"{key}: {value:.2f}%"
        )

    print("\nEvaluation completed successfully.")

    print("\nGenerated files:")

    print(
        "results/controller_comparison.csv"
    )

    print(
        "results/metrics/controller_evaluation.json"
    )

    print(
        "results/figures/"
    )


if __name__ == "__main__":
    run_evaluation()