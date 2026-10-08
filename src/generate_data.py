from pathlib import Path

import numpy as np
import pandas as pd

from .driving import generate_driving_cycle
from .teacher import teacher_action


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ROOT /
    "data" /
    "raw"
)

PROCESSED_DIR = (
    ROOT /
    "data" /
    "processed"
)


def create_demand_dataset(
    dataframe
):

    dataframe = dataframe.copy()

    for lag in range(1, 7):

        dataframe[
            f"power_lag_{lag}"
        ] = dataframe[
            "power_demand_kw"
        ].shift(lag)

    dataframe = (
        dataframe
        .dropna()
        .reset_index(drop=True)
    )

    return dataframe


def create_policy_dataset(
    n_samples=20000,
    seed=42
):

    rng = np.random.default_rng(seed)

    demand = rng.uniform(
        5,
        150,
        n_samples
    )

    battery_soc = rng.uniform(
        0.10,
        0.95,
        n_samples
    )

    supercap_soc = rng.uniform(
        0.10,
        0.95,
        n_samples
    )

    battery_power = []

    for d, b_soc, s_soc in zip(
        demand,
        battery_soc,
        supercap_soc
    ):

        action = teacher_action(
            d,
            b_soc,
            s_soc
        )

        battery_power.append(
            action
        )

    return pd.DataFrame({

        "demand_kw":
            demand,

        "battery_soc":
            battery_soc,

        "supercap_soc":
            supercap_soc,

        "battery_power_kw":
            battery_power
    })


if __name__ == "__main__":

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    driving_data = (
        generate_driving_cycle()
    )

    driving_data.to_csv(
        RAW_DIR /
        "driving_cycle.csv",
        index=False
    )

    demand_dataset = (
        create_demand_dataset(
            driving_data
        )
    )

    demand_dataset.to_csv(
        PROCESSED_DIR /
        "demand_dataset.csv",
        index=False
    )

    policy_dataset = (
        create_policy_dataset()
    )

    policy_dataset.to_csv(
        PROCESSED_DIR /
        "policy_dataset.csv",
        index=False
    )

    print(
        "Driving data created."
    )

    print(
        "Demand dataset created."
    )

    print(
        "Policy dataset created."
    )