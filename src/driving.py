import numpy as np
import pandas as pd


def generate_driving_cycle(
    n_steps=3600,
    seed=42
):

    rng = np.random.default_rng(seed)

    time = np.arange(n_steps)

    speed = (
        45
        + 18 * np.sin(time / 140)
        + 12 * np.sin(time / 37)
        + 5 * np.sin(time / 11)
        + rng.normal(0, 1.8, n_steps)
    )

    speed = np.clip(
        speed,
        0,
        110
    )

    acceleration = np.gradient(
        speed
    )

    # Simplified illustrative vehicle power model
    power_demand = (
        12
        + 0.65 * speed
        + 10.0 * np.maximum(
            acceleration,
            0
        )
        + 3.0 * np.maximum(
            -acceleration,
            0
        )
        + rng.normal(
            0,
            2.5,
            n_steps
        )
    )

    power_demand = np.clip(
        power_demand,
        5,
        150
    )

    return pd.DataFrame({

        "time_s":
            time,

        "speed_kmh":
            speed,

        "acceleration_kmh_s":
            acceleration,

        "power_demand_kw":
            power_demand
    })