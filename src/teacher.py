import numpy as np
from . import config


def estimate_battery_loss(power_kw):
    power_w = abs(power_kw) * 1000.0
    current_a = power_w / config.BATTERY_VOLTAGE

    return (
        current_a ** 2 *
        config.BATTERY_RESISTANCE_OHM
    )


def estimate_supercap_loss(power_kw):
    power_w = abs(power_kw) * 1000.0
    current_a = power_w / config.BATTERY_VOLTAGE

    return (
        current_a ** 2 *
        config.SUPERCAP_RESISTANCE_OHM
    )


def teacher_action(
    demand_kw,
    battery_soc,
    supercap_soc
):
    """
    Constrained teacher policy for the HESS.

    Strategy:

    High supercapacitor SOC:
        Battery provides approximately 55% of demand.

    Medium supercapacitor SOC:
        Battery contribution gradually increases.

    Low supercapacitor SOC:
        Battery provides most of the demand.

    Minimum supercapacitor SOC:
        Battery supplies the demand.
    """

    demand_kw = max(
        0.0,
        float(demand_kw)
    )

    battery_soc = float(battery_soc)
    supercap_soc = float(supercap_soc)

    # ---------------------------------------------------------
    # Maximum battery power
    # ---------------------------------------------------------

    max_battery_power = min(
        demand_kw,
        config.BATTERY_MAX_POWER_KW
    )

    # ---------------------------------------------------------
    # Determine desired battery fraction
    # ---------------------------------------------------------

    # Normal operating point
    normal_fraction = 0.55

    # Supercapacitor reserve region
    reserve_soc = 0.25

    # Minimum SOC
    minimum_soc = config.SOC_MIN

    if supercap_soc <= minimum_soc:

        # Supercapacitor cannot discharge
        desired_fraction = 1.0

    elif supercap_soc < reserve_soc:

        # Gradually increase battery contribution as
        # supercapacitor SOC approaches its minimum.

        soc_ratio = (
            supercap_soc - minimum_soc
        ) / (
            reserve_soc - minimum_soc
        )

        desired_fraction = (
            1.0 -
            soc_ratio * (1.0 - normal_fraction)
        )

    else:

        desired_fraction = normal_fraction

    desired_battery_power = (
        demand_kw *
        desired_fraction
    )

    desired_battery_power = min(
        desired_battery_power,
        max_battery_power
    )

    # ---------------------------------------------------------
    # Candidate battery powers around desired value
    # ---------------------------------------------------------

    lower_bound = max(
        0.0,
        desired_battery_power - 10.0
    )

    upper_bound = min(
        max_battery_power,
        desired_battery_power + 10.0
    )

    candidates = np.linspace(
        lower_bound,
        upper_bound,
        51
    )

    best_battery_power = desired_battery_power
    best_cost = float("inf")

    # ---------------------------------------------------------
    # Evaluate candidates
    # ---------------------------------------------------------

    for battery_power in candidates:

        supercap_power = (
            demand_kw -
            battery_power
        )

        # Physical power constraints
        if supercap_power < 0:
            continue

        if supercap_power > config.SUPERCAP_MAX_POWER_KW:
            continue

        # -----------------------------------------------------
        # Estimate battery SOC after timestep
        # -----------------------------------------------------

        battery_energy_kwh = (
            battery_power *
            config.DT_HOURS
        )

        battery_soc_after = (
            battery_soc -
            battery_energy_kwh /
            config.BATTERY_CAPACITY_KWH
        )

        # -----------------------------------------------------
        # Estimate supercap SOC after timestep
        # -----------------------------------------------------

        supercap_energy_kwh = (
            supercap_power *
            config.DT_HOURS /
            config.SUPERCAP_EFFICIENCY
        )

        supercap_soc_after = (
            supercap_soc -
            supercap_energy_kwh /
            config.SUPERCAP_CAPACITY_KWH
        )

        # -----------------------------------------------------
        # Hard SOC constraints
        # -----------------------------------------------------

        if battery_soc_after < config.SOC_MIN:
            continue

        if supercap_soc_after < config.SOC_MIN:
            continue

        # -----------------------------------------------------
        # Losses
        # -----------------------------------------------------

        battery_loss = estimate_battery_loss(
            battery_power
        )

        supercap_loss = estimate_supercap_loss(
            supercap_power
        )

        # -----------------------------------------------------
        # Battery stress
        # -----------------------------------------------------

        battery_stress = (
            battery_power /
            config.BATTERY_MAX_POWER_KW
        ) ** 2

        # -----------------------------------------------------
        # Desired allocation penalty
        # -----------------------------------------------------

        allocation_error = (
            (
                battery_power -
                desired_battery_power
            )
            /
            config.BATTERY_MAX_POWER_KW
        ) ** 2

        # -----------------------------------------------------
        # Supercap reserve penalty
        # -----------------------------------------------------

        reserve_penalty = max(
            0.0,
            reserve_soc - supercap_soc_after
        ) ** 2

        # -----------------------------------------------------
        # Battery low-SOC penalty
        # -----------------------------------------------------

        battery_low_penalty = max(
            0.0,
            0.30 - battery_soc_after
        ) ** 2

        # -----------------------------------------------------
        # Total cost
        # -----------------------------------------------------

        cost = (
            1.0 * battery_loss
            + 0.25 * supercap_loss
            + 10.0 * battery_stress
            + 50.0 * allocation_error
            + 200.0 * reserve_penalty
            + 100.0 * battery_low_penalty
        )

        if cost < best_cost:

            best_cost = cost
            best_battery_power = battery_power

    return float(best_battery_power)