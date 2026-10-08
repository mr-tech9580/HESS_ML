from .battery import Battery
from .supercapacitor import Supercapacitor
from . import config


class HESS:

    def __init__(self):

        self.battery = Battery(
            capacity_kwh=config.BATTERY_CAPACITY_KWH,
            max_power_kw=config.BATTERY_MAX_POWER_KW,
            soc=config.BATTERY_INITIAL_SOC,
            efficiency=config.BATTERY_EFFICIENCY,
            resistance_ohm=config.BATTERY_RESISTANCE_OHM
        )

        self.supercap = Supercapacitor(
            capacity_kwh=config.SUPERCAP_CAPACITY_KWH,
            max_power_kw=config.SUPERCAP_MAX_POWER_KW,
            soc=config.SUPERCAP_INITIAL_SOC,
            efficiency=config.SUPERCAP_EFFICIENCY,
            resistance_ohm=config.SUPERCAP_RESISTANCE_OHM
        )

    def get_max_safe_supercap_power(self):
        """
        Maximum supercapacitor discharge power that can be
        used during the current timestep without going below
        SOC_MIN.
        """

        available_energy_kwh = (
            self.supercap.soc - config.SOC_MIN
        ) * self.supercap.capacity_kwh

        if available_energy_kwh <= 0:
            return 0.0

        # Account for discharge efficiency
        max_power_kw = (
            available_energy_kwh
            * self.supercap.efficiency
            / config.DT_HOURS
        )

        return min(
            max_power_kw,
            self.supercap.max_power_kw
        )

    def step(self, demand_kw, battery_power_kw):

        demand_kw = max(0.0, float(demand_kw))

        # ---------------------------------------------------------
        # Clamp requested battery power
        # ---------------------------------------------------------

        battery_power_kw = max(
            0.0,
            min(
                float(battery_power_kw),
                self.battery.max_power_kw
            )
        )

        # ---------------------------------------------------------
        # Calculate required supercapacitor power
        # ---------------------------------------------------------

        requested_supercap_power = (
            demand_kw - battery_power_kw
        )

        requested_supercap_power = max(
            0.0,
            requested_supercap_power
        )

        # ---------------------------------------------------------
        # Enforce physical supercapacitor limits
        # ---------------------------------------------------------

        max_safe_supercap_power = (
            self.get_max_safe_supercap_power()
        )

        supercap_power_kw = min(
            requested_supercap_power,
            max_safe_supercap_power
        )

        # ---------------------------------------------------------
        # If supercapacitor cannot provide enough power,
        # increase battery power.
        # ---------------------------------------------------------

        remaining_power = (
            demand_kw - supercap_power_kw
        )

        battery_power_kw = min(
            max(
                battery_power_kw,
                remaining_power
            ),
            self.battery.max_power_kw
        )

        # ---------------------------------------------------------
        # If battery + supercap still cannot satisfy demand,
        # record unmet power.
        # ---------------------------------------------------------

        delivered_power_kw = (
            battery_power_kw +
            supercap_power_kw
        )

        unmet_power_kw = max(
            0.0,
            demand_kw - delivered_power_kw
        )

        # ---------------------------------------------------------
        # Update storage systems
        # ---------------------------------------------------------

        battery_result = self.battery.step(
            battery_power_kw
        )

        supercap_result = self.supercap.step(
            supercap_power_kw
        )

        return {
            "demand_kw": demand_kw,
            "delivered_power_kw": delivered_power_kw,

            "battery_power_kw": battery_power_kw,
            "supercap_power_kw": supercap_power_kw,

            "battery_soc": battery_result["soc"],
            "supercap_soc": supercap_result["soc"],

            "battery_loss_w": battery_result["loss_w"],
            "supercap_loss_w": supercap_result["loss_w"],

            "unmet_power_kw": unmet_power_kw
        }