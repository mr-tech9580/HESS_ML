from dataclasses import dataclass

from .config import (
    DT_HOURS,
    SOC_MIN,
    SOC_MAX,
    BATTERY_VOLTAGE
)


@dataclass
class Supercapacitor:

    capacity_kwh: float
    max_power_kw: float
    soc: float
    efficiency: float
    resistance_ohm: float

    def step(
        self,
        power_kw: float,
        dt_hours: float = DT_HOURS
    ):

        power_kw = max(
            -self.max_power_kw,
            min(power_kw, self.max_power_kw)
        )

        if power_kw >= 0:

            energy_terminal = (
                power_kw * dt_hours
            )

            energy_from_storage = (
                energy_terminal /
                self.efficiency
            )

            self.soc -= (
                energy_from_storage /
                self.capacity_kwh
            )

        else:

            charge_power = abs(power_kw)

            energy_stored = (
                charge_power *
                dt_hours *
                self.efficiency
            )

            self.soc += (
                energy_stored /
                self.capacity_kwh
            )

        self.soc = max(
            SOC_MIN,
            min(SOC_MAX, self.soc)
        )

        current_a = abs(
            power_kw * 1000 /
            BATTERY_VOLTAGE
        )

        loss_w = (
            current_a ** 2 *
            self.resistance_ohm
        )

        return {
            "power_kw": power_kw,
            "soc": self.soc,
            "loss_w": loss_w
        }