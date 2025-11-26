# battery_simulator/battery_models/experiment/experiment.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class Experiment:
    def __init__(self, params, ocv_data, Ri=0.02, R_rc=0.015, C_rc=2400, dt=1):
        self.params = params
        self.ocv_data = ocv_data
        self.soc_arr = np.linspace(1, 0, len(ocv_data))
        self.Ri = Ri
        self.R_rc = R_rc
        self.C_rc = C_rc
        self.dt = dt

    def soc_to_ocv(self, soc):
        return np.interp(soc, self.soc_arr[::-1], self.ocv_data[::-1])

    def run(self, cycles=3):
        timestamp = datetime.now()
        soc = 1.0  # 초기 SOC
        V_rc = 0  # RC 회로 초기 전압
        data = []

        for _ in range(cycles):
            # 방전 과정
            current = -self.params["discharge_C_rate"] * self.params["nominal_capacity"]
            while (
                soc > 0
                and self.soc_to_ocv(soc) > self.params["discharge_cutoff_voltage"]
            ):
                ocv = self.soc_to_ocv(soc)
                V_terminal = ocv + current * self.Ri + V_rc
                data.append([timestamp, V_terminal, current])
                soc += current / self.params["nominal_capacity"] / 3600 * self.dt
                V_rc = V_rc * np.exp(
                    -self.dt / (self.R_rc * self.C_rc)
                ) + current * self.R_rc * (
                    1 - np.exp(-self.dt / (self.R_rc * self.C_rc))
                )
                timestamp += timedelta(seconds=self.dt)

            # 휴지 (Rest)
            rest_time = self.params["rest_time_hours"] * 3600
            for _ in range(int(rest_time)):
                V_rc *= np.exp(-self.dt / (self.R_rc * self.C_rc))
                ocv = self.soc_to_ocv(soc)
                V_terminal = ocv + V_rc
                data.append([timestamp, V_terminal, 0])
                timestamp += timedelta(seconds=self.dt)

            # 충전 과정
            current = self.params["charge_C_rate"] * self.params["nominal_capacity"]
            while (
                soc < 1 and self.soc_to_ocv(soc) < self.params["charge_cutoff_voltage"]
            ):
                ocv = self.soc_to_ocv(soc)
                V_terminal = ocv + current * self.Ri + V_rc
                data.append([timestamp, V_terminal, current])
                soc += current / self.params["nominal_capacity"] / 3600 * self.dt
                V_rc = V_rc * np.exp(
                    -self.dt / (self.R_rc * self.C_rc)
                ) + current * self.R_rc * (
                    1 - np.exp(-self.dt / (self.R_rc * self.C_rc))
                )
                timestamp += timedelta(seconds=self.dt)

            # 충전 후 휴지 (Rest)
            for _ in range(int(rest_time)):
                V_rc *= np.exp(-self.dt / (self.R_rc * self.C_rc))
                ocv = self.soc_to_ocv(soc)
                V_terminal = ocv + V_rc
                data.append([timestamp, V_terminal, 0])
                timestamp += timedelta(seconds=self.dt)

        df = pd.DataFrame(data, columns=["TIMESTAMP", "voltage", "current"])
        return df
