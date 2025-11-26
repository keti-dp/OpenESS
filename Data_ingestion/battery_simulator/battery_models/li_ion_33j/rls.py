import numpy as np
import pandas as pd


class RLS:
    """
    LiIon33J 모델에 대한 RLS 분석 모델 클래스입니다.

    이 클래스는 LiIon33J 배터리 모델 인스턴스를 입력받아,
    해당 배터리의 충방전 데이터와 ocv 데이터를 기반으로 RLS 알고리즘을 수행합니다.

    Parameters:
        battery (LiIon33J): LiIon33J 배터리 모델 인스턴스 (ocv_data 정보 포함).
        init_Ri (float): 초기 Ri 값.
        init_Rdiff (float): 초기 R_diff 값.
        init_Cdiff (float): 초기 C_diff 값.
    """

    def __init__(self, battery, init_Ri, init_Rdiff, init_Cdiff):
        self.battery = battery
        self.init_Ri = init_Ri
        self.init_Rdiff = init_Rdiff
        self.init_Cdiff = init_Cdiff

    def simulate(self):
        """
        배터리 충방전 데이터에 대해 RLS 알고리즘을 수행하여
        각 타임스탬프별로 R_diff와 C_diff 값을 계산합니다.

        ocv 테이블은 self.battery.ocv_data를 이용합니다.
        (ocv_data가 DataFrame 형태로 존재할 경우, 'ocv' 열의 값을 시간 길이에 맞게 보간하여 사용)

        Returns:
            pandas.DataFrame: TIMESTAMP, Rdiff, Cdiff, voltage, current 열을 포함하는 결과 DataFrame.
        """
        # 배터리 데이터 추출 (DataFrame에는 'timestamp', 'voltage', 'current' 열이 포함되어야 함)
        voltage_data = self.battery.data["voltage"].values
        current_data = self.battery.data["current"].values

        if "timestamp" in self.battery.data.columns:
            timestamps = pd.to_datetime(self.battery.data["timestamp"]).values
        else:
            timestamps = np.arange(len(voltage_data))

        # ocv_data 처리: battery 인스턴스의 ocv_data (DataFrame, 'ocv' 열 포함)를
        # voltage_data 길이에 맞게 선형 보간하여 사용
        if hasattr(self.battery, "ocv_data"):
            ocv_df = self.battery.ocv_data
            if "ocv" in ocv_df.columns:
                ocv_values = ocv_df["ocv"].values
                # ocv_values 길이는 예를 들어 21개일 수 있으므로, 보간하여 voltage_data와 동일한 길이로 만듭니다.
                x = np.linspace(0, 1, len(ocv_values))
                x_new = np.linspace(0, 1, len(voltage_data))
                OCV_data = np.interp(x_new, x, ocv_values)
            else:
                OCV_data = voltage_data.copy()
        else:
            OCV_data = voltage_data.copy()

        # RLS 알고리즘 초기화
        P = np.eye(3)  # 초기 오차 공분산 행렬
        ForgettingFactor = 0.99999999

        theta = np.array(
            [
                self.init_Ri,
                (
                    -self.init_Ri
                    + 1 / self.init_Cdiff
                    + self.init_Ri / (self.init_Rdiff * self.init_Cdiff)
                ),
                (1 / (self.init_Rdiff * self.init_Cdiff) - 1),
            ]
        ).reshape(-1, 1)

        results = []
        ECM_voltage = np.zeros(len(voltage_data))
        I_diff_1RC = np.zeros(len(voltage_data))

        for t in range(1, len(voltage_data)):
            voltage = voltage_data[t]
            current = current_data[t]
            prev_current = current_data[t - 1]
            ocv = OCV_data[t]
            timestamp = timestamps[t]

            overpotential = ocv - voltage
            phi = np.array([current, prev_current, overpotential]).reshape(-1, 1)

            RlsEstimatedVoltage = theta.T @ phi
            RlsVoltageError = voltage - ocv - RlsEstimatedVoltage

            P_phi = P @ phi
            gain_denominator = ForgettingFactor + phi.T @ P @ phi
            gain = P_phi / gain_denominator
            P = (P - gain @ phi.T @ P) / ForgettingFactor
            P = np.clip(P, 0, 1e-11)

            theta = theta + gain @ RlsVoltageError
            theta = np.abs(theta)

            Ri = theta[0, 0]
            Rdiff = (theta[1, 0] - theta[2, 0] * theta[0, 0]) / (1 - theta[2, 0])
            Cdiff = 1 / (theta[1, 0] - theta[2, 0] * theta[0, 0])

            Ri = np.abs(Ri)
            Rdiff = np.abs(Rdiff)
            Cdiff = np.abs(Cdiff)

            Vi = current * Ri
            I_diff_1RC[t] = I_diff_1RC[t - 1] * np.exp(
                -1 / (Rdiff * Cdiff)
            ) + prev_current * (1 - np.exp(-1 / (Rdiff * Cdiff)))
            Vdiff = I_diff_1RC[t] * Rdiff
            ECM_voltage[t] = ocv + Vi + Vdiff

            results.append(
                {
                    "TIMESTAMP": timestamp,
                    "overpotential": overpotential,
                    "Ri": Ri,
                    "Rdiff": Rdiff,
                    "Cdiff": Cdiff,
                    "voltage": voltage,
                    "current": current,
                }
            )

        return pd.DataFrame(results)
