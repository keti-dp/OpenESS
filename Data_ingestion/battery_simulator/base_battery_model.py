import pandas as pd
import numpy as np


class BaseBatteryModel:
    """
    모든 배터리 모델이 상속하는 기본 클래스입니다.

    배터리 충방전 데이터를 CSV 파일 경로나 pandas DataFrame 형태로 입력받습니다.
    데이터는 반드시 'timestamp', 'voltage', 'current' 열을 포함해야 합니다.
    만약 data 인자가 제공되지 않으면, 기본 임의 충방전 데이터를 생성합니다.
    """

    def __init__(self, data=None):
        if data is None:
            # 기본 임의 데이터 생성: 10개의 데이터 포인트
            timestamps = pd.date_range(start="2021-01-01", periods=10, freq="H")
            voltage = np.linspace(4.2, 3.0, num=10)  # 예: 4.2V에서 3.0V까지 선형 감소
            current = np.full(10, 1.0)  # 예: 일정한 1.0 A
            self.data = pd.DataFrame(
                {"timestamp": timestamps, "voltage": voltage, "current": current}
            )
        else:
            if isinstance(data, str):
                # CSV 파일 경로가 입력된 경우
                self.data = pd.read_csv(data)
            elif isinstance(data, pd.DataFrame):
                self.data = data
            else:
                raise ValueError(
                    "data는 CSV 파일 경로나 pandas DataFrame이어야 합니다."
                )

    def simulate(self):
        """
        배터리 모델의 시뮬레이션 로직을 구현해야 합니다.
        각 서브클래스에서 구체적인 시뮬레이션을 구현하세요.
        """
        raise NotImplementedError("simulate() 메소드를 서브클래스에서 구현하세요.")
