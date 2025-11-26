from battery_simulator.base_battery_model import BaseBatteryModel


class LiIon41J(BaseBatteryModel):
    """
    리튬이온 41J 배터리 모델 클래스입니다.

    입력:
        data: 배터리 충방전 데이터 (CSV 파일 경로나 pandas DataFrame)
              반드시 'timestamp', 'voltage', 'current' 열을 포함해야 합니다.
              입력하지 않으면 기본 임의 데이터를 사용합니다.
    """

    def __init__(self, data=None):
        super().__init__(data)
