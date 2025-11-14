# battery_simulator/battery_models/li_ion_41j/rls.py

from .battery_model import LiIon41J


class RLS:
    """
    LiIon41J 모델에 대한 RLS 분석 모델 클래스입니다.

    이 클래스는 배터리 파라미터와 RLS 전용 파라미터를 입력받아 시뮬레이션을 수행합니다.

    Attributes:
        battery (LiIon41J): 배터리 모델 인스턴스
        rls_param1: RLS 분석 파라미터1
        rls_param2: RLS 분석 파라미터2
    """

    def __init__(self, battery_params, rls_param1, rls_param2):
        self.battery = LiIon41J(**battery_params)
        self.rls_param1 = rls_param1
        self.rls_param2 = rls_param2

    def simulate(self):
        """
        예시 시뮬레이션 메소드: 시간에 따른 전압 감소를 가정합니다.

        Returns:
            time (list): 시간 데이터
            data (list): 시뮬레이션 결과 전압 데이터
        """
        time = list(range(10))
        data = [self.battery.voltage * (1 - i * 0.1) for i in time]
        return time, data
