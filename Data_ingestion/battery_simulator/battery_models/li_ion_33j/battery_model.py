import pandas as pd
from battery_simulator.base_battery_model import BaseBatteryModel


class LiIon33J(BaseBatteryModel):
    """
    리튬이온 33J 배터리 모델 클래스
    """

    def __init__(self, data=None, ocv_path=None):
        super().__init__(data)
        if ocv_path is None:
            from battery_simulator.parameters.ocv import get_ocv_cell

            self.ocv_data = get_ocv_cell()
        else:
            self.ocv_path = ocv_path
            self.ocv_data = pd.read_csv(ocv_path)

    def rls(
        self,
        init_Ri=0.0310707902382320,
        init_Rdiff=0.0190371443335961,
        init_Cdiff=6093.350870660123,
    ):
        from .rls import RLS

        return RLS(self, init_Ri=init_Ri, init_Rdiff=init_Rdiff, init_Cdiff=init_Cdiff)

    def experiment(
        self,
        steps,
        nominal_capacity,
        v_max,
        v_min,
        ocv_data,
        Ri,  # li_ion_33j 모델의 실제 내부저항 값으로 설정
        R_rc,  # li_ion_33j 모델의 실제 RC 저항 값으로 설정
        C_rc,  # li_ion_33j 모델의 실제 RC 커패시턴스 값으로 설정
        dt,  # 데이터 간격 (초)
    ):
        from .experiment import Experiment

        return Experiment(
            self,
            steps=steps,
            nominal_capacity=nominal_capacity,
            v_max=v_max,
            v_min=v_min,
            ocv_data=ocv_data,
            Ri=Ri,  # li_ion_33j 모델의 실제 내부저항 값으로 설정
            R_rc=R_rc,  # li_ion_33j 모델의 실제 RC 저항 값으로 설정
            C_rc=C_rc,  # li_ion_33j 모델의 실제 RC 커패시턴스 값으로 설정
            dt=dt,  # 데이터 간격 (초)
        )
