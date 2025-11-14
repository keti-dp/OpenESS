# battery_simulator/battery_models/li_ion_33j/experiment.py

from battery_simulator.battery_models.experiment.experiment import (
    Experiment as BaseExperiment,
)
from battery_simulator.parameters.ocv import get_ocv_cell


class Experiment(BaseExperiment):
    def __init__(self, params):
        ocv_data = get_ocv_cell()["ocv"].values  # 'ocv' 컬럼만 1차원 배열로 추출
        super().__init__(
            params=params,
            ocv_data=ocv_data,
            Ri=0.02,  # 모델에 맞는 값 설정
            R_rc=0.015,
            C_rc=2400,
            dt=1,
        )
