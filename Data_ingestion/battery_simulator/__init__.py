"""
battery_simulator 패키지
이 패키지는 배터리 모델링과 시뮬레이션을 위한 모듈들을 포함합니다.

모듈:
    - base_battery_model: 배터리 모델의 기본 클래스
    - simulation: 시뮬레이션 실행 함수
    - battery_models: 개별 배터리 모델 (예: li_ion_33j, li_ion_41j)
    - parameters: OCV 등의 파라미터 관련 모듈
"""

__version__ = "0.1"

from .base_battery_model import BaseBatteryModel
from .battery_models import li_ion_33j
from .battery_models import li_ion_41j


__all__ = ["BaseBatteryModel", "li_ion_33j", "li_ion_41j"]
