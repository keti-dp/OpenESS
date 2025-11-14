# import sys
# import os

# # 현재 battery_models/__init__.py 파일의 절대 경로
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 상위 디렉토리 (즉, battery_simulator 폴더)를 찾습니다.
# parent_dir = os.path.dirname(current_dir)

# # battery_simulator(프로젝트 루트)의 경로가 sys.path에 없으면 추가합니다.
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# # 하위 모듈들을 재노출하여 편리하게 사용할 수 있도록 합니다.
# from .li_ion_33j import battery_model as li_ion_33j_battery_model, rls as li_ion_33j_rls
# from .li_ion_41j import battery_model as li_ion_41j_battery_model, rls as li_ion_41j_rls


from . import li_ion_33j
from . import li_ion_41j
from . import experiment

__all__ = ["li_ion_33j", "li_ion_41j", "experiment"]
