"""
data_modifier.py : 배터리 충방전 데이터 열화 및 가상 셀 데이터 생성을 위한 유틸리티 코드

        ---------------------------------------------------------------------------
        Copyright(C) 2025, 윤태일 / KETI / taeil777@keti.re.kr

        아파치 라이선스 버전 2.0(라이선스)에 따라 라이선스가 부여됩니다.
        라이선스를 준수하지 않는 한 이 파일을 사용할 수 없습니다.
        다음에서 라이선스 사본을 얻을 수 있습니다.

        http://www.apache.org/licenses/LICENSE-2.0

        관련 법률에서 요구하거나 서면으로 동의하지 않는 한 소프트웨어
        라이선스에 따라 배포되는 것은 '있는 그대로' 배포되며,
        명시적이든 묵시적이든 어떠한 종류의 보증이나 조건도 제공하지 않습니다.
        라이선스에 따른 권한 및 제한 사항을 관리하는 특정 언어는 라이선스를 참조하십시오.
        ---------------------------------------------------------------------------

        ---------------------------------------------------------------------------
        The MIT License

        Copyright(C) 2025, 윤태일 / KETI / taeil777@keti.re.kr

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in
        all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
        THE SOFTWARE.
        ---------------------------------------------------------------------------


최신 테스트 버전 : 1.0.0 ver
최신 안정화 버전 : 1.0.0 ver

배터리 충방전 데이터를 입력받아 열 이름 변경, 간단한 열화 효과 적용, 구간별(piecewise) 열화 시뮬레이션,
전압 기울기 조정 등을 수행하여 가상 열화 데이터 및 가상 셀 데이터를 생성하기 위한 코드입니다.

실험 데이터나 시뮬레이션 데이터를 기반으로, 시간축과 전압/전류 파형을 재구성하여
다양한 열화 시나리오를 빠르게 생성·검증하는 데 활용함.

[외부 오픈소스 라이브러리 및 라이선스 안내]
이 파일은 다음과 같은 외부 오픈소스 라이브러리를 사용함.
  - pandas     : BSD-3-Clause License
  - NumPy      : BSD-3-Clause License
  - SciPy      : BSD-3-Clause License

각 외부 라이브러리의 상세 라이선스 전문은 해당 프로젝트의 LICENSE 파일을 참조함.

"""

import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.signal import savgol_filter


class DataModifier:
    """
    배터리 충방전 데이터를 입력받아, 간단한 열화 및 가상 셀 데이터 생성 기능을 제공합니다.

    주요 기능:
      - 열 이름 변경 (rename_columns)
      - 노화 효과 적용 (apply_aging_effect)
      - 구간별(piecewise) 열화 효과 시뮬레이션 (simulate_degradation_piecewise)
      - 전압 기울기 변경 (adjust_voltage_slopes)
    """

    def __init__(self, data):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("data는 CSV 파일 경로나 pandas DataFrame이어야 합니다.")
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"]).dt.tz_localize(
                None
            )

    def rename_columns(self, new_names: dict):
        self.df = self.df.rename(columns=new_names)
        return self

    def apply_aging_effect(self, column: str, aging_factor: float):
        if column not in self.df.columns:
            raise ValueError(f"'{column}' 열이 데이터에 없습니다.")
        self.df[column] *= aging_factor
        return self

    def get_dataframe(self):
        return self.df

    def simulate_degradation_piecewise(
        self,
        charging_time_factor_early=1.0,
        charging_time_factor_plateau=1.0,
        plateau_voltage_factor=0.98,
        discharging_time_factor_early=1.0,
        discharging_time_factor_late=1.0,
        internal_resistance_increase_early=0.01,
        internal_resistance_increase_late=0.015,
        hysteresis_factor=0.005,
        bridging_ratio=0.5,
        smoothing_window=7,
        smoothing_polyorder=2,
        delta_ocv=0.0,  # 새로 추가된 파라미터 (초기 OCV 보정)
    ):
        df = self.df.copy()
        df["phase"] = df["current"].apply(
            lambda i: "charging" if i > 0 else ("discharging" if i < 0 else "rest")
        )

        groups, curr_grp, curr_phase = [], [df.index[0]], df.at[df.index[0], "phase"]
        for idx in df.index[1:]:
            ph = df.at[idx, "phase"]
            if ph == curr_phase:
                curr_grp.append(idx)
            else:
                groups.append((curr_grp, curr_phase))
                curr_grp, curr_phase = [idx], ph
        groups.append((curr_grp, curr_phase))

        out, t_cursor = [], df.at[df.index[0], "timestamp"]

        def assign_ts(n, duration, start):
            if n == 1:
                return [start]
            step = duration / (n - 1)
            return [start + timedelta(seconds=i * step) for i in range(n)]

        def smooth(vals, win, poly):
            return savgol_filter(vals, win, poly) if len(vals) >= win else vals

        for idxs, phase in groups:
            seg = df.loc[idxs].copy()
            t0, t1 = seg["timestamp"].iloc[0], seg["timestamp"].iloc[-1]
            dur = (t1 - t0).total_seconds()

            if phase == "charging":
                mid = len(seg) // 2
                early, plat = seg.iloc[:mid].copy(), seg.iloc[mid:].copy()
                de = (dur / 2) * charging_time_factor_early
                dp = (dur / 2) * charging_time_factor_plateau

                early["timestamp"] = assign_ts(len(early), de, t_cursor)
                early["voltage"] -= (
                    internal_resistance_increase_early * early["current"].abs()
                )
                t_cursor += timedelta(seconds=de)

                plat["timestamp"] = assign_ts(len(plat), dp, t_cursor)
                plat["voltage"] *= plateau_voltage_factor
                t_cursor += timedelta(seconds=dp)

                new_seg = pd.concat([early, plat])
                new_seg["voltage"] += delta_ocv  # 초기 OCV 보정 추가

            elif phase == "discharging":
                mid = len(seg) // 2
                early, late = seg.iloc[:mid].copy(), seg.iloc[mid:].copy()
                de = (dur / 2) * discharging_time_factor_early
                dl = (dur / 2) * discharging_time_factor_late

                early["timestamp"] = assign_ts(len(early), de, t_cursor)
                early["voltage"] -= (
                    internal_resistance_increase_early * early["current"].abs()
                )
                early["voltage"] += hysteresis_factor
                t_cursor += timedelta(seconds=de)

                late["timestamp"] = assign_ts(len(late), dl, t_cursor)
                late["voltage"] -= (
                    internal_resistance_increase_late * late["current"].abs()
                )
                late["voltage"] -= hysteresis_factor
                gap = late["voltage"].iloc[0] - early["voltage"].iloc[-1]
                late["voltage"] -= gap * bridging_ratio
                bc = max(3, int(0.2 * len(late)))
                late.iloc[:bc, late.columns.get_loc("voltage")] = smooth(
                    late["voltage"].iloc[:bc].values,
                    min(smoothing_window, bc // 2 * 2 + 1),
                    smoothing_polyorder,
                )
                t_cursor += timedelta(seconds=dl)
                new_seg = pd.concat([early, late])

            else:
                new_seg = seg.copy()
                new_seg["timestamp"] = assign_ts(len(new_seg), dur, t_cursor)
                t_cursor += timedelta(seconds=dur)

            out.append(new_seg)

        self.df = pd.concat(out).sort_index().drop(columns="phase")
        return self

    def adjust_voltage_slopes(
        self, charging_slope_factor=1.07, discharging_slope_factor=1.05
    ):
        df = self.df.copy()
        df["phase"] = df["current"].apply(
            lambda x: "charging" if x > 0 else ("discharging" if x < 0 else "rest")
        )

        groups = []
        curr_idx = df.index[0]
        curr_phase = df.at[curr_idx, "phase"]
        curr_group = [curr_idx]
        for idx in df.index[1:]:
            ph = df.at[idx, "phase"]
            if ph == curr_phase:
                curr_group.append(idx)
            else:
                groups.append((curr_group, curr_phase))
                curr_group, curr_phase = [idx], ph
        groups.append((curr_group, curr_phase))

        for idxs, phase in groups:
            factor = (
                charging_slope_factor
                if phase == "charging"
                else discharging_slope_factor
            )
            seg = df.loc[idxs]
            v0 = seg["voltage"].iloc[0]
            df.loc[idxs, "voltage"] = v0 + (seg["voltage"] - v0) * factor

        self.df = df.drop(columns=["phase"])
        return self
