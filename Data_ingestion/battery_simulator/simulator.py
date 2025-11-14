"""
simulator.py : 배터리 시뮬레이션 결과를 시각화하고 CSV로 저장하기 위한 인터랙티브 플로팅 도구

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

배터리 시뮬레이션 결과 DataFrame을 입력받아, 다중 서브플롯과 슬라이더를 활용한
시간축 진행 시각화 및 CSV 다운로드 기능을 제공하는 인터랙티브 시뮬레이터 도구입니다.

TIMESTAMP와 여러 지표(전압, 전류, SOC 등)를 동시에 그려보고, 슬라이더로 인덱스를 이동하며
시점별 변화를 확인할 수 있으며, 결과를 CSV 파일로 저장하여 후속 분석에 활용함.

[외부 오픈소스 라이브러리 및 라이선스 안내]
이 파일은 다음과 같은 외부 오픈소스 라이브러리를 사용함.
  - Matplotlib : BSD-style License
  - pandas     : BSD-3-Clause License
  - NumPy      : BSD-3-Clause License

(추가로, tkinter는 Python 표준 라이브러리로 포함되어 있음.)

각 외부 라이브러리의 상세 라이선스 전문은 해당 프로젝트의 LICENSE 파일을 참조함.

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from matplotlib.widgets import Slider, Button
import math
import datetime
import tkinter as tk
from tkinter import filedialog


class Simulator:
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df

    def plot_dynamic_subplots_with_slider(self):
        df = self.df.copy()

        # TIMESTAMP 열이 있으면 날짜로, 없으면 인덱스를 사용
        if "TIMESTAMP" in df.columns:
            timestamps = pd.to_datetime(df["TIMESTAMP"])
        else:
            timestamps = pd.Series(df.index, name="Index")

        columns = [col for col in df.columns if col != "TIMESTAMP"]
        special_cols = columns[-2:]  # 예: voltage, current
        normal_cols = columns[:-2]

        total_plots = len(columns)
        nrows = 2
        ncols = math.ceil(total_plots / nrows)

        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axs = axs.flatten()

        # 서브플롯 배치 위치를 계산
        positions = []
        normal_idx = 0
        for i in range(nrows * ncols):
            if i == ncols - 1:
                positions.append(special_cols[0])  # 첫 행 마지막: voltage
            elif i == (2 * ncols) - 1:
                positions.append(special_cols[1])  # 두 번째 행 마지막: current
            elif normal_idx < len(normal_cols):
                positions.append(normal_cols[normal_idx])
                normal_idx += 1
            else:
                positions.append(None)

        lines = []
        vertical_lines = []

        for idx, (ax, col) in enumerate(zip(axs, positions)):
            if col is None:
                fig.delaxes(ax)
                lines.append(None)
                vertical_lines.append(None)
                continue

            if col in special_cols:
                (line,) = ax.plot(
                    timestamps,
                    df[col],
                    marker="o",
                    markersize=4,
                    linewidth=1,
                    color="orange" if col == "voltage" else "red",
                    label=col,
                )
                v_line = ax.axvline(
                    timestamps.iloc[0], color="k", linestyle="--", linewidth=1
                )
                vertical_lines.append(v_line)
            else:
                (line,) = ax.plot(
                    timestamps[:1],
                    df[col][:1],
                    marker="o",
                    markersize=4,
                    linewidth=1,
                    label=col,
                )
                vertical_lines.append(None)

            ax.set_title(f"{col} over Time", fontsize=10)
            ax.set_xlabel("Timestamp", fontsize=9)
            ax.set_ylabel(col, fontsize=9)
            ax.grid(True)
            ax.legend(fontsize=8)

            if "TIMESTAMP" in df.columns:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
            ax.tick_params(axis="x", rotation=45, labelsize=8)

            lines.append(line)

        plt.subplots_adjust(bottom=0.2, top=0.92, hspace=0.4, wspace=0.3)

        # Slider 생성
        ax_slider = plt.axes([0.15, 0.05, 0.6, 0.03])
        slider = Slider(
            ax=ax_slider,
            label="Time Index",
            valmin=0,
            valmax=len(timestamps) - 1,
            valinit=0,
            valstep=1,
        )

        def update(val):
            idx = int(slider.val)
            current_time = timestamps.iloc[idx]
            for i, col in enumerate(positions):
                if col is None:
                    continue
                if col in special_cols:
                    vertical_lines[i].set_xdata([current_time, current_time])
                else:
                    lines[i].set_data(timestamps[: idx + 1], df[col][: idx + 1])
                    axs[i].relim()
                    axs[i].autoscale_view()
            fig.canvas.draw_idle()

        slider.on_changed(update)

        # Download 버튼 생성 (파일 저장 경로 지정)
        ax_download = plt.axes([0.8, 0.02, 0.15, 0.05])
        btn_download = Button(ax_download, "Download CSV")

        def download(event):
            # Tkinter를 이용한 파일 저장 대화상자 열기
            root = tk.Tk()
            root.withdraw()  # Tk 창 숨김
            # 기본 파일 이름 생성 (현재 시각 기반)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"simulation_results_{current_time}.csv"
            filepath = filedialog.asksaveasfilename(
                initialfile=default_filename,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save simulation results as...",
            )
            if filepath:
                self.df.to_csv(filepath, index=False)
                print(f"Results downloaded successfully to {filepath}")
            root.destroy()

        btn_download.on_clicked(download)

        plt.show()
