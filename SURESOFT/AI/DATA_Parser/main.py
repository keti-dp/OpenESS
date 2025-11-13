import pandas as pd

from src.visualization.history_plot import plot_usad_history, plot_deepant_history
from src.visualization.signal_plot import plot_rack_signals
from src.utils.label_utils import label_anomaly_ranges

from src.utils.prediction_utils import load_prediction_log
from logistic_function import logistic_scaled  # 기존 함수



def main():
    # 학습 History Plot
    plot_usad_history("./data/USAD_history.csv")
    plot_deepant_history("./data/DeepAnT_history.csv")
    # Anomaly Validation Data Load
    df = pd.read_csv("data/rack_anomaly_251111.csv", engine="pyarrow").rename(
        columns={"Time": "timestamp"}
    )
    # Raw Data Plotting
    fig = plot_rack_signals(
        df,
        columns=["max_runtime", "temp_dt", "stack_use", "sio_interrupt4"],
        save_path="output/rack_signals.html",
    )
    fig.show()
    # Add Answer Col for Validation Data
    anomaly_ranges = [
        ("2024-11-12 12:30:00+09", "2024-11-12 12:36:00+09"),
        ("2024-11-13 09:20:00+09", "2024-11-13 09:27:00+09"),
        ("2024-11-14 06:55:00+09", "2024-11-14 07:01:00+09"),
    ]

    df = label_anomaly_ranges(df, anomaly_ranges=anomaly_ranges, save_path="./output/rack_anomaly_251111.csv")


    # USAD
    usad = load_prediction_log(
        json_path="./data/USAD_Prediction.txt",
        logistic_fn=logistic_scaled,
        k=1,
        x0=0.5,
        time_shift_sec=128  # USAD 특성
    )

    # DeepAnT
    deepant = load_prediction_log(
        json_path="./data/DeepAnt_Prediction.txt",
        logistic_fn=logistic_scaled,
        k=0.45,
        x0=1.5,
        time_shift_sec=0
    )

if __name__ == "__main__":
    main()
