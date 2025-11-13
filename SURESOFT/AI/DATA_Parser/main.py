import pandas as pd

from src.visualization.history_plot import plot_usad_history, plot_deepant_history
from src.visualization.signal_plot import plot_rack_signals
from src.utils.label_utils import label_anomaly_ranges


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


if __name__ == "__main__":
    main()
