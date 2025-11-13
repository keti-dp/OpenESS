import pandas as pd

from src.visualization.history_plot import plot_usad_history, plot_deepant_history
from src.visualization.signal_plot import plot_rack_signals
def main():

    plot_usad_history("./data/USAD_history.csv")
    plot_deepant_history("./data/DeepAnT_history.csv")
    df = pd.read_csv("data/rack_anomaly_251111.csv").rename(columns={"Time": "timestamp"})



    fig = plot_rack_signals(
        df,
        columns=['max_runtime', 'temp_dt', 'stack_use', 'sio_interrupt4'],
        save_path="output/rack_signals.html"
    )
    fig.show()



if __name__ == "__main__":
    main()
