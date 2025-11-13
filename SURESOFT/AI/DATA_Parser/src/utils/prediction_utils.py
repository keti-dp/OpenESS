import numpy as np
import pandas as pd
import datetime

def __logistic_scaled(x, k=10, x0=0.3):
    return 1 / (1 + np.exp(-k*(x - x0)))


def load_prediction_log(
        json_path: str,
        k: float,
        x0: float,
        time_shift_sec: int = 0,
        timezone: str = "Asia/Seoul"
    ):
    """
    Load a prediction log (JSON series), fix timezone, apply time shift,
    logistic scaling, and normalize columns to TIMESTAMP / VALUE.

    Parameters
    ----------
    json_path : str
        Path to JSON file containing anomaly score series.
    logistic_fn : callable
        Logistic scaling function (logistic_scaled).
    k : float
        Logistic function parameter.
    x0 : float
        Logistic function parameter.
    time_shift_sec : int
        Seconds to shift timestamps (USAD = 128 sec).
    timezone : str
        Target timezone (default: Asia/Seoul).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ['TIMESTAMP', 'VALUE']
    """

    # Load JSON as Series
    series = pd.read_json(json_path, typ="series")

    # Timezone 처리
    series.index = (
        series.index.tz_localize("UTC")
                    .tz_convert(timezone)
    )

    # 시간 보정
    if time_shift_sec != 0:
        series.index = series.index - datetime.timedelta(seconds=time_shift_sec)

    # Logistic scaling
    series = __logistic_scaled(series, k=k, x0=x0)

    # Reset index & rename
    df = series.reset_index()
    df = df.rename(columns={"index": "TIMESTAMP", 0: "VALUE"})

    return df


