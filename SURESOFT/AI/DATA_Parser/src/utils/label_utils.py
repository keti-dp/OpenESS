import pandas as pd

def label_anomaly_ranges(
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        anomaly_ranges=None,
        timezone: str = "Asia/Seoul",
        save_path: str = None
    ):
    """
    Label anomaly ranges (Answer = 1/0) based on given timestamp intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    timestamp_col : str
        Name of the timestamp column
    anomaly_ranges : list of tuples
        Each tuple: (start_time, end_time) in string format
        Example:
        [
            ("2024-11-12 12:30:00+9", "2024-11-12 12:36:00+9"),
            ("2024-11-13 09:20:00+9", "2024-11-13 09:27:00+9"),
        ]
    timezone : str
        Timezone for timestamp conversion ("Asia/Seoul" recommended)
    save_path : str
        Optional CSV save path

    Returns
    -------
    df : pd.DataFrame
        Dataframe with new column 'Answer'
    """

    if anomaly_ranges is None:
        raise ValueError("anomaly_ranges must be provided")

    # timestamp 변환 (timezone-aware)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True).dt.tz_convert(timezone)

    # Answer 초기화
    df["Answer"] = 0

    # 구간 라벨링
    for start, end in anomaly_ranges:
        start_t = pd.to_datetime(start)
        end_t  = pd.to_datetime(end)

        df.loc[
            (df[timestamp_col] >= start_t) &
            (df[timestamp_col] <= end_t),
            "Answer"
        ] = 1

    # 저장 옵션
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"📁 Labeled file saved: {save_path}")

    return df
