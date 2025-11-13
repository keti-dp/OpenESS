import pandas as pd
import plotly.graph_objects as go


def plot_rack_signals(
    df: pd.DataFrame,
    columns=None,
    timestamp_col: str = "timestamp",
    title: str = "Rack Anomaly Signals",
    save_path: str = None,
):
    """
    Plot rack anomaly-related signals using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing timestamp + signal columns
    columns : list
        List of columns to plot
    timestamp_col : str
        Timestamp column name
    title : str
        Title of the plot
    save_path : str
        If provided, the plot will be saved as an HTML file

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    # Default columns
    if columns is None:
        columns = ["max_runtime", "temp_dt", "stack_use", "sio_interrupt4"]

    # Timestamp 처리
    if timestamp_col in df.columns:
        x_axis = pd.to_datetime(df[timestamp_col])
    else:
        x_axis = df.index

    # Plotly figure 생성
    fig = go.Figure()

    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=x_axis, y=df[col], mode="lines", name=col))
        else:
            print(f"⚠️ Warning: column {col} not found in dataframe.")

    # Layout 설정
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        legend_title="Signals",
        hovermode="x unified",
    )

    # 저장 옵션 (HTML로 저장)
    if save_path:
        fig.write_html(save_path)
        print(f"📁 Figure saved to: {save_path}")

    return fig
