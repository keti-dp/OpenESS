import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score

plt.style.use("seaborn-v0_8")


def plot_usad_history(csv_path: str, save_path: str = "./output/USAD_history"):
    """
    USAD 모델 학습 이력 시각화
    Expecting columns: ['val_loss1', 'val_loss2']
    """
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df["val_loss1"], label="Val Loss 1", linewidth=2)
    plt.plot(df["val_loss2"], label="Val Loss 2", linewidth=2)

    plt.title("USAD Validation Loss", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def plot_deepant_history(csv_path: str, save_path: str = "./output/DeepAnT_history"):
    """
    DeepAnT 모델 학습 이력 시각화
    Expecting columns: ['val_loss']
    """
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df["val_loss"], label="Val Loss", linewidth=2, color="tab:orange")

    plt.title("DeepAnT Validation Loss", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def plot_threshold_metrics(
    df: pd.DataFrame,
    name: str = "Model",
    score_col: str = "VALUE",
    label_col: str = "Answer",
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_step: float = 0.005,
    save_path: str = None,
):
    """
    Plot Precision / Recall / F1 over a range of thresholds,
    and mark the best threshold (max F1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing score and label columns
    name : str
        Name of the model (plot title)
    score_col : str
        Column name of anomaly score (default: "VALUE")
    label_col : str
        Column name of ground-truth label (default: "Answer")
    threshold_min : float
        Lower bound of threshold range
    threshold_max : float
        Upper bound of threshold range
    threshold_step : float
        Step size for threshold sweeping
    save_path : str
        Optional path to save the HTML plot

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated figure
    best_threshold : float
        Threshold with best F1 score
    best_f1 : float
        Best F1 score
    """

    y_true = df[label_col].values
    scores = df[score_col].values

    thresholds = np.arange(
        threshold_min, threshold_max + threshold_step, threshold_step
    )

    precisions, recalls, f1s = [], [], []

    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    # ----------------------------------------------
    # Best threshold from max F1
    # ----------------------------------------------
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    print(f"{name} Best Threshold = {best_threshold:.4f} / Best F1 = {best_f1:.4f}")

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=thresholds, y=precisions, mode="lines", name="Precision")
    )
    fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode="lines", name="Recall"))
    fig.add_trace(go.Scatter(x=thresholds, y=f1s, mode="lines", name="F1 Score"))

    # Best Threshold V-line
    fig.add_vline(
        x=best_threshold,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Best Th = {best_threshold:.3f}",
        annotation_position="top right",
    )

    # Best F1 H-line
    fig.add_hline(
        y=best_f1,
        line_width=2,
        line_dash="dot",
        line_color="blue",
        annotation_text=f"Best F1 = {best_f1:.3f}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"{name}: Threshold vs Precision / Recall / F1",
        xaxis_title="Threshold",
        yaxis_title="Metric",
        template="plotly_white",
        hovermode="x unified",
    )

    # Save if requested
    if save_path:
        fig.write_html(save_path)
        print(f"📁 Plot saved to: {save_path}")

    return fig, best_threshold, best_f1
