import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

def plot_usad_history(csv_path: str, save_path: str = "./output/USAD_history"):
    """
    USAD 모델 학습 이력 시각화
    Expecting columns: ['val_loss1', 'val_loss2']
    """
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df['val_loss1'], label='Val Loss 1', linewidth=2)
    plt.plot(df['val_loss2'], label='Val Loss 2', linewidth=2)

    plt.title("USAD Validation Loss", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
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
    plt.plot(df['val_loss'], label='Val Loss', linewidth=2, color='tab:orange')

    plt.title("DeepAnT Validation Loss", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
