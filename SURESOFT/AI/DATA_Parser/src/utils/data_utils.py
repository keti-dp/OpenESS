import numpy as np


def __logistic_scaled(x, k=10, x0=0.3):
    return 1 / (1 + np.exp(-k*(x - x0)))

