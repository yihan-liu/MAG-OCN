# utils.py

import numpy as np

def r2(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute the coefficient of determination R^2.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0