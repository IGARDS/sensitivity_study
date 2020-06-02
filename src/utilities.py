from scipy import stats
import numpy as np

def kendall_tau(r1, r2):
    # Calculates the kendall tau between two ranking vectors
    tau, _ = stats.kendalltau(np.argsort(r1), np.argsort(r2))
    return tau

def spearman_r(r1, r2):
    # Calculates the kendall tau between two ranking vectors
    rho, _ = stats.spearmanr(np.argsort(r1), np.argsort(r2))
    return rho
