import numpy as np
import pandas as pd
from utils.datastructs import ModelParameters

def generate_synthetic_log_returns(params: ModelParameters, y_0=0, n=300):
    h_1 = np.random.normal(params.mu, (params.sigma_eta**2)/(1-params.phi**2))

    n_steps = n
    y_array = np.empty(n_steps + 1)
    h_array = np.empty(n_steps + 1)

    y_array[0] = y_0
    h_array[0] = h_1

    eps = np.random.standard_t(df=3, size=n_steps)
    eta = np.random.normal(size=n_steps)

    for t in range(1, n_steps + 1):
        h_array[t] = params.mu + params.phi * (h_array[t-1] - params.mu) + params.sigma_eta * eta[t-1]
        y_array[t] = (
            params.beta_0
            + params.beta_1 * y_array[t-1]
            + params.beta_2 * np.exp(h_array[t-1])
            + np.exp(h_array[t-1] / 2) * eps[t-1]
        )

    return pd.DataFrame({
        'Log_Return': y_array
    })