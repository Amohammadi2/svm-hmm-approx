import numpy as np
import pandas as pd
from utils.datastructs import SVMParameters

def generate_synthetic_log_returns(params: SVMParameters, y_0=0.0, n=300):
    # 1. Draw initial stationary log-volatility h_1 ~ N(mu, sigma_eta^2 / (1 - phi^2))
    stationary_std = params.sigma_eta / np.sqrt(1.0 - params.phi**2)
    h_current = np.random.normal(loc=params.mu, scale=stationary_std)
    
    y_current = y_0
    y_returns = np.empty(n)
    h_states = np.empty(n)
    
    # 2. Draw random innovations
    eps = np.random.standard_t(df=10, size=n) 
    eta = np.random.normal(loc=0.0, scale=1.0, size=n)

    # 3. Simulate process t = 1, ..., n
    for t in range(n):
        # Store current log-volatility h_t
        h_states[t] = h_current
        
        # Calculate return y_t = beta0 + beta1 * y_{t-1} + beta2 * exp(h_t) + exp(h_t / 2) * eps_t
        exp_h = np.exp(h_current)
        y_t = (
            params.beta0
            + params.beta1 * y_current
            + params.beta2 * exp_h
            + np.sqrt(exp_h) * eps[t]
        )
        y_returns[t] = y_t
        
        # Transition to next log-volatility h_{t+1} = mu + phi * (h_t - mu) + sigma_eta * eta_t
        h_current = params.mu + params.phi * (h_current - params.mu) + params.sigma_eta * eta[t]
        y_current = y_t

    return pd.DataFrame({
        'Log_Return': y_returns,
        'Log_Volatility': h_states  # Optional: useful for debugging/validation
    })