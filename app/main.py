import os
import sys
import inspect
from scipy import stats
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from numpy.random import default_rng as rng

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:
    sys.path.insert(0, parentdir) 

from datalink.scraper import TGJUScraper
from datalink.processor import calculate_tgju_log_returns
from datalink import params_cache as pc
from utils.datastructs import SVMParameters
from inference.estimator import Hyperparameters, FastBayesianSVMEstimator
from inference.var_calculator import (
    create_var_calculator,
)
from utils.presentation import create_var_violation_plot


st.set_page_config(
    page_title="VaR Dashboard",
    page_icon="👋",
)

st.title("VaR Analysis Dashboard")

st.markdown(
    "This is an expository web application built on top of our data processing core to demonstrate"
    " our real data pipeline from scraping to visual analysis")

# --- Controls ---
st.header("Data Scraper", divider="rainbow")

st.markdown(
    "This web app scrapes financial data from this website: https://www.tgju.org/profile/tgju_gold_irg18/history. "
    "Each page contains 30 data points (roughly one month) and you can configure how many pages to scrape."
)

pages = st.slider(
    "Number of pages to scrape",
    min_value=1,
    max_value=24,
    value=3,
)

if "raw_data" not in st.session_state:
    scraper = TGJUScraper(headless=True, max_pages=pages)
    st.session_state.raw_data = scraper.load_or_scrape()

if st.button("🔄 Scrape data", type="primary"):
    with st.spinner(f"Scraping {pages} pages..."):
        scraper = TGJUScraper(headless=True, max_pages=pages)
        st.session_state.raw_data = scraper.scrape()

    st.success("Data successfully rescraped!")

# --- Display ---
st.header("Raw Dataframe", divider="rainbow")
st.markdown("This part previews the cached data upon application startup, you can update to the latest data by rescraping.")
st.dataframe(
    st.session_state.raw_data,
    use_container_width=True,
)

st.header("Model Hyperparameters", divider="rainbow")
st.markdown("You can configure these hyperparameters manually in order to blanace between "
            "precision & speed.")

m = st.number_input(label="Number of grid points (m)", min_value=0)
b_limit = st.number_input(label="Grid boundaries to cover around zero (interval of b)", min_value=0.0)
is_samples = st.number_input(label="Effective samples during importance sampling", min_value=500)

st.header("Model Parameters", divider="rainbow")
st.markdown("You can either enter the values manually or fit the parameters against the data")

def student_t_logpdf(y: float, mu: np.ndarray, sigma: np.ndarray, nu: float) -> np.ndarray:
    """
    Log-density of the Student-t distribution parameterized by location, scale, and df.
    """
    # Using scipy stats, passing arrays for vectorized operations over the grid
    return stats.t.logpdf(y, df=nu, loc=mu, scale=sigma)

c1, c2, c3 = st.columns(3)

if c1.button("Fit Parameters", type="primary", width="stretch"):
    log_returns = st.session_state.raw_data.pipe(calculate_tgju_log_returns)
    log_returns_np = log_returns['Log_Return'].dropna().to_numpy()

    st.session_state.log_returns = log_returns
    st.session_state.log_returns_np = log_returns_np

        # 1. Define the SMN log-density function (e.g., Student-t distribution)

    # 3. Configure Hyperparameters
    config = Hyperparameters(
        m=150,               # Lowered to 50 for speed in MWE, paper uses 100-200
        b_limit=2.5,        # Grid boundaries
        is_samples=500      # Samples for IS
    )

    # 4. Instantiate and run the estimator
    estimator = FastBayesianSVMEstimator(
        data=log_returns_np,
        smn_logpdf=student_t_logpdf,
        hyperparams=config
    )

    with st.spinner(f"Estimating parameters, wait a few minutes..."):

        # 5. Extract Results
        result = estimator.estimate()

        params = SVMParameters(**result.map_estimate_con)

        st.session_state.beta_0 = params.beta0
        st.session_state.beta_1 = params.beta1
        st.session_state.beta_2 = params.beta2
        st.session_state.mu = params.mu
        st.session_state.phi = params.phi
        st.session_state.sigma_eta = params.sigma_eta
        st.session_state.nu = params.nu
        st.session_state.params = params

        pc.save_params_to_cache(params)

        st.rerun()
        st.success("Successfully estimated the params")
if c2.button("Load cached params", width="stretch"):
    log_returns = st.session_state.raw_data.pipe(calculate_tgju_log_returns)
    log_returns_np = log_returns['Log_Return'].dropna().to_numpy()

    st.session_state.log_returns = log_returns
    st.session_state.log_returns_np = log_returns_np
    if (params:=pc.load_cached_params()) is not None:
        st.session_state.beta_0 = params.beta0
        st.session_state.beta_1 = params.beta1
        st.session_state.beta_2 = params.beta2
        st.session_state.mu = params.mu
        st.session_state.phi = params.phi
        st.session_state.sigma_eta = params.sigma_eta
        st.session_state.nu = params.nu
        st.success("Loaded successfully")
    else:
        st.error("Cache file is empty")
if c3.button("Save to cache", width="stretch"):
    s = st.session_state
    pc.save_params_to_cache(SVMParameters(s.beta_0, s.beta_1, s.beta_2, s.mu, s.phi, s.sigma_eta, s.nu))
    st.success("Saved to cache successfully")


beta_0: float = st.number_input(label="beta_0", key="beta_0", format="%.17g")
beta_1: float = st.number_input(label="beta_1", key="beta_1", min_value=-1.0, max_value=1.0, format="%.17g")
beta_2: float = st.number_input(label="beta_2", key="beta_2", format="%.17g")
mu: float = st.number_input(label="mu", key="mu", format="%.17g")
phi: float = st.number_input(label="phi", key="phi", min_value=-1.0, max_value=1.0, format="%.17g")
sigma_eta: float = st.number_input(label="sigma_eta", key="sigma_eta",min_value=0.0, format="%.17g")
nu: int = st.number_input(label="nu", key="nu", min_value=1, format="%.17g")

st.header("VaR Chart", divider="rainbow")

alpha = st.slider("Alpha", min_value=0.01, max_value=0.2)

if st.button("Render chart"):
    # Construct the HMM-based VaR calculator.
    calculator = create_var_calculator(
        parameters=SVMParameters(beta_0, beta_1, beta_2, mu, phi, sigma_eta, nu),
        model="t",
        lower=-b_limit,
        upper=b_limit,
        n_states=m,
    )

    log_returns = st.session_state.log_returns
    log_returns_np = st.session_state.log_returns_np

    var_alpha_pct = calculator.calculate(
        log_returns_np,
        alpha=alpha,
    )


    dates = pd.to_datetime(log_returns["Date"].iloc[1:]).reset_index(drop=True)

    st.pyplot(create_var_violation_plot(dates, log_returns_np, var_alpha_pct, shift_steps=0))

    violations = log_returns_np < var_alpha_pct

    violation_rate = violations[~np.isnan(var_alpha_pct)].mean()

    st.markdown(f"Observed violation rate: :blue[{violation_rate:.4%}]")