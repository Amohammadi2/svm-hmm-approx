import os
import sys
import inspect
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import chi2
from numpy.random import default_rng as rng

# Pathing setup
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:
    sys.path.insert(0, parentdir) 

from datalink.scraper import TGJUScraper
from datalink.processor import calculate_tgju_log_returns
from datalink import params_cache as pc
from utils.datastructs import SVMParameters
from inference.estimator import Hyperparameters, FastBayesianSVMEstimator
from inference.var_calculator import create_var_calculator

# --------------------------------------------------------------------------------
# 1. Backtesting & Metrics Module
# --------------------------------------------------------------------------------
class VaRBacktester:
    """
    Encapsulates industry-standard backtesting metrics for Value at Risk models.
    """
    @staticmethod
    def run_tests(violations: np.ndarray, alpha: float) -> dict:
        """
        Calculates Kupiec's Unconditional Coverage (UC), Christoffersen's 
        Independence (IND), and Conditional Coverage (CC) tests.
        """
        # Clean NaNs from predictions
        valid_violations = violations[~np.isnan(violations)].astype(int)
        N = len(valid_violations)
        
        if N == 0:
            return {"UC_p": np.nan, "IND_p": np.nan, "CC_p": np.nan, "violation_rate": np.nan}

        # 1. Kupiec's Unconditional Coverage (UC)
        n1 = np.sum(valid_violations)
        n0 = N - n1
        pi_hat = n1 / N

        if n1 == 0:
            lr_uc = -2 * np.log((1 - alpha)**N) + 2 * np.log(1.0)
        elif n0 == 0:
            lr_uc = -2 * np.log(alpha**N) + 2 * np.log(1.0)
        else:
            lr_uc = -2 * np.log(((1 - alpha)**n0) * (alpha**n1)) + \
                     2 * np.log(((1 - pi_hat)**n0) * (pi_hat**n1))
        
        uc_p_value = 1 - chi2.cdf(lr_uc, df=1)

        # 2. Christoffersen's Independence (IND) Test
        n00 = n01 = n10 = n11 = 0
        for i in range(1, N):
            if valid_violations[i-1] == 0 and valid_violations[i] == 0: n00 += 1
            elif valid_violations[i-1] == 0 and valid_violations[i] == 1: n01 += 1
            elif valid_violations[i-1] == 1 and valid_violations[i] == 0: n10 += 1
            elif valid_violations[i-1] == 1 and valid_violations[i] == 1: n11 += 1

        pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi_mat = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0

        try:
            term1 = ((1 - pi_mat)**(n00 + n10)) * (pi_mat**(n01 + n11))
            term2 = ((1 - pi_0)**n00) * (pi_0**n01) * ((1 - pi_1)**n10) * (pi_1**n11)
            lr_ind = -2 * np.log(term1) + 2 * np.log(term2)
            ind_p_value = 1 - chi2.cdf(lr_ind, df=1)
        except (ValueError, ZeroDivisionError):
            lr_ind = 0
            ind_p_value = np.nan

        # 3. Conditional Coverage (CC) Test
        lr_cc = lr_uc + lr_ind
        cc_p_value = 1 - chi2.cdf(lr_cc, df=2)

        return {
            "violation_rate": pi_hat,
            "UC_p": uc_p_value,
            "IND_p": ind_p_value,
            "CC_p": cc_p_value
        }

# --------------------------------------------------------------------------------
# 2. Interactive Plotting Module
# --------------------------------------------------------------------------------
class InteractiveVaRPlotter:
    """
    Independent plotting class that generates Streamlit-native interactive Plotly charts, 
    mapping both log returns and real-world asset prices.
    """
    @staticmethod
    def plot_price(dates: pd.Series, actual_prices: np.ndarray, predicted_var_prices: np.ndarray) -> go.Figure:
        violations = actual_prices < predicted_var_prices
        fig = go.Figure()

        # Calculate reasonable bounds based on the asset price, ignoring VaR outliers
        upper_limit = np.percentile(actual_prices, 99.5) * 1.05
        lower_limit = np.min(actual_prices) * 0.95

        # Risk Zone (Shaded Area Below VaR boundary)
        y_min_buffer = min(actual_prices.min(), predicted_var_prices.min()) * 0.95
        
        # We concatenate dates forward and backward to create a closed polygon for the fill
        x_fill = pd.concat([dates, dates.iloc[::-1]])
        y_fill = np.concatenate([predicted_var_prices, np.full(len(dates), y_min_buffer)])
        
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill,
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Risk Zone (Violations)',
            hoverinfo='skip'
        ))

        # Actual Prices (Point-wise Trend Styling)
        fig.add_trace(go.Scatter(
            x=dates, y=actual_prices,
            mode='lines+markers', name='Actual Asset Price',
            line=dict(color='#00E6C3', width=1.5),
            marker=dict(size=4)
        ))

        # Predicted VaR Boundary
        fig.add_trace(go.Scatter(
            x=dates, y=predicted_var_prices,
            mode='lines', name='Worst-Case VaR Boundary',
            line=dict(color='#d62728', width=2, dash='dot')
        ))

        # Violation Markers
        fig.add_trace(go.Scatter(
            x=dates[violations], y=actual_prices[violations],
            mode='markers', name='VaR Breach',
            marker=dict(color='red', size=8, symbol='x', line=dict(width=1, color='darkred'))
        ))

        fig.update_layout(
            title="Real-Time Price Dynamics vs. VaR Threshold",
            xaxis_title="Date",
            yaxis_title="Asset Price",
            yaxis_range=[lower_limit, upper_limit],
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @staticmethod
    def plot_log_returns(dates: pd.Series, log_returns: np.ndarray, var_limit: np.ndarray) -> go.Figure:
        violations = log_returns < var_limit
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates, y=log_returns,
            mode='lines', name='Log Returns',
            line=dict(color='#1f77b4', width=1.5)
        ))

        fig.add_trace(go.Scatter(
            x=dates, y=var_limit,
            mode='lines', name='VaR Limit (Log Scale)',
            line=dict(color='#d62728', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=dates[violations], y=log_returns[violations],
            mode='markers', name='Violations',
            marker=dict(color='red', size=6, symbol='circle')
        ))

        fig.update_layout(
            title="Log Returns vs. Estimated VaR",
            xaxis_title="Date",
            yaxis_title="Log Return",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

# --------------------------------------------------------------------------------
# 3. Object-Oriented UI & App Logic
# --------------------------------------------------------------------------------
class VaRDashboard:
    def __init__(self):
        st.set_page_config(page_title="VaR Risk Pipeline", page_icon="📈", layout="wide")
        self.initialize_state()

    def initialize_state(self):
        if "raw_data" not in st.session_state:
            scraper = TGJUScraper(headless=True, max_pages=3)
            st.session_state.raw_data = scraper.load_or_scrape()
        
        # Initialize default UI states for parameters if missing
        if "b0_input" not in st.session_state:
            self.update_session_params(SVMParameters(0.0, 0.0, 0.0, 0.0, 0.95, 0.1, 10.0))

    def update_session_params(self, params: SVMParameters):
        """Helper to sync parameter objects with Streamlit UI state variables."""
        st.session_state.b0_input = float(params.beta0)
        st.session_state.b1_input = float(params.beta1)
        st.session_state.b2_input = float(params.beta2)
        st.session_state.mu_input = float(params.mu)
        st.session_state.phi_input = float(params.phi)
        st.session_state.sig_input = float(params.sigma_eta)
        st.session_state.nu_input = float(params.nu)
        st.session_state.params = params

    def render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Data Pipeline", divider="rainbow")
            st.markdown("Configure scraping for TGJU financial data.")
            pages = st.slider("Pages to scrape (30 pts/page)", 1, 24, 3)

            if st.button("🔄 Rescrape Data", use_container_width=True):
                with st.spinner(f"Scraping {pages} pages..."):
                    scraper = TGJUScraper(headless=True, max_pages=pages)
                    st.session_state.raw_data = scraper.scrape()
                st.success("Data rescraped!")

            st.header("🧮 Model Configuration", divider="rainbow")
            st.session_state.m = st.number_input("Grid points (m)", min_value=10, value=150)
            st.session_state.b_limit = st.number_input("Grid boundaries (b)", min_value=0.1, value=2.5)
            st.session_state.is_samples = st.number_input("IS Samples", min_value=100, value=500)

            st.header("💾 Parameter Management")
            col1, col2 = st.columns(2)
            if col1.button("Load Cache"):
                if (params := pc.load_cached_params()) is not None:
                    self.update_session_params(params)
                    st.success("Loaded")
                else:
                    st.error("Empty Cache")
            if col2.button("Fit Model"):
                self.fit_parameters()

            st.header("🎛️ Manual Parameters", divider="rainbow")
            st.number_input("beta_0", key="b0_input", format="%.6f")
            st.number_input("beta_1", key="b1_input", format="%.6f")
            st.number_input("beta_2", key="b2_input", format="%.6f")
            st.number_input("mu", key="mu_input", format="%.6f")
            st.number_input("phi", key="phi_input", format="%.6f")
            st.number_input("sigma_eta", key="sig_input", format="%.6f")
            st.number_input("nu (df)", key="nu_input", format="%.6f")

            st.session_state.params = SVMParameters(
                st.session_state.b0_input,
                st.session_state.b1_input,
                st.session_state.b2_input,
                st.session_state.mu_input,
                st.session_state.phi_input,
                st.session_state.sig_input,
                st.session_state.nu_input
            )

            if st.button("💾 Save Manual Params to Cache", use_container_width=True):
                pc.save_params_to_cache(st.session_state.params)
                st.success("Manual parameters saved successfully!")

    def fit_parameters(self):
        log_returns = st.session_state.raw_data.pipe(calculate_tgju_log_returns)
        log_returns_np = log_returns['Log_Return'].dropna().to_numpy()
        
        st.session_state.log_returns = log_returns
        st.session_state.log_returns_np = log_returns_np

        def student_t_logpdf(y, mu, sigma, nu):
            return stats.t.logpdf(y, df=nu, loc=mu, scale=sigma)

        config = Hyperparameters(
            m=st.session_state.m, 
            b_limit=st.session_state.b_limit, 
            is_samples=st.session_state.is_samples
        )
        estimator = FastBayesianSVMEstimator(
            data=log_returns_np, smn_logpdf=student_t_logpdf, hyperparams=config
        )

        with st.spinner("Estimating parameters via Fast Bayesian Inference..."):
            result = estimator.estimate()
            params = SVMParameters(**result.map_estimate_con)
            
            self.update_session_params(params)
            pc.save_params_to_cache(params)
            
            st.success("Model fitted and saved to cache automatically!")

    def render_main_dashboard(self):
        st.title("Risk Radar: Actionable VaR Insights")
        st.markdown("Value at Risk (VaR) summarizes the potential loss of a financial position resulting from adverse market movements. It represents the worst expected loss over a given time horizon at a specified confidence level.")

        with st.expander("📊 View Raw Data & Log Returns"):
            st.dataframe(st.session_state.raw_data, use_container_width=True)

        st.header("📈 Risk Projections & Backtesting", divider="rainbow")
        
        if st.session_state.params is None:
            st.info("👈 Please load cached parameters or fit the model in the sidebar to view VaR.")
            return

        alpha = st.slider("Select Confidence Level (Alpha)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
        
        if st.button("Generate VaR Chart & Validation", type="primary"):
            self.generate_forecast(alpha)

    def generate_forecast(self, alpha: float):
        # 1. Setup Data
        if "log_returns_np" not in st.session_state:
            log_returns = st.session_state.raw_data.pipe(calculate_tgju_log_returns)
            st.session_state.log_returns = log_returns
            st.session_state.log_returns_np = log_returns['Log_Return'].dropna().to_numpy()

        df_returns = st.session_state.log_returns.dropna(subset=['Log_Return'])
        returns = df_returns['Log_Return'].to_numpy()
        dates = pd.to_datetime(df_returns["Date"]).reset_index(drop=True)

        # Retrieve actual prices safely. TGJU normally maps this to 'Close'.
        price_cols = [c for c in df_returns.columns if c.lower() in ['close', 'price', 'value']]
        price_col = price_cols[0] if price_cols else df_returns.columns[1]
        
        # To calculate Price VaR (P_t = P_{t-1} * exp(VaR)), we need previous day prices.
        # df_returns dropped the first NaN row, so we pull the shifted series directly from the original dataframe.
        actual_prices = df_returns[price_col].to_numpy()
        prev_prices = st.session_state.log_returns[price_col].shift(1).dropna().to_numpy()

        # 2. Calculate VaR
        calculator = create_var_calculator(
            parameters=st.session_state.params,
            model="t",
            lower=-st.session_state.b_limit,
            upper=st.session_state.b_limit,
            n_states=st.session_state.m,
        )
        var_limit_log = calculator.calculate(returns, alpha=alpha)

        # 3. Transform VaR to Real Prices
        predicted_var_prices = prev_prices * np.exp(var_limit_log)
        
        # Tomorrow's projection (assuming the last known price is shocked by tomorrow's VaR limit)
        tomorrow_worst_case_price = actual_prices[-1] * np.exp(var_limit_log[-1])

        # 4. Interactive Plotting (Streamlit Native via Plotly)
        st.subheader("VaR Projections")
        tab1, tab2 = st.tabs(["Real Price Trajectory", "Log Returns Base"])
        
        with tab1:
            price_fig = InteractiveVaRPlotter.plot_price(dates, actual_prices, predicted_var_prices)
            st.plotly_chart(price_fig, use_container_width=True)
        with tab2:
            log_fig = InteractiveVaRPlotter.plot_log_returns(dates, returns, var_limit_log)
            st.plotly_chart(log_fig, use_container_width=True)

        # 5. Metrics & Validation
        violations = returns < var_limit_log
        results = VaRBacktester.run_tests(violations, alpha)

        st.subheader("Model Validation Metrics")
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        
        # Focus the metric on the tangible business outcome (Real Price)
        m1.metric(
            "Predicted Worst-Case Price (Tomorrow)", 
            f"{tomorrow_worst_case_price:,.2f}", 
            help=f"Calculated from last closing price ({actual_prices[-1]:,.2f}) * exp(VaR_log)"
        )
        m2.metric("Target Alpha vs Actual", f"{alpha:.1%} ➔ {results['violation_rate']:.1%}")
        m3.metric("Kupiec UC (p-value)", f"{results['UC_p']:.4f}", 
                  help="H0: Empirical rate matches nominal rate. >0.05 is good.")
        m4.metric("Christoffersen IND (p-value)", f"{results['IND_p']:.4f}", 
                  help="H0: Violations are independent over time. >0.05 is good.")

        # Conditional Actionable Interpretations
        self.render_interpretations(results, alpha)

    def render_interpretations(self, results: dict, alpha: float):
        uc_pass = results['UC_p'] >= 0.05
        ind_pass = results['IND_p'] >= 0.05
        cc_pass = results['CC_p'] >= 0.05

        st.divider()
        st.header("💡 Actionable Insights & Interpretations")

        st.subheader("🔬 Technical Validation")
        if uc_pass and ind_pass:
            st.success("**Model is Well-Calibrated:** The Kupiec Unconditional Coverage (UC) test, a likelihood ratio test that verifies if the model's actual empirical violation rate matches the chosen nominal level, fails to reject the null hypothesis of correct coverage. Furthermore, the Christoffersen Independence (IND) test fails to reject the null hypothesis, demonstrating that the VaR violations do not exhibit significant temporal clustering.")
        elif not uc_pass and ind_pass:
            st.warning("**Coverage Mismatch:** The Kupiec UC test rejects the null hypothesis. The model is either overestimating or underestimating tail risk at this specific $\\alpha$ level, though violations remain serially independent (IND test passed).")
        elif uc_pass and not ind_pass:
            st.warning("**Volatility Clustering Detected:** While the overall frequency of violations is correct (UC passed), the Christoffersen IND test indicates that violations are clustering together over time, suggesting uncaptured leverage effects or regime shifts in the latent volatility process.")
        else:
            st.error("**Model Rejected:** Both the Conditional Coverage (CC) and Independence tests reject the null hypothesis. The current hyperparameters or chosen distributional assumption (SMN) fail to capture the tail dynamics at this confidence level.")

        st.subheader("💼 Stakeholder Takeaway")
        if uc_pass and ind_pass:
            st.success("**Reliable Risk Limits:** We can trust this model to establish baseline capital reserves. It accurately forecasts how often we will face severe market drops and confirms that a severe market shock today does not artificially break the model's independent predictive capability for tomorrow.")
        else:
            st.warning("**Caution Advised on Capital Reserves:** The model is currently failing historical stress tests. *Note: Value at Risk (VaR) is an industry-standard baseline for quantifying tail risk, but it fundamentally ignores the magnitude of losses beyond that threshold*. Because it will not inform a portfolio manager how devastating a true market crash will be once the limit is breached, it is strongly recommended to pair this analysis with Expected Shortfall (ES) metrics.")

if __name__ == "__main__":
    app = VaRDashboard()
    app.render_sidebar()
    app.render_main_dashboard()