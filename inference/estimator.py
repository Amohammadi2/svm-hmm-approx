import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize, approx_fprime
import logging
from dataclasses import dataclass, field
from typing import Callable, Tuple, Dict, Optional

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SVMEstimator")


# =============================================================================
# Dataclasses for Configuration
# =============================================================================
@dataclass
class Hyperparameters:
    m: int = 100                 # Number of grid points for HMM
    b_limit: float = 5.0         # Limits for log-volatility grid [-b_limit, b_limit]
    is_samples: int = 2000       # Number of draws for Importance Sampling

@dataclass
class PriorConfig:
    # Means and standard deviations defined on the UNCONSTRAINED scale
    mu_0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0]))
    sigma_0: np.ndarray = field(default_factory=lambda: np.array([10.0, 3.16, 10.0, 10.0, 1.0, 1.0, 10.0]))

@dataclass
class EstimationResult:
    map_estimate_unc: np.ndarray
    map_estimate_con: dict
    inv_hessian: np.ndarray
    posterior_mean_con: dict
    success: bool
    message: str


# =============================================================================
# Parameter Transformation Management
# =============================================================================
class ParameterTransformer:
    """
    Handles bijective transformations between constrained parameter space 
    and the unconstrained real line R^d.
    Theta unconstrained: [beta0, gamma, beta2, mu, psi, omega, xi]
    """
    def __init__(self, nu_bounds: Tuple[float, float] = (2.0, 40.0)):
        self.nu_min, self.nu_max = nu_bounds
        self.nu_a = (self.nu_max - self.nu_min) / 2.0
        self.nu_c = (self.nu_max + self.nu_min) / 2.0

    def to_constrained(self, theta_u: np.ndarray) -> np.ndarray:
        """ Maps unconstrained vector to constrained vector. """
        beta0, gamma, beta2, mu, psi, omega, xi = theta_u
        
        beta1 = np.tanh(gamma / 2.0)
        phi = np.tanh(psi / 2.0)
        sigma_eta = np.exp(omega)
        # Bounded transformation for nu using tanh
        nu = self.nu_a * np.tanh(xi) + self.nu_c 
        
        return np.array([beta0, beta1, beta2, mu, phi, sigma_eta, nu])

    def to_dict(self, theta_c: np.ndarray) -> Dict[str, float]:
        keys = ['beta0', 'beta1', 'beta2', 'mu', 'phi', 'sigma_eta', 'nu']
        return dict(zip(keys, theta_c))


# =============================================================================
# Main Estimator Class
# =============================================================================
class FastBayesianSVMEstimator:
    def __init__(self, 
                 data: np.ndarray, 
                 smn_logpdf: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
                 hyperparams: Hyperparameters = Hyperparameters(),
                 priors: PriorConfig = PriorConfig()):
        """
        data: 1D array of daily returns.
        smn_logpdf: Callable function representing the log-density of the SMN observation.
                    Must accept (y, mean, std_dev, nu) and return log-probabilities.
        """
        self.y = np.asarray(data)
        self.T = len(self.y)
        self.smn_logpdf = smn_logpdf
        self.hp = hyperparams
        self.priors = priors
        self.transformer = ParameterTransformer()
        
        # HMM Grid Setup
        self.b_grid = np.linspace(-self.hp.b_limit, self.hp.b_limit, self.hp.m)
        self.delta_b = self.b_grid[1] - self.b_grid[0]

    def _build_hmm_matrices(self, theta_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Constructs the transition matrix Gamma and initial distribution delta. """
        _, _, _, mu, phi, sigma_eta, _ = theta_c
        
        # Transition matrix (rows sum to 1)
        grid_diff = self.b_grid[:, None] - (mu + phi * (self.b_grid[None, :] - mu))
        gamma_mat = stats.norm.pdf(grid_diff, loc=0, scale=sigma_eta) * self.delta_b
        
        # Numerical resilience: ensure rows sum exactly to 1 and prevent zeros
        row_sums = gamma_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-12 
        gamma_mat = gamma_mat / row_sums
        
        # Initial distribution
        stat_var = (sigma_eta ** 2) / (1.0 - phi ** 2 + 1e-12)
        delta = stats.norm.pdf(self.b_grid, loc=mu, scale=np.sqrt(stat_var)) * self.delta_b
        delta_sum = delta.sum()
        delta = delta / delta_sum if delta_sum > 0 else np.ones(self.hp.m) / self.hp.m
        
        return gamma_mat, delta

    def _log_likelihood(self, theta_u: np.ndarray) -> float:
        """ Scaled Forward Algorithm to evaluate log P(Y | theta). """
        theta_c = self.transformer.to_constrained(theta_u)
        beta0, beta1, beta2, mu, phi, sigma_eta, nu = theta_c
        
        gamma_mat, alpha = self._build_hmm_matrices(theta_c)
        log_L = 0.0
        
        # Precompute standard deviations for the observation equation
        sigmas = np.exp(self.b_grid / 2.0)
        
        for t in range(1, self.T):
            y_t = self.y[t]
            y_t_1 = self.y[t-1]
            
            # Conditional mean for all m states
            mus = beta0 + beta1 * y_t_1 + beta2 * np.exp(self.b_grid)
            
            # Emission probabilities (exponentiating the injected SMN logpdf)
            log_obs_probs = self.smn_logpdf(y_t, mus, sigmas, nu)
            
            # Prevent catastrophic underflow in emission probabilities
            max_log = np.max(log_obs_probs)
            obs_probs = np.exp(log_obs_probs - max_log) 
            
            # Forward step: alpha_t = (alpha_{t-1} * Gamma) \odot P(y_t)
            alpha = (alpha @ gamma_mat) * obs_probs
            
            # Scaling step to prevent underflow
            c_t = np.sum(alpha)
            if c_t <= 0 or np.isnan(c_t):
                # Mitigation: Extreme parameter regions yield impossible likelihoods
                return -1e10 
            
            alpha = alpha / c_t
            # Adjust log-likelihood (add back the max_log extracted earlier)
            log_L += np.log(c_t) + max_log
            
        return log_L

    def _log_prior(self, theta_u: np.ndarray) -> float:
        """ Evaluates the log-prior directly on the unconstrained space. """
        return np.sum(stats.norm.logpdf(theta_u, loc=self.priors.mu_0, scale=self.priors.sigma_0))

    def _negative_log_posterior(self, theta_u: np.ndarray) -> float:
        """ Objective function for L-BFGS. """
        ll = self._log_likelihood(theta_u)
        if ll == -1e10:
            return 1e10
        lp = self._log_prior(theta_u)
        return - (ll + lp)

    def _ensure_positive_definite(self, matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """ Mitigation strategy: Adds ridge to diagonal if matrix isn't positive definite. """
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            logger.warning("Matrix is not positive definite. Applying ridge regularization.")
            min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
            return matrix + (abs(min_eig) + epsilon) * np.eye(matrix.shape[0])

    def estimate(self, init_theta_u: Optional[np.ndarray] = None) -> EstimationResult:
        """ Performs MAP estimation and Importance Sampling. """
        if init_theta_u is None:
            init_theta_u = self.priors.mu_0.copy()
            
        logger.info("Starting numerical maximization (L-BFGS-B)...")
        opt_res = minimize(
            self._negative_log_posterior,
            init_theta_u,
            method='L-BFGS-B',
            options={'ftol': 1e-6, 'disp': False}
        )
        
        if not opt_res.success:
            logger.warning(f"Optimization warning: {opt_res.message}. Proceeding with best found mode.")
            
        map_u = opt_res.x
        logger.info(f"MAP optimization complete. Unconstrained Mode: {np.round(map_u, 3)}")
        
        # ---------------------------------------------------------
        # Hessian Extraction & Error Mitigation
        # ---------------------------------------------------------
        try:
            # Attempt to extract inverse Hessian from L-BFGS-B
            inv_hessian = opt_res.hess_inv.todense()
        except AttributeError:
            logger.info("Extracting dense Hessian failed. Computing numerical Hessian...")
            # Fallback: Compute numerical Hessian
            epsilon = np.sqrt(np.finfo(float).eps)
            hessian = approx_fprime(map_u, 
                                    lambda x: approx_fprime(x, self._negative_log_posterior, epsilon), 
                                    epsilon)
            try:
                inv_hessian = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                logger.error("Hessian inversion failed. Falling back to spherical covariance.")
                inv_hessian = np.eye(len(map_u)) * 1e-4
                
        inv_hessian = self._ensure_positive_definite(inv_hessian)

        # ---------------------------------------------------------
        # Importance Sampling [cite: 529, 531-532]
        # ---------------------------------------------------------
        logger.info(f"Drawing {self.hp.is_samples} samples for Importance Sampling inference...")
        proposal_dist = stats.multivariate_normal(mean=map_u, cov=inv_hessian)
        samples_u = proposal_dist.rvs(size=self.hp.is_samples)
        
        log_weights = np.zeros(self.hp.is_samples)
        samples_c = np.zeros((self.hp.is_samples, 7))
        
        for i in range(self.hp.is_samples):
            th_u = samples_u[i]
            samples_c[i] = self.transformer.to_constrained(th_u)
            
            # w = P(theta | Y) / q(theta) => log(w) = log P(theta|Y) - log q(theta)
            log_p = -self._negative_log_posterior(th_u) 
            log_q = proposal_dist.logpdf(th_u)
            log_weights[i] = log_p - log_q
            
        # Log-Sum-Exp Trick for numerical stability of weights
        max_log_w = np.max(log_weights)
        weights = np.exp(log_weights - max_log_w)
        weights /= np.sum(weights)
        
        # Compute posterior means on the constrained scale
        posterior_mean_c = np.sum(samples_c * weights[:, None], axis=0)
        
        logger.info("Importance Sampling complete.")
        
        return EstimationResult(
            map_estimate_unc=map_u,
            map_estimate_con=self.transformer.to_dict(self.transformer.to_constrained(map_u)),
            inv_hessian=inv_hessian,
            posterior_mean_con=self.transformer.to_dict(posterior_mean_c),
            success=opt_res.success,
            message=opt_res.message
        )