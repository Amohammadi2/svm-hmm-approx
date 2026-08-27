"""
One-step-ahead Value at Risk for the SVM-HMM model.

This module implements the HMM-based forecasting methodology described in:

    Holtz, Abanto-Valle, Ehlers & Rodríguez
    "Stochastic Volatility in Mean Models with Heavy Tails:
     A Fast Approximate Bayesian Inference Using Hidden Markov Models."

The SVM model is

    y_t = beta_0
        + beta_1 * y_{t-1}
        + beta_2 * exp(h_t)
        + exp(h_t / 2) * epsilon_t

with latent volatility

    h_{t+1} = mu + phi * (h_t - mu) + sigma_eta * eta_t,

where eta_t ~ N(0, 1).

The continuous latent volatility process is approximated by an HMM
on an equidistant grid of m intervals. One-step-ahead prediction is

    xi_t = phi_{t-1} @ Gamma

and the predictive distribution is

    F_t(y) = sum_i xi_t[i] * F_i(y).

VaR at tail probability alpha is therefore

    VaR_t(alpha) = F_t^{-1}(alpha).

The implementation supports:

    - SVM-N
    - SVM-t
    - SVM-S
    - SVM-VG

Dependencies
------------
numpy
scipy
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import gammainc, gamma, kv
from scipy.stats import norm
from scipy.stats import t as student_t
from scipy.integrate import quad_vec
from scipy.optimize import brentq


Array = np.ndarray

from utils.datastructs import SVMParameters

# ============================================================================
# Volatility grid
# ============================================================================


@dataclass(frozen=True, slots=True)
class VolatilityGrid:
    """Equidistant discretization of the latent log-volatility process.

    The paper partitions [b_0, b_m] into m equidistant intervals and
    represents each interval by its midpoint b_i^*.

    Parameters
    ----------
    lower:
        Lower grid boundary.
    upper:
        Upper grid boundary.
    n_states:
        Number of HMM states.
    """

    lower: float = -2.5
    upper: float = 2.5
    n_states: int = 200

    def __post_init__(self) -> None:
        if self.upper <= self.lower:
            raise ValueError("upper must be greater than lower.")

        if self.n_states < 2:
            raise ValueError("n_states must be at least 2.")

    @property
    def width(self) -> float:
        """Width of each discretization interval."""
        return (self.upper - self.lower) / self.n_states

    @property
    def midpoints(self) -> Array:
        """Midpoints of the discretization intervals."""
        return (
            self.lower
            + (np.arange(self.n_states, dtype=float) + 0.5)
            * self.width
        )


# ============================================================================
# Innovation distributions
# ============================================================================


class InnovationDistribution(ABC):
    """Interface for the standardized SMN innovation distribution."""

    @abstractmethod
    def pdf(self, x: Array) -> Array:
        """Evaluate the standardized innovation density."""

    @abstractmethod
    def cdf(self, x: Array) -> Array:
        """Evaluate the standardized innovation CDF."""

    @abstractmethod
    def ppf(self, probability: float) -> float:
        """Evaluate the standardized innovation quantile."""


class NormalInnovation(InnovationDistribution):
    """Standard normal innovation used by SVM-N."""

    def pdf(self, x: Array) -> Array:
        return norm.pdf(x)

    def cdf(self, x: Array) -> Array:
        return norm.cdf(x)

    def ppf(self, probability: float) -> float:
        return float(norm.ppf(probability))


class StudentTInnovation(InnovationDistribution):
    """Student-t innovation used by SVM-t."""

    def __init__(self, nu: float) -> None:
        if nu <= 0.0:
            raise ValueError("nu must be positive.")

        self.nu = float(nu)

    def pdf(self, x: Array) -> Array:
        return student_t.pdf(x, df=self.nu)

    def cdf(self, x: Array) -> Array:
        return student_t.cdf(x, df=self.nu)

    def ppf(self, probability: float) -> float:
        return float(student_t.ppf(probability, df=self.nu))


class SlashInnovation(InnovationDistribution):
    """Slash innovation used by SVM-S.

    The paper defines the Slash distribution through

        lambda ~ Beta(nu, 1)

        epsilon | lambda ~ N(0, 1 / lambda).

    Its density is evaluated using the lower incomplete gamma function.
    """

    def __init__(
        self,
        nu: float,
        quadrature_epsabs: float = 1e-9,
        quadrature_epsrel: float = 1e-9,
    ) -> None:
        if nu <= 0.0:
            raise ValueError("nu must be positive.")

        self.nu = float(nu)
        self.quadrature_epsabs = quadrature_epsabs
        self.quadrature_epsrel = quadrature_epsrel

    def pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)

        result = np.empty_like(x)

        nonzero = x != 0.0

        if np.any(nonzero):
            x_nz = x[nonzero]
            z = 0.5 * x_nz**2
            shape = self.nu + 0.5

            # f(x) =
            #   nu / sqrt(2*pi)
            #   * (2/x^2)^(nu + 1/2)
            #   * gamma(nu + 1/2, x^2/2)
            #
            # scipy.special.gammainc is the regularized lower
            # incomplete gamma function, so multiply by Gamma(shape).

            result[nonzero] = (
                self.nu
                / np.sqrt(2.0 * np.pi)
                * (2.0 / x_nz**2) ** shape
                * gammainc(shape, z)
                * gamma(shape)
            )

        if np.any(~nonzero):
            result[~nonzero] = (
                self.nu
                / (
                    np.sqrt(2.0 * np.pi)
                    * (self.nu + 0.5)
                )
            )

        return result

    def cdf(self, x: Array) -> Array:
        """Evaluate the Slash CDF.

        The CDF is represented directly through the SMN mixture:

            F(x) =
                integral_0^1
                Phi(x * sqrt(lambda))
                * nu * lambda^(nu - 1)
                d lambda.

        quad_vec evaluates the integral for the complete state vector
        simultaneously, avoiding m independent Python-level quadratures.
        """
        x = np.asarray(x, dtype=float)

        def integrand(lam: float) -> Array:
            return (
                self.nu
                * lam ** (self.nu - 1.0)
                * norm.cdf(x * np.sqrt(lam))
            )

        value, _ = quad_vec(
            integrand,
            0.0,
            1.0,
            epsabs=self.quadrature_epsabs,
            epsrel=self.quadrature_epsrel,
        )

        return np.asarray(value, dtype=float)

    def ppf(self, probability: float) -> float:
        return _numerical_ppf(
            self.cdf,
            probability,
        )


class VarianceGammaInnovation(InnovationDistribution):
    """Variance-Gamma innovation used by SVM-VG.

    The paper uses

        lambda ~ IG(nu / 2, nu / 2)

        epsilon | lambda ~ N(0, 1 / lambda).

    The density is evaluated using the modified Bessel function of
    the second kind.
    """

    def __init__(
        self,
        nu: float,
        quadrature_epsabs: float = 1e-9,
        quadrature_epsrel: float = 1e-9,
    ) -> None:
        if nu <= 0.0:
            raise ValueError("nu must be positive.")

        self.nu = float(nu)
        self.quadrature_epsabs = quadrature_epsabs
        self.quadrature_epsrel = quadrature_epsrel

    def pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)

        result = np.empty_like(x)

        nonzero = x != 0.0

        if np.any(nonzero):
            x_nz = x[nonzero]

            order = (self.nu - 1.0) / 2.0
            argument = np.sqrt(self.nu) * np.abs(x_nz)

            result[nonzero] = (
                np.sqrt(self.nu / np.pi)
                / (
                    2.0 ** ((self.nu - 1.0) / 2.0)
                    * gamma(self.nu / 2.0)
                )
                * argument**order
                * kv(order, argument)
            )

        if np.any(~nonzero):
            if self.nu <= 1.0:
                # The paper gives the finite x -> 0 expression
                # for nu > 1. For nu <= 1 the density is singular
                # at zero.
                result[~nonzero] = np.inf
            else:
                result[~nonzero] = (
                    0.5
                    * np.sqrt(self.nu / np.pi)
                    * gamma((self.nu - 1.0) / 2.0)
                    / gamma(self.nu / 2.0)
                )

        return result

    def cdf(self, x: Array) -> Array:
        """Evaluate the Variance-Gamma CDF.

        The CDF follows directly from the SMN representation:

            F(x) =
                integral_0^infinity
                Phi(x * sqrt(lambda))
                IG(lambda; nu/2, nu/2)
                d lambda.

        The vector-valued quadrature evaluates the complete HMM state
        vector in a single numerical integration.
        """
        x = np.asarray(x, dtype=float)

        shape = self.nu / 2.0
        scale = self.nu / 2.0

        def integrand(lam: float) -> Array:
            # Inverse-Gamma density:
            #
            # scale^shape / Gamma(shape)
            # * lambda^(-shape-1)
            # * exp(-scale / lambda)

            density = (
                scale**shape
                / gamma(shape)
                * lam ** (-shape - 1.0)
                * np.exp(-scale / lam)
            )

            return norm.cdf(x * np.sqrt(lam)) * density

        value, _ = quad_vec(
            integrand,
            0.0,
            np.inf,
            epsabs=self.quadrature_epsabs,
            epsrel=self.quadrature_epsrel,
        )

        return np.asarray(value, dtype=float)

    def ppf(self, probability: float) -> float:
        return _numerical_ppf(
            self.cdf,
            probability,
        )


# ============================================================================
# Numerical utilities
# ============================================================================


def _numerical_ppf(
    cdf,
    probability: float,
    initial_width: float = 8.0,
    max_width: float = 1.0e5,
) -> float:
    """Numerically invert a scalar CDF.

    The CDF passed here must accept a NumPy array and return a NumPy array.
    This helper is primarily used for the non-closed-form innovation
    quantiles.

    Parameters
    ----------
    cdf:
        CDF callable.
    probability:
        Target probability.
    initial_width:
        Initial symmetric bracketing interval.
    max_width:
        Maximum bracketing width.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(
            "probability must satisfy 0 < probability < 1."
        )

    width = initial_width

    while width <= max_width:
        lower = -width
        upper = width

        lower_value = float(cdf(np.array([lower]))[0])
        upper_value = float(cdf(np.array([upper]))[0])

        if lower_value <= probability <= upper_value:
            root = brentq(
                lambda value: float(
                    cdf(np.array([value]))[0]
                ) - probability,
                lower,
                upper,
                xtol=1e-10,
                rtol=1e-10,
            )

            return float(root)

        width *= 2.0

    raise RuntimeError(
        "Could not bracket the requested innovation quantile."
    )


# ============================================================================
# HMM
# ============================================================================


class SVMHMM:
    """Hidden Markov approximation of the SVM latent process.

    This class owns the latent-volatility grid, transition matrix, and
    stationary initial distribution.
    """

    def __init__(
        self,
        parameters: SVMParameters,
        grid: VolatilityGrid,
    ) -> None:
        self.parameters = parameters
        self.grid = grid

        self.h = grid.midpoints
        self.exp_h = np.exp(self.h)
        self.exp_half_h = np.exp(0.5 * self.h)

        self.transition_matrix = self._build_transition_matrix()
        self.initial_distribution = self._build_initial_distribution()

    def _build_transition_matrix(self) -> Array:
        """Construct the discretized latent-volatility transition matrix.

        The continuous transition is

            h_t | h_{t-1}
                ~ N(
                    mu + phi * (h_{t-1} - mu),
                    sigma_eta^2
                ).

        The paper approximates the continuous state space using
        equidistant intervals represented by their midpoints.
        """
        p = self.parameters

        conditional_means = (
            p.mu
            + p.phi * (self.h - p.mu)
        )

        standardized = (
            self.h[None, :]
            - conditional_means[:, None]
        ) / p.sigma_eta

        transition = (
            norm.pdf(standardized)
            * self.grid.width
            / p.sigma_eta
        )

        # The finite grid truncates the continuous transition density.
        # Normalize each row so Gamma is a valid transition matrix.
        row_sums = transition.sum(axis=1)

        if np.any(row_sums <= 0.0):
            raise RuntimeError(
                "Transition matrix contains an invalid row."
            )

        transition /= row_sums[:, None]

        return np.ascontiguousarray(transition)

    def _build_initial_distribution(self) -> Array:
        """Construct the stationary initial state distribution.

        The paper assumes

            h_1 ~ N(
                mu,
                sigma_eta^2 / (1 - phi^2)
            ).

        The density is evaluated at the grid midpoints and multiplied
        by the interval width.
        """
        p = self.parameters

        stationary_sd = (
            p.sigma_eta / np.sqrt(1.0 - p.phi**2)
        )

        probability = (
            norm.pdf(
                self.h,
                loc=p.mu,
                scale=stationary_sd,
            )
            * self.grid.width
        )

        total = probability.sum()

        if total <= 0.0:
            raise RuntimeError(
                "Initial state distribution is invalid."
            )

        probability /= total

        return np.ascontiguousarray(probability)


# ============================================================================
# VaR calculator
# ============================================================================


class VaRCalculator:
    """One-step-ahead SVM-HMM Value at Risk calculator.

    The filtering/prediction sequence is

        phi_{t-1}
            -> phi_{t-1} Gamma
            -> predictive return distribution
            -> VaR_t
            -> observe y_t
            -> phi_t.

    Crucially, y_t is never used to construct VaR_t.

    Parameters
    ----------
    parameters:
        Fitted SVM parameters.
    innovation:
        Standardized innovation distribution.
    grid:
        Latent-volatility discretization.
    """

    def __init__(
        self,
        parameters: SVMParameters,
        innovation: InnovationDistribution,
        grid: Optional[VolatilityGrid] = None,
    ) -> None:
        self.parameters = parameters
        self.innovation = innovation

        self.grid = (
            grid
            if grid is not None
            else VolatilityGrid()
        )

        self.hmm = SVMHMM(
            parameters=parameters,
            grid=self.grid,
        )

        self._h = self.hmm.h
        self._exp_h = self.hmm.exp_h
        self._scale = self.hmm.exp_half_h

        self._transition = self.hmm.transition_matrix

    # ------------------------------------------------------------------
    # Observation model
    # ------------------------------------------------------------------

    def _state_locations(
        self,
        previous_return: float,
    ) -> Array:
        """Return the conditional mean for every latent state.

        mu_i,t =
            beta0
            + beta1 * y_{t-1}
            + beta2 * exp(h_i).
        """
        p = self.parameters

        return (
            p.beta0
            + p.beta1 * previous_return
            + p.beta2 * self._exp_h
        )

    def _state_standardized_values(
        self,
        value: float,
        previous_return: float,
    ) -> Array:
        """Standardize an observation under every latent state."""
        locations = self._state_locations(previous_return)

        return (
            value - locations
        ) / self._scale

    def _emission_pdf(
        self,
        value: float,
        previous_return: float,
    ) -> Array:
        """Evaluate p(y_t | h_t = h_i, y_{t-1}) for all states."""
        standardized = self._state_standardized_values(
            value,
            previous_return,
        )

        # Transformation:
        #
        # p(y | h_i)
        #   = f_epsilon(z_i) / exp(h_i / 2)

        return (
            self.innovation.pdf(standardized)
            / self._scale
        )

    def _emission_cdf(
        self,
        value: float,
        previous_return: float,
    ) -> Array:
        """Evaluate F_i(value) for every latent state."""
        standardized = self._state_standardized_values(
            value,
            previous_return,
        )

        return self.innovation.cdf(standardized)

    # ------------------------------------------------------------------
    # Forward filtering
    # ------------------------------------------------------------------

    def _update_filter(
        self,
        filtered: Array,
        observed_return: float,
        previous_return: float,
    ) -> Array:
        """Update the HMM filtering distribution.

        Given

            phi_{t-1} = P(X_{t-1} | y_1, ..., y_{t-1}),

        first predict

            xi_t = phi_{t-1} Gamma,

        and then condition on y_t:

            phi_t proportional to
                xi_t * p(y_t | X_t).

        This is the scaled forward recursion recommended in the paper.
        """
        predicted = filtered @ self._transition

        emission = self._emission_pdf(
            observed_return,
            previous_return,
        )

        updated = predicted * emission

        normalization = updated.sum()

        if not np.isfinite(normalization) or normalization <= 0.0:
            raise FloatingPointError(
                "HMM filtering failed. "
                "Consider enlarging the volatility grid."
            )

        return updated / normalization

    # ------------------------------------------------------------------
    # Predictive distribution
    # ------------------------------------------------------------------

    def _forecast_state_probabilities(
        self,
        filtered: Array,
    ) -> Array:
        """Compute the one-step-ahead latent-state probabilities.

        xi_t = phi_{t-1} Gamma.
        """
        forecast = filtered @ self._transition

        total = forecast.sum()

        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError(
                "Invalid forecast-state probabilities."
            )

        return forecast / total

    def predictive_cdf(
        self,
        value: float,
        previous_return: float,
        forecast_state_probabilities: Array,
    ) -> float:
        """Evaluate the HMM one-step-ahead predictive CDF.

        F_t(y) =
            sum_i xi_{t,i} F_i(y).
        """
        state_cdfs = self._emission_cdf(
            value,
            previous_return,
        )

        return float(
            np.dot(
                forecast_state_probabilities,
                state_cdfs,
            )
        )

    # ------------------------------------------------------------------
    # Predictive quantile
    # ------------------------------------------------------------------

    def _predictive_quantile(
        self,
        alpha: float,
        previous_return: float,
        forecast_state_probabilities: Array,
    ) -> float:
        """Numerically invert the HMM predictive CDF."""
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                "alpha must satisfy 0 < alpha < 1."
            )

        locations = self._state_locations(previous_return)

        # Construct an initial bracket using the innovation quantile.
        innovation_quantile = self.innovation.ppf(alpha)

        candidate = (
            locations
            + self._scale * innovation_quantile
        )

        lower = float(np.min(candidate))
        upper = float(np.max(candidate))

        # A mixture quantile is not necessarily between the same-state
        # alpha quantiles. Expand the bracket until it encloses alpha.
        def predictive_cdf(value: float) -> float:
            return self.predictive_cdf(
                value=value,
                previous_return=previous_return,
                forecast_state_probabilities=(
                    forecast_state_probabilities
                ),
            )

        lower_cdf = predictive_cdf(lower)
        upper_cdf = predictive_cdf(upper)

        width = max(upper - lower, 1.0)

        for _ in range(50):
            if lower_cdf <= alpha <= upper_cdf:
                break

            if lower_cdf > alpha:
                lower -= width
                lower_cdf = predictive_cdf(lower)

            if upper_cdf < alpha:
                upper += width
                upper_cdf = predictive_cdf(upper)

            width *= 2.0

        else:
            raise RuntimeError(
                "Could not bracket the predictive VaR."
            )

        return float(
            brentq(
                lambda value: predictive_cdf(value) - alpha,
                lower,
                upper,
                xtol=1e-9,
                rtol=1e-9,
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        log_returns: Array,
        alpha: float = 0.01,
        initial_return: Optional[float] = None,
    ) -> Array:
        """Calculate one-step-ahead VaR through the complete sample.

        Parameters
        ----------
        log_returns:
            One-dimensional array containing y_1, ..., y_T.

        alpha:
            Left-tail probability. For example:

                alpha=0.01 -> 1% VaR
                alpha=0.05 -> 5% VaR

        initial_return:
            Optional y_0.

            If supplied, VaR is calculated for every element of
            ``log_returns``.

            If omitted, log_returns[0] is treated as y_0 and the
            returned VaR array has NaN at index 0.

        Returns
        -------
        numpy.ndarray
            One-step-ahead VaR for each observation.

        Notes
        -----
        At time t the ordering is:

            1. forecast h_t using information through t-1;
            2. construct predictive distribution of y_t;
            3. calculate VaR_t;
            4. observe y_t;
            5. update the filtering distribution.

        Thus the realized y_t never leaks into VaR_t.
        """
        y = np.asarray(log_returns, dtype=float)

        if y.ndim != 1:
            raise ValueError(
                "log_returns must be one-dimensional."
            )

        if y.size == 0:
            raise ValueError(
                "log_returns cannot be empty."
            )

        if not np.all(np.isfinite(y)):
            raise ValueError(
                "log_returns contains NaN or infinite values."
            )

        if not 0.0 < alpha < 1.0:
            raise ValueError(
                "alpha must satisfy 0 < alpha < 1."
            )

        var = np.full(y.size, np.nan)

        filtered = self.hmm.initial_distribution.copy()

        if initial_return is None:
            if y.size == 1:
                return var

            previous_return = float(y[0])
            start = 1

        else:
            previous_return = float(initial_return)
            start = 0

        for t in range(start, y.size):
            forecast_probabilities = (
                self._forecast_state_probabilities(
                    filtered
                )
            )

            var[t] = self._predictive_quantile(
                alpha=alpha,
                previous_return=previous_return,
                forecast_state_probabilities=(
                    forecast_probabilities
                ),
            )

            filtered = self._update_filter(
                filtered=filtered,
                observed_return=float(y[t]),
                previous_return=previous_return,
            )

            previous_return = float(y[t])

        return var


# ============================================================================
# Factory
# ============================================================================


def create_var_calculator(
    parameters: SVMParameters,
    model: str = "t",
    lower: float = -2.5,
    upper: float = 2.5,
    n_states: int = 200,
) -> VaRCalculator:
    """Create a VaR calculator for one of the paper's SVM models.

    Parameters
    ----------
    parameters:
        SVM model parameters.

    model:
        One of:

            "n" / "normal" / "gaussian"
            "t" / "student-t"
            "s" / "slash"
            "vg" / "variance-gamma"

    lower, upper:
        Latent log-volatility grid boundaries.

    n_states:
        Number of HMM states.
    """
    normalized_model = model.lower().replace("_", "-")

    if normalized_model in {
        "n",
        "normal",
        "gaussian",
    }:
        innovation = NormalInnovation()

    elif normalized_model in {
        "t",
        "student-t",
    }:
        if parameters.nu is None:
            raise ValueError(
                "nu is required for SVM-t."
            )

        innovation = StudentTInnovation(
            parameters.nu
        )

    elif normalized_model in {
        "s",
        "slash",
    }:
        if parameters.nu is None:
            raise ValueError(
                "nu is required for SVM-S."
            )

        innovation = SlashInnovation(
            parameters.nu
        )

    elif normalized_model in {
        "vg",
        "variance-gamma",
    }:
        if parameters.nu is None:
            raise ValueError(
                "nu is required for SVM-VG."
            )

        innovation = VarianceGammaInnovation(
            parameters.nu
        )

    else:
        raise ValueError(
            f"Unknown SVM model: {model!r}. "
            "Expected 'n', 't', 's', or 'vg'."
        )

    grid = VolatilityGrid(
        lower=lower,
        upper=upper,
        n_states=n_states,
    )

    return VaRCalculator(
        parameters=parameters,
        innovation=innovation,
        grid=grid,
    )


__all__ = [
    "SVMParameters",
    "VolatilityGrid",
    "InnovationDistribution",
    "NormalInnovation",
    "StudentTInnovation",
    "SlashInnovation",
    "VarianceGammaInnovation",
    "SVMHMM",
    "VaRCalculator",
    "create_var_calculator",
]