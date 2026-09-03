from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelParameters:
    """Dataclass encapsulating the core 6 model parameters exchanged between

    Python and C++ IPC executables.
    """

    beta_0: float = 0.01
    beta_1: float = 0.05
    beta_2: float = -0.02
    mu: float = -0.5
    phi: float = 0.95
    sigma_eta: float = 0.25

    def __post_init__(self):
        """Validates fundamental statistical parameter bounds."""
        if self.sigma_eta <= 0:
            raise ValueError(
                f"sigma_eta must be positive (> 0), got {self.sigma_eta}"
            )
        if not (-1.0 < self.phi < 1.0):
            raise ValueError(
                f"phi must lie in (-1.0, 1.0) for stationarity, got {self.phi}"
            )
        if not (-1.0 < self.beta_1 < 1.0):
            raise ValueError(
                f"beta_1 must lie in (-1.0, 1.0), got {self.beta_1}"
            )

    @classmethod
    def from_list(cls, params: List[float]) -> "ModelParameters":
        """Constructs ModelParameters from an ordered list of 6 floats."""
        if len(params) != 6:
            raise ValueError(
                f"Expected exactly 6 parameters, received {len(params)}."
            )
        return cls(
            beta_0=params[0],
            beta_1=params[1],
            beta_2=params[2],
            mu=params[3],
            phi=params[4],
            sigma_eta=params[5],
        )

    def to_list(self) -> List[float]:
        """Converts model parameters to an ordered list for IPC stdout/stdin

        streams.
        """
        return [
            self.beta_0,
            self.beta_1,
            self.beta_2,
            self.mu,
            self.phi,
            self.sigma_eta,
        ]


@dataclass(frozen=True, slots=True)
class SVMParameters:
    """Parameters of the SVM model.

    Parameters
    ----------
    beta0:
        Intercept of the conditional mean equation.
    beta1:
        AR(1) coefficient of the return process.
    beta2:
        Volatility-in-mean coefficient.
    mu:
        Unconditional mean of the latent log-volatility process.
    phi:
        Persistence parameter of the latent log-volatility process.
    sigma:
        Innovation standard deviation of the latent log-volatility process.
    nu:
        Tail parameter for SVM-t, SVM-S, and SVM-VG.
        Must be omitted for SVM-N.
    """

    beta0: float
    beta1: float
    beta2: float
    mu: float
    phi: float
    sigma_eta: float
    nu: Optional[float] = None

    def __post_init__(self) -> None:
        if not abs(self.beta1) < 1.0:
            raise ValueError("beta1 must satisfy |beta1| < 1.")

        if not abs(self.phi) < 1.0:
            raise ValueError("phi must satisfy |phi| < 1.")

        if self.sigma_eta <= 0.0:
            raise ValueError("sigma must be positive.")

        if self.nu is not None and self.nu <= 0.0:
            raise ValueError("nu must be positive.")