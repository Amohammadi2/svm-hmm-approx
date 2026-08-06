from dataclasses import dataclass, asdict
import os
import subprocess
import sys
from typing import List, Union, Dict, Any, Tuple, Optional
import numpy as np
from .datastructs import ModelParameters



class ParameterEstimatorIPC:
    """Manages stdin/stdout/stderr IPC communication with the C++ parameter

    estimation executable.
    """

    def __init__(self, executable_path: str):
        """Initialize and validate the executable path."""
        self.executable_path = os.path.abspath(executable_path)
        if not os.path.isfile(self.executable_path):
            raise FileNotFoundError(
                f"Executable not found at path: {self.executable_path}"
            )
        if not os.access(self.executable_path, os.X_OK):
            raise PermissionError(
                f"File at {self.executable_path} lacks execution permissions."
            )

    def estimate(
        self, log_returns: Union[List[float], np.ndarray]
    ) -> ModelParameters:
        """Sends log returns to the C++ executable via stdin and returns the

        estimated parameters packaged inside a ModelParameters instance.

        :param log_returns: Collection of floating-point log returns.
        :return: ModelParameters dataclass containing the 6 estimated parameters.
        """
        n = len(log_returns)
        if n == 0:
            raise ValueError("Data array `log_returns` cannot be empty.")

        # Spawn the subprocess with pipe redirection
        process = subprocess.Popen(
            [self.executable_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Handle string encoding automatically
            bufsize=1,  # Line buffered
        )

        try:
            # 1. Format stdin payload: first line is length N, followed by N numbers
            payload_lines = [str(n)]
            payload_lines.extend(f"{val:.10g}" for val in log_returns)
            input_data = "\n".join(payload_lines) + "\n"

            # 2. Communicate payload and await output safely
            stdout_data, stderr_data = process.communicate(input=input_data)

        except Exception as e:
            process.kill()
            process.wait()
            raise RuntimeError(f"Subprocess execution failed: {e}") from e

        # 3. Check process exit code
        if process.returncode != 0:
            raise RuntimeError(
                f"C++ process exited with non-zero status code ({process.returncode}).\n"
                f"Stderr Output:\n{stderr_data.strip()}"
            )

        # 4. Parse output parameters
        output_lines = stdout_data.split("\n")
        cleaned_lines = [line.strip() for line in output_lines if line.strip()]

        if len(cleaned_lines) != 6:
            raise ValueError(
                f"Expected exactly 6 parameters from C++, but received {len(cleaned_lines)}.\n"
                f"Raw Stdout Output:\n{repr(stdout_data)}\n"
                f"Stderr Output:\n{stderr_data.strip()}"
            )

        # Convert output floats and wrap in ModelParameters
        try:
            parsed_floats = [float(val) for val in cleaned_lines]
            return ModelParameters.from_list(parsed_floats)
        except ValueError as e:
            raise ValueError(
                f"Failed to parse C++ stdout as floats: {cleaned_lines}"
            ) from e


class VaRCalculatorIPC:
    """Synchronous IPC Stub to communicate with the C++ VaR calculator

    executable.
    """

    def __init__(self, executable_path: str):
        self.executable_path = os.path.abspath(executable_path)
        if not os.path.isfile(self.executable_path):
            raise FileNotFoundError(
                f"Executable not found at path: {self.executable_path}"
            )
        if not os.access(self.executable_path, os.X_OK):
            raise PermissionError(
                f"File at {self.executable_path} lacks execution permissions."
            )

    def calculate_var(
        self,
        returns: Union[List[float], np.ndarray],
        params: Optional[ModelParameters] = None,
        m: int = 100,
        std_dv_rng: float = 4.0,
    ) -> Tuple[List[float], str]:
        """Sends return series and ModelParameters to the C++ executable and

        retrieves predicted VaR values and debug logs synchronously.

        :param returns: List or array of daily log returns.
        :param params: ModelParameters instance (uses standard defaults if None).
        :param m: Number of integration steps / simulation grid size.
        :param std_dv_rng: Standard deviation integration range limit.
        :return: Tuple of (List of calculated VaRs, stderr debug log output).
        """
        # Fallback to default dataclass instance if none provided
        if params is None:
            params = ModelParameters()

        # 1. Format payload string matching C++ cin order in NDEBUG mode
        # Streams the 6 core parameters, then execution settings, then return count & series
        payload_lines = [
            f"{params.beta_0}",
            f"{params.beta_1}",
            f"{params.beta_2}",
            f"{params.mu}",
            f"{params.phi}",
            f"{params.sigma_eta}",
            f"{int(m)}",
            f"{std_dv_rng}",
            f"{len(returns)}",
        ]

        # Append return values line by line
        payload_lines.extend(str(r) for r in returns)

        # Prepare input data block
        input_data = "\n".join(payload_lines) + "\n"

        # 2. Run C++ executable synchronously via subprocess.run
        process = subprocess.run(
            [self.executable_path],
            input=input_data,
            text=True,  # Handles string encoding/decoding automatically
            capture_output=True,  # Captures both stdout and stderr
            check=False,
        )

        # 3. Check for execution errors
        if process.returncode != 0:
            raise RuntimeError(
                f"C++ Executable failed with exit code {process.returncode}:\n{process.stderr}"
            )

        # 4. Parse stdout line-by-line
        var_estimates: List[float] = []
        for line in process.stdout.strip().splitlines():
            line = line.strip()
            if line:
                try:
                    var_estimates.append(float(line))
                except ValueError:
                    # Ignores human-readable debug text if present
                    continue

        return var_estimates, process.stderr