from typing import List, Optional
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd


def create_var_violation_plot(
    dates: pd.Series,
    log_returns: npt.ArrayLike,
    var_estimates: List[float],
    shift_steps: int = 1,  # Positive = Shift Forward (+1)
) -> plt.Figure:
    # -----------------------------
    # Data Preparation & Alignment
    # -----------------------------
    # Convert inputs to pandas Series to ensure consistent indexing
    dates_s = pd.Series(dates).reset_index(drop=True)
    returns_s = pd.Series(log_returns).reset_index(drop=True)
    var_s = pd.Series(var_estimates).reset_index(drop=True)

    # Safety check on raw lengths
    assert (
        len(dates_s) == len(returns_s) == len(var_s)
    ), "All inputs must have the same length."

    # Apply the shift (e.g., shift(1) moves forecast made at t to t+1)
    if shift_steps != 0:
        var_s = var_s.shift(shift_steps)

    # Filter out NaNs created by shifting so matplotlib doesn't misbehave
    valid_mask = ~var_s.isna() & ~returns_s.isna()
    dates_plot = dates_s[valid_mask]
    returns_plot = returns_s[valid_mask]
    var_plot = var_s[valid_mask]

    # -----------------------------
    # Figure
    # -----------------------------
    fig, ax = plt.subplots(figsize=(15, 7), dpi=140)

    # Actual returns
    ax.plot(
        dates_plot,
        returns_plot,
        color="#1f77b4",
        linewidth=1.2,
        label="Log Returns",
        zorder=3,
    )

    # VaR estimate
    ax.plot(
        dates_plot,
        var_plot,
        color="#d62728",
        linewidth=2.2,
        linestyle="--",
        label=f"VaR Estimate (Shifted +{shift_steps})"
        if shift_steps
        else "VaR Estimate",
        zorder=4,
    )

    # Highlight VaR violations (Actual return strictly below VaR threshold)
    violations = returns_plot < var_plot

    ax.scatter(
        dates_plot[violations],
        returns_plot[violations],
        color="crimson",
        s=28,
        edgecolors="white",
        linewidths=0.6,
        label="VaR Violations",
        zorder=5,
    )

    # Shade the danger region
    ax.fill_between(
        dates_plot,
        var_plot,
        returns_plot,
        where=violations,
        interpolate=True,
        color="crimson",
        alpha=0.18,
    )

    # Zero return reference line
    ax.axhline(
        0,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.8,
    )

    # -----------------------------
    # Formatting
    # -----------------------------
    ax.set_title(
        "Daily Log Returns vs Value-at-Risk (VaR)",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )

    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Log Return", fontsize=13)

    # Date formatting
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Grid
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
    )

    # Remove top/right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        fontsize=11,
    )

    ax.margins(x=0.01)
    plt.tight_layout()

    return fig