import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import numpy.typing as npt
from typing import List

def create_var_violation_plot(dates: pd.Series[pd.Timestamp], log_returns: npt.ArrayLike, var_estimates: List[float]):

    # Safety check
    assert len(dates) == len(log_returns) == len(var_estimates)

    # -----------------------------
    # Figure
    # -----------------------------
    fig, ax = plt.subplots(figsize=(15, 7), dpi=140)

    # Actual returns
    ax.plot(
        dates,
        log_returns,
        color="#1f77b4",
        linewidth=1.2,
        label="Log Returns",
        zorder=3,
    )

    # VaR estimate
    ax.plot(
        dates,
        var_estimates,
        color="#d62728",
        linewidth=2.2,
        linestyle="--",
        label="VaR Estimate",
        zorder=4,
    )

    # Highlight VaR violations
    violations = log_returns < var_estimates

    ax.scatter(
        dates[violations],
        log_returns[violations],
        color="crimson",
        s=28,
        edgecolors="white",
        linewidths=0.6,
        label="VaR Violations",
        zorder=5,
    )

    # Shade the danger region
    ax.fill_between(
        dates,
        var_estimates,
        log_returns,
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

    # Nice date formatting
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

    # Slight margins
    ax.margins(x=0.01)

    plt.tight_layout()
    return plt