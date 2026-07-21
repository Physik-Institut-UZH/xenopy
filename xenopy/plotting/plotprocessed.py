import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


def plot_histogram(arrays, labels, colors=None, bins=125, range_=None,
                   highlight_last=True, shade_region=None, shade_label="",
                   xlabel="", bin_width=None, dpi=150):
    """
    Overlay step histograms for n arrays.

    Parameters
    ----------
    arrays        : list of array-like  — values to histogram
    labels        : list of str         — legend labels, one per array
    colors        : list of str or None — hex colours; defaults to a built-in palette
    bins          : int                 — number of bins
    range_        : (float, float) or None — histogram range; if None, auto from data
    highlight_last: bool                — fill the last array with alpha
    shade_region  : (float, float) or None — shade this x-region (e.g. drift window)
    shade_label   : str                 — legend label for the shaded region
    xlabel        : str                 — x-axis label
    bin_width     : float or None       — used for the y-axis label; auto if range_ given
    dpi           : int
    """
    if colors is None:
        default_palette = ["#a5a1a1", "#4477aa", "#9a0505", "#228833", "#aa3377"]
        colors = default_palette[:len(arrays)]

    if bin_width is None and range_ is not None:
        bin_width = (range_[1] - range_[0]) / bins

    fig, ax = plt.subplots(dpi=dpi)

    for data, color, label in zip(arrays, colors, labels):
        ax.hist(data, bins=bins, range=range_, histtype="step",
                color=color, label=f"{label} (n={len(data)})", linewidth=1.2)

    if highlight_last:
        ax.hist(arrays[-1], bins=bins, range=range_,
                histtype="stepfilled", color=colors[-1], alpha=0.4)

    if shade_region is not None:
        ax.axvspan(shade_region[0], shade_region[1], color="burlywood",
                   alpha=0.4, label=shade_label or "region")

    ax.set_xlabel(xlabel)
    ylabel = f"Events / {bin_width} µs" if bin_width is not None else "Events"
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig, ax


def plot_drift_time(arrays, labels, colors=None, bins=125, range_=(0, 2500),
                    highlight_last=True, drift_window=None, bin_width_us=None):
    """
    Drift-time histogram: thin wrapper around plot_histogram with drift-specific
    labels and shading. arrays are drift times in µs (already extracted).
    """
    return plot_histogram(
        arrays, labels, colors=colors, bins=bins, range_=range_,
        highlight_last=highlight_last,
        shade_region=drift_window, shade_label="Drift window",
        xlabel="Drift time [µs]", bin_width=bin_width_us,
    )