import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from xenopy.processing.spectra import _gaussian


# single spectrum 

def plot_spectrum(charge: np.ndarray,
                  bins: int = 400,
                  range: Optional[tuple[float, float]] = (-100, 2000),
                  log_y: bool = False,
                  title: str = '',
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot a charge spectrum for one tile.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(charge, bins=bins, range=range,
            histtype='step', linewidth=1.2, color='royalblue')
    ax.set_xlabel(r"$\mathrm{Charge~[ADC~counts]}$")
    ax.set_ylabel(r"$\mathrm{Counts}$")
    if log_y:
        ax.set_yscale('log')
    if title:
        ax.set_title(title)
    return ax


# overlay spectra for different LED voltages 

def plot_spectra_vs_led(charge_by_led: dict,
                        bins: int = 400,
                        range: tuple[float, float] = (-100, 2000),
                        log_y: bool = False,
                        title: str = '',
                        cmap: str = 'tab10',
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Overlay charge spectra for different LED voltages on one tile.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sorted_items = sorted(charge_by_led.items(),
                          key=lambda kv: float(kv[0].replace(" ", "").rstrip("V")))
    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.85, len(sorted_items)))

    for (led, charge), color in zip(sorted_items, colors):
        ax.hist(charge, bins=bins, range=range,
                histtype='step', linewidth=1.2,
                color=color, alpha=0.8, label=f"{led}")

    ax.set_xlabel(r"$\mathrm{Charge~[ADC~counts]}$")
    ax.set_ylabel(r"$\mathrm{Counts}$")
    if log_y:
        ax.set_yscale('log')
    if title:
        ax.set_title(title)
    ax.legend(title="LED voltage", ncol=2)
    return ax

def plot_spectrum_fit(fit_result: dict,
                      title: str = "",
                      ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot histogram plus fitted 0-PE and 1-PE Gaussians """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    hist = fit_result["hist"]
    bin_edges = fit_result["bin_edges"]
    bin_centers = fit_result["bin_centers"]

    ax.hist(bin_centers, bins=bin_edges, weights=hist,
            histtype='step', linewidth=1.2, color='black', label='Data')

    x_plot = np.linspace(bin_edges[0], bin_edges[-1], 2000)

    fit0 = fit_result.get("fit_0pe")
    if fit0 is not None:
        p0 = fit0["params"]
        ax.plot(x_plot, _gaussian(x_plot, *p0), color='dodgerblue', lw=2, alpha=0.8,
            label="Pedestal fit")

    fit1 = fit_result["fit_1pe"]
    p1 = fit1["params"]
    ax.plot(x_plot, _gaussian(x_plot, *p1), color='orangered', lw=2, alpha=0.8,
            label="1-PE peak fit")

    ax.set_xlabel(r"$\mathrm{Charge~[ADC~counts]}$")
    ax.set_ylabel(r"$\mathrm{Counts}$")
    if title:
        ax.set_title(title)
    ax.legend(loc='best')
    return ax
