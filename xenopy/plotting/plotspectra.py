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

def plot_occupancy_vs_led(
        data: dict,
        title: str = '',
        occ_range: Optional[tuple[float, float]] = (1, 3),
        color: str = '#9a0505',
        colors: Optional[list] = None,
        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot occupancy vs LED voltage — single or multiple curves.

    Parameters
    ----------
    data : dict
        Single curve: ``{led_str: (occ, occ_err)}``
        Multiple curves: ``{label: {led_str: (occ, occ_err)}}``
        As returned by looping ``xs.compute_occupancy()`` over LED voltages.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    multi = isinstance(next(iter(data.values())), dict)

    def _plot_one(occ_by_led, label, col):
        x      = [float(led[:-1])   for led in sorted(occ_by_led, key=lambda k: float(k[:-1]))]
        occ_vals  = [occ_by_led[led][0] for led in sorted(occ_by_led, key=lambda k: float(k[:-1]))]
        occ_errs  = [occ_by_led[led][1] for led in sorted(occ_by_led, key=lambda k: float(k[:-1]))]
        kw = dict(fmt='o-', capsize=4)
        if col is not None:
            kw['color'] = col
        if label is not None:
            kw['label'] = label
        ax.errorbar(x, occ_vals, yerr=occ_errs, **kw)

    if multi:
        _colors = colors if colors is not None else [None] * len(data)
        for (label, occ_by_led), col in zip(data.items(), _colors):
            _plot_one(occ_by_led, label, col)
        ax.legend(loc='upper left')
    else:
        _plot_one(data, None, color)

    if occ_range is not None:
        ax.axhspan(*occ_range, alpha=0.15, color='darkseagreen', label='target range')

    ax.set_xlabel('LED voltage [V]')
    ax.set_ylabel('λ [PE / trigger]')
    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax