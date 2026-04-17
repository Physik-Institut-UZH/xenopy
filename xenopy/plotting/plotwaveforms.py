import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from typing import Optional


def plot_waveform(waveform_array: np.ndarray, full_y: bool | tuple[float, float] = False,
                  full_x: bool = True, pe: bool = False,
                  title: str = '',
                  baseline_range: Optional[tuple[int, int]] = None,
                  signal_range: Optional[tuple[int, int]] = None,
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot a waveform from its array.

    Args:
        waveform_array (np.ndarray): waveform array. Elements of the array
            correspond to the sample of the waveform and its value the recorded ADC
            counts.
        full_y (bool | tuple[float, float], optional): plot full range of ADC amplitude or specify y-axis limits. Defaults
            to False.
        full_x (bool, optional): plot full range of ADC samples. Defaults
            to True.
        pe (bool, optional): parse that the waveform is in PE/s (for peaks).
            Defaults to False.
        ax (plt.Axes, optional): axes to plot into. Defaults to None.

    Returns:
        plt.Axes: axes with plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    n_samples = len(waveform_array)
    x = np.arange(0, n_samples)
    ax.plot(x, waveform_array, color='royalblue')
    baseline_rough = np.median(waveform_array[:50])
    std_rough = np.std(waveform_array[:50])
    ax.axhline(baseline_rough, ls='--', lw=1, c='gray', alpha=0.8, label=r"Baseline")
    ax.axhline(baseline_rough - std_rough, ls='--', lw=1.2, c='darkorange', alpha=0.8, label=r"Baseline ± 1$\sigma$")
    ax.axhline(baseline_rough + std_rough, ls='--', lw=1.2, c='darkorange', alpha=0.8)
    ax.axhline(baseline_rough - std_rough * 5, ls='--', lw=1.2, c='firebrick', alpha=0.8, label=r"Baseline ± 5$\sigma$")
    ax.axhline(baseline_rough + std_rough * 5, ls='--', lw=1.2, c='firebrick', alpha=0.8)
    if baseline_range is not None:
        ax.axvspan(*baseline_range, alpha=0.15, color='gray', label=f"Baseline [{baseline_range[0]}:{baseline_range[1]}]")
    if signal_range is not None:
        ax.axvspan(*signal_range, alpha=0.15, color='green', label=f"Signal [{signal_range[0]}:{signal_range[1]}]")
    if full_x != True:
        ax.set_xlim(full_x)
    if isinstance(full_y, tuple):
        ax.set_ylim(full_y)                 
    elif full_y is True:
        ax.set_ylim(0, 2**14)               
    else:                                   
        ax.set_ylim(min(waveform_array) - std_rough*6,
                    max(waveform_array) + std_rough*6)
    ax.set_xlabel(r"$\mathrm{Sample~number}$")
    ax.set_ylabel(r"$\mathrm{ADC~counts}$")
    ax.legend(loc='best')

    if title:
        ax.set_title(title)
    if pe == True:
        ax.set_ylabel('PE/s')
    return ax


def plot_all_tiles_average(tiles: dict, title: str = '') -> plt.Figure:
    """Plot average waveforms for all tiles overlaid in a single axes.

    Args:
        tiles: dict as returned by ``load_xenodaq_run`` (third return value),
            i.e. ``{tile_name: {'waveforms': ndarray, ...}}``.
        title: optional figure title.

    Returns:
        plt.Figure: figure with all tiles on one plot.
    """
    tile_names = sorted(tiles.keys())
    fig, ax = plt.subplots(figsize=(7, 4))

    for tile in tile_names:
        wf = tiles[tile]["waveforms"].mean(axis=0)
        x = np.arange(len(wf))
        ax.plot(x, wf, label=f'Tile {tile.replace("tile_", "").upper()}', alpha=0.8)

    ax.set_xlabel(r'$\mathrm{Sample~number}$')
    ax.set_ylabel(r'$\mathrm{ADC~counts}$')
    ax.legend(loc='best', ncol=3, frameon=True)

    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_pulses(waveform: np.ndarray, pulse_list: list,
                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot the identified pulses in a waveform.

    Args:
        waveform (np.ndarray): waveform array. Elements of the array
            correspond to the sample of the waveform and its value the recorded ADC
            counts.
        pulse_list (list): list of pulses, as the output of
            pulse_processing.find_pulses_simple.
        ax (plt.Axes, optional): axes to plot into. Defaults to None.

    Returns:
        plt.Axes: axes with plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax = plot_waveform(waveform, ax=ax)
    for pulse in pulse_list:
        if len(pulse) > 1:
            ax.fill_betweenx(y=np.linspace(0, 16000, 100),
                             x1=pulse[0], x2=pulse[-1],
                             alpha=0.2, color='cyan')
    return ax