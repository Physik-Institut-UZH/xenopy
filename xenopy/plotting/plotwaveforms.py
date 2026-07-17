import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from typing import Optional
from pathlib import Path
import matplotlib.cm as cm



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
    colors = [cm.tab20(i) for i in range(len(tile_names))]
    
    for tile, color in zip(tile_names, colors):
        wf = tiles[tile]["waveforms"].mean(axis=0)
        x = np.arange(len(wf))
        ax.plot(x, wf, color = color, label=f'Tile {tile.replace("tile_", "").upper()}', alpha=0.8)

    ax.set_xlabel(r'$\mathrm{Sample~number}$')
    ax.set_ylabel(r'$\mathrm{ADC~counts}$')
    ax.legend(loc = "best", ncol=2, frameon=True)

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


def plot_event(event_idx, wf, muon1, muon2, date_str, offset=0,
                t0=900, t1=1100, s0=140_000, s1=190_000,
                dt_ns=10, save_path=None):
    """
    Plot trigger region + drift window for a single event, with x-axis in
    drift time (us) instead of raw sample index.

    Parameters
    ----------
    event_idx : int
        Global event number.
    wf, muon1, muon2 : array-like
        Waveform arrays as loaded (0-indexed locally, starting at `offset`).
    date_str : str
        Label for the run/date, shown in the title.
    offset : int
        entry_start used when loading wf/muon1/muon2.
    t0, t1 : int
        Sample range for the trigger region (left panel).
    s0, s1 : int
        Sample range for the drift window (right panel).
    dt_ns : float
        Sample period in nanoseconds (default 10 ns/sample).
    save_path : str or Path, optional
        If given, saves the figure there instead of showing it.

    Returns
    -------
    fig, (ax_left, ax_right)
    """
    local_idx = event_idx - offset
    if not (0 <= local_idx < len(wf)):
        raise IndexError(
            f"Event {event_idx} not in loaded range "
            f"[{offset}, {offset + len(wf)})"
        )

    dt_us = dt_ns / 1000  # ns -> us

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 3), dpi=120)
    fig.suptitle(f'Event {event_idx}  —  {date_str}', fontsize=14, fontweight='bold')

    # left: trigger region with muon channels
    t1_wf = min(t1, len(wf[local_idx]))
    t1_m1 = min(t1, len(muon1[local_idx]))
    t1_m2 = min(t1, len(muon2[local_idx]))
    ax_left.plot(np.arange(t0, t1_wf) * dt_us, wf[local_idx][t0:t1_wf],
                 color='#4477aa', lw=1.1, label='summed tiles')
    ax_left.plot(np.arange(t0, t1_m1) * dt_us, muon1[local_idx][t0:t1_m1],
                 color='#9a0505', lw=1.1, alpha=0.7, label='muon1')
    ax_left.plot(np.arange(t0, t1_m2) * dt_us, muon2[local_idx][t0:t1_m2],
                 color='#228833', lw=1.1, alpha=0.7, label='muon2')
    ax_left.set_xlim(t0 * dt_us, t1 * dt_us)
    ax_left.set_ylabel('ADC counts', fontsize=11)
    ax_left.set_xlabel('Drift time [µs]', fontsize=11)
    ax_left.legend(fontsize=11, loc='upper right')
    ax_left.set_title('Trigger region', fontsize=13, pad=6)

    # right: drift window region
    s1_wf = min(s1, len(wf[local_idx]))
    ax_right.plot(np.arange(s0, s1_wf) * dt_us, wf[local_idx][s0:s1_wf],
                  color='#4477aa', lw=0.8, label='summed tiles')
    ax_right.set_xlim(s0 * dt_us, s1 * dt_us)
    ax_right.set_ylabel('ADC counts', fontsize=11)
    ax_right.set_xlabel('Drift time [µs]', fontsize=11)
    ax_right.legend(fontsize=11, loc='upper right')
    ax_right.set_title('Drift window', fontsize=13, pad=6)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    return fig, (ax_left, ax_right)


def plot_events(event_indices, wf, muon1, muon2, date_str, offset=0,
                 t0=900, t1=1100, s0=140_000, s1=190_000,
                 dt_ns=10, save_dir=None):
    """
    Plot trigger region + drift window for multiple events.
    """
    for idx in event_indices:
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f'event_{idx}.png'
        plot_event(idx, wf, muon1, muon2, date_str, offset=offset,
                   t0=t0, t1=t1, s0=s0, s1=s1, dt_ns=dt_ns, save_path=save_path)