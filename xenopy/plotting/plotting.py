import numpy as np
from typing import List
from typing import Optional

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Circle


def load_layout():
    array_layout = np.loadtxt('/disk/gfs_atp/xenoscope/tpc/tiles_layout.txt')
    array_labels = ['A', 'B', 'C', 'D', 'E', 'F',
                    'G', 'H', 'J', 'K', 'L', 'M']
        
    return array_layout, array_labels

def plot_hitpattern(hitpattern: [np.ndarray, List[float]],
                    layout: np.ndarray,
                    labels: List[str] = None,
                    r_tpc: float = None,
                    cmap: str = 'gnuplot',
                    log: bool = False,
                    ax=None):
    """Plot a beautiful hitpattern.

    Args:
        hitpattern (Union[np.ndarray, List[float]]): array with the area per sensor.
        layout (np.ndarray): layout of the sensor array (x1,x2,y1,y2) corners.
        labels (Optional[List[str]], optional): ordered labels to put in the
            center of each sensor. Defaults to None.
        r_tpc (Optional[float], optional): plot a line at the tpc edge.
            Defaults to None.
        cmap (str, optional): name of colormap to use. Defaults to 'gnuplot'.
        log (bool, optional): plot the log10 of pe instead of pe. Defaults to False.
        ax (_type_, optional): axis where to draw the hitpattern. Defaults to None.

    Returns:
        (fig, axis, mappable): figure (or None if ax was provided), axis with
            the hitpattern drawn, and the mappable for a colorbar.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    cm = plt.get_cmap(cmap)

    hitpattern = np.array(hitpattern, dtype=float)
    if log:
        hitpattern = np.log10(np.clip(hitpattern, a_min=1e-10, a_max=None))

    color_max = np.max(hitpattern)
    color_min = np.min(hitpattern)

    color_range = color_max - color_min
    if color_range == 0:
        color_range = 1.0  # all patches will map to the midpoint colour

    for i, _sensor in enumerate(layout):
        pe = hitpattern[i]

        xy = (_sensor[0], _sensor[2])
        width = _sensor[1] - _sensor[0]
        height = _sensor[3] - _sensor[2]
        ax.add_patch(Rectangle(xy, width, height, fill=True,
                               edgecolor='k',
                               facecolor=cm((pe - color_min) / color_range)))
        if labels is not None:
            ax.text(xy[0] + width / 2, xy[1] + height / 2, labels[i],
                    ha='center', va='center', zorder=10)

    if r_tpc is not None:
        ax.add_patch(Circle((0, 0), r_tpc, color='r', fill=False,
                            label='TPC edge'))

    norm = matplotlib.colors.Normalize(vmin=color_min, vmax=color_max)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    return (fig, ax, mappable)


def plot_waveform_withpattern(plottitle, data, n_wf):
    labels = {
        'tile_H': 'Tile H', 'tile_J': 'Tile J',
        'tile_K': 'Tile K', 'tile_L': 'Tile L',
        'tile_M': 'Tile M',
        'tile_A': 'Tile A', 'tile_B': 'Tile B',
        'tile_C': 'Tile C', 'tile_D': 'Tile D',
        'tile_E': 'Tile E', 'tile_F': 'Tile F',
        'tile_G': 'Tile G'
    }

    array_layout, array_labels = load_layout()
    fig = plt.figure(figsize=(14, 7), dpi=200)
    gs = GridSpec(3, 2, width_ratios=[1, 0.8], height_ratios=[1, 1, 1])

    ax_trigger = fig.add_subplot(gs[0, 1])
    ax_wf = fig.add_subplot(gs[0, 0])
    ax_mod1 = fig.add_subplot(gs[1, 0], sharex=ax_wf)
    ax_mod0 = fig.add_subplot(gs[2, 0], sharex=ax_wf)
    ax_hitp = fig.add_subplot(gs[1:3, 1])

    n_samples = len(data["singleChannels"]["tile_A"][n_wf])
    _x = np.linspace(0, 2500, n_samples)


    plt.subplots_adjust(hspace=0.2)
    trigger_label = ["Trigger 1", "Trigger 2"]
    linewidth = 1

    for i, tile in enumerate(['muon1', 'muon2']):
        ax_trigger.plot(_x, data["singleChannels"][tile][n_wf],
                        label=trigger_label[i], linewidth=linewidth)

    for tile in ['tile_A', 'tile_B', 'tile_C', 'tile_D', 'tile_E', 'tile_F']:
        ax_mod0.plot(_x, data["singleChannels"][tile][n_wf],
                     label=labels[tile], linewidth=linewidth)
    
    for tile in ['tile_G','tile_H', 'tile_J', 'tile_K', 'tile_L', 'tile_M']:
        ax_mod1.plot(_x, data["singleChannels"][tile][n_wf],
                     label=labels[tile], linewidth=linewidth)
        
    n_samples = len(data["rawWf"][n_wf])
    x_large = np.arange(0, n_samples / 100, 0.01)
    ax_wf.plot(x_large, data["rawWf"][n_wf], label="Total Waveform", linewidth=linewidth)

    if data["nPulses"][n_wf] > 0:
        cmap_binary = plt.get_cmap('binary_r')
        n_colors = data["nPulses"][n_wf] - 1

        # FIX 4: data[n_wf]["pulseStart_us"] → data["pulseStart_us"][n_wf]
        ax_wf.axvspan(data["pulseStart_us"][n_wf][0], data["pulseEnd_us"][n_wf][0],
                      alpha=0.3, color="Red", label="Prominent Pulse")

        if data["nPulses"][n_wf] > 1:
            colors = [cmap_binary(i) for i in np.linspace(0.3, 0.7, n_colors)]
            for pulseStart, pulseEnd, color in zip(
                data["pulseStart_us"][n_wf][1:],
                data["pulseEnd_us"][n_wf][1:],
                colors
            ):
                ax_wf.axvspan(pulseStart, pulseEnd, alpha=0.3, color=color)

        width = data["fwhm_us"][n_wf][0]   # FIX 4 (same): corrected indexing
        area = data["area"][n_wf][0]
        ax_wf.plot(0, 0, ",", color="white", label=f"FWHM = {width:.0f} us")
        ax_wf.plot(0, 0, ",", color="white", label=f"Area = {area:.2e} PE")

    ax_wf.legend()
    ax_mod0.set_xlabel('Time [us]')
    ax_mod0.set_ylabel('Amplitude [PE/40ns]')
    ax_mod1.set_ylabel('Amplitude [PE/40ns]')
    ax_wf.set_ylabel('Amplitude [PE/10ns]')

    ax_mod0.legend()
    ax_mod1.legend()
    ax_trigger.legend()
    ax_trigger.set_ylabel('Amplitude [ADC value]')

    # Build hitpattern 
    hitpattern_area = []

    for tile in ['tile_A', 'tile_B', 'tile_C', 'tile_D', 'tile_E', 'tile_F']:
        _wf = data["singleChannels"][tile][n_wf]
        hitpattern_area.append(np.sum(_wf))

    for tile in ['tile_G','tile_H', 'tile_J', 'tile_K', 'tile_L', 'tile_M']:
        _wf = data["singleChannels"][tile][n_wf]
        hitpattern_area.append(np.sum(_wf))

    _, ax, mappable = plot_hitpattern(
        hitpattern=hitpattern_area,
        layout=array_layout,
        labels=array_labels,
        r_tpc=160 / 2,
        cmap='cividis',
        log=False,
        ax=ax_hitp)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_aspect('equal')
    fig.colorbar(mappable, label='Total waveform area [PE]')

    ax_hitp.set_xlabel('x [mm]')
    ax_hitp.set_ylabel('y [mm]')

    fig.suptitle(f'{plottitle}, Event {n_wf}', y=0.92, x=0.5,
                 horizontalalignment='center', verticalalignment='center',
                 transform=fig.transFigure)
    fig.show()



def plot_waveform(waveform_array: np.ndarray, full_y: bool | tuple[float, float] = False,
                  full_x: bool = True, pe: bool = False,
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot a waveform from its array.

    Args:
        waveform_array (np.ndarray): waveform array. Elements of the array
            correspond to the sample of the waveform and its value the recorded ADC
            counts.
        full_y (bool, optional): plot full range of ADC amplitude. Defaults
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
        fig, ax = plt.subplots(1, 1)
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

    if pe == True:
        ax.set_ylabel('PE/s')
    return ax


def plot_average_waveform(waveform_array: np.ndarray,
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot an average waveform with baseline reference lines.

    Args:
        waveform_array (np.ndarray): average waveform array (1-D).
        ax (plt.Axes, optional): axes to plot into. Defaults to None.

    Returns:
        plt.Axes: axes with plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    n_samples = len(waveform_array)
    x = np.arange(n_samples)
    ax.plot(x, waveform_array, color='royalblue')
    baseline_rough = np.median(waveform_array[:50])
    std_rough = np.std(waveform_array[:50])
    ax.axhline(baseline_rough, ls='--', c='gray', lw=1, alpha=0.8, label=r"Baseline")
    ax.axhline(baseline_rough - std_rough * 5, ls='--', c='firebrick', lw=1, alpha=0.8, label=r"Baseline ± 5$\sigma$")
    ax.axhline(baseline_rough + std_rough * 5, ls='--', c='firebrick', lw=1, alpha=0.8)
    ax.set_xlabel(r"$\mathrm{Sample~number}$")
    ax.set_ylabel(r"$\mathrm{ADC~counts}$")
    ax.legend(loc='best')
    return ax


def plot_all_tiles_average(avg_waveforms: dict, title: str = '') -> plt.Figure:
    """Plot average waveforms for all tiles overlaid in a single axes.

    Args:
        avg_waveforms: dict of ``{tile_name: np.ndarray}`` as returned by
            ``get_all_average_waveforms``.
        title: optional figure title.

    Returns:
        plt.Figure: figure with all tiles on one plot.
    """
    tiles = sorted(avg_waveforms.keys())
    fig, ax = plt.subplots(figsize=(7, 4))

    for tile in tiles:
        wf = avg_waveforms[tile]
        x = np.arange(len(wf))
        ax.plot(x, wf, label=f'Tile {tile[-1]}', alpha=0.8)

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