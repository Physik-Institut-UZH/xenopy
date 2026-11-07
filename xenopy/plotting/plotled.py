import matplotlib.pyplot as plt

__all__ = ["plot_gain_evolution", "TILES", "TILE_COLORS", "EXPOSURE_COLOR", "FULL_WIDTH"]

TILES = ['tile_A', 'tile_B', 'tile_C', 'tile_D', 'tile_E', 'tile_F',
         'tile_G', 'tile_H', 'tile_J', 'tile_K', 'tile_L', 'tile_M']

TILE_COLORS = ['#882255', '#BBCC33', '#4477AA', '#DD8844', '#AA4499', '#88CCEE',
               '#117733', '#DDCC77', '#332288', '#CC6677', '#999933', '#44AA99']
EXPOSURE_COLOR = '#9F3030'

FULL_WIDTH = 6.8   


def plot_gain_evolution(dates, gains, gain_errs, exposure=None, tiles=TILES,
                        tile_colors=TILE_COLORS, figsize=(FULL_WIDTH, 2.8),
                        ylim=(1e6, 1.8e6), legend_ncol=7):
    """Gain of each tile vs date, with the cumulative muon-coincidence exposure on a twin axis.

    Parameters
    ----------
    dates : array-like of datetime
    gains, gain_errs : dict
        Per-tile arrays, keyed by tile name (e.g. from `load_led_gain_files`).
    exposure : (times, cumulative_days) or None
        Step curve from `load_muon_exposure`; None to omit it.
    figsize : (width, height) in inches
        Default is the full text width. Font sizes come from the style; nothing is rescaled.
    """
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for i, tile in enumerate(tiles):
        ax.errorbar(dates, gains[tile], yerr=gain_errs[tile], fmt='o',
                    color=tile_colors[i % len(tile_colors)], alpha=0.7,
                    label=tile.replace('tile_', 'Tile '))
    ax.set_xlabel("Date")
    ax.set_ylabel("SiPM gain")
    ax.set_ylim(*ylim)

    handles, labels = ax.get_legend_handles_labels()
    if exposure is not None:
        t_exp, c_exp = exposure
        ax2 = ax.twinx()
        ax2.plot(t_exp, c_exp, color=EXPOSURE_COLOR, alpha=0.8, label="Muon-coincidence mode")
        ax2.set_ylabel("Cumulative exposure [days]", color=EXPOSURE_COLOR)
        ax2.tick_params(axis='y', colors=EXPOSURE_COLOR)
        ax2.set_ylim(bottom=0)
        ax.set_zorder(ax2.get_zorder() + 1)   # keep the gain points above the exposure line
        ax.patch.set_visible(False)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2

    ax.legend(handles, labels, ncol=legend_ncol, loc='lower center',
              bbox_to_anchor=(0.5, 1.02), columnspacing=0.8, handletextpad=0.3)
    fig.autofmt_xdate()
    return fig, ax
