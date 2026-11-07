import numpy as np
import matplotlib.pyplot as plt
from xenopy.processing.driftfit import gauss_plus_bkg, gauss_plus_bkg_binned, poisson_errors

def plot_fit(dt, r, fit_range, bin_width_us, range_=(0, 2500)):
    """Data with Poisson errors, fit curve, and standardised residuals in the fit window."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c_fit, c_band1, c_band2 = cycle[6], cycle[2], cycle[4]

    mu, sigma = r["popt"][1], abs(r["popt"][2])
    mu_err, sigma_err = r["perr"][1], r["perr"][2]

    bins = int((range_[1] - range_[0]) / bin_width_us)
    counts, edges = np.histogram(dt, bins=bins, range=range_)
    centres = 0.5 * (edges[:-1] + edges[1:])
    m = (centres >= fit_range[0]) & (centres <= fit_range[1])
    mod = gauss_plus_bkg_binned(centres, *r["popt"], bin_width_us)
    with np.errstate(invalid="ignore", divide="ignore"):
        std_res = (counts - mod) / np.sqrt(mod)
    t_fine = np.linspace(*fit_range, 1000)

    w, h = plt.rcParams["figure.figsize"]
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(w, 1.1 * h), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    elo, ehi = poisson_errors(counts)
    ax.errorbar(centres, counts, yerr=[elo, ehi], fmt="o", ms=1.5, color="black",
                alpha=0.6, elinewidth=0.5, capsize=0, label="Selected events", zorder=2)
    ax.plot(t_fine, gauss_plus_bkg(t_fine, *r["popt"]), color=c_fit, zorder=3,
            label="Gaussian + bkg fit")
    ax.text(0.03, 0.97, rf"μ = {mu:.1f} ± {mu_err:.1f} µs" "\n"
                        rf"σ = {sigma:.1f} ± {sigma_err:.1f} µs",
            transform=ax.transAxes, ha="left", va="top")
    ax.set_ylabel(f"Events / {bin_width_us:g} µs")
    ax.set_xlim(*range_)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))

    for x in fit_range:
        for a in (ax, ax_res):
            a.axvline(x, color="gray", lw=1, ls=":", zorder=0)

    ax_res.axhline(0, color="black", lw=0.8)
    ax_res.fill_between(fit_range, -2, 2, color=c_band2, alpha=0.40, linewidth=0)
    ax_res.fill_between(fit_range, -1, 1, color=c_band1, alpha=0.45, linewidth=0)
    ax_res.scatter(centres[m], std_res[m], s=8, color="black", alpha=0.8, linewidths=0, zorder=3)
    ax_res.set_ylim(-3.5, 3.5)
    ax_res.set_xlabel("Drift time [µs]")
    ax_res.set_ylabel("Std. residuals")
    fig.tight_layout()
    return fig, (ax, ax_res)
