import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.special import erf
from scipy.stats import chi2 as chi2_dist


def gauss_plus_bkg(t, A, mu, sigma, C):
    """Point model (used for the smooth curve in the plot)."""
    return A * np.exp(-0.5 * ((t - mu) / sigma) ** 2) + C


def gauss_plus_bkg_binned(t, A, mu, sigma, C, bw):
    """Model averaged over a bin of width bw centred at t (counts per bin)."""
    s = np.sqrt(2) * sigma
    gauss = A * sigma * np.sqrt(np.pi / 2) / bw * (erf((t + bw / 2 - mu) / s) - erf((t - bw / 2 - mu) / s))
    return gauss + C


def fit_drift_peak(dt, fit_range, bin_width_us, weighted=True):
    """Least-squares fit in `fit_range`; weights are sqrt(model) (Poisson) if `weighted`.

    Returns popt/perr for (A, mu, sigma, C), the counts, chi2 (Pearson) and ndf.
    """
    bins = int((fit_range[1] - fit_range[0]) / bin_width_us)
    counts, edges = np.histogram(dt, bins=bins, range=fit_range)
    centres = 0.5 * (edges[:-1] + edges[1:])
    model = lambda t, A, mu, sigma, C: gauss_plus_bkg_binned(t, A, mu, sigma, C, bin_width_us)

    p0 = [counts.max() - counts.min(), centres[counts.argmax()], 20, counts.min()]
    popt, pcov = curve_fit(model, centres, counts, p0=p0)
    if weighted:
        sig = np.sqrt(np.maximum(model(centres, *popt), 1))
        popt, pcov = curve_fit(model, centres, counts, p0=popt, sigma=sig, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    mod = model(centres, *popt)
    chi2 = np.sum((counts - mod) ** 2 / mod)
    return dict(popt=popt, perr=perr, counts=counts, centres=centres,
                chi2=chi2, ndf=len(counts) - len(popt))


def fit_drift_peak_poisson(dt, fit_range, bin_width_us):
    """Binned Poisson-likelihood fit (cross-check). chi2 is the Poisson deviance."""
    bins = int((fit_range[1] - fit_range[0]) / bin_width_us)
    counts, edges = np.histogram(dt, bins=bins, range=fit_range)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mean = lambda p: np.maximum(gauss_plus_bkg_binned(centres, *p, bin_width_us), 1e-9)
    nll = lambda p: np.sum(mean(p) - counts * np.log(mean(p)))

    p0 = [counts.max(), centres[counts.argmax()], 15, max(np.median(counts), 0.1)]
    res = minimize(nll, p0, method="L-BFGS-B",
                   bounds=[(0, None), (None, None), (1, 100), (1e-6, None)])
    perr = np.sqrt(np.diag(res.hess_inv.todense()))

    mu_i = mean(res.x)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.sum(mu_i - counts + np.where(counts > 0, counts * np.log(counts / mu_i), 0))
    return dict(popt=res.x, perr=perr, counts=counts, centres=centres,
                chi2=dev, ndf=len(counts) - len(p0))


def poisson_errors(n, cl=0.6827):
    """Asymmetric Poisson (Garwood) error bars; returns (lower, upper) distances."""
    a = 1 - cl
    lo = np.where(n > 0, chi2_dist.ppf(a / 2, 2 * np.maximum(n, 1)) / 2, 0.0)
    hi = chi2_dist.ppf(1 - a / 2, 2 * (n + 1)) / 2
    return n - lo, hi - n


def report(name, r):
    """One-line summary for the paper text."""
    A, mu, sigma, C = r["popt"]
    _, mu_e, sigma_e, C_e = r["perr"]
    print(f"{name}: mu = {mu:.1f} ± {mu_e:.1f} µs, sigma = {abs(sigma):.1f} ± {sigma_e:.1f} µs, "
          f"C = {C:.2f} ± {C_e:.2f}/bin, chi2/ndf = {r['chi2']:.1f}/{r['ndf']}, "
          f"{int(r['counts'].sum())} events in {len(r['counts'])} bins")
