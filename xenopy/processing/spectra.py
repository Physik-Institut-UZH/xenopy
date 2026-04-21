import numpy as np
from typing import Optional
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

def compute_charge(waveforms: np.ndarray,
                   baseline_range: tuple[int, int] = (0, 50),
                   signal_range: tuple[int, int] = (110, 140)) -> np.ndarray:
    """Compute integrated charge for each waveform
    by baseline-subtracting and summing over the signal region (can be set)

    """
    b0, b1 = baseline_range
    s0, s1 = signal_range
    baseline = waveforms[:, b0:b1].mean(axis=1, keepdims=True)
    charge = -(waveforms[:, s0:s1] - baseline).sum(axis=1)
    return charge


def compute_charge_all_tiles(tiles: dict,
                             baseline_range: tuple[int, int] = (0, 50),
                             signal_range: tuple[int, int] = (110, 140)) -> dict:
    """Compute charge for all tiles
    """
    return {
        tile: compute_charge(tiles[tile]["waveforms"], baseline_range, signal_range)
        for tile in tiles
    }


def _gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _fit_gaussian_in_window(hist: np.ndarray,
                            bin_centers: np.ndarray,
                            window: tuple[float, float],
                            mu_guess: float) -> Optional[dict]:
    mask = (bin_centers >= window[0]) & (bin_centers <= window[1])
    x = bin_centers[mask]
    y = hist[mask]
    if x.size < 5 or np.sum(y) <= 0:
        return None

    p0 = [float(np.max(y)), mu_guess, max((window[1] - window[0]) / 6.0, 1.0)]
    bounds = ([0.0, window[0], 0.1], [np.inf, window[1], np.inf])
    try:
        popt, pcov = curve_fit(_gaussian, x, y, p0=p0, bounds=bounds, maxfev=20000)
        return {"params": popt, "cov": pcov, "x": x, "y": y}
    except Exception:
        return None

def fit_spectrum(charge: np.ndarray,
                 bins: int = 500,
                 range: tuple[float, float] = (-100, 5000),
                 window_0pe: Optional[tuple[float, float]] = None,
                 window_1pe: Optional[tuple[float, float]] = None,
                 min_peak_separation: float = 80.0) -> dict:
    """SiPM peak fit: find 0-PE near zero, then 1-PE to the right of the pedestal"""
    hist, bin_edges = np.histogram(charge, bins=bins, range=range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if hist.sum() == 0:
        raise ValueError("Empty histogram")

    span = range[1] - range[0]
    bin_width = span / bins

    # search near zero by default 
    if window_0pe is None:
        z0, z1 = max(-150.0, range[0]), min(150.0, range[1])
        m = (bin_centers >= z0) & (bin_centers <= z1)
        peak0_guess = float(bin_centers[np.where(m)[0][int(np.argmax(hist[m]))]]) if np.any(m) and np.max(hist[m]) > 0 else 0.0
        halfw0 = max(40.0, 0.05 * span)
        window_0pe = (peak0_guess - halfw0, peak0_guess + halfw0)
    else:
        m = (bin_centers >= window_0pe[0]) & (bin_centers <= window_0pe[1])
        peak0_guess = float(bin_centers[np.where(m)[0][int(np.argmax(hist[m]))]]) if np.any(m) and np.max(hist[m]) > 0 else 0.0

    fit0 = _fit_gaussian_in_window(hist, bin_centers, window_0pe, peak0_guess)

    if fit0 is None:
        raise RuntimeError(
            "Pedestal fit failed"
        )

    center_for_1pe = float(fit0["params"][1])
    if window_1pe is None:
        search_start = center_for_1pe + max(min_peak_separation, 3.0 * bin_width)
        mask1search = bin_centers >= search_start
        idxs = np.where(mask1search)[0]
        peak1_guess = None
        if idxs.size >= 5:
            h_smooth = uniform_filter1d(hist[idxs].astype(float), size=5)
            peaks, _ = find_peaks(h_smooth, height=np.max(h_smooth) * 0.05)
            if peaks.size > 0:
                # take the first peak found (lowest charge = 1-PE)
                peak1_guess = float(bin_centers[idxs[peaks[0]]])
        if peak1_guess is None:
            if np.any(mask1search) and np.max(hist[mask1search]) > 0:
                idx1 = np.where(mask1search)[0][int(np.argmax(hist[mask1search]))]
                peak1_guess = float(bin_centers[idx1])
            else:
                peak1_guess = center_for_1pe + max(0.15 * span, min_peak_separation)
        halfw1 = max(60.0, 0.06 * span)
        window_1pe = (peak1_guess - halfw1, peak1_guess + halfw1)
    else:
        mask1 = (bin_centers >= window_1pe[0]) & (bin_centers <= window_1pe[1])
        if np.any(mask1) and np.max(hist[mask1]) > 0:
            idx1 = np.where(mask1)[0][int(np.argmax(hist[mask1]))]
            peak1_guess = float(bin_centers[idx1])
        else:
            peak1_guess = center_for_1pe + max(0.15 * span, min_peak_separation)

    fit1 = _fit_gaussian_in_window(hist, bin_centers, window_1pe, peak1_guess)

    if fit1 is None:
        raise RuntimeError(
            "1-PE fit failed"
        )

    mean_1pe = float(fit1["params"][1])
    sigma_1pe = float(abs(fit1["params"][2]))
    mean_0pe = float(fit0["params"][1])
    sigma_0pe = float(abs(fit0["params"][2]))

    if mean_1pe <= mean_0pe:
        raise RuntimeError(
            "Auto peak assignment gave mean_1pe <= mean_0pe. ")


    return {
        "hist": hist,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "window_0pe": window_0pe,
        "window_1pe": window_1pe,
        "fit_0pe": fit0,
        "fit_1pe": fit1,
        "mean_0pe": mean_0pe,
        "sigma_0pe": sigma_0pe,
        "mean_1pe": mean_1pe,
        "sigma_1pe": sigma_1pe
    }

