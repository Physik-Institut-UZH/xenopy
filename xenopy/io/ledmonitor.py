import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_led_gain_files", "load_muon_exposure"]

_TILES = ['tile_A', 'tile_B', 'tile_C', 'tile_D', 'tile_E', 'tile_F',
          'tile_G', 'tile_H', 'tile_J', 'tile_K', 'tile_L', 'tile_M']


def load_led_gain_files(gain_dir, tiles=_TILES, pattern="gain_*.json"):
    """Read all gain json files; returns dates and per-tile arrays (NaN if a tile is missing)."""
    files = sorted(Path(gain_dir).glob(pattern))
    keys = dict(gain='gain', gain_err='gain_std', spe='SPE', spe_err='SPE_sigma',
                occupancy='occupancy', occupancy_err='occupancy_std',
                pedestal='Pedestal', pedestal_err='Pedestal_sigma')
    out = {k: {t: [] for t in tiles} for k in keys}
    dates = []
    for f in files:
        dates.append(datetime.strptime(f.stem.replace('gain_', ''), '%Y%m%d'))
        with open(f) as fh:
            data = json.load(fh)
        for t in tiles:
            for k, jk in keys.items():
                out[k][t].append(data[t][jk] if t in data else np.nan)
    out = {k: {t: np.array(v) for t, v in d.items()} for k, d in out.items()}
    return pd.to_datetime(dates), out


def load_muon_exposure(json_path):
    """Cumulative exposure [days] as a step curve: flat between datasets, rising during each.

    Returns (times, cumulative_days, total_days, total_events).
    """
    with open(json_path) as f:
        exp = pd.DataFrame(json.load(f))
    exp["start"] = pd.to_datetime(exp["run_start_utc"])
    exp["end"] = exp["start"] + pd.to_timedelta(exp["total_exposure_seconds"], unit="s")
    exp = exp.sort_values("start").reset_index(drop=True)

    days = exp["total_exposure_seconds"] / 86400
    cum_after = days.cumsum()
    cum_before = cum_after - days
    t, c = [], []
    for s, e, cb, ca in zip(exp["start"], exp["end"], cum_before, cum_after):
        t += [s, e]
        c += [cb, ca]
    return t, c, cum_after.iloc[-1], int(exp["total_events"].sum())
