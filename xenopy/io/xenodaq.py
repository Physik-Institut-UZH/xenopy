"""
XenoDAQ ROOT file loader for Xenoscope analysis.

This module provides functions to load waveform data from XenoDAQ ROOT files
produced by the FMCDAQ system with multiple digitizers.
"""
import re
import numpy as np
import uproot
import gc
import pandas as pd
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional, Tuple
import json


def load_description(dataset: str, datadir: str = 'datasets') -> Optional[Dict[str, str]]:
    """
    Load and parse the Description field from a dataset's JSON config.

    The description string is expected to have the form:
        ``"LED calibration -  BV: 50V, LED: 5V, F: 1.5 kHz , Width: 100ns"``

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. '20260414_180123')
    datadir : str, default 'datasets'
        Base directory containing datasets

    Returns
    -------
    dict or None
        Parsed fields, e.g.
        ``{'BV': '50V', 'LED': '5V', 'F': '1.5 kHz', 'Width': '100ns', 'raw': '...'}``
        Returns None if no JSON is found or no Description key exists.
    """
    dirpath = Path(datadir) / dataset
    json_files = glob(str(dirpath / f"*{dataset}.json"))
    if not json_files:
        return None
    with open(json_files[0]) as f:
        config = json.load(f)
    desc = config.get("Description")
    if desc is None:
        return None
    parsed = {"raw": desc}
    # extract key: value pairs like "BV: 50V"
    for match in re.finditer(r'(\w+)\s*:\s*([^,}]+)', desc):
        key = match.group(1).strip()
        val = match.group(2).strip()
        parsed[key] = val
    return parsed


def find_datasets(
    datadir: str,
    date: Optional[str] = None,
    bv: Optional[str] = None,
    led: Optional[str] = None,
) -> List[Dict]:
    """
    Scan a data directory and return datasets matching the given filters.
    Filters are applied to the Description field inside each dataset's JSON.

    Returns
    -------
    list of dict
        Each entry contains:
        ``{'dataset': str, 'description': dict, 'path': str}``
        Sorted by dataset name.

    """
    import os
    base = Path(datadir)
    results = []
    for name in sorted(os.listdir(base)):
        subdir = base / name
        if not subdir.is_dir():
            continue
        if date and not name.startswith(date):
            continue
        desc = load_description(name, datadir)
        if desc is None:
            continue
        if bv and not _match_value(desc.get("BV", ""), bv):
            continue
        if led and not _match_value(desc.get("LED", ""), led):
            continue
        results.append({
            "dataset": name,
            "description": desc,
            "path": str(subdir),
        })
    return results


def _match_value(field: str, target: str) -> bool:
    """Check if *target* matches *field*, ignoring a trailing 'V'."""
    field_clean = field.replace(" ", "").rstrip("V").lower()
    target_clean = target.replace(" ", "").rstrip("V").lower()
    return field_clean == target_clean


def load_xenodaq_run(
    dataset: str,
    datadir: str = 'datasets',
    filenumbers: List[int] = [0],
    wfs_to_load: Optional[Dict[str, List[str]]] = None,
    channel_map: Optional[Dict] = None
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Optional[Dict]]:
    """
    Load xenodaq root file, extract waveforms, metadata, and tile mapping.

    When called with no optional arguments, it automatically:
    1. Loads the channel map from the dataset JSON
    2. Loads all waveforms from the ROOT file
    3. Maps waveforms to physical tile names

    datadir : str, default 'datasets'
        Base directory containing datasets
    filenumbers : list, default 0
    wfs_to_load : dict, optional
        Dictionary mapping digitizer number to list of waveform names.
        Example: {"0": ["wf0", "wf1"], "1": ["wf9", "wf10"]}
        If None, loads all available waveforms from all digitizers.
        When used, tiles will be None (no mapping performed).
    channel_map : dict, optional
        Custom ChsMap dict mapping physical names to DAQ locations.
        Entries with null tree/channel are ignored.
        Cannot be used together with wfs_to_load.

    Returns
    -------
    wfs : dict
        Dictionary of waveform arrays {wf_name: ndarray of shape (n_events, n_samples)}
    wfs_df : pd.DataFrame
        DataFrame with metadata columns:
        - {wf}_evid: Event counter for each waveform
        - {wf}_ttt: Time trigger tag for each waveform
        - {wf}_rt: Run time for each waveform (may be None for some digitizers)
    tiles : dict or None
        Tile-mapped data ``{tile_name: {'waveforms': ..., 'evid': ..., ...}}``.
        None when ``wfs_to_load`` is given explicitly (raw mode).
    """
    dirpath = Path(datadir) / dataset
    ds_flist = glob(str(dirpath / "*.root"))

    if not ds_flist:
        raise FileNotFoundError(f"No ROOT files found in {dirpath}")

    rootfile_paths = []
    for num in filenumbers:
        suffix = f"_{num:04d}.root"
        filepath = [f for f in ds_flist if f.endswith(suffix)]
        rootfile_paths.extend(filepath)

    if channel_map is not None and wfs_to_load is not None:
        raise ValueError("Provide either 'channel_map' or 'wfs_to_load', not both.")

    # Determine the tile map to use for mapping at the end
    tile_map = None  # will stay None when wfs_to_load is given explicitly
    raw_mode = wfs_to_load is not None  # user asked for specific branches

    if channel_map is not None:
        tile_map = channel_map
        wfs_to_load = _channel_map_to_wfs_to_load(channel_map)
        if not wfs_to_load:
            raise ValueError("channel_map has no valid (non-null) entries.")

    if wfs_to_load is None:
        # Try JSON config first, fall back to auto-detect
        json_map = load_channel_map(dataset, datadir)
        if json_map:
            tile_map = json_map
            wfs_to_load = _channel_map_to_wfs_to_load(json_map)
            print(f"Loaded waveforms from JSON ChsMap: {wfs_to_load}")
        else:
            wfs_to_load = detect_waveforms(rootfile_paths[0])
            print(f"No JSON ChsMap found — auto-detected waveforms: {wfs_to_load}")

    wfs = {}
    wfs_df = {}

    for rootfile_path in rootfile_paths:
        with uproot.open(rootfile_path) as rootfile:
            print(f"Loading File {rootfile_path}")
            for dig_n in wfs_to_load.keys():
                tree = rootfile[f'dig_{dig_n}']
                branches = wfs_to_load[dig_n]
                meta_branches = [f"EvCounter_{dig_n}", f"TimeTrigTag_{dig_n}"]
                if 'RunTime' in tree.keys():
                    meta_branches.append("RunTime")

                for batch in tree.iterate(branches + meta_branches, step_size=50, library="np"):
                    evcounters = batch[f"EvCounter_{dig_n}"].astype(np.uint32)
                    ttts       = batch[f"TimeTrigTag_{dig_n}"].astype(np.uint32)
                    runtimes = batch["RunTime"].astype(np.float32) if "RunTime" in batch else np.full(len(evcounters), np.nan, dtype=np.float32)

                    for wf_name in branches:
                        arr = batch[wf_name].astype(np.uint32)
                        wfs.setdefault(wf_name, []).append(arr)
                        wfs_df.setdefault(f'{wf_name}_evid', []).append(evcounters)
                        wfs_df.setdefault(f'{wf_name}_ttt',  []).append(ttts)
                        wfs_df.setdefault(f'{wf_name}_rt',   []).append(runtimes)

    wfs = {k: np.concatenate(v, axis=0) for k, v in wfs.items()}
    wfs_df = {k: np.concatenate(v, axis=0) for k, v in wfs_df.items()}

    wfs_df = pd.DataFrame(wfs_df)

    # Map to tiles automatically (unless raw mode)
    tiles = None
    if tile_map is not None and not raw_mode:
        tiles = map_channels_to_tiles(wfs, wfs_df, tile_map)
        print(f"Mapped {len(tiles)} tiles: {sorted(tiles.keys())}")

    return wfs, wfs_df, tiles


def average_xenodaq_run(
    dataset: str,
    datadir: str = 'datasets',
    filenumbers: Optional[List[int]] = None,
    batch_size: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Memory-efficient averaged waveform loader across multiple ROOT files.

    Returns
    -------
    dict
        {wf_name: avg_waveform (n_samples,)}
    """
    filenums = filenumbers if filenumbers is not None else get_all_filenumbers(dataset, datadir)
    dirpath  = Path(datadir) / dataset
    ds_flist = glob(str(dirpath / "*.root"))

    wfs_to_load = detect_waveforms(ds_flist[0])

    running_sum   = {}
    running_count = {}

    for filenum in filenums:
        suffix = f"_{filenum:04d}.root"
        paths  = [f for f in ds_flist if f.endswith(suffix)]
        if not paths:
            print(f"  file {filenum} not found, skipping")
            continue

        print(f"  Reading file {filenum}...")
        with uproot.open(paths[0]) as rootfile:
            for dig_n, branches in wfs_to_load.items():
                tree = rootfile[f'dig_{dig_n}']
                for batch in tree.iterate(branches, step_size=batch_size, library="np"):
                    for wf_name in branches:
                        arr = batch[wf_name].astype(np.float64)
                        if wf_name not in running_sum:
                            running_sum[wf_name]   = np.zeros(arr.shape[1], dtype=np.float64)
                            running_count[wf_name] = 0
                        running_sum[wf_name]   += arr.sum(axis=0)
                        running_count[wf_name] += arr.shape[0]
        gc.collect()

    averaged = {wf: running_sum[wf] / running_count[wf] for wf in running_sum}
    total_events = list(running_count.values())[0]
    print(f"Done — averaged {total_events} events across {len(filenums)} file(s)")
    return averaged

def get_all_filenumbers(dataset: str, datadir: str = 'datasets') -> List[int]:
    """
    Scan a dataset directory and return all available ROOT file numbers.
    -------
    list of int
        Sorted list of file numbers found (e.g. [0, 1, 2])
    """
    dirpath = Path(datadir) / dataset
    root_files = glob(str(dirpath / "*.root"))
    numbers = []
    for f in root_files:
        match = re.search(r'_(\d{4})\.root$', f)
        if match:
            numbers.append(int(match.group(1)))
    return sorted(numbers)

def detect_waveforms(rootfile_path: str) -> Dict[str, List[str]]:
    """
    Auto-detect available waveforms in a xenodaq root file.
    
    Parameters
    ----------
    rootfile_path : str
        Path to the ROOT file
    
    Returns
    -------
    dict
        Dictionary mapping digitizer number to list of waveform branch names
        Example: {"0": ["wf0", "wf1", ...], "1": ["wf9", "wf10", ...]}
    """
    wfs_map = {}
    
    with uproot.open(rootfile_path) as rootfile:
        # Find all digitizer trees (get latest cycle only)
        dig_trees = [key for key in rootfile.keys() if key.startswith('dig_')]
        
        # Extract unique digitizer numbers
        dig_nums = set(key.split(';')[0].split('_')[1] for key in dig_trees)
        
        for dig_n in sorted(dig_nums):
            # uproot automatically uses latest cycle when no cycle specified
            tree = rootfile[f'dig_{dig_n}']
            wf_branches = [b for b in tree.keys() if b.startswith('wf')]
            wfs_map[dig_n] = sorted(wf_branches)
    
    return wfs_map


def _channel_map_to_wfs_to_load(channel_map: Dict) -> Dict[str, List[str]]:
    """
    Convert a ChsMap dict to the wfs_to_load format for load_xenodaq_run.

    Entries with null tree or channel are silently skipped.
    """
    wfs_to_load: Dict[str, List[str]] = {}
    for loc in channel_map.values():
        tree = loc.get('tree')
        channel = loc.get('channel')
        if tree is None or channel is None:
            continue
        dig_n = tree.split('_')[1]
        if dig_n not in wfs_to_load:
            wfs_to_load[dig_n] = []
        if channel not in wfs_to_load[dig_n]:
            wfs_to_load[dig_n].append(channel)
    return wfs_to_load


def get_file_info(dataset: str, datadir: str = 'datasets') -> Dict:
    """
    Get detailed information about a xenodaq root file structure.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    datadir : str, default 'datasets'
        Base directory containing datasets
    
    Returns
    -------
    dict
        Dictionary containing file information:
        - filepath: Path to the ROOT file
        - digitizers: Dict with info for each digitizer tree
        - metadata_keys: List of metadata entries in file
    
    Examples
    --------
    >>> info = get_file_info('20260226_160258')
    >>> print(f"File: {info['filepath']}")
    >>> for tree_name, tree_info in info['digitizers'].items():
    ...     print(f"{tree_name}: {tree_info['num_entries']} events")
    """
    dirpath = Path(datadir) / dataset
    ds_flist = glob(str(dirpath / "*.root"))
    
    if not ds_flist:
        raise FileNotFoundError(f"No ROOT files found in {dirpath}")
    
    rootfile_path = ds_flist[0]
    
    info = {
        'filepath': rootfile_path,
        'digitizers': {},
        'metadata_keys': []
    }
    
    with uproot.open(rootfile_path) as rootfile:
        # Get metadata
        metadata_keys = [key for key in rootfile.keys() if key.startswith('metadata')]
        info['metadata_keys'] = metadata_keys
        
        # Get digitizer info (all cycles)
        dig_trees = [key for key in rootfile.keys() if key.startswith('dig_')]
        for tree_name in dig_trees:
            tree = rootfile[tree_name]
            
            wf_branches = [b for b in tree.keys() if b.startswith('wf')]
            other_branches = [b for b in tree.keys() if not b.startswith('wf')]
            
            info['digitizers'][tree_name] = {
                'num_entries': tree.num_entries,
                'waveforms': sorted(wf_branches),
                'metadata_branches': sorted(other_branches),
                'all_branches': list(tree.keys())
            }
    
    return info


def print_file_structure(dataset: str, datadir: str = 'datasets') -> None:
    """
    Print a summary of the root file structure.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    datadir : str
        Base directory containing datasets
    """
    info = get_file_info(dataset, datadir)
    
    print(f"File: {info['filepath']}\n")
    
    if info['metadata_keys']:
        print("Metadata entries:")
        for key in info['metadata_keys']:
            print(f"  - {key}")
        print()
    
    print("Digitizers:")
    for tree_name, tree_info in info['digitizers'].items():
        print(f"\n  {tree_name}:")
        print(f"    Entries: {tree_info['num_entries']}")
        print(f"    Waveforms ({len(tree_info['waveforms'])}): {tree_info['waveforms']}")
        print(f"    Metadata: {tree_info['metadata_branches']}")


def load_channel_map(dataset: str, datadir: str = 'datasets') -> Optional[Dict]:
    """
    Load the channel map from the dataset JSON config file.

    Looks for a file matching ``*<dataset>.json`` inside the dataset
    directory 

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. '20260226_160258')
    datadir : str, default 'datasets'
        Base directory containing datasets

    Returns
    -------
    dict or None
        ChsMap dictionary, or None if no JSON config file is found.

    Examples
    --------
    >>> chs = load_channel_map('20260226_160258')
    >>> print(chs['tile_A'])  # {'tree': 'dig_0', 'channel': 'wf0'}
    """
    dirpath = Path(datadir) / dataset
    json_files = glob(str(dirpath / f"*{dataset}.json"))
    if not json_files:
        return None
    with open(json_files[0]) as f:
        config = json.load(f)
    return config.get('ChsMap')


def map_channels_to_tiles(
    wfs: Dict[str, np.ndarray],
    wfs_df: pd.DataFrame,
    channel_map: Dict
) -> Dict:
    """
    Map loaded waveforms to physical names using a channel map.
    Distinct from load_channel_map() which only loads the mapping info, we need this to use fallback back if not provided 
    Parameters
    ----------
    wfs : dict
        Waveform arrays as returned by load_xenodaq_run
    wfs_df : pd.DataFrame
        Metadata DataFrame as returned by load_xenodaq_run
    channel_map : dict
        ChsMap with entries to map (null entries are skipped)

    Returns
    -------
    dict
        ``{tile_name: {'waveforms': ndarray, 'evid': ndarray,
                       'ttt': ndarray, 'tree': str, 'channel': str}}``

    Examples
    --------
    >>> tiles = map_channels_to_tiles(wfs, wfs_df, tile_channel_map)
    >>> print(tiles['tile_A']['waveforms'].shape)
    """
    tiles = {}
    for tile_name, loc in channel_map.items():
        tree = loc.get('tree')
        channel = loc.get('channel')
        if tree is None or channel is None:
            continue
        if channel not in wfs:
            continue
        evid_col = f'{channel}_evid'
        ttt_col = f'{channel}_ttt'
        tiles[tile_name] = {
            'waveforms': wfs[channel],
            'evid': wfs_df[evid_col].values if evid_col in wfs_df.columns else None,
            'ttt': wfs_df[ttt_col].values if ttt_col in wfs_df.columns else None,
            'tree': tree,
            'channel': channel,
        }
    return tiles