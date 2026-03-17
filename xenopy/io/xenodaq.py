"""
XenoDAQ ROOT file loader for Xenoscope analysis.

This module provides functions to load waveform data from XenoDAQ ROOT files
produced by the FMCDAQ system with multiple digitizers.
"""
import numpy as np
import uproot
import pandas as pd
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional, Tuple
import json


def load_xenodaq_run(
    dataset: str,
    datadir: str = 'datasets',
    wfs_to_load: Optional[Dict[str, List[str]]] = None,
    channel_map: Optional[Dict] = None
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Load xenodaq root file and extract waveforms and metadata.
    
    Parameters
    ----------
    dataset : str
        Dataset name (e.g., '20260226_160258')
    datadir : str, default 'datasets'
        Base directory containing datasets
    wfs_to_load : dict, optional
        Dictionary mapping digitizer number to list of waveform names.
        Example: {"0": ["wf0", "wf1"], "1": ["wf9", "wf10"]}
        If None, loads all available waveforms from all digitizers.
    channel_map : dict, optional
        ChsMap dict mapping physical names to DAQ locations, as returned
        by load_channel_map(). Entries with null tree/channel are ignored.
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
    
    Examples
    --------
    >>> # Load specific waveforms
    >>> wfs, df = load_xenodaq_run('20260226_160258', 
    ...                            wfs_to_load={"0": ["wf0"], "1": ["wf9"]})
    >>> print(f"Loaded {len(df)} events")
    >>> print(f"Waveform shape: {wfs['wf0'].shape}")
    
    >>> # Load all waveforms
    >>> wfs, df = load_xenodaq_run('20260226_160258')
    """
    dirpath = Path(datadir) / dataset
    ds_flist = glob(str(dirpath / "*.root"))
    
    if not ds_flist:
        raise FileNotFoundError(f"No ROOT files found in {dirpath}")
    
    rootfile_path = ds_flist[0]

    if channel_map is not None and wfs_to_load is not None:
        raise ValueError("Provide either 'channel_map' or 'wfs_to_load', not both.")

    if channel_map is not None:
        wfs_to_load = _channel_map_to_wfs_to_load(channel_map)
        if not wfs_to_load:
            raise ValueError("channel_map has no valid (non-null) entries.")

    if wfs_to_load is None:
        # Try JSON config first, fall back to auto-detect
        json_map = load_channel_map(dataset, datadir)
        if json_map:
            wfs_to_load = _channel_map_to_wfs_to_load(json_map)
            print(f"Loaded waveforms from JSON ChsMap: {wfs_to_load}")
        else:
            wfs_to_load = detect_waveforms(rootfile_path)
            print(f"No JSON ChsMap found — auto-detected waveforms: {wfs_to_load}")
    
    wfs = {}
    wfs_df = {}
    
    with uproot.open(rootfile_path) as rootfile:
        for dig_n in wfs_to_load.keys():
            tree = rootfile[f'dig_{dig_n}']
            
            # Load metadata
            evcounters = tree[f"EvCounter_{dig_n}"].array(library="np").astype(np.uint32)
            ttts = tree[f"TimeTrigTag_{dig_n}"].array(library="np").astype(np.uint32)
            runtimes = tree["RunTime"].array(library="np").astype(np.float32) if 'RunTime' in tree.keys() else None
            
            # Load waveforms
            for wf_name in wfs_to_load[dig_n]:
                wfs[wf_name] = tree[wf_name].array(library="np").astype(np.uint32)
                wfs_df[f'{wf_name}_evid'] = evcounters
                wfs_df[f'{wf_name}_ttt'] = ttts
                wfs_df[f'{wf_name}_rt'] = runtimes
    
    wfs_df = pd.DataFrame(wfs_df)
    return wfs, wfs_df


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