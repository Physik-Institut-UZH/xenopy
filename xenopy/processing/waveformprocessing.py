import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from scipy.ndimage import gaussian_filter1d
import csv
from xenopy.io import load_xenodaq_run, print_file_structure
from matplotlib.gridspec import GridSpec
import json
import os
import glob
from datetime import datetime
import uproot

import logging
logger = logging.getLogger(__name__)


# ------------- Input Data Processing ------------- #

###### Baselines ######

def get_baseline_channel(wf):
    """ average of first 700 samples, then median over all events in data_dict """

    wf_initial = wf[:, 0:700] # get baseline from 500 samples (trigger at 1000)
    baselines = np.average(wf_initial, axis = 1) # changed this to average, and then take the median over events! ( basically super noise events ( events where something happens) will not determine the baseline!)
    assert len(baselines) == wf_initial.shape[0]
    std = np.std(wf_initial, axis = 1)
    return baselines, std

def get_avgbaseline_all_channels(wfs):
    avg_baselines = {key: [] for key in wfs}
    avg_stds = {key: [] for key in wfs}

    for key in wfs.keys():
        _baselines, _stds = get_baseline_channel(np.array(wfs[key]["waveforms"])) 
        avg_baselines[key] = np.median(_baselines)
        avg_stds[key] = np.std(_baselines)/len(_baselines)*1.253 # unsure how true this is...

    return avg_baselines, avg_stds

def baseline_correction(tiles):
    """Apply to baseline correction to tiles array from loading function"""

    gain, _ = load_gain()

    baseline, _ = get_avgbaseline_all_channels(tiles)

    data_baselinecorrected = {
        key: (baseline[key] - np.array(tiles[key]["waveforms"])[:, :])/gain[key]  
        for key in tiles.keys()}

    return data_baselinecorrected

###### Rebin ######

def bin_single_waveform(y, bin_size):
    """Bin y data into bins of size `bin_size`."""
    n_bins = len(y) // bin_size
    y_binned = y[:n_bins * bin_size].reshape(n_bins, bin_size).sum(axis=1)
    return y_binned

def bin_multiple_waveforms(arr, factor):
    """
    Bin a 2D array along the second axis (timepoints).
    Returns a new array of shape (rows, timepoints // factor).
    """
    n_rows, n_cols = arr.shape
    n_bins = n_cols // factor
    arr = arr[:, :n_bins * factor] # trim excess if not divisible
    return arr.reshape(n_rows, n_bins, factor).sum(axis=2)

###### Gain correction ######
def load_gain(dataset):
    """
    Loads the gain file of the day / closest in date
    Returns a dict with the gains and the path to the used gain file
    """

    gain = {}
    tile_keys = ['tile_A', 'tile_B', 'tile_C', 'tile_D', 'tile_E', 'tile_F',
                  'tile_G', 'tile_H', 'tile_J', 'tile_K', 'tile_L', 'tile_M']
    
    gain_directory = "/disk/gfs_atp/xlzd/xenoscope/proc_data/Run6/LED/gains/"
    date_str = dataset.split("_")[0]
    date_target = datetime.strptime(date_str, "%Y%m%d") 
    
    gain_files = glob.glob(os.path.join(gain_directory, "gain_*.json"))
    if not gain_files:
        raise FileNotFoundError(f"No gain files found in {gain_directory}")
    
    gain_file = min(gain_files,
        key=lambda f: abs(datetime.strptime(os.path.basename(f), "gain_%Y%m%d.json") - date_target))

    with open(gain_file, "r") as f:
        gain_json = json.load(f)
        for key in tile_keys:
            gain[key] = gain_json[key]["SPE"] - gain_json[key]["Pedestal"]

    return gain, gain_file


def save_processed_waveforms(filename, outputdir, waveforms, metadata):

    """ Save the processed waveforms stored in an awkward array in a new root file
    Args:
        filename [string]: name of the outputfile
        waveforms [ak.Array]: array with the single waveforms, summed waveform and event id
        metadata [ak.Array]: array with the metadata to be saved
    """

    with uproot.recreate(os.path.join(outputdir, f"{filename}.root")) as f:
        f["events"] = waveforms

    meta_dict = {k: ak.to_list(metadata[k]) for k in metadata.fields}
    with open(os.path.join(outputdir, f"{filename}_metadata.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)


    logger.info(f"Saved: {filename}.root")
    logger.info(f"  [events]   {len(waveforms)} entries | branches: {waveforms.fields}")

    return


def correct_rawWf(dataset, datadir, filenumbers=[0]):
    """
    Gain and baseline corrects waveforms.
    Returns an array containing the summed waveform per event, the single channels and the baseline.

    Args:
        data (str): Date of the input file
        datadir (str): Name of the directiory where the files are stored
        filenumbers (List(int)): Filenumbers to load
    """
    
    # Load waveforms
    _, _, tiles = load_xenodaq_run(dataset, datadir, filenumbers)

    logger.info("Gain and baseline correcting each event")

    tile_keys = ['tile_A', 'tile_B', 'tile_C', 'tile_D', 'tile_E', 'tile_F',
                  'tile_G', 'tile_H', 'tile_J', 'tile_K', 'tile_L', 'tile_M']

    ## Load gain file closest in time
    gain, gain_file = load_gain(dataset)
    
    # Set muon panel scintillator triggers to gain 1
    gain["muon1"] = 1
    gain["muon2"] = 1
    gain["muon3"] = 1

    # gain = {key: 1 for key in tiles.keys()}

    baseline, _ = get_avgbaseline_all_channels(tiles)

    corrected_waveforms = {
        key: (baseline[key] - np.array(tiles[key]["waveforms"])[:, :])/gain[key]  
        for key in tiles.keys()}
    

    ## sum only tiles and not muons!
    total_corrected_waveform = np.sum(
        np.stack([
            corrected_waveforms[key]
            for key in tile_keys
            ], axis=0),  # Stack along a new axis: [n_wf_total, n_events, waveform_len]
        axis=0       # Sum over mod+wf → result shape: [n_events, waveform_len]
    )
    
    # change baseline to PE for later processing of events
    baseline_PE ={
        key: baseline[key]/gain[key]
        for key in baseline.keys()}

    return total_corrected_waveform, corrected_waveforms, baseline_PE, gain_file


def process_rawWf(dataset, datadir, filenumbers = [0], outputdir = "", filenumber_in_filename = False):
    """ 
     Process raw Waveforms and saves them to new root files. Add event ID and metadata.
     Use the date and time as a file name.

     Args:
        data (str): Date of the input file
        datadir (str): Name of the directiory where the files are stored
        filenumbers (List(int)): Filenumbers to load
        outputdir (str): Where the root file should be saved
        filenumber_in_filename (bool): Whether to put filenumber in filename, mainly used for background as large files
    """
     
    total_corrected_waveform, corrected_waveforms, baseline_PE, gain_file = correct_rawWf(dataset, datadir, filenumbers)

    waveforms = {key: arr for key, arr in corrected_waveforms.items()}
    waveforms["summed_tiles"] = total_corrected_waveform
    waveforms["eventID"]      = list(range(len(total_corrected_waveform)))

    metadata = {"original_file":        [dataset],
                "original_filepath":    [datadir],
                "original_filenumbers": [filenumbers], 
                "gain_file":            [gain_file], 
                "baseline":             [baseline_PE],
                "timestamp":            [datetime.now().isoformat()]}

    waveforms = ak.Array(waveforms)
    metadata = ak.Array(metadata)
    
    if filenumber_in_filename:
        filename = dataset + f"_{filenumbers[:]}"
    else:
        filename = dataset

    save_processed_waveforms(filename, outputdir, waveforms, metadata)

    return
