import json
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

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

###### PE Conversion ######
def convert_to_pe(waveforms, gain_file):
    """Convert waveforms to PE units using a gain JSON file.

    Args:
        waveforms (dict): accepts tiles as returned by ``load_xenodaq_run``
        gain_file (str): Path to a JSON file with gain data.

    Returns:
        dict: ``{channel: array}`` with waveforms in PE units.
    """
    with open(gain_file, 'r') as f:
        gain_data = json.load(f)
    waveforms_pe = {}
    for channel, data in waveforms.items():
        if channel in gain_data:
            spe = gain_data[channel]['SPE']
            waveforms_pe[channel] = -np.array(data['waveforms']) / spe
        else:
            print(f"Warning: no gain data for {channel}")
    return waveforms_pe


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


# ------------- Pulse Processing - updated for Run 6 ------------- #

def DoGPulseFinder(rawWf, sigma_1=100, sigma_2=500,
                   deriv_thresh=0.001, amp_thresh=0.5,
                   peak_thresh=5, scaling=1, area_thresh=1):
    """
    Difference-of-Gaussians peak finder. Returns (starts, ends, peaks).
    Parameters exposed so they can be swept for the robustness study.
    """
    filteredWf = gaussian_filter1d(rawWf, sigma_1) - gaussian_filter1d(rawWf, sigma_2)
    d1 = np.gradient(filteredWf)
    d2 = np.gradient(d1)

    boundaries = np.where((np.abs(d1) < deriv_thresh * scaling) &
                          (d2 > 0) &
                          (filteredWf < -amp_thresh * scaling))[0]
    cand_peaks = np.where((np.abs(d1) < deriv_thresh * scaling) &
                          (d2 < 0) &
                          (rawWf > peak_thresh * scaling))[0]

    if len(boundaries) == 0:
        return [], [], []

    starts, ends, peaks = [], [], []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        p_in = cand_peaks[(cand_peaks > s) & (cand_peaks < e)]
        if len(p_in) > 0 and np.sum(rawWf[s:e]) / (e - s) > area_thresh:
            starts.append(s)
            ends.append(e)
            peaks.append(p_in[np.argmax(filteredWf[p_in])])
    return starts, ends, peaks
    

def mergePulses(rawWf, starts, ends, peaks, gap_tol=100):
    """
    Merge consecutive peaks into pulses if their boundaries touch,
    i.e. the gap between ends[i] and starts[i+1] is <= gap_tol.

    gap_tol : int
        Max gap (samples) between one boundary's end and the next's start
        to still count as 'touching'. 0 = strictly contiguous.
    """
    if len(starts) == 0:
        return [], [], []

    fs, fe, fp = [], [], []
    cur = starts[0]
    for i in range(len(starts) - 1):
        gap = starts[i + 1] - ends[i]
        if gap > gap_tol:                      
            fs.append(cur)
            fe.append(ends[i])
            fp.append(np.argmax(rawWf[cur:ends[i]]) + cur)
            cur = starts[i + 1]
    fs.append(cur)
    fe.append(ends[-1])
    fp.append(np.argmax(rawWf[cur:ends[-1]]) + cur)
    return fs, fe, fp


def getFWHM(rawWf, start, end, peak):

    """ FWHM = time from first halfmax to last halfmax in the pulse.
        Returns (nan, nan, nan) if no half-max crossing exists (e.g. peak
        coincides with a boundary), so one bad pulse doesn't discard the event. """
    
    halfMax = rawWf[peak] / 2
    left_hits = np.where(rawWf[start:peak] >= halfMax)[0]
    right_hits = np.where(rawWf[peak:end] >= halfMax)[0]
    if len(left_hits) == 0 or len(right_hits) == 0:
        return np.nan, np.nan, np.nan
    left_index = left_hits[0] + start
    right_index = right_hits[-1] + peak
    return right_index - left_index, left_index, right_index


def getAFT(rawWf, start, end, inital = 0.1, final = 0.9):

    totalArea = np.sum(rawWf[start:end])
    area_ = np.cumsum(rawWf[start:end])

    left_index = np.where(area_ - totalArea*inital > 0)[0][0] + start
    right_index = np.where(area_ - totalArea*final > 0)[0][0] + start

    aft = right_index - left_index

    return aft, left_index, right_index

def getAFT50(rawWf, start, end):

    totalArea = np.sum(rawWf[start:end])
    area_ = np.cumsum(rawWf[start:end])

    aft50 = np.where(area_ - totalArea*0.5 > 0)[0][0] + start

    return aft50

def getCoincidence(single_channels, start, end):
    nChannels = 0
    for key in single_channels.fields:
        nChannels += (np.sum(single_channels[key][start:end]) > 50)

    return nChannels

def getSaturation(single_channels, baseline, start, end):

    chSaturated = {}
    nSaturatedChannels = 0
    for key in single_channels.fields:
        nSaturatedChannels += (ak.any(single_channels[key][start:end] >= baseline[key]))
        chSaturated[key] = ak.any(single_channels[key][start:end] >= baseline[key])

    return nSaturatedChannels, ak.Array([chSaturated])

def getXYPosition(maxChannel):
    # WIP!
    # A-> Box: Choose coordinate system that Window is positiv, box negative and towards gassystem negative in y
    
    channelPositions = {'wf1': [-70.8, 0.0],
                        'wf2': [-35.4, -35.4],
                        'wf3': [-35.4, 0.0],
                        'wf4': [-35.4, 35.4],
                        'wf5': [0.0, -53.1],
                        'wf6': [0.0, -17.7],
                        'wf7': [0.0, 17.7],
                        'wf8': [0.0, 53.1],
                        'wf9': [35.4, -35.4],
                        'wf10': [35.4, 0.0],
                        'wf11': [35.4, 35.4],
                        'wf12': [70.8, 0.0]}
    
    return channelPositions[maxChannel]
    

def getMaxChannel(single_channels, start, end):
    # not really usefull maybe because the saturation is very different for the channels??
    # so like if multiple are saturated the max will be given by the highest saturating channel
    max_per_channel = {}
    for channel in single_channels.fields:
        max_per_channel[channel] = max(single_channels[channel][start:end])

    maxChannel = max(max_per_channel, key=max_per_channel.get)

    return maxChannel
    

def process_pulses(filename, entry_start=None, entry_stop=None,
                   sigma_1=150, sigma_2=750,
                   gap_tol=100,
                   **finder_kwargs):
    """
    Full processing of events with pulse finder tuned for muon events, 
    takes the gain and baseline corrected waveforms as input.

    Args:
        filename [string]: name and path of the file
        entry_start [int]: event where to start processing
        entry_stop [int]: event where to end processing

    Output: awkward array of pulse shape variables.
    """
    
    if entry_start is None:
        waveforms = uproot.open(filename + ":events").arrays(filter_name=["summed_tiles", "eventID", "muon1", "muon2", "muon3"])
        logger.info(f"Loading all waveforms from {filename}")

    else:
        with uproot.open(filename) as f:
            tree = f["events"]
            waveforms = tree.arrays(filter_name=["summed_tiles", "eventID", "muon1", "muon2", "muon3"], entry_start= entry_start, entry_stop=entry_stop)
        logger.info(f"Loading waveforms from {filename}, from event {entry_start} to event {entry_stop}")
    if len(waveforms) == 0: # in case the selected entry start is larger than the number of events in the file
        print("No events to process.")
        return waveforms

    summed_channels = waveforms["summed_tiles"]
    eventID = waveforms["eventID"]
    # single_channels = {key: waveforms[key] for key in waveforms.fields if "tile_" in key}
    try:
        muon_channels = {key: waveforms[key] for key in waveforms.fields if "muon" in key}
        muon_channels = ak.Array(muon_channels)    

    except:
        logger.info("No Muons in the file saved")
        muon_channels = ak.Array([])

    with open(filename[:-5] + "_metadata.json") as f:
        metadata = json.load(f)
    baseline = metadata["baseline"][0]
    gain_file = metadata["gain_file"]

    # t=0 reference: fixed muon trigger position
    trigger_sample = 975          # center of the 950:1000 trigger window (see triggerSelection)
    samples_per_us = 100          
    trigger_time_us = trigger_sample / samples_per_us

    logger.info("Calculating pulse shape variables")
    # single_channels = ak.Array(single_channels)    

    ## read in additional muon panel information if given
    window_muonpanel = finder_kwargs.pop('window_muonpanel', None)

    muon_kwargs = {}
    if window_muonpanel is not None:
        muon_kwargs['window'] = window_muonpanel

    ## read in additional muon panel trigger information if given
    threshold_muonpanel = finder_kwargs.pop('threshold_muonpanel', None)
    trigger_kwargs = {}
    if threshold_muonpanel is not None:
        trigger_kwargs['threshold'] = threshold_muonpanel
    

    pulses = []
    merge = True

    for n, rawWf in enumerate(tqdm(summed_channels)):
        starts, ends, peaks = DoGPulseFinder(rawWf, sigma_1=sigma_1, sigma_2=sigma_2,
                                             **finder_kwargs)
        if merge:
            starts, ends, peaks = mergePulses(rawWf, starts, ends, peaks,
                                               gap_tol=gap_tol)
        
        # Convert to numpy arrays for vectorized operations
        starts, ends, peaks = np.array(starts), np.array(ends), np.array(peaks)

        if len(starts) == 0:
            pulses.append({
                # "rawWf": rawWf,
                # "singleChannels": single_channels_binned[n],
                "pulseStart": starts,
                "pulseEnd": ends,
                "peak": peaks,
                "area": np.array([]),
                "width": np.array([]),
                "nPulses": 0,
                "totalWfArea": np.sum(rawWf),
                "fwhm": np.array([]),
                "aft": np.array([]), 
                "fwhmLeft": np.array([]), 
                "aftLeft": np.array([]),
                "fwhm_us": np.array([]), 
                "aft_us": np.array([]), 
                "fwhmLeft_us": np.array([]), 
                "aftLeft_us": np.array([]),
                "pulseStart_us": np.array([]), 
                "pulseEnd_us": np.array([]),
                "driftTime_us": np.array([]),
                "maxima": np.array([]), 
                # "coincidence": np.array([]),
                # "nSaturatedChannels": np.array([]),
                # "chSaturated": ak.Array({field: [] for field in single_channels.fields}),
                "maxima_over_fwhm": np.array([]), 
                "peaktime_us": np.array([]), 
                "aft50": np.array([]),
                "baseline": baseline,
                "eventID": eventID[n],
                **getMuonAmplitudes(muon_channels, n, **muon_kwargs),
                **getMuonArea(muon_channels, n, **muon_kwargs)
            })
            continue

        try: 

            maximas = rawWf[peaks]
            areas = np.array([np.sum(rawWf[s:e]) for s, e in zip(starts, ends)])
            widths = ends - starts

            fwhms, fwhms_left, _ = zip(*[getFWHM(rawWf, s, e, p) for s, e, p in zip(starts, ends, peaks)])
            afts, afts_left, _ = zip(*[getAFT(rawWf, s, e) for s, e in zip(starts, ends)])
            aft50 = [getAFT50(rawWf, s, e) for s, e in zip(starts, ends)]
            
            # coincidence = [getCoincidence(single_channels[n], s, e) for s, e in zip(starts, ends)]
            # nSaturatedChannels, chSaturated_list = zip(*[getSaturation(single_channels[n], baseline, s, e) for s, e in zip(starts, ends)])
            # chSaturated = ak.concatenate(chSaturated_list)


        except Exception as e:
        
            pulses.append({
                # "rawWf": rawWf,
                # "singleChannels": single_channels_binned[n],
                "pulseStart": starts, "pulseEnd": ends, "peak": peaks,
                "area": np.array([]), "width": np.array([]), "nPulses": 0,
                "totalWfArea": np.sum(rawWf),
                "fwhm": np.array([]), "aft": np.array([]), "fwhmLeft": np.array([]), "aftLeft": np.array([]),
                "fwhm_us": np.array([]), "aft_us": np.array([]), "fwhmLeft_us": np.array([]), "aftLeft_us": np.array([]),
                "pulseStart_us": np.array([]), "pulseEnd_us": np.array([]),
                "driftTime_us": np.array([]),
                "maxima": np.array([]), 
                # "coincidence": np.array([]),
                # "nSaturatedChannels": np.array([]),
                # "chSaturated": ak.Array({field: [] for field in single_channels.fields}),
                "maxima_over_fwhm": np.array([]), "peaktime_us": np.array([]),
                "aft50": np.array([]),
                "baseline": baseline,
                "eventID": eventID[n],
                **getMuonAmplitudes(muon_channels, n, **muon_kwargs),
                **getMuonArea(muon_channels, n, **muon_kwargs)
            })
            print(f"Empty event {n} because there are unresolved issues. Error: {e}")            
            continue


        # Sort all the arrays by area
        sorted_indices = np.argsort(areas)[::-1]
        
        # Apply the sorting to all the calculated arrays
        starts_sorted = starts[sorted_indices]
        ends_sorted = ends[sorted_indices]
        peaks_sorted = peaks[sorted_indices]
        areas_sorted = areas[sorted_indices]
        widths_sorted = widths[sorted_indices]
        maximas_sorted = maximas[sorted_indices]
        fwhms_sorted = np.array(fwhms)[sorted_indices]
        fwhms_left_sorted = np.array(fwhms_left)[sorted_indices]
        afts_sorted = np.array(afts)[sorted_indices]
        afts_left_sorted = np.array(afts_left)[sorted_indices]
        # coincidence_sorted = np.array(coincidence)[sorted_indices]
        # nSaturatedChannels_sorted = np.array(nSaturatedChannels)[sorted_indices]
        # chSaturated_sorted = ak.Array({key: chSaturated[key][sorted_indices] for key in chSaturated.fields})
        aft50_sorted = np.array(aft50)[sorted_indices]

        ## Check which cuts the event passes

        pulses.append({
            # "rawWf": rawWf,
            # "singleChannels": single_channels_binned[n],
            "pulseStart": starts_sorted,
            "pulseEnd": ends_sorted,
            "peak": peaks_sorted,
            "area": areas_sorted,
            "width": widths_sorted,
            "nPulses": len(areas),
            "totalWfArea": np.sum(rawWf),
            "fwhm": fwhms_sorted,
            "aft": afts_sorted,
            "fwhmLeft": fwhms_left_sorted,
            "aftLeft": afts_left_sorted,
            "fwhm_us": fwhms_sorted / 100,
            "aft_us": afts_sorted / 100,
            "fwhmLeft_us": fwhms_left_sorted / 100,
            "aftLeft_us": afts_left_sorted / 100,
            "pulseStart_us": starts_sorted / 100,
            "pulseEnd_us": ends_sorted / 100,
            "driftTime_us": (starts_sorted / 100) - trigger_time_us,
            "maxima": maximas_sorted,
            # "coincidence": coincidence_sorted,
            # "nSaturatedChannels": nSaturatedChannels_sorted,
            # "chSaturated": chSaturated_sorted,
            "maxima_over_fwhm": maximas_sorted / fwhms_sorted,
            "peaktime_us": (starts_sorted - peaks_sorted) / 100,
            "aft50": aft50_sorted,
            "baseline": baseline,
            "eventID": eventID[n],
            **getMuonAmplitudes(muon_channels, n, **muon_kwargs),
            **getMuonArea(muon_channels, n, **muon_kwargs)
        })


    # Attach gain file as metadata on each event
    for pulse in pulses:
        pulse["gain_file"] = gain_file
        pulse["sigma_1"] = sigma_1
        pulse["sigma_2"] = sigma_2
        pulse["gap_tol"] = gap_tol

    pulses = ak.Array(pulses)

    ## Add which cuts they pass!
    cut_rqs(pulses)

    try: 
        pulses["cut_trigger"] = triggerSelection(pulses, trigger_kwargs)
        pulses["cut_antiMuonVeto"] = antiMuonVeto(pulses)
    except:
        print("No muons in the file available")

    return pulses

def find_root_files(dataset_folder, datadir):
    """Find all .root files in a dataset folder."""

    pattern = "*.root"
    filepath = os.path.join(datadir, dataset_folder, pattern)

    return sorted(glob.glob(filepath))


def processEventsFromMultipleFiles(datasets, datadir,
                                   sigma_1=100, sigma_2=500, gap_tol=100,
                                   **finder_kwargs):
    """
    Function to process multiple files at once (use this function if the files are small!)
    """
    pulses = []
    for dataset in datasets:
        logger.info(f"Processing dataset {dataset}")
        root_files = find_root_files(dataset, datadir)
        if not root_files:
            logger.warning(f"No ROOT files found in {dataset}")
            continue

        for root_file in root_files:
            logger.info(f"Processing file {root_file}")
            pulses_tmp = process_pulses(root_file,
                                        sigma_1=sigma_1, sigma_2=sigma_2,
                                        gap_tol=gap_tol, **finder_kwargs)
            pulses.append(pulses_tmp)

    pulses = ak.Array(pulses)
    pulses = ak.concatenate(pulses)
    logger.info("Processed all files.")
    return pulses

   
def cut_rqs(pulses):
    """ Add cut variables to the awkward arrays in order"""
    
    # Number of pulses
    cut_nPulses = pulses["nPulses"] <= 2

    # Prominence
    cut_prominence = ak.any(pulses["area"]/pulses["totalWfArea"] >= 0.4, axis=1)

    # Width check
    non_empty_mask = [len(x) > 0 for x in pulses["fwhm_us"]]
    valid_fwhm = pulses["fwhm_us"][non_empty_mask]
    cut_valid_data = (valid_fwhm[:, 0] > 3) & (valid_fwhm[:, 0] < 40)
    cut_width = np.zeros(len(pulses["fwhm_us"]), dtype=bool)
    cut_width[non_empty_mask] = cut_valid_data
    
    # combine all the cuts
    cut_all = cut_nPulses & cut_prominence & cut_width

    # Add the cuts as new fields to the pulses array
    pulses["cut_nPulses"] = cut_nPulses
    pulses["cut_prominence"] = cut_prominence
    pulses["cut_width"] = cut_width
    pulses["cut_all"] = cut_all

    return pulses


def getMuonAmplitudes(muon_channels, event_idx, window=(950, 1000)):
    """
    Extract muon trigger peak amplitudes in the fixed trigger window.
    In case we need to check the validity of the threshold directly from the processed data.

    Returns a dict of peak ADC values (and their sample positions) per channel.
    """
    w0, w1 = window
    out = {}
    for ch in ("muon1", "muon2", "muon3"):
        if ch in muon_channels.fields:
            seg = np.array(muon_channels[ch][event_idx][w0:w1])
            out[f"{ch}_amp"] = float(seg.max())
            out[f"{ch}_amp_sample"] = int(np.argmax(seg)) + w0
        else:
            out[f"{ch}_amp"] = np.nan
            out[f"{ch}_amp_sample"] = -1
    return out


def getMuonArea(muon_channels, event_idx, window=(950, 1000), threshold = 10):
    """
    Extract muon trigger area in the fixed trigger window.
    In case we need to check the validity of the threshold directly from the processed data.

    Returns a dict of area ADC values per channel.
    """
    w0, w1 = window
    out = {}
    for ch in ("muon1", "muon2", "muon3"):
        if ch in muon_channels.fields:
            waveform_window = muon_channels[ch][event_idx][w0:w1]
            if len(waveform_window[waveform_window > threshold]) == 0:
                out[f"{ch}_area"] = 0
            else:
                area = np.sum(waveform_window[waveform_window > threshold])
                out[f"{ch}_area"] = area
        else:
            out[f"{ch}_area"] = np.nan

    return out

def triggerSelection(pulses, threshold = 1000):
    " Trigger selection based on the area in both panels"

    mask = (pulses["muon1_area"] + pulses["muon2_area"]) >= threshold

    return ak.fill_none(mask, False)

def antiMuonVeto(pulses, threshold = 140):

    maskmuon3 = pulses["muon3_area"] < threshold

    return ak.fill_none(maskmuon3, False)


def data_selection(pulses):
    """ Apply data selection """
    pulses = cut_rqs(pulses)
    clean_events = pulses[pulses["cut_all"] == True]

    return ak.Array(clean_events)


##------------ Drift Veloctiy from HeXe ------------##

def drift_velocity(E):
    # From [2109.13735] all in units of V/cm and mm/µs
    A1, A2 = -1.38, -0.95
    B1, B2 = 38, 1000
    C = 2.33
    return A1 * np.exp(-E/B1) + A2 * np.exp(-E/B2) + C
