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


# ------------- Pulse Processing ------------- #

## REMARK: Currently this is only plausible for muons as the pulse finder was tuned for muons!

def DoGPulseFinder(rawWf):

    """ Pulse finder based on the difference of gaussians """

    # Sigmas are tuned to muons in xenoscope, redo more carefully with clean data
    sigma_1 = 200
    sigma_2 = 1000

    filteredWf = gaussian_filter1d(rawWf, sigma_1) - gaussian_filter1d(rawWf, sigma_2)

    derivative1 = np.gradient(filteredWf)
    derivative2 = np.gradient(derivative1)

    # potential Boundaries where the derivative is around 0 and the second derivative is larger then zero -> minimum 
    # potnteial Peaks: first derivative around 0 and second < zero
    # Why not exactly zero? won't have the value == 0 in the array but close to zero!
    # filteredWf has to be lower then -25 to reduce noise!
    scaling = 1 # since the waveforms are not gain corrected anymore rescale the factors (gain ~O(1000))
    potentialBoundaries = np.where((np.abs(derivative1) < 0.001*scaling) & (derivative2 > 0) & (filteredWf < -0.5*scaling))[0]
    potentialPeaks = np.where((np.abs(derivative1) < 0.001*scaling) & (derivative2 < 0) & (rawWf > 5*scaling))[0]

    if len(potentialBoundaries) == 0:
 
        starts = []
        ends = []
        peaks = []

        return starts, ends, peaks

    starts = []
    ends = []
    peaks = []

    for i in range(0, len(potentialBoundaries[:-1])):
        tmp_start =  potentialBoundaries[i]
        tmp_end = potentialBoundaries[i+1]
        tmp_peaks = potentialPeaks[(potentialPeaks > tmp_start) & (potentialPeaks < tmp_end)]
        area = np.sum(rawWf[tmp_start:tmp_end])
        if (len(tmp_peaks) > 0) and (area / (tmp_end -tmp_start) > 1): # second argument is to not accept tails as pulses!
            starts.append(potentialBoundaries[i])
            ends.append(potentialBoundaries[i+1])
            peaks.append(tmp_peaks[np.argmax(filteredWf[tmp_peaks])])
    
    return starts, ends, peaks
    

def mergePulses(rawWf, starts, ends, peaks):
    
    if len(starts) == 0:
        return [], [], []

    final_starts = []
    final_ends = []
    final_peaks = []
    current_start = starts[0]
    current_peak = peaks[0]
    for i in range(0,len(starts)-1):
        next_peak = peaks[i+1]
        scaling = 1 # remove scaling as gains are applied
        if ak.any(rawWf[current_peak:next_peak] < (max([rawWf[current_peak],rawWf[next_peak]])/10/scaling)):# or ((next_peak- current_start) > 10000): # maybe make this better?
            final_starts.append(current_start)
            final_ends.append(ends[i])
            final_peaks.append(np.argmax(rawWf[current_start:ends[i]]) + current_start )
            current_start = starts[i+1]
            current_peak = peaks[i+1]
        else:
            if rawWf[next_peak] > rawWf[current_peak]:
                current_peak = next_peak
            continue

    # finish the last pulse! 
    final_starts.append(current_start)
    final_ends.append(ends[-1])
    final_peaks.append(current_peak)

    return final_starts, final_ends, final_peaks


def getFWHM(rawWf, start, end, peak):
    """ Define FWHM as the time from the first halfmax until the last halfmax in the pulse """

    halfMax = rawWf[peak]/2
    left_index = np.where(rawWf[start:peak] >= halfMax)[0][0] + start
    right_index = np.where(rawWf[peak:end] >= halfMax)[0][-1] + peak
    fwhm = right_index - left_index

    return fwhm, left_index, right_index


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
    

def process_pulses(dataset, datadir):
    """
    Full processing of events with pusle finder tuned for muon events, 
    takes the gain and baseline corrected waveforms as input.

    Args:
        dataset [string]: name of the file
        datadir [string]: name of the folder

    Output: awkward array of pulse shape variables.
    """
    
    logger.info(f"Loading waveforms from {dataset}")
    filename = os.path.join(datadir, dataset)

    waveforms = uproot.open(filename + ".root:events").arrays()
    summed_channels = waveforms["summed_tiles"]
    eventID = waveforms["eventID"]
    single_channels = {key: waveforms[key] for key in waveforms.fields if "tile_" in key}
    muon_channels = {key: waveforms[key] for key in waveforms.fields if "muon" in key}

    with open(filename + "_metadata.json") as f:
       metadata = json.load(f)
    baseline = metadata["baseline"][0]
    gain_file = metadata["gain_file"]


    logger.info("Calculating pulse shape variables")
    single_channels = ak.Array(single_channels)    
    muon_channels = ak.Array(muon_channels)    


    pulses = []
    merge = True

    for n, rawWf in enumerate(summed_channels):
        starts, ends, peaks = DoGPulseFinder(rawWf)
        if merge:
            starts, ends, peaks = mergePulses(rawWf, starts, ends, peaks)
        
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
                "maxima": np.array([]), 
                "coincidence": np.array([]),
                "nSaturatedChannels": np.array([]),
                "chSaturated": ak.Array({field: [False] for field in single_channels.fields}),
                "maxima_over_fwhm": np.array([]), 
                "peaktime_us": np.array([]), 
                "aft50": np.array([]),
            })
            continue

        try: 

            maximas = rawWf[peaks]
            areas = np.array([np.sum(rawWf[s:e]) for s, e in zip(starts, ends)])
            widths = ends - starts

            fwhms, fwhms_left, _ = zip(*[getFWHM(rawWf, s, e, p) for s, e, p in zip(starts, ends, peaks)])
            afts, afts_left, _ = zip(*[getAFT(rawWf, s, e) for s, e in zip(starts, ends)])
            aft50 = [getAFT50(rawWf, s, e) for s, e in zip(starts, ends)]
            
            coincidence = [getCoincidence(single_channels[n], s, e) for s, e in zip(starts, ends)]
            nSaturatedChannels, chSaturated_list = zip(*[getSaturation(single_channels[n], baseline, s, e) for s, e in zip(starts, ends)])
            chSaturated = ak.concatenate(chSaturated_list)


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
                "maxima": np.array([]), "coincidence": np.array([]),
                "nSaturatedChannels": np.array([]),
                "chSaturated": ak.Array({field: [False] for field in single_channels.fields}),
                "maxima_over_fwhm": np.array([]), "peaktime_us": np.array([]),
                "aft50": np.array([])
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
        coincidence_sorted = np.array(coincidence)[sorted_indices]
        nSaturatedChannels_sorted = np.array(nSaturatedChannels)[sorted_indices]
        chSaturated_sorted = ak.Array({key: chSaturated[key][sorted_indices] for key in chSaturated.fields})
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
            "maxima": maximas_sorted,
            "coincidence": coincidence_sorted,
            "nSaturatedChannels": nSaturatedChannels_sorted,
            "chSaturated": chSaturated_sorted,
            "maxima_over_fwhm": maximas_sorted / fwhms_sorted,
            "peaktime_us": (starts_sorted - peaks_sorted) / 100,
            "aft50": aft50_sorted,
            "baseline": baseline,
            "eventID": eventID
        })
    #


    # Attach gain file as metadata on each event
    for pulse in pulses:
        pulse["gain_file"] = gain_file

    pulses = ak.Array(pulses)

    ## Add which cuts they pass!
    cut_rqs(pulses)
    pulses["cut_antiMuonVeto"] = anitMuonVeto(muon_channels)
    pulses["cut_trigger"] = triggerSelection(muon_channels)


    return pulses


def processEventsFromMultipleFiles(datasets, datadir):

    pulses = []
    for dataset in datasets:
        logger.info(f"Processing dataset {dataset}")
        pulses_tmp = process_pulses(dataset,  )
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


def triggerSelection(muon_channels):

    maskmuon1 = np.any(muon_channels["muon1"][:,200:300] > 500, axis = 1)
    maskmuon2 = np.any(muon_channels["muon2"][:,200:300] > 500, axis = 1)

    return (maskmuon1 & maskmuon2)

def anitMuonVeto(muon_channels):

    maskmuon3 = np.any(muon_channels["muon3"] < 100, axis = 1)

    return maskmuon3


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
