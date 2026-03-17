import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import pylars


def load_layout():
    array_layout = np.loadtxt('/disk/gfs_atp/xenoscope/tpc/tiles_layout.txt')
    array_labels = ['A', 'B', 'C', 'D', 'E', 'F',
                    'G', 'H', 'J', 'K', 'L', 'M']
    return array_layout, array_labels


def plot_waveform_withpattern(plottitle, single_channels, data, n_wf):
    ## Adjust this!!
    labels = {'mod0' : {'wf1': 'wf1 | Tile H', 'wf2': 'wf2 | Tile J', 
                        'wf3': 'wf3 | Tile K', 'wf4': 'wf4 | Tile L',
                        'wf5': 'wf5 | Tile M'},
              'mod1' : {'wf1': 'wf1 | Tile A', 'wf2': 'wf2 | Tile B', 
                        'wf3': 'wf3 | Tile C', 'wf4': 'wf4 | Tile D',
                        'wf5': 'wf5 | Tile E', 'wf6': 'wf6 | Tile F', 
                        'wf7': 'wf7 | Tile G' }
             }
    
    labels = {{'wf1': 'wf1 | Tile H', 'wf2': 'wf2 | Tile J', 
                        'wf3': 'wf3 | Tile K', 'wf4': 'wf4 | Tile L',
                        'wf5': 'wf5 | Tile M', 'wf6': 'wf6 | Tile A', 'wf7': 'wf7 | Tile B', 
                        'wf8': 'wf8 | Tile C', 'wf9': 'wf9 | Tile D',
                        'wf10': 'wf10 | Tile E', 'wf11': 'wf11 | Tile F', 
                        'wf12': 'wf12 | Tile G' }
                        }

    array_layout, array_labels = load_layout()
    fig = plt.figure(figsize = (14,7), dpi = 200)
    gs = GridSpec(3, 2, width_ratios=[1, 0.8], height_ratios=[1,1, 1])

    ax_trigger = fig.add_subplot(gs[0,1])
    ax_wf = fig.add_subplot(gs[0,0])
    ax_mod1 = fig.add_subplot(gs[1,0], sharex = ax_wf)
    ax_mod0 = fig.add_subplot(gs[2,0], sharex = ax_wf)
    ax_hitp = fig.add_subplot(gs[1:3,1])
    
    n_samples = len(single_channels["wf1"][0])
    _x = np.arange(0, n_samples/100, 0.01)


    plt.subplots_adjust(hspace=0.2)
    trigger_label = ["Trigger 1", "Trigger 2"]
    linewidth = 1
    for i,_ch in enumerate(['wf6', 'wf7']):
        ax_trigger.plot(_x, single_channels[_ch][n_wf], 
                    label = trigger_label[i], linewidth = linewidth)
    
    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5',]:
        ax_mod0.plot(_x, single_channels[_ch][n_wf], 
                    label = labels[_ch], linewidth = linewidth)

    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5', 'wf7']:
        ax_mod1.plot(_x, single_channels[_ch][n_wf], 
                    label = labels[_ch], linewidth = linewidth)
        
    ax_wf.plot(_x, data["rawWf"][n_wf], label = "Total Waveform", linewidth = linewidth)

    if data["nPulses"][n_wf] > 0:

        cmap = plt.get_cmap('binary_r')
        n_colors = data["nPulses"][n_wf] -1
        colors = [cmap(i) for i in np.linspace(0.3,0.7,n_colors)]
        ax_wf.axvspan(data[n_wf]["pulseStart_us"][0], data[n_wf]["pulseEnd_us"][0],alpha = 0.3, color = "Red", label= f"Prominent Pulse")
        if data["nPulses"][n_wf] > 1:
            for pulseStart, pulseEnd,  color in zip(data[n_wf]["pulseStart_us"][1:],data[n_wf]["pulseEnd_us"][1:], colors) :
                ax_wf.axvspan(pulseStart, pulseEnd,alpha = 0.3, color = color)
        width = data[n_wf]["fwhm_us"][0]
        area = data[n_wf]["area"][0]
        ax_wf.plot(0, 0, ",", color = "white", label = f"FWHM = {width:.0f} us")
        ax_wf.plot(0, 0, ",", color = "white", label = f"Area = {area:.2e} PE")

    ax_wf.legend()

    ax_mod0.set_xlabel('Time [us]')
    ax_mod1.set_ylabel('Amplitude [PE/20ns]')
    ax_mod0.legend()
    ax_mod1.legend()
    ax_trigger.legend()
    ax_trigger.set_ylabel('Amplitude [ADC value]')

    
    #Make hitpattern
    hitpattern_area = []
    for key in single_channels.keys():
        _wf = single_channels[key][n_wf]
        _area = np.sum(_wf)
        hitpattern_area.append(_area)


    ax, _map = pylars.plotting.plot_hitpattern(
        hitpattern = hitpattern_area,
        layout = array_layout,
        labels = array_labels,
        r_tpc = 160/2,
        cmap = 'cividis',
        log = False,
        ax = ax_hitp)
    
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    ax.set_aspect('equal')
    fig.colorbar(_map, label = 'Total waveform area [PE]')
    
    ax_hitp.set_xlabel('x [mm]')
    ax_hitp.set_ylabel('y [mm]')
    
    # fig.legend(ncol = 5, loc = 'lower center', 
    #            bbox_to_anchor  = (0,0.9,1,0))
    fig.suptitle(f'{plottitle}, Event {n_wf}', y = 0.92, x = 0.5, horizontalalignment='center',
                 verticalalignment='center', transform=fig.transFigure)
    fig.show()

    return