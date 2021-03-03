import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob


def show_raster_bars(t_start, t_stop, n_rec, frac_to_plot, path, n_scaling, k_scaling):

    # List of spike arrays, one entry for each population
    spikes = []

    # Read out spikes for each population
    layer_list = ['L23', 'L4', 'L5', 'L6']
    pop_list = ['E', 'I']

    for i in range(8):
        layer = int(i / 2)
        pop = i % 2
        filestart = path + 'spikes_' + str(layer_list[layer]) + '_' + str(pop_list[pop]) + '*'
        filelist = glob.glob(filestart)
        pop_spike_array = np.empty((0, 2))
        last_id = 0
        for file_name in filelist:
            spike_array = np.loadtxt(file_name)
            spike_array[:, 1] = spike_array[:, 1] + last_id
            pop_spike_array = np.vstack((pop_spike_array, spike_array))
            last_id = pop_spike_array[-1, 1]
        spikes.append(pop_spike_array)

    # Plot spike times in raster plot and bar plot with the average firing rates of each population

    color = ['#595289', '#af143c']
    pops = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']
    rates = np.zeros(8)
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axarr = []
    axarr.append(fig.add_subplot(121))
    axarr.append(fig.add_subplot(122))

    # Plot raster plot
    id_count = 0
    print("Mean rates")
    for i in range(8)[::-1]:
        layer = int(i / 2)
        pop = i % 2
        rate = 0.0
        t_spikes = spikes[i][:, 0]
        ids = spikes[i][:, 1] + (id_count + 1)
        filtered_times_indices = [np.where((t_spikes > t_start) & (t_spikes < t_stop))][0]
        t_spikes = t_spikes[filtered_times_indices]
        ids = ids[filtered_times_indices]

        # Compute rates with all neurons
        rate = 1000 * len(t_spikes) / (t_stop - t_start) * 1 / float(n_rec[layer][pop])
        rates[i] = rate
        #print(pops[-i] + np.round(rate, 2))
        # Reduce data for raster plot
        num_neurons = frac_to_plot * np.unique(ids).size
        t_spikes = t_spikes[np.where(ids < num_neurons + id_count + 1)[0]]
        ids = ids[np.where(ids < num_neurons + id_count + 1)[0]]
        axarr[0].plot(t_spikes, ids, '.', color=color[pop], markersize=1)
        print('ids for layer %s, pop %s: %s'%(layer, pop, ids))
        if len(ids)>0:
            id_count = ids[-1]

    # Plot bar plot
    axarr[1].barh(np.arange(0, 8, 1), rates[::-1], color=color[::-1] * 4)

    # Set labels
    axarr[0].set_ylim((0.0, id_count))
    axarr[0].set_yticklabels([])
    axarr[0].set_xlabel('time (ms)')
    axarr[1].set_ylim((-0.5, 7.5))
    axarr[1].set_yticks(np.arange(0, 8, 1.0))
    axarr[1].set_yticklabels(pops[::-1])
    axarr[1].set_xlabel('rate (spikes/s)')

    plt.title("Network, N scaling: %s, K scaling: %s"%(n_scaling,k_scaling), x=-.2)

    plt.savefig(path + 'result.png')
