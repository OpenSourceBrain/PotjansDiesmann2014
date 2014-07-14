# -*- coding: utf-8 -*-
#
# plot_rates.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
import glob

# relative path to the data files
path = 'data'

# beginning and end of time interval to analyze
t_start = 300
t_stop = 1000

# get area from network_params.sli

f = open(path + '/network_params.sli', 'r')
for line in f:
    if '/area' in line:
        area = float(line.split()[1])
f.close()

# get fraction or numbers of neurons recorded from sim_params.sli

f = open(path + '/sim_params.sli', 'r')
for line in f:
    if '/record_fraction_neurons_spikes' in line:
      record_frac = line.split()[1]
f.close()
f = open(path + '/sim_params.sli', 'r')
for line in f:
    if record_frac == 'true':
        if 'frac_rec_spikes' in line:
            frac_rec = float(line.split()[1])
    else:    
        if 'n_rec_spikes' in line:
            n_rec = int(line.split()[1])
f.close()

# numbers of neurons recorded in each population

if record_frac == 'true':
    pop_sizes = np.array([[20683,5834],[21915,5479],[4850,1065],[14395,2948]])*area*frac_rec
else:
    pop_sizes = [[n_rec]*2]*4

# list of dictionaries, one  dictionary containing neuron number and spike times
spikes = [{} for i in range(8)]

# read out spikes within the specified time interval
    
for i in range(8):
    layer = i/2
    pop = i%2
    filestart = path + '/spikes_' + str(layer) + '_' + str(pop) + '*'
    filelist = glob.glob(filestart)

    for filename in filelist:
        input_file = open(filename,'r')
        while True:
            line = input_file.readline()
            data = line.split()
            if len(data) == 0:
                break
            t_spike = float(data[1])
            if t_spike > t_stop:
                break
            neuron_gid = int(data[0])
            if t_spike>=t_start:
                if neuron_gid in spikes[i]:
                    spikes[i][neuron_gid].append(t_spike)
                else:
                    spikes[i][neuron_gid] = [t_spike]

# plot spike times in raster plot, and bar plot with the average firing rates of each population

color = ['#595289', '#af143c'] 
pops = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']
rates = np.zeros(8)
fig, axarr = plt.subplots(1,2)

# plot raster plot
gid_count = 0
for i in range(8)[::-1]:
    layer = i/2
    pop = i%2
    rate = 0.0
    for neuron_gid in spikes[i]:
        t_spikes = spikes[i][neuron_gid]
        nr_spikes = len(t_spikes)
        rate += 1000*nr_spikes/(t_stop-t_start)*1/float(pop_sizes[layer][pop])
        gid = gid_count*np.ones(nr_spikes)
        axarr[0].plot(t_spikes, gid, '.', color=color[pop])
        gid_count += 1
    rates[i] = rate

# plot bar plot
axarr[1].barh(np.arange(0,8,1)+0.1,rates[::-1],color=color[::-1]*4)

# set labels
axarr[0].set_ylim((0.0,gid_count))
axarr[0].set_yticklabels([])
axarr[0].set_xlabel('time (ms)')
axarr[1].set_ylim((0.0,8.5))
axarr[1].set_yticks(np.arange(0.5,8.5,1.0))
axarr[1].set_yticklabels(pops[::-1])
axarr[1].set_xlabel('rate (spikes/s)')

plt.show()
