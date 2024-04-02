# -*- coding: utf-8 -*-
"""
PyNN microcircuit example
---------------------------

Example file to run the microcircuit.

Based on original PyNEST version by Hendrik Rothe, Hannah Bos, Sacha van Albada; May 2016
Adapted for PyNN by Andrew Davison, December 2017

"""

import time
import numpy as np
import network
from network_params import net_dict
from sim_params import sim_dict
from stimulus_params import stim_dict


# Initialize the network and pass parameters to it.
tic = time.time()
net = network.Network(sim_dict, net_dict, stim_dict)
toc = time.time() - tic
print("Time to initialize the network: %.2f s" % toc)
# Connect all nodes.
tic = time.time()
net.setup()
toc = time.time() - tic
print("Time to create the connections: %.2f s" % toc)
# Simulate.
tic = time.time()
net.simulate()
toc = time.time() - tic
print("Time to simulate: %.2f s" % toc)
tic = time.time()
net.write_data()
toc = time.time() - tic
print("Time to write data: %.2f s" % toc)
# Plot a raster plot of the spikes of the simulated neurons and the average
# spike rate of all populations. For visual purposes only spikes 100 ms
# before and 100 ms after the thalamic stimulus time are plotted here by
# default. The computation of spike rates discards the first 500 ms of
# the simulation to exclude initialization artifacts.
raster_plot_time_idx = np.array(
    [stim_dict['th_start'] - 100.0, stim_dict['th_start'] + 100.0]
    )
fire_rate_time_idx = np.array([500.0, sim_dict['t_sim']])
net.evaluate(raster_plot_time_idx, fire_rate_time_idx)
