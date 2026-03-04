# -*- coding: utf-8 -*-
"""
PyNN microcircuit example
---------------------------

Example file to run the microcircuit on SpiNNaker.

Based on original PyNEST version by Hendrik Rothe, Hannah Bos, Sacha van Albada; May 2016
Adapted for PyNN by Andrew Davison, December 2017

"""

import time
import numpy as np
import os

import network
from network_params import net_dict
from stimulus_params import stim_dict

import pyNN.spiNNaker as sim


def setup_spinnaker():
    # Simulation resolution
    sim_resolution = 0.1

    if not os.path.exists("spynnaker.cfg"):
        with open("spynnaker.cfg", "w") as f:
            print("[Machine]", file=f)
            print("timeScaleFactor = 1", file=f)
            print("enable_reinjection = False", file=f)
            print("", file=f)
            print("[Buffers]", file=f)
            print("use_auto_pause_and_resume = False", file=f)
            print("", file=f)
            print("[Simulation]", file=f)
            print("incoming_spike_buffer_size = 1024", file=f)
            print("drop_late_spikes = False", file=f)
            print("", file=f)
            print("[Reports]", file=f)
            print("read_router_provenance_data = True", file=f)
            print("read_graph_provenance_data = True", file=f)
            print("read_placements_provenance_data = True", file=f)
            print("write_provenance = True", file=f)
    
    sim.setup(timestep=sim_resolution)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)
    sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 64)
    sim.set_allow_delay_extensions(sim.IF_curr_exp, False)


sim_dict = {
    # Simulation time (in ms).
    't_sim': 1000.0,
    # Path to save the output data.
    'data_path': os.path.join(os.getcwd(), 'data_spinnaker'),
    # Recording interval of the membrane potential (in ms).
    'rec_V_int': 1.0,
    # Fraction of neurons from which to record membrane potentials
    'frac_record_v' : 0.0,
    # Type of voltage init type between 'random' and 'pop_random'
    'v_init_type': 'pop_random',
    # Function to call to setup
    'setup_func': setup_spinnaker
}

# Run at full scale!
net_dict['K_scaling'] = 1.0
net_dict['N_scaling'] = 1.0

# SpiNNaker Poisson at min-delay will use direct input mode,
# which is more efficient with high-rate sources
net_dict['poisson_delay'] = 0.1,


# Initialize the network and pass parameters to it.
tic = time.time()
net = network.Network(sim, sim_dict, net_dict, stim_dict)
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
