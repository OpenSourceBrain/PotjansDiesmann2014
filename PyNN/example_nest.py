# -*- coding: utf-8 -*-
"""
PyNN microcircuit example
---------------------------

Example file to run the microcircuit on NEST.

Based on original PyNEST version by Hendrik Rothe, Hannah Bos, Sacha van Albada; May 2016
Adapted for PyNN by Andrew Davison, December 2017

"""

import time
import numpy as np
import os

import network
from network_params import net_dict
from stimulus_params import stim_dict

import pyNN.nest as sim


def setup_nest():
    # Masterseed for PyNN and NumPy.
    master_seed = 55
    # Number of threads per MPI process.
    local_num_threads = 1
    # Simulation resolution
    sim_resolution = 0.1
    
    N_tp = sim.num_processes() * local_num_threads
    rng_seeds = list(range(master_seed + 1 + N_tp, master_seed + 1 + (2 * N_tp)))
    grng_seed = master_seed + N_tp
    pyrngs = [np.random.RandomState(s) 
              for s in list(range(master_seed, master_seed + N_tp))]
    sim.setup(timestep=sim_resolution,
              threads=local_num_threads,
              grng_seed=grng_seed,
              rng_seeds=rng_seeds)
    if sim.rank() == 0:
        print('Master seed: %i ' % master_seed)
        print('Number of total processes: %i' % N_tp)
        print('Seeds for random number generators of virtual processes: %r' % rng_seeds)
        print('Global random number generator seed: %i' % grng_seed)


sim_dict = {
    # Simulation time (in ms).
    't_sim': 1000.0,
    # Path to save the output data.
    'data_path': os.path.join(os.getcwd(), 'data_nest'),
    # Recording interval of the membrane potential (in ms).
    'rec_V_int': 1.0,
    # Fraction of neurons from which to record membrane potentials
    'frac_record_v' : 0.1,
    # Type of voltage init type between 'random' and 'pop_random'
    'v_init_type': 'random',
    # Function to call to setup
    'setup_func': setup_nest
}


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
