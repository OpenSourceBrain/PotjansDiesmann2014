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
import os


# Initialize the network and pass parameters to it.
tic = time.time()

net_dict['N_scaling'] = 0.1
net_dict['to_record'] = ['spikes', 'v']
sim_dict['data_path'] = os.path.join(os.getcwd(), 'results')


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
