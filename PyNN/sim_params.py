# -*- coding: utf-8 -*-
"""
Simulation parameters for the microcircuit.

Based on original PyNEST version by Hendrik Rothe, Hannah Bos, Sacha van Albada; May 2016
Adapted for PyNN by Andrew Davison, December 2017
"""

import os
from datetime import datetime

sim_dict = {
    # Simulator
    'simulator': 'nest',
    # Simulation time (in ms).
    't_sim': 1000.0,
    # Resolution of the simulation (in ms).
    'sim_resolution': 0.1,
    # Path to save the output data.
    'data_path': os.path.join(os.getcwd(), 'data', datetime.now().strftime("%Y%m%d-%H%M%S")),
    # Masterseed for PyNN and NumPy.
    'master_seed': 55,
    # Number of threads per MPI process.
    'local_num_threads': 1,
    # Recording interval of the membrane potential (in ms).
    'rec_V_int': 1.0,
}
