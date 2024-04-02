# -*- coding: utf-8 -*-
"""
PyNN microcircuit example
---------------------------

Export to NeuroML of the microcircuit.

"""

import time
import numpy as np
import network
from network_params import net_dict
from sim_params import sim_dict
from stimulus_params import stim_dict
import os

from test import setup

def export(net):

    # Export.
    tic = time.time()
    print("Exporting...")
    net.simulate()
    net.sim.end()
    toc = time.time() - tic
    print("Time to export: %.2f s" % toc)
   


if __name__ == "__main__":
    net, net_dict, sim_dict = setup(simulator='neuroml', 
                N_scaling=0.002)
    export(net)