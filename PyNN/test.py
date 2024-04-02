# -*- coding: utf-8 -*-
"""
PyNN microcircuit example
---------------------------

Test file for the microcircuit.

"""

import time
import numpy as np
import network
from network_params import net_dict
from sim_params import sim_dict
from stimulus_params import stim_dict
import os

def setup(simulator='nest', N_scaling=0.1):
    # Initialize the network and pass parameters to it.
    tic = time.time()

    net_dict['N_scaling'] = N_scaling

    net_dict['to_record'] = ['spikes', 'v']
    sim_dict['data_path'] = os.path.join(os.getcwd(), 'results')

    sim_dict['simulator'] = simulator

    net = network.Network(sim_dict, net_dict, stim_dict)
    toc = time.time() - tic
    print("Time to initialize the network: %.2f s" % toc)
    
    # Connect all nodes.
    tic = time.time()
    extra_setup_params = {}
    if simulator=='neuroml':
        extra_setup_params['reference']='Microcircuit_%spcnt'%(str(N_scaling*100).replace('.','_'))

    net.setup(extra_setup_params)
    toc = time.time() - tic
    print("Time to create the connections: %.2f s" % toc)
    return net, net_dict, sim_dict


def run(net):
    
    # Simulate.
    tic = time.time()
    net.simulate()
    toc = time.time() - tic
    print("Time to simulate: %.2f s" % toc)
    tic = time.time()
    net.write_data()
    toc = time.time() - tic
    print("Time to write data: %.2f s" % toc)


if __name__ == "__main__":
    net, net_dict, sim_dict = setup(N_scaling=0.1)
    run(net)