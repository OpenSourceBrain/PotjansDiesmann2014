# Potjans-Diesmann 2014 Cortical Microcircuit model: PyNN implementation

Authors: Andrew Davison, based on the PyNEST implementation by Hendrik Rothe, Hannah Bos, Sacha van Albada   
December 2017

## Description ##
This is a PyNN implementation of the microcircuit model by Potjans and Diesmann (2014): The cell-type specific
cortical microcircuit: relating structure and activity in a full-scale spiking
network model. Cerebral Cortex: doi:10.1093/cercor/bhs358

* Files:  
	* `helpers.py`  
	Helper functions for the simulation and evaluation of the microcircuit.	
	* `network.py`  
	Gathers all parameters and connects the different nodes with each other.
	* `network_params.py`  
	Contains the parameters for the network.
	* `sim_params.py`  
	Contains the simulation parameters.
	* `stimulus_params.py`  
	Contains the parameters for the stimuli.
	* `example.py`  
   Use this script to try out the microcircuit.
   
How to use the example:

To run the microcircuit on a local machine, adjust the variables `N_scaling` and `K_scaling` in `network_params.py` to `0.1`. `N_scaling` adjusts the number of neurons and `K_scaling` the number of connections to be simulated. The full network can be run by adjusting these values to 1. If this is done, the option to print the time progress should be set to False in the file `sim_params.py`. For running, use `python example.py`. The output will be saved in the directory `data`.

If using NEST as the backend simulator, the code can be parallelized using OpenMP and MPI, if NEST has been built with these applications [(Parallel computing with NEST)](http://www.nest-simulator.org/parallel_computing/). The number of threads (per MPI process) can be chosen by adjusting `local_num_threads` in `sim_params.py`. The number of MPI process can be set by choosing a reasonable value for `num_mpi_prc` and then running the script with the command `mpirun -n num_mpi_prc` `python` `example.py`.

If using NEURON as the backend simulator, the code can be parallelized using MPI, if NEURON has been built with that option.
The command to run the script is as for NEST.

The default version of the simulation uses Poisson input, which is defined in the file `network_params.py` to excite neuronal populations of the microcircuit. If no Poisson input is provided, DC input is calculated which should approximately compensate the Poisson input. It is also possible to add thalamic stimulation to the microcircuit or drive it with constant DC input. This can be defined in the file `stimulus_params.py`.

Tested configuration:
This version has been tested with PyNN 0.9.2, NEST 2.14.0, NEURON 7.4, Python 3.5.2, NumPy 1.13.0, mpi4py 2.0.0, Neo 0.5.2, matplotlib 2.0.2.
