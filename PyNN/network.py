# -*- coding: utf-8 -*-
"""
Main file for the microcircuit.

Based on original PyNEST version by Hendrik Rothe, Hannah Bos, Sacha van Albada; May 2016
Adapted for PyNN by Andrew Davison, December 2017
"""

from importlib import import_module
import numpy as np
import os
from helpers import adj_w_ext_to_K
from helpers import synapses_th_matrix
from helpers import get_total_number_of_synapses
from helpers import get_weight
from helpers import plot_raster
from helpers import fire_rate
from helpers import boxplot
from helpers import compute_DC
from pyNN.random import RandomDistribution
from pyNN.space import RandomStructure, Cuboid
import math


class Network:
    """ Handles the setup of the network parameters and
    provides functions to connect the network and devices.

    Arguments
    ---------
    sim_dict
        dictionary containing all parameters specific to the simulation
        such as the directory the data is stored in and the seeds
        (see: sim_params.py)
    net_dict
         dictionary containing all parameters specific to the neurons
         and the network (see: network_params.py)

    Keyword Arguments
    -----------------
    stim_dict
        dictionary containing all parameter specific to the stimulus
        (see: stimulus_params.py)

    """
    def __init__(self, sim_dict, net_dict, stim_dict=None):
        self.sim_dict = sim_dict
        self.net_dict = net_dict
        if stim_dict is not None:
            self.stim_dict = stim_dict
        else:
            self.stim_dict = None
        self.sim = import_module("pyNN.%s" % sim_dict["simulator"])
        self.data_path = sim_dict['data_path']
        

    def setup_pyNN(self, extra_setup_params):
        """ Reset and configure the simulator.

        Where the simulator is NEST,
        the number of seeds for the NEST-kernel is computed, based on the
        total number of MPI processes and threads of each.
        """

        master_seed = self.sim_dict['master_seed']
        if self.sim_dict['simulator'] == "spiNNaker":
            N_tp = 1
        else:
            N_tp = self.sim.num_processes() * self.sim_dict['local_num_threads']
        rng_seeds = list(range(master_seed + 1 + N_tp, master_seed + 1 + (2 * N_tp)))
        grng_seed = master_seed + N_tp
        self.pyrngs = [np.random.RandomState(s) 
                       for s in list(range(master_seed, master_seed + N_tp))]
        self.sim_resolution = self.sim_dict['sim_resolution']
        self.sim.setup(timestep=self.sim_resolution,
                       threads=self.sim_dict['local_num_threads'],
                       grng_seed=grng_seed,
                       rng_seeds=rng_seeds,
                       **extra_setup_params)
        if self.sim.rank() == 0:
            print('Master seed: %i ' % master_seed)
            print('Number of total processes: %i' % N_tp)
            print('Seeds for random number generators of virtual processes: %r' % rng_seeds)
            print('Global random number generator seed: %i' % grng_seed)
            if os.path.isdir(self.sim_dict['data_path']):
                print('data directory already exists')
            else:
                os.makedirs(self.sim_dict['data_path'])
                print('data directory created')
            print('Data will be written to %s' % self.data_path)


    def create_populations(self):
        """ Creates the neuronal populations.

        The neuronal populations are created and the parameters are assigned
        to them. The initial membrane potential of the neurons is drawn from a
        normal distribution. Scaling of the number of neurons and of the
        synapses is performed. If scaling is performed extra DC input is added
        to the neuronal populations.

        """
        self.N_full = self.net_dict['N_full']
        self.N_scaling = self.net_dict['N_scaling']
        self.K_scaling = self.net_dict['K_scaling']
        self.synapses = get_total_number_of_synapses(self.net_dict)
        self.synapses_scaled = self.synapses * self.K_scaling
        self.nr_neurons = self.N_full * self.N_scaling
        self.K_ext = self.net_dict['K_ext'] * self.K_scaling
        self.w_from_PSP = get_weight(self.net_dict['PSP_e'], self.net_dict)
        self.weight_mat = get_weight(
            self.net_dict['PSP_mean_matrix'], self.net_dict
            )
        self.weight_mat_std = self.net_dict['PSP_std_matrix']
        self.w_ext = self.w_from_PSP
        if self.net_dict['poisson_input']:
            self.DC_amp_e = np.zeros(len(self.net_dict['populations']))
        else:
            if self.sim.rank() == 0:
                print(
                    '''
                    no poisson input provided
                    calculating dc input to compensate
                    '''
                    )
            self.DC_amp_e = compute_DC(self.net_dict, self.w_ext)

        if self.sim.rank() == 0:
            print(
                'The number of neurons is scaled by a factor of: %.2f'
                % self.N_scaling
                )
            print(
                'The number of synapses is scaled by a factor of: %.2f'
                % self.K_scaling
                )

        # Scaling of the synapses.
        if self.K_scaling != 1:
            synapses_indegree = self.synapses / (
                self.N_full.reshape(len(self.N_full), 1) * self.N_scaling)
            self.weight_mat, self.w_ext, self.DC_amp_e = adj_w_ext_to_K(
                synapses_indegree, self.K_scaling, self.weight_mat,
                self.w_from_PSP, self.DC_amp_e, self.net_dict, self.stim_dict
                )

        # Create cortical populations.
        self.pops = []
        neuron_model = getattr(self.sim, self.net_dict['neuron_model'])
        parameters = {
            'tau_syn_E': self.net_dict['neuron_params']['tau_syn_ex'],
            'tau_syn_I': self.net_dict['neuron_params']['tau_syn_in'],
            'v_rest': self.net_dict['neuron_params']['E_L'],
            'v_thresh': self.net_dict['neuron_params']['V_th'],
            'v_reset':  self.net_dict['neuron_params']['V_reset'],
            'tau_refrac': self.net_dict['neuron_params']['t_ref'],
            'cm': self.net_dict['neuron_params']['C_m'] * 0.001,  # pF --> nF
            'tau_m': self.net_dict['neuron_params']['tau_m']
        }
        v_init = RandomDistribution("normal",
                                    [self.net_dict['neuron_params']['V0_mean'],
                                     self.net_dict['neuron_params']['V0_sd']],
                                   )  # todo: specify rng
        layer_structures = {}

        x_dim_scaled = self.net_dict['x_dimension'] * math.sqrt(self.N_scaling)
        z_dim_scaled = self.net_dict['z_dimension'] * math.sqrt(self.N_scaling)
        
        default_cell_radius = 10 # for visualisation 
        default_input_radius = 5 # for visualisation 
        layer_thicknesses = self.net_dict["layer_thicknesses"]


        for i, pop in enumerate(self.net_dict['populations']):
            layer = pop[:-1]

            y_offset = 0
            if layer == 'L6': y_offset = layer_thicknesses['L6']/2
            elif layer == 'L5': y_offset = layer_thicknesses['L6']+layer_thicknesses['L5']/2
            elif layer == 'L4': y_offset = layer_thicknesses['L6']+layer_thicknesses['L5']+layer_thicknesses['L4']/2
            elif layer == 'L23': y_offset = layer_thicknesses['L6']+layer_thicknesses['L5']+layer_thicknesses['L4']+layer_thicknesses['L23']/2
            else:
                raise Exception("Problem with %s"%layer)

            layer_volume = Cuboid(x_dim_scaled,layer_thicknesses[layer],z_dim_scaled)
            layer_structures[layer] = RandomStructure(layer_volume, origin=(0,y_offset,0))

            parameters['i_offset'] = self.DC_amp_e[i] * 0.001   # pA --> nA
            population = self.sim.Population(int(self.nr_neurons[i]),
                                             neuron_model(**parameters),
                                             structure=layer_structures[layer], 
                                             label=pop)
            population.initialize(v=v_init)
            # Store whether population is inhibitory or excitatory
            population.annotate(type=pop[-1:])
                
            population.annotate(radius=default_cell_radius)
            population.annotate(structure=str(layer_structures[layer]))


            try:
                import opencortex.utils.color as occ
                print('Adding color for %s'%pop)
                if 'L23' in pop:
                    if 'E' in pop: color = occ.L23_PRINCIPAL_CELL
                    if 'I' in pop: color = occ.L23_INTERNEURON
                if 'L4' in pop:
                    if 'E' in pop: color = occ.L4_PRINCIPAL_CELL
                    if 'I' in pop: color = occ.L4_INTERNEURON
                if 'L5' in pop:
                    if 'E' in pop: color = occ.L5_PRINCIPAL_CELL
                    if 'I' in pop: color = occ.L5_INTERNEURON
                if 'L6' in pop:
                    if 'E' in pop: color = occ.L6_PRINCIPAL_CELL
                    if 'I' in pop: color = occ.L6_INTERNEURON
                        
                population.annotate(color=color)
            except Exception as e:
                print(e)
                # Don't worry about it, it's just metadata
                pass

            self.pops.append(population)

    def create_devices(self):
        """
        Setup recording
        """

        if self.sim.rank() == 0:
            print('Recording {} (frac_record_v: {}), {}'.format(self.net_dict['to_record'],self.sim_dict['frac_record_v'],self.sim_dict['rec_V_int']))
        
        for i, pop in enumerate(self.pops):
            if 'spikes' in self.net_dict['to_record']:
                pop.record('spikes')
            if 'v' in self.net_dict['to_record']:
                if self.sim_dict['frac_record_v']:
                    num_v = max(1,int(round(pop.size * self.sim_dict['frac_record_v'])))
                else:
                    num_v = pop.size
                print(num_v)
                pop[0:num_v].record('v', sampling_interval=self.sim_dict['rec_V_int'])


 
    def create_thalamic_input(self):
        """ This function creates the thalamic neuronal population if this
        is specified in stimulus_params.py.

        """
        if self.stim_dict['thalamic_input']:
            if self.sim.rank() == 0:
                print('Thalamic input provided')
            self.thalamic_population = self.sim.Population(
                                                self.stim_dict['n_thal'],
                                                self.sim.PoissonSpikeSource(
                                                    rate=self.stim_dict['th_rate'],
                                                    start=self.stim_dict['th_start'],
                                                    duration=self.stim_dict['th_duration']),
                                                label="Thalamic input")
            self.thalamic_weight = get_weight(
                self.stim_dict['PSP_th'], self.net_dict
                )
            self.nr_synapses_th = synapses_th_matrix(
                self.net_dict, self.stim_dict
                )
            if self.K_scaling != 1:
                self.thalamic_weight = self.thalamic_weight / (self.K_scaling ** 0.5)
                self.nr_synapses_th = self.nr_synapses_th * self.K_scaling
        else:
            if self.sim.rank() == 0:
                print('Thalamic input not provided')

    def create_poisson(self):
        """ Creates the Poisson generators.

        If Poissonian input is provided, the Poissonian generators are created
        and the parameters needed are passed to the Poissonian generator.

        """
        if self.net_dict['poisson_input']:
            if self.sim.rank() == 0:
                print('Poisson background input created')
            rate_ext = self.net_dict['bg_rate'] * self.K_ext
            self.poisson = []
            for i, target_pop in enumerate(self.pops):
                poisson = self.sim.Population(target_pop.size,
                                              self.sim.SpikeSourcePoisson(rate=rate_ext[i]),
                                              label="Input_to_{}".format(target_pop.label))
                self.poisson.append(poisson)

    def create_dc_generator(self):
        """ Creates a DC input generator.

        If DC input is provided, the DC generators are created and the
        necessary parameters are passed to them.

        """
        if self.stim_dict['dc_input']:
            if self.sim.rank() == 0:
                print('DC generator created')
            dc_amp_stim = self.net_dict['K_ext'] * self.stim_dict['dc_amp']
            self.dc = []
            if self.sim.rank() == 0:
                print('DC_amp_stim', dc_amp_stim)
            for i in range(len(self.pops)):
                dc = sim.DCSource(
                        amplitude=dc_amp_stim[i],
                        start=self.stim_dict['dc_start'],
                        stop=self.stim_dict['dc_start'] + self.stim_dict['dc_dur'])
                self.dc.append(dc)

    def create_connections(self):
        """ Creates the recurrent connections.

        The recurrent connections between the neuronal populations are created.

        """
        if self.sim.rank() == 0:
            print('Recurrent connections are being established')
        mean_delays = self.net_dict['mean_delay_matrix']
        std_delays = self.net_dict['std_delay_matrix']
        self.projections = []
        for i, target_pop in enumerate(self.pops):
            for j, source_pop in enumerate(self.pops):
                synapse_nr = int(self.synapses_scaled[i][j])
                if synapse_nr > 0:
                    w_mean = 0.001 * self.weight_mat[i][j]  # pA --> nA
                    w_sd = abs(w_mean * self.weight_mat_std[i][j])
                    if w_mean < 0:
                        high = 0.0
                        low = -np.inf
                    else:
                        high = np.inf
                        low = 0.0
                    weight = RandomDistribution('normal_clipped',
                                                mu=w_mean,
                                                sigma=w_sd,
                                                low=low,
                                                high=high)
                    delay = RandomDistribution('normal_clipped',
                                               mu=mean_delays[i][j],
                                               sigma=std_delays[i][j],
                                               low=self.sim_resolution,
                                               high=mean_delays[i][j] + 10 * std_delays[i][j])
                    if self.sim_dict["simulator"] == "spiNNaker":
                        connector_params = {"num_synapses": synapse_nr}
                    else:
                        connector_params = {"n": synapse_nr}
                    self.projections.append(
                        self.sim.Projection(
                            source_pop,
                            target_pop,
                            self.sim.FixedTotalNumberConnector(**connector_params),
                            synapse_type=self.sim.StaticSynapse(weight=weight,
                                                                delay=delay))
                    )
                    if self.sim.rank() == 0:
                        if self.sim_dict["simulator"] == "spiNNaker":
                            # at present Projection.label is not defined in SpyNNaker
                            label = "{}-{}".format(source_pop.label,
                                                   target_pop.label)
                        else:
                            label = self.projections[-1].label
                        print(
                            "{:10} {:9} connections, weight = {:6.3f} +/- {:5.3f} nA, delay = {:4.2f} +/- {:5.3f} ms".format(
                                label + ":", synapse_nr,
                                w_mean, w_sd,
                                mean_delays[i][j], std_delays[i][j])
                        )

    def connect_poisson(self):
        """ Connects the Poisson generators to the microcircuit."""
        if self.sim.rank() == 0:
            print('Poisson background input is connected')
        for i, target_pop in enumerate(self.pops):
            self.projections.append(
                self.sim.Projection(
                    self.poisson[i],
                    target_pop,
                    self.sim.OneToOneConnector(),
                    self.sim.StaticSynapse(weight=0.001 * self.w_ext,
                                           delay=self.net_dict['poisson_delay']))
            )

    def connect_thalamus(self):
        """ Connects the thalamic population to the microcircuit."""
        if self.sim.rank() == 0:
            print('Thalamus connection established')
        
        weight = RandomDistribution('normal_clipped',
                                    mu=0.001 * self.thalamic_weight,
                                    sigma=self.thalamic_weight * self.net_dict['PSP_sd'],
                                    low=0.0, high=np.inf)

        for i, target_pop in enumerate(self.pops):
            synapse_nr = int(self.nr_synapses_th[i])
            if self.sim_dict["simulator"] == "spiNNaker":
                connector_params = {"num_synapses": synapse_nr}
            else:
                connector_params = {"n": synapse_nr}

            mu_d = self.stim_dict['delay_th'][i]
            s_d = self.stim_dict['delay_th_sd'][i]
            delay = RandomDistribution('normal_clipped',
                                       mu=mu_d,
                                       sigma=s_d,
                                       low=self.sim_resolution,
                                       high=mu_d + 10 * s_d)

            self.projections.append(
                self.sim.Projection(
                    self.thalamic_population,
                    target_pop,
                    self.sim.FixedTotalNumberConnector(**connector_params),
                    self.sim.StaticSynapse(weight=weight, delay=delay)
                )
            )

    def connect_dc_generator(self):
        """ Connects the DC generator to the microcircuit."""
        if self.sim.rank() == 0:
            print('DC Generator connection established')
        for i, target_pop in enumerate(self.pops):
            if self.stim_dict['dc_input']:
                self.dc[i].inject_into(target_pop)

    def setup(self, extra_setup_params = {}):
        """ 
        Execute subfunctions of the network.

        This function executes several subfunctions to create neuronal
        populations, devices and inputs, connects the populations with
        each other and with devices and input nodes.

        """
        self.setup_pyNN(extra_setup_params)
        self.create_populations()
        self.create_devices()
        self.create_thalamic_input()
        self.create_poisson()
        self.create_dc_generator()
        self.create_connections()
        if self.net_dict['poisson_input']:
            self.connect_poisson()
        if self.stim_dict['thalamic_input']:
            self.connect_thalamus()
        if self.stim_dict['dc_input']:
            self.connect_dc_generator()

    def write_data(self):

        import time
        start_writing = time.time()
        self.output_data = {}
        for pop in self.pops:
            self.output_data[pop.label] = "{}/{}.pkl".format(self.data_path, pop.label)
            pop.write_data(self.output_data[pop.label], gather=True)

        record_v = 'v' in self.net_dict['to_record']
                
        for pop_obj in self.pops:
            layer = pop_obj.label[:-1]
            pop = pop_obj.label[-1]
            print('Saving spikes of %s%s'%(layer, pop))
  
            spikes = pop_obj.get_data('spikes', gather=False)
            
            spiketrains = spikes.segments[0].spiketrains
       
            spikes_file2 = self.data_path \
                + "/spikes_" + layer + '_' + pop + '_' + str(self.sim.rank()) + ".spikes"
            #print('Saving data recorded for spikes in %s%s, indices: %s to %s'%(layer, pop, [s.annotations for s in spiketrains], spikes_file2))
            ff = open(spikes_file2, 'w')

            def get_source_id(spiketrain):
                if 'source_id' in spiketrain.annotations:
                    return spiketrain.annotations['source_id']

                elif 'channel_id' in spiketrain.annotations: # See https://github.com/NeuralEnsemble/PyNN/pull/762
                    return spiketrain.annotations['channel_id']
                
            for spiketrain in spiketrains:
                source_id = get_source_id(spiketrain)
                source_index = pop_obj.id_to_index(source_id)
                    
                '''print("Writing spike data for cell %s[%s] (gid: %i): %i spikes: [%s,...,%s] "% \
                        (pop,
                        source_index, 
                        source_id, 
                        len(spiketrain),
                        spiketrain[0] if len(spiketrain)>1 else '-',
                        spiketrain[-1] if len(spiketrain)>1 else '-'))'''
                for t in spiketrain:
                    ff.write('%s\t%i\n'%(t.magnitude,source_index))
            ff.close()
                
            if record_v :
                import numpy
                vm = pop_obj.get_data('v', gather=False)

                analogsignal = vm.segments[0].analogsignals[0]
                name = analogsignal.name
                source_ids = analogsignal.annotations['source_ids']

                print('Saving data recorded for %s in pop %s%s, global ids: %s'%(name, layer, pop, source_ids))
                filename=self.data_path+"/vm_%s_%s_%s.%s.dat"%(layer, pop, self.sim.rank(),self.sim_dict["simulator"])
                times_vm_a = []
                tt = numpy.array([t*self.sim.get_time_step()/1000. for t in range(len(analogsignal.transpose()[0]))])
                times_vm_a.append(tt)
                for i in range(len(source_ids)):
                    glob_id = source_ids[i]
                    index_in_pop = pop_obj.id_to_index(glob_id)
                    #print("Writing data for cell %i = %s[%s] (gid: %i) to %s "%(i, pop,index_in_pop, glob_id, filename))
                    vm = analogsignal.transpose()[i]
                    times_vm_a.append(vm/1000.)

                times_vm = numpy.array(times_vm_a).transpose()
                numpy.savetxt(filename, times_vm , delimiter = '\t', fmt='%s')    

        end_writing = time.time()
        print("Writing data took %g s" % (end_writing - start_writing,))

    def simulate(self):
        """ Simulates the microcircuit."""
        self.sim.run(self.sim_dict['t_sim'])

    def evaluate(self, raster_plot_time_idx, fire_rate_time_idx):
        """ Displays output of the simulation.

        Calculates the firing rate of each population,
        creates a spike raster plot and a box plot of the
        firing rates.

        """
        if self.sim.rank() == 0:
            annotation = "Simulated with pyNN.{}".format(
                            self.sim_dict["simulator"])
            print(
                'Interval to compute firing rates: %s ms'
                % np.array2string(fire_rate_time_idx)
                )
            fire_rate(
                self.output_data,
                fire_rate_time_idx[0], fire_rate_time_idx[1],
                self.data_path
                )
            print(
                'Interval to plot spikes: %s ms'
                % np.array2string(raster_plot_time_idx)
                )
            plot_raster(
                self.output_data,
                raster_plot_time_idx[0], raster_plot_time_idx[1],
                self.data_path,
                annotation=annotation
                )
            boxplot(self.net_dict, self.data_path,
                    annotation=annotation)
