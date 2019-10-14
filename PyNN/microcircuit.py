###################################################
###     	Main script			###        
###################################################

import sys
from sim_params import simulator_params, system_params
sys.path.append(system_params['backend_path'])
sys.path.append(system_params['pyNN_path'])
from network_params import *
# import logging # TODO! Remove if it runs without this line
import pyNN
import time
#from neo.io import PyNNTextIO
import plotting


# prepare simulation
# logging.basicConfig() # TODO! Remove if it runs without this line
exec('import pyNN.%s as sim' % simulator)
sim.setup(**simulator_params[simulator])
print('Starting microcircuit model in %s' % simulator)
import network

# create network
start_netw = time.time()
n = network.Network(sim)

n.setup(sim)
end_netw = time.time()
print('Creating the network took %g s on rank %i (of %i total)' % (end_netw - start_netw,sim.rank(),sim.num_processes()))

# simulate
if sim.rank() == 0 :
    print("Simulating...")
start_sim = time.time()
t = sim.run(simulator_params[simulator]['sim_duration'])
end_sim = time.time()
if sim.rank() == 0 :
    print('Simulation took %g s' % (end_sim - start_sim,))

 
start_writing = time.time()
for layer in n.pops :
    for pop in n.pops[layer] :
        ## Note: disabling PyNNTextIO save option, as this is not supported with later versions of Neo...
        ##spikes_file = system_params['output_path'] \
        ##     + "/spikes_" + layer + '_' + pop + '_' + str(sim.rank()) + ".txt"
        #print('Writing %s'%spikes_file)
        ##io = PyNNTextIO(filename=spikes_file)
        spikes = n.pops[layer][pop].get_data('spikes', gather=False)
        ##for segment in spikes.segments :
        ##    io.write_segment(segment)
        
        spiketrains = spikes.segments[0].spiketrains
        spikes_file2 = system_params['output_path'] \
             + "/spikes_" + layer + '_' + pop + '_' + str(sim.rank()) + ".spikes"
        #print('Saving data recorded for spikes in pop %s, indices: %s to %s'%(pop, [s.annotations['source_id'] for s in spiketrains], spikes_file2))
        ff = open(spikes_file2, 'w')
            
        for spiketrain in spiketrains:
            source_id = spiketrain.annotations['source_id']
            source_index = spiketrain.annotations['source_index']
                
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
            ## Note: disabling PyNNTextIO save option, as this is not supported with later versions of Neo...
            ##v_file = system_params['output_path'] \
            ##     + "/vm_" + layer + '_' + pop + '_' + str(sim.rank()) + ".txt"
            ##io = PyNNTextIO(filename=v_file)
            #print('Writing %s'%v_file)
            vm = n.pops[layer][pop].get_data('v', gather=False)
            ##for segment in vm.segments :
            ##    try :
            ##        io.write_segment(segment)
            ##    except AssertionError :
            ##        pass

            analogsignal = vm.segments[0].analogsignals[0]
            name = analogsignal.name
            source_ids = analogsignal.annotations['source_ids']

            print('Saving data recorded for %s in pop %s%s, global ids: %s'%(name, layer, pop, source_ids))
            filename=system_params['output_path']+"/vm_%s_%s_%s.%s.dat"%(layer, pop, sim.rank(),simulator)
            times_vm_a = []
            tt = numpy.array([t*sim.get_time_step()/1000. for t in range(len(analogsignal.transpose()[0]))])
            times_vm_a.append(tt)
            for i in range(len(source_ids)):
                glob_id = source_ids[i]
                index_in_pop = n.pops[layer][pop].id_to_index(glob_id)
                #print("Writing data for cell %i = %s[%s] (gid: %i) to %s "%(i, pop,index_in_pop, glob_id, filename))
                vm = analogsignal.transpose()[i]
                times_vm_a.append(vm/1000.)

            times_vm = numpy.array(times_vm_a).transpose()
            numpy.savetxt(filename, times_vm , delimiter = '\t', fmt='%s')


end_writing = time.time()
print("Writing data took %g s" % (end_writing - start_writing,))

if create_raster_plot and sim.rank() == 0 :
    # Numbers of neurons from which spikes were recorded
    n_rec = [[0] * n_pops_per_layer for i in range(n_layers)]
    for layer, i in layers.items() :
        for pop, j in pops.items() :
            if record_fraction:
                n_rec[i][j] = round(N_full[layer][pop] * N_scaling * frac_record_spikes)
            else:
                n_rec[i][j] = n_record
    if n_rec > 0:
        plotting.show_raster_bars(raster_t_min, raster_t_max, n_rec, frac_to_plot,
                              system_params['output_path'] + '/', N_scaling, K_scaling)

sim.end()
