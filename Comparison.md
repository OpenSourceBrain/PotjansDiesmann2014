## Comparison of model behaviour across implementations

### NEST SLI

The model was originally implemented in [NEST SLI format](/NEST_SLI) (taken from [here](https://github.com/nest/nest-simulator/tree/master/examples/nest/Potjans_2014)). 

**10% scaled down version**

Note 1: 10% of cells in the network are included in rasterplots on the left.<br/>
Note 2: rasterplots include first 200ms to enable comparison between implementations at startup, but firing rates on right are calculated on 200-1000ms.
<table>
<tr>
<td><img alt="NEST SLI raster" src="images/nestsli_rasterplot_0.1.png" height="300"/></td>
<td><img alt="NEST SLI firing rates" src="images/nestsli_firing_rates_0.1.png" height="300"/></td>
</tr>
</table>

**Full scale version**
<table>
<tr>
<td><img alt="NEST SLI raster" src="images/nestsli_rasterplot_1.0.png" height="300"/></td>
<td><img alt="NEST SLI firing rates" src="images/nestsli_firing_rates_1.0.png" height="300"/></td>
</tr>
</table>


### PyNEST

**10% scaled down version**
<table>
<tr>
<td><img alt="PyNEST raster" src="images/pynest_rasterplot_0.1.png" height="300"/></td>
<td><img alt="PyNEST firing rates" src="images/pynest_firing_rates_0.1.png" height="300"/></td>
</tr>
</table>


**Full scale version**
<table>
<tr>
<td><img alt="PyNEST raster" src="images/pynest_rasterplot_1.0.png" height="300"/></td>
<td><img alt="PyNEST firing rates" src="images/pynest_firing_rates_1.0.png" height="300"/></td>
</tr>
</table>

### PyNN: NEST

**10% scaled down version**
<table>
<tr>
<td><img alt="PyNN NEST plots" src="images/pynn_nest_plots_0.1.png" height="300"/></td>
</tr>
</table>

**Full scale version**
<table>
<tr>
<td><img alt="PyNN NEST plots" src="images/pynn_nest_plots_1.0.png" height="300"/></td>
</tr>
</table>

### PyNN: NEURON

**10% scaled down version**
<table>
<tr>
<td><img alt="PyNN NEURON plots" src="images/pynn_neuron_plots_0.1.png" height="300"/></td>
</tr>
</table>

**Full scale version**
<table>
<tr>
<td><img alt="PyNN NEURON plots" src="images/pynn_neuron_plots_1.0.png" height="300"/></td>
</tr>
</table>


