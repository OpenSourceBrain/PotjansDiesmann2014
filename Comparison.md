## Comparison of model behaviour across implementations

### NEST SLI

The model was originally implemented in [NEST SLI format](/NEST_SLI) (taken from [here](https://github.com/nest/nest-simulator/tree/master/examples/nest/Potjans_2014)). 

**10% scaled down version**

Note 1: 10% of cells recorded & plotted on left.<br/>
Note 2: rasterplot includes first 200ms to enable comparison between implementations at startup, but firing rates are calculated on 200-1000ms.
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

<table>
<tr>
<td><img alt="PyNN NEST plots" src="images/pynn_nest_plots_1.0.png" height="300"/></td>
</tr>
</table>