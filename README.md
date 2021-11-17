### Large scale model of cortical network from Potjans and Diesmann, 2014

[Potjans and Diesmann (2014)](https://academic.oup.com/cercor/article/24/3/785/398560) describes a microcircuit model of early sensory cortex, displaying asynchronous irregular activity with layer-specific firing rates similar to the activity observed in cortex in the awake spontaneous condition. The inhibitory neurons have higher firing rates than the excitatory neurons, despite being modeled with identical intrinsic properties. Hence, this feature arises due to the connectivity of the network.

The model represents the approximately 80,000 neurons under a square mm of cortical surface, organized into layers 2/3, 4, 5, and 6. Each layer contains an excitatory and an inhibitory population of leaky integrate-and-fire neurons with current-based synapses, connected with delays. The population-specific connectivity and external Poisson inputs are based on the integration of anatomical and physiological data mainly from cat V1 and rat S1. The possibility is included of providing an additional thalamic stimulus to layers 4 and 6, which propagates across layers in a realistic fashion partly due to the target specificity of feedback projections across layers.

<table>
<tr>
<td><img alt="Raster and bar plots of spiking activity" src="https://raw.githubusercontent.com/OpenSourceBrain/PotjansDiesmann2014/master/images/pynn_nest_plots_1.0.png" height="300"/></td>
</tr>
</table>

[![Continuous builds](https://github.com/OpenSourceBrain/PotjansDiesmann2014/actions/workflows/main.yml/badge.svg)](https://github.com/OpenSourceBrain/PotjansDiesmann2014/actions/workflows/main.yml)

[![DOI](https://www.zenodo.org/badge/21612755.svg)](https://www.zenodo.org/badge/latestdoi/21612755)

### Reusing this model

The code in this repository is provided under the terms of the [software license](LICENSE) included with it. If you use this model in your research, we respectfully ask you to cite the references outlined in the
 [CITATION](CITATION.md) file.
