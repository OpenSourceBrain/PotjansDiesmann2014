# Smaller scale PyNN net to test
sed -i -e "s/1., # N_scaling/0.03, # N_scaling/g" PyNN/network_params.py
sed -i -e "s/K_scaling' : 0.5,/K_scaling' : 0.03,/g" PyNN/network_params.py
mkdir -p PyNN/results

# Exit before plotting in PyNEST
###sed -i -e 's/# Plot a raster/exit() # Plot a raster/g'  PyNEST/example.py 
