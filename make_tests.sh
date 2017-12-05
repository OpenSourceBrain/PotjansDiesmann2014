# Smaller scale PyNN net to test
sed -i -e "s/1., # N_scaling/0.1, # N_scaling/g" PyNN/network_params.py

# Exit before plotting in PyNEST
sed -i -e 's/# Plot a raster/exit() # Plot a raster/g'  PyNEST/example.py 