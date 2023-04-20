# Create a smaller scale PyNN network to test...
sed -i -e "s/1., # N_scaling/0.1, # N_scaling/g" PyNN/network_params.py
sed -i -e "s/K_scaling' : 0.5,/K_scaling' : 0.1,/g" PyNN/network_params.py
mkdir -p PyNN/results

