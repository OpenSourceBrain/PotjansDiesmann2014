#!/bin/bash
set -ex

cd ../PyNN
python test_neuroml.py

cp *.nml LEMS*ml ../NeuroML2
