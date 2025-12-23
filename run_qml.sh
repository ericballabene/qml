#!/bin/bash
set -e

echo "Running on $(hostname)"
echo "Start time: $(date)"

# Print current working directory
echo "Current directory: $(pwd)"

# Go to the directory where venv is
cd /eos/user/e/eballabe

#source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh
#source /cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt/setup.sh

source venv_QNN/bin/activate

which python3
python3 --version
pip list

cd /eos/user/e/eballabe/Quantum/qml
python3 main.py

echo "End time: $(date)"
