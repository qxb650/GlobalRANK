#!/bin/bash

curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh

bash Anaconda3-2025.06-1-Linux-x86_64.sh -b

echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc

eval "$(~/anaconda3/bin/conda shell.bash hook)"

pip install EconModel Consav
pip install -U "jax[cuda12]"
pip install optimistix
pip install flax
pip install optax
pip install orbax-checkpoint
pip install nvidia-ml-py
pip install line_profiler
pip install papermill

exec bash