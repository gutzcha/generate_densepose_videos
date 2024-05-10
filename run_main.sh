#!/bin/bash

# Set up Conda environment
source /opt/conda/etc/profile.d/conda.sh  # Adjust the path as necessary
conda activate dense

# Run your job
python /lustre/home/ygoussha/one-click-dense-pose/one-click-dense-pose/utils/main.py
