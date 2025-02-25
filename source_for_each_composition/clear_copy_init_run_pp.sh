#!/bin/bash

python3 t_tilde_max_adaptor.py

mkdir overal_pp

rm -rf "overal_pp"/*

# Read the value of N from the N_runs.csv file
N=$(<N_runs.csv)


declare -i N0=1

# Create directories
for i in $(seq $N0 $N); do
    mkdir -p "run_$i"
done

# Clear the folder contents
for i in $(seq $N0 $N); do
    rm -rf "run_$i"/*
done

# Copy all contents from source into the run_$i folders
for i in $(seq $N0 $N); do
    cp -r source/* "run_$i"
done

# Run the Python initialization script
python3 init_cell_number_maker_v2.py

