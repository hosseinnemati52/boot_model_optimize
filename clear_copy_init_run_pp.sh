#!/bin/bash

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
python3 init_cell_number_maker.py

# Run the initialization executable in each directory
#for i in $(seq $N0 $N); do
#    cd "run_$i"
#    ./initialization.sh
#    cd ..
#done

#./terminal_opener.sh


# Open terminals and run scripts
#./terminal_opener.sh

# Wait for all terminals to close
#sleep 4m

# Run the final Python script after all terminals are closed
#python3 org_pp_over_runs_v5.py
