#!/bin/bash

# Function to run a file in a new terminal with a custom title
run_file_in_terminal() {
  subfolder=$1
  file=$2
  title=$3
  # Open a new terminal window in the specified subfolder and run the file with a custom title
  gnome-terminal --working-directory="$PWD/$subfolder" --title="$title" -- sh -c "echo "$PWD/$subfolder"; ./$file; exit"
}

# first runs all WT stuff in this terminal and waits until it ends
cd WT
echo "WT"
./do_all_over_runs.sh
sleep 5
cd ..
# first runs all WT stuff in this terminal and waits until it ends


# then opens a new terminal for mix, and does all the stuff of it there
folder="mix"
title="mix"
run_file_in_terminal "$folder" "./do_all_over_runs.sh" "$title"
sleep 5
# then opens a new terminal for mix, and does all the stuff of it there

# third, runs all C stuff in this terminal and waits until it ends
cd C
echo "C"
./do_all_over_runs.sh
sleep 5
cd ..
# third, runs all C stuff in this terminal and waits until it ends
