#!/bin/bash

# Function to run a file in a new terminal with a custom title
run_file_in_terminal() {
  subfolder=$1
  file=$2
  title=$3
  # Open a new terminal window in the specified subfolder and run the file with a custom title
  gnome-terminal --working-directory="$PWD/$subfolder" --title="$title" -- sh -c "echo "$PWD/$subfolder"; ./$file; exit"
}

folder="minus_dh"
title="minus_dh"
run_file_in_terminal "$folder" "./threefold_caller.sh" "$title"
sleep 5

exit
