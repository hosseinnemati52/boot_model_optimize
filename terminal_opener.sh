#!/bin/bash

# Function to run a file in a new terminal with a custom title
run_file_in_terminal() {
  subfolder=$1
  file=$2
  title=$3

  # Open a new terminal window in the specified subfolder and run the file with a custom title
  gnome-terminal --working-directory="$PWD/$subfolder" --title="$title" -- sh -c "./$file; exit"
}

N=$(<N_runs.csv)  # Total number of runs from the CSV file
MAX_TERMINALS=20  # Number of terminals to open at a time
THRESHOLD=5       # Minimum number of terminals to trigger new batch
INTERVAL=60       # Time interval to check in seconds
started=0         # Counter for started scripts

# Function to count open gnome-terminal instances
count_open_terminals() {
  ls /dev/pts/  | wc -l
}

# Main loop
while (( started < N )); do
  # Count currently open terminals
  open_terminals=$(count_open_terminals)
	
  echo $(count_open_terminals)
  
  # Only proceed if the number of open terminals is below the threshold
  if (( open_terminals < THRESHOLD )); then
    remaining=$((N - started))      # Tasks remaining
    to_start=$((MAX_TERMINALS))    # Number of terminals to start in this batch

    # Ensure not to exceed the total number of tasks
    if (( remaining < MAX_TERMINALS )); then
      to_start=$remaining
    fi

    # Start the next batch of terminals
    for (( i = 0; i < to_start; i++ )); do
      folder="run_$((started + 1))"
      title="Terminal $((started + 1))"
      run_file_in_terminal "$folder" "./do_all.sh" "$title" &
      started=$((started + 1))
    done
  fi

  # Wait for the specified interval before checking again
  sleep $INTERVAL
done

# Wait for all terminals to finish
wait

