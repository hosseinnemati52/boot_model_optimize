#!/bin/bash

read -r composition < compos.txt
if [ "$composition" = "WT" ]; then

	condition=1  # no spaces around '=' in Bash variable assignments

	while [ "$condition" -eq 1 ]
	do
	    # Run your scripts
	    ./clear_copy_init_run_pp.sh
	    ./terminal_opener.sh
	    ./pp_plus_cost.sh

	    # Read a text file into a variable (e.g., 'result.txt')
	    # Make sure 'result.txt' exists or handle errors as needed
	    string=$(< t_sufficiency.txt)

	    # Check if the string read from the file is "success"
	    if [ "$string" = "success" ]; then
		condition=0
	    fi
	done

else

	    ./clear_copy_init_run_pp.sh
	    ./terminal_opener.sh
	    ./pp_plus_cost.sh
fi
