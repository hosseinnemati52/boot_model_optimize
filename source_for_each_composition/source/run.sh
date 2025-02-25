#!/bin/bash
condition=$(cat eq_condition.txt)

if [[ "$condition" -eq 1 ]]; then
    ./Organoid_boot norm
else
    # Log an error to a file
    echo "Error: Not equilibrated. Condition=$condition" >> error_log.txt
fi

