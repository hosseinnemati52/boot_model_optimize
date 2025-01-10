#!/bin/bash

for i in {8..30}; do
    old_name="run_$i"
    new_name="run_$((i + 30))"
    
    if [ -d "$old_name" ]; then
        mv "$old_name" "$new_name"
        echo "Renamed $old_name to $new_name"
    else
        echo "$old_name does not exist."
    fi
done

