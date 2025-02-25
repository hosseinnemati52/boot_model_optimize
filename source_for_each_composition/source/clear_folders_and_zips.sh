#!/bin/bash

# Ensure the script exits on error
set -e

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#echo "Clearing all folders and matching zip files in directory: $SCRIPT_DIR"

# Remove all folders in the directory
for item in "$SCRIPT_DIR"/*; do
    if [ -d "$item" ]; then
        #echo "Deleting folder: $item"
        rm -rf "$item"
    fi
done

# Remove all zip files matching the pattern
for zip_file in "$SCRIPT_DIR"/data*.zip; do
    if [ -f "$zip_file" ]; then
        #echo "Deleting zip file: $zip_file"
        rm -f "$zip_file"
    fi
done

#echo "All folders and matching zip files cleared!"

