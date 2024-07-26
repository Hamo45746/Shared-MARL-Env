#!/bin/bash

# Run memray to profile memory usage
python TA_autoencoder/memory_profile_autoencoder.py

# After interruption, generate memray visualizations
# Use a wildcard to match the generated .bin file(s)
for file in *.bin; do
    if [ -f "$file" ]; then
        base_name=$(basename "$file" .bin)
        
        echo "Generating flamegraph for $file"
        memray flamegraph "$file" -o "${base_name}_flame.html"
        
        echo "Generating table for $file"
        memray table "$file" -o "${base_name}_table.txt"
        
        echo "Generating summary for $file"
        memray summary "$file" -o "${base_name}_summary.txt"
    fi
done